import argparse
import json
import math
import random
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from data_provider.data_provider import get_test, get_train
from models.jepas.mmd_jepa_1.denoiser import Denoiser
from utils.utils_drift import count_parameters

try:
    import wandb
except ImportError:
    wandb = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_int_list(value: Optional[str]) -> Optional[List[int]]:
    if value is None or value == "":
        return None
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected a comma-separated list of integers.")
    return values


def load_dataset_configs(args: argparse.Namespace) -> List[Dict[str, Any]]:

    config = {
        "name": args.dataset_name,
        "data": args.data,
        "datasets_dir": args.datasets_dir,
        "seq_len": args.ts_seq_len,
    }
    if args.rel_path is not None or args.rel_path_train is not None:
        config["rel_path"] = args.rel_path or args.rel_path_train
    if args.input_channels is not None:
        config["input_channels"] = args.input_channels
    if args.rel_path_train is not None:
        config["rel_path_train"] = args.rel_path_train
    if args.rel_path_valid is not None:
        config["rel_path_valid"] = args.rel_path_valid
    return [config]


def extract_time_series(sample: Any) -> torch.Tensor:
    if isinstance(sample, (list, tuple)):
        sample = sample[0]
    if not torch.is_tensor(sample):
        sample = torch.as_tensor(sample, dtype=torch.float32)
    sample = sample.to(torch.float32)
    if sample.ndim == 1:
        sample = sample.unsqueeze(-1)
    if sample.ndim != 2:
        raise ValueError(f"Expected one sample with shape (T, C), got {tuple(sample.shape)}.")
    return sample


def resize_time_series(sample: torch.Tensor, seq_len: int, channels: Optional[int]) -> torch.Tensor:
    sample = sample[:seq_len]
    if sample.shape[0] < seq_len:
        sample = F.pad(sample, (0, 0, 0, seq_len - sample.shape[0]))
    if channels is not None:
        sample = sample[:, :channels]
        if sample.shape[1] < channels:
            sample = F.pad(sample, (0, channels - sample.shape[1], 0, 0))
    return sample


def infer_channels(dataset: Dataset) -> int:
    if len(dataset) == 0:
        raise ValueError("Cannot infer channel count from an empty dataset.")
    return int(extract_time_series(dataset[0]).shape[-1])


def fit_minmax_stats(dataset: Dataset, seq_len: int, channels: int) -> Tuple[torch.Tensor, torch.Tensor]:
    data_min = None
    data_max = None
    for idx in range(len(dataset)):
        sample = resize_time_series(extract_time_series(dataset[idx]), seq_len, channels)
        sample_min = sample.amin(dim=0)
        sample_max = sample.amax(dim=0)
        data_min = sample_min if data_min is None else torch.minimum(data_min, sample_min)
        data_max = sample_max if data_max is None else torch.maximum(data_max, sample_max)
    if data_min is None or data_max is None:
        raise ValueError("Cannot fit min-max statistics on an empty dataset.")
    return data_min, data_max


class NormalizedTimeSeriesDataset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        seq_len: int,
        channels: int,
        data_min: torch.Tensor,
        data_max: torch.Tensor,
        normalize: bool = True,
    ):
        self.base_dataset = base_dataset
        self.seq_len = seq_len
        self.channels = channels
        self.data_min = data_min.to(torch.float32)
        self.data_max = data_max.to(torch.float32)
        self.normalize = normalize
        self.denom = torch.clamp(self.data_max - self.data_min, min=1e-6)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        sample = resize_time_series(
            extract_time_series(self.base_dataset[idx]),
            self.seq_len,
            self.channels,
        )
        if self.normalize:
            sample = torch.clamp((sample - self.data_min) / self.denom, 0.0, 1.0)
            sample = sample * 2.0 - 1.0
        return sample


def load_data(args: argparse.Namespace) -> Tuple[Dataset, Dataset, int]:
    configs = load_dataset_configs(args)
    raw_trainsets = []
    raw_valsets = []
    max_channels = 0

    for config in configs:
        config = dict(config)
        config["seq_len"] = args.ts_seq_len
        config.setdefault("datasets_dir", args.datasets_dir)
        trainset = get_train(config.copy())
        valset = get_test(config.copy())
        raw_trainsets.append((config, trainset))
        raw_valsets.append((config, valset))
        max_channels = max(max_channels, infer_channels(trainset), infer_channels(valset))
        print(
            f"{config['name']} | train={len(trainset)} | val={len(valset)} | "
            f"channels={infer_channels(trainset)}"
        )

    channels = args.input_channels or max_channels
    trainsets = []
    valsets = []
    for (_, trainset), (_, valset) in zip(raw_trainsets, raw_valsets):
        data_min, data_max = fit_minmax_stats(trainset, args.ts_seq_len, channels)
        trainsets.append(
            NormalizedTimeSeriesDataset(
                trainset,
                seq_len=args.ts_seq_len,
                channels=channels,
                data_min=data_min,
                data_max=data_max,
                normalize=not args.no_normalize,
            )
        )
        valsets.append(
            NormalizedTimeSeriesDataset(
                valset,
                seq_len=args.ts_seq_len,
                channels=channels,
                data_min=data_min,
                data_max=data_max,
                normalize=not args.no_normalize,
            )
        )

    train_dataset = trainsets[0] if len(trainsets) == 1 else ConcatDataset(trainsets)
    val_dataset = valsets[0] if len(valsets) == 1 else ConcatDataset(valsets)
    return train_dataset, val_dataset, channels


def resolve_modality_splits(args: argparse.Namespace, total_channels: int) -> List[int]:
    splits = parse_int_list(args.modality_channel_splits)
    if splits is not None:
        if sum(splits) != total_channels:
            raise ValueError(
                f"--modality_channel_splits sums to {sum(splits)}, but data has {total_channels} channels."
            )
        return splits

    if args.num_modalities is None:
        return [1] * total_channels
    if total_channels % args.num_modalities != 0:
        raise ValueError(
            f"Cannot evenly split {total_channels} channels into {args.num_modalities} modalities. "
            "Use --modality_channel_splits to specify the split explicitly."
        )
    return [total_channels // args.num_modalities] * args.num_modalities


def batch_to_modalities(batch: Any, splits: Sequence[int], device: torch.device) -> List[torch.Tensor]:
    x = batch[0] if isinstance(batch, (list, tuple)) else batch
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=torch.float32)
    x = x.to(device=device, dtype=torch.float32)
    if x.ndim == 2:
        x = x.unsqueeze(-1)
    if x.ndim != 3:
        raise ValueError(f"Expected batch shape (B, T, C), got {tuple(x.shape)}.")

    modalities = []
    start = 0
    for width in splits:
        end = start + width
        modalities.append(x[:, :, start:end].contiguous())
        start = end
    return modalities


def delay_image_size(seq_len: int, delay: int, embedding: int) -> int:
    cols = 0
    while cols * delay + embedding <= seq_len:
        cols += 1
    if cols < embedding and cols * delay != seq_len and cols * delay + embedding > seq_len:
        cols += 1
    return max(embedding, max(cols, 1))


def build_model_args(args: argparse.Namespace, splits: Sequence[int]) -> SimpleNamespace:
    if len(set(splits)) != 1:
        raise ValueError(
            "The current JiT diffusion target has one fixed in_channels value, so every modality "
            "must have the same channel width."
        )

    img_size = args.img_size
    if img_size is None:
        img_size = delay_image_size(args.ts_seq_len, args.ts_delay, args.ts_embedding)

    return SimpleNamespace(
        num_modalities=len(splits),
        jepa_input_dims=list(splits),
        ts_seq_len=args.ts_seq_len,
        ts_delay=args.ts_delay,
        ts_embedding=args.ts_embedding,
        device=args.device,
        jepa_hidden_size=args.jepa_hidden_size,
        jepa_encoder_layers=args.jepa_encoder_layers,
        jepa_embed_dim=args.jepa_embed_dim,
        jepa_latent_downsample=args.jepa_latent_downsample,
        jepa_encoder_dropout=args.jepa_encoder_dropout,
        jepa_predictor_dim=args.jepa_predictor_dim,
        jepa_predictor_layers=args.jepa_predictor_layers,
        jepa_predictor_heads=args.jepa_predictor_heads,
        jepa_predictor_mlp_ratio=args.jepa_predictor_mlp_ratio,
        jepa_predictor_dropout=args.jepa_predictor_dropout,
        jepa_ema_momentum=args.jepa_ema_momentum,
        jepa_max_len=args.jepa_max_len,
        hidden_channels=args.hidden_channels,
        img_size=img_size,
        patch_size=args.patch_size,
        in_channels=splits[0],
        depth=args.depth,
        num_heads=args.num_heads,
        attn_dropout=args.attn_dropout,
        proj_dropout=args.proj_dropout,
        bottleneck_dim=args.bottleneck_dim,
        in_context_start=args.in_context_start,
        P_mean=args.P_mean,
        P_std=args.P_std,
        t_eps=args.t_eps,
        noise_scale=args.noise_scale,
        ema_decay1=args.ema_decay1,
        ema_decay2=args.ema_decay2,
        sampling_method=args.sampling_method,
        num_sampling_steps=args.num_sampling_steps,
        cfg=args.cfg,
        interval_min=args.interval_min,
        interval_max=args.interval_max,
    )


def unpack_model_loss(output: Any) -> Tuple[torch.Tensor, Dict[str, float]]:
    if torch.is_tensor(output):
        return output, {"loss": float(output.detach().item())}
    if isinstance(output, (list, tuple)):
        loss = output[0]
        metrics = {"loss": float(loss.detach().item())}
        if len(output) > 1:
            metrics["diffusion_loss"] = float(output[1])
        if len(output) > 2:
            metrics["jepa_loss"] = float(output[2])
        return loss, metrics
    raise TypeError(f"Unsupported model output type: {type(output)}")


@torch.no_grad()
def evaluate_loss(
    model: Denoiser,
    loader: DataLoader,
    splits: Sequence[int],
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    totals: Dict[str, float] = {}
    n = 0
    for batch in loader:
        modalities = batch_to_modalities(batch, splits, device)
        _, metrics = unpack_model_loss(model(modalities))
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        n += 1
    return {key: value / max(n, 1) for key, value in totals.items()}


@torch.no_grad()
def evaluate_imputation(
    model: Denoiser,
    loader: DataLoader,
    splits: Sequence[int],
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    sum_sq_by_modality = torch.zeros(model.num_modalities, dtype=torch.float64)
    sum_abs_by_modality = torch.zeros(model.num_modalities, dtype=torch.float64)
    count_by_modality = torch.zeros(model.num_modalities, dtype=torch.float64)

    for batch in loader:
        modalities = batch_to_modalities(batch, splits, device)
        bsz = modalities[0].shape[0]
        normalized_modalities = model._jepa_modalities(modalities)

        for modality_idx in range(model.num_modalities):
            missing_modalities = torch.full(
                (bsz,),
                modality_idx,
                dtype=torch.long,
                device=device,
            )
            target_img, _ = model._select_diffusion_target(
                normalized_modalities,
                missing_modalities,
            )
            generated_img = model.generate(
                modalities=modalities,
                missing_modalities=missing_modalities,
            )

            generated_ts = model.delay_embedder.img_to_ts(generated_img)
            target_ts = model.delay_embedder.img_to_ts(target_img)
            diff = generated_ts - target_ts

            sum_sq_by_modality[modality_idx] += diff.square().sum().double().cpu()
            sum_abs_by_modality[modality_idx] += diff.abs().sum().double().cpu()
            count_by_modality[modality_idx] += diff.numel()

    metrics = {}
    total_sq = float(sum_sq_by_modality.sum().item())
    total_abs = float(sum_abs_by_modality.sum().item())
    total_count = max(float(count_by_modality.sum().item()), 1.0)
    metrics["mse"] = total_sq / total_count
    metrics["mae"] = total_abs / total_count

    for modality_idx in range(model.num_modalities):
        count = max(float(count_by_modality[modality_idx].item()), 1.0)
        metrics[f"modality_{modality_idx}/mse"] = float(sum_sq_by_modality[modality_idx].item()) / count
        metrics[f"modality_{modality_idx}/mae"] = float(sum_abs_by_modality[modality_idx].item()) / count
    return metrics


def save_checkpoint(
    path: Path,
    model: Denoiser,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    args: argparse.Namespace,
    splits: Sequence[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_val_loss": best_val_loss,
            "args": vars(args),
            "modality_channel_splits": list(splits),
        },
        path,
    )


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    args.device = str(device)
    print(f"Using device: {device}")

    train_dataset, val_dataset, total_channels = load_data(args)
    splits = resolve_modality_splits(args, total_channels)
    print(f"Resolved modality channel splits: {splits}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
    )

    model = Denoiser(build_model_args(args, splits)).to(device)
    print(f"Trainable parameters: {count_parameters(model):,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wb = None
    if not args.disable_wandb:
        if wandb is None:
            raise ImportError("wandb is not installed. Install wandb or pass --disable_wandb.")
        wb = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            config={
                **vars(args),
                "resolved_num_modalities": len(splits),
                "resolved_modality_channel_splits": list(splits),
            },
            dir=str(output_dir),
        )

    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        totals: Dict[str, float] = {}
        n_steps = 0

        for batch in train_loader:
            modalities = batch_to_modalities(batch, splits, device)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = unpack_model_loss(model(modalities))
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            model.jepa.update_target_encoders()

            n_steps += 1
            global_step += 1
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value

        train_metrics = {key: value / max(n_steps, 1) for key, value in totals.items()}
        val_metrics = evaluate_loss(model, val_loader, splits, device)
        val_loss = val_metrics.get("loss", float("inf"))
        is_best = val_loss <= best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        elapsed = time.time() - epoch_start
        print(
            f"Epoch {epoch + 1} done in {elapsed:.1f}s | "
            f"train_loss={train_metrics.get('loss', math.nan):.6f} | "
            f"val_loss={val_loss:.6f} | best_val_loss={best_val_loss:.6f}"
        )

        epoch_log = {
            "epoch": epoch + 1,
            "epoch/time_sec": elapsed,
            "train/avg_loss": train_metrics.get("loss", math.nan),
            "val/avg_loss": val_loss,
            "val/best_loss": best_val_loss,
            "train/lr": optimizer.param_groups[0]["lr"],
        }
        for key in ("diffusion_loss", "jepa_loss"):
            if key in train_metrics:
                epoch_log[f"train/{key}_epoch"] = train_metrics[key]
            if key in val_metrics:
                epoch_log[f"val/{key}_epoch"] = val_metrics[key]

        if args.imputation_eval_interval > 0 and (epoch + 1) % args.imputation_eval_interval == 0:
            imputation_metrics = evaluate_imputation(
                model,
                val_loader,
                splits,
                device,
            )
            epoch_log.update({f"imputation/{key}": value for key, value in imputation_metrics.items()})
            print(
                f"Imputation | mse={imputation_metrics['mse']:.6f} | "
                f"mae={imputation_metrics['mae']:.6f}"
            )

        if wb is not None:
            wandb.log(epoch_log, step=global_step)

        if is_best:
            save_checkpoint(
                output_dir / "checkpoint_best.pt",
                model,
                optimizer,
                epoch,
                global_step,
                best_val_loss,
                args,
                splits,
            )

        if args.save_interval > 0 and (epoch + 1) % args.save_interval == 0:
            save_checkpoint(
                output_dir / f"checkpoint_epoch{epoch + 1}.pt",
                model,
                optimizer,
                epoch,
                global_step,
                best_val_loss,
                args,
                splits,
            )

    save_checkpoint(
        output_dir / "checkpoint_final.pt",
        model,
        optimizer,
        args.epochs - 1,
        global_step,
        best_val_loss,
        args,
        splits,
    )
    if wb is not None:
        wb.finish()
    print(f"Training complete. Final checkpoint saved to {output_dir / 'checkpoint_final.pt'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train multimodal time-series JEPA + JiT imputation model.")

    parser.add_argument("--dataset_name", type=str, default="dataset")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--datasets_dir", type=str, required=True)
    parser.add_argument("--rel_path", type=str, default=None)
    parser.add_argument("--rel_path_train", type=str, default=None)
    parser.add_argument("--rel_path_valid", type=str, default=None)

    parser.add_argument("--output_dir", type=str, default="./outputs/mm_jepa_jit")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--save_interval", type=int, default=10)

    parser.add_argument("--input_channels", type=int, default=None)
    parser.add_argument("--num_modalities", type=int, default=None)
    parser.add_argument(
        "--modality_channel_splits",
        type=str,
        default=None,
        help="Comma-separated modality channel widths. Default: each channel is one modality.",
    )
    parser.add_argument("--no_normalize", action="store_true")

    parser.add_argument("--ts_seq_len", type=int, default=128)
    parser.add_argument("--ts_delay", type=int, default=12)
    parser.add_argument("--ts_embedding", type=int, default=12)

    parser.add_argument("--jepa_hidden_size", type=int, default=64)
    parser.add_argument("--jepa_encoder_layers", type=int, default=2)
    parser.add_argument("--jepa_embed_dim", type=int, default=64)
    parser.add_argument("--jepa_latent_downsample", type=int, default=8)
    parser.add_argument("--jepa_encoder_dropout", type=float, default=0.0)
    parser.add_argument("--jepa_predictor_dim", type=int, default=128)
    parser.add_argument("--jepa_predictor_layers", type=int, default=2)
    parser.add_argument("--jepa_predictor_heads", type=int, default=4)
    parser.add_argument("--jepa_predictor_mlp_ratio", type=float, default=4.0)
    parser.add_argument("--jepa_predictor_dropout", type=float, default=0.0)
    parser.add_argument("--jepa_ema_momentum", type=float, default=0.996)
    parser.add_argument("--jepa_max_len", type=int, default=10000)

    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--patch_size", type=int, default=2)
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--attn_dropout", type=float, default=0.0)
    parser.add_argument("--proj_dropout", type=float, default=0.0)
    parser.add_argument("--bottleneck_dim", type=int, default=64)
    parser.add_argument("--in_context_start", type=int, default=0)
    parser.add_argument("--P_mean", type=float, default=-1.2)
    parser.add_argument("--P_std", type=float, default=1.2)
    parser.add_argument("--t_eps", type=float, default=1e-5)
    parser.add_argument("--noise_scale", type=float, default=1.0)
    parser.add_argument("--ema_decay1", type=float, default=0.999)
    parser.add_argument("--ema_decay2", type=float, default=0.9999)
    parser.add_argument("--sampling_method", type=str, default="euler", choices=["euler", "heun"])
    parser.add_argument("--num_sampling_steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--interval_min", type=float, default=0.0)
    parser.add_argument("--interval_max", type=float, default=1.0)

    parser.add_argument("--imputation_eval_interval", type=int, default=5)

    parser.add_argument("--disable_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="mm-jepa-jit")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default=None, choices=["online", "offline", "disabled"])

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
