import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.vqvae_simple import VQVAE1D
from utils.utils_dataset import get_dataset


try:
    import wandb
except ImportError:
    wandb = None


class CalorieVQVAEDataset(Dataset):
    """Expose normalized calorie sequences from the glucose/calorie dataset."""

    def __init__(self, base_dataset: Dataset, interpolate_to_glucose_time: bool = True):
        self.base_dataset = base_dataset
        self.interpolate_to_glucose_time = interpolate_to_glucose_time

    def __len__(self) -> int:
        return len(self.base_dataset)

    @staticmethod
    def _interpolate_to_glucose_time(item: dict) -> tuple[torch.Tensor, torch.Tensor]:
        activity_mask = item["activity_mask"].bool().squeeze(-1)
        glucose_time = item["glucose_time_local"]

        if not activity_mask.any():
            x = torch.zeros(glucose_time.numel(), 1, dtype=torch.float32)
            mask = torch.zeros_like(x)
            return x, mask

        source_time = item["activity_time_local"][activity_mask].cpu().numpy().astype(np.float64)
        source_values = (
            item["activity_calorie"][activity_mask]
            .squeeze(-1)
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        target_time = glucose_time.cpu().numpy().astype(np.float64)
        interpolated = np.interp(target_time, source_time, source_values).astype(np.float32)

        x = torch.from_numpy(interpolated).view(-1, 1)
        mask = torch.ones_like(x)
        return x, mask

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.base_dataset[idx]
        if self.interpolate_to_glucose_time:
            x, mask = self._interpolate_to_glucose_time(item)
            return {
                "x": x,
                "mask": mask,
            }

        return {
            "x": item["activity_calorie"].float(),
            "mask": item["activity_mask"].float(),
        }


def parse_int_tuple(value: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def masked_mse(x_hat: torch.Tensor, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum().clamp_min(1.0) * x.size(-1)
    return ((x_hat - x).square() * mask).sum() / denom


def codebook_usage(ids: torch.Tensor, num_codes: int) -> tuple[int, float]:
    counts = torch.bincount(ids.reshape(-1).detach().cpu(), minlength=num_codes).float()
    probs = counts / counts.sum().clamp_min(1.0)
    used = int((counts > 0).sum().item())
    entropy = -(probs[probs > 0] * torch.log(probs[probs > 0])).sum()
    perplexity = float(torch.exp(entropy).item())
    return used, perplexity


def make_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, dict]:
    min_activity_events = args.min_activity_events
    if min_activity_events is None:
        min_activity_events = 0 if args.allow_empty_activity else 1

    dataset_cfg = {
        "ts_seq_len": args.seq_len,
        "ts_delay": args.ts_delay,
        "ts_embedding": args.ts_embedding,
        "ts_stride": args.stride,
        "max_activity_events": max(args.max_activity_events, args.seq_len)
        if args.interpolate_to_glucose_time
        else args.max_activity_events,
        "require_activity": not args.allow_empty_activity,
        "min_activity_events": min_activity_events,
        "return_dict": True,
    }
    train_base, val_base = get_dataset(
        "glucose_calorie_imputation",
        dataset_cfg,
        root=args.data_root,
        seed=args.seed,
    )
    train_dataset = CalorieVQVAEDataset(
        train_base,
        interpolate_to_glucose_time=args.interpolate_to_glucose_time,
    )
    val_dataset = CalorieVQVAEDataset(
        val_base,
        interpolate_to_glucose_time=args.interpolate_to_glucose_time,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=args.drop_last,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    metadata = {
        "calorie_min": getattr(train_base, "calorie_min", None),
        "calorie_max": getattr(train_base, "calorie_max", None),
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "vqvae_seq_len": args.seq_len
        if args.interpolate_to_glucose_time
        else args.max_activity_events,
        "interpolate_to_glucose_time": args.interpolate_to_glucose_time,
        "min_activity_events": min_activity_events,
    }
    return train_loader, val_loader, metadata


def make_model(args: argparse.Namespace) -> VQVAE1D:
    model = VQVAE1D(
        in_channels=1,
        encoder_channels=args.encoder_channels,
        decoder_channels=args.decoder_channels,
        code_dim=args.code_dim,
        num_codes=args.num_codes,
        down_ratio=args.down_ratio,
        up_ratio=args.up_ratio,
        code_len=args.code_len,
        seq_len=args.seq_len if args.interpolate_to_glucose_time else args.max_activity_events,
    )
    model.quantizer.beta = args.beta
    return model


def run_epoch(
    model: VQVAE1D,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    recon_weight: float,
    grad_clip: float,
    num_codes: int,
    max_batches: int | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)

    totals = {
        "loss": 0.0,
        "recon_loss": 0.0,
        "vq_loss": 0.0,
        "used_codes": 0.0,
        "perplexity": 0.0,
    }
    n_batches = 0

    pbar = tqdm(
        loader,
        desc="train" if training else "val",
        leave=False,
        mininterval=1.0,
        dynamic_ncols=True,
    )
    for batch in pbar:
        x = batch["x"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        import matplotlib
        matplotlib.use("TkAgg")  # 或者 MacOSX
        import matplotlib.pyplot as plt
        for i_plot, (x_i, mask_i) in enumerate(zip(x, mask)):
            plt.plot(mask_i, label="mask")
            plt.plot(x_i, label="calorie")
            plt.legend()
            plt.show()
            if i_plot > 10:
                break

        with torch.set_grad_enabled(training):
            x_hat, ids, vq_loss = model(x, mask)
            recon_loss = masked_mse(x_hat, x, mask)
            loss = recon_weight * recon_loss + vq_loss

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        used_codes, perplexity = codebook_usage(ids, num_codes)
        totals["loss"] += float(loss.item())
        totals["recon_loss"] += float(recon_loss.item())
        totals["vq_loss"] += float(vq_loss.item())
        totals["used_codes"] += float(used_codes)
        totals["perplexity"] += perplexity
        n_batches += 1

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            recon=f"{recon_loss.item():.4f}",
            vq=f"{vq_loss.item():.4f}",
            codes=used_codes,
        )

        if max_batches is not None and n_batches >= max_batches:
            break

    return {key: value / max(n_batches, 1) for key, value in totals.items()}


def save_checkpoint(
    path: Path,
    model: VQVAE1D,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    args: argparse.Namespace,
    dataset_metadata: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "args": vars(args),
            "dataset_metadata": dataset_metadata,
        },
        path,
    )


def log_metrics(prefix: str, metrics: dict[str, float], epoch: int) -> None:
    pieces = [f"{key}: {value:.6f}" for key, value in metrics.items()]
    print(f"{prefix} epoch {epoch}: " + " | ".join(pieces))


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a 1D VQ-VAE on calorie sequences from the glucose imputation dataset."
    )

    parser.add_argument("--data_root", type=str, default="./AI-READI")
    parser.add_argument("--save_dir", type=str, default="./outputs/vqvae_calorie")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--max_activity_events", type=int, default=128)
    parser.add_argument("--ts_delay", type=int, default=12)
    parser.add_argument("--ts_embedding", type=int, default=12)
    parser.add_argument("--allow_empty_activity", action="store_true")
    parser.add_argument(
        "--min_activity_events",
        type=int,
        default=60,
        help="Only keep glucose windows with at least this many raw calorie events.",
    )
    parser.add_argument(
        "--no_interpolate_to_glucose_time",
        dest="interpolate_to_glucose_time",
        action="store_false",
        help="Use sparse padded activity events instead of interpolating calorie onto glucose timestamps.",
    )
    parser.set_defaults(interpolate_to_glucose_time=True)

    parser.add_argument("--encoder_channels", type=parse_int_tuple, default=(64, 64, 64))
    parser.add_argument("--decoder_channels", type=parse_int_tuple, default=(64, 64, 64))
    parser.add_argument("--code_dim", type=int, default=8)
    parser.add_argument("--num_codes", type=int, default=1000)
    parser.add_argument("--code_len", type=int, default=4)
    parser.add_argument("--down_ratio", type=int, default=2)
    parser.add_argument("--up_ratio", type=int, default=2)
    parser.add_argument("--beta", type=float, default=0.25)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--recon_weight", type=float, default=1.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--drop_last", action="store_true")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="vqvae-calorie")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    return parser.parse_args()


def main() -> None:
    args = get_args()
    set_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    train_loader, val_loader, dataset_metadata = make_loaders(args)
    model = make_model(args).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    if args.wandb:
        if wandb is None:
            raise ImportError("wandb is not installed")
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={**vars(args), **dataset_metadata},
        )

    print(f"device: {device}")
    print(f"train windows: {dataset_metadata['train_size']} | val windows: {dataset_metadata['val_size']}")
    print(f"vqvae sequence length: {dataset_metadata['vqvae_seq_len']}")
    print(f"interpolate calorie to glucose time: {dataset_metadata['interpolate_to_glucose_time']}")
    print(f"calorie range: [{dataset_metadata['calorie_min']}, {dataset_metadata['calorie_max']}]")

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            args.recon_weight,
            args.grad_clip,
            args.num_codes,
            args.max_train_batches,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            None,
            args.recon_weight,
            args.grad_clip,
            args.num_codes,
            args.max_val_batches,
        )

        log_metrics("train", train_metrics, epoch)
        log_metrics("val", val_metrics, epoch)

        if args.wandb:
            wandb.log(
                {
                    **{f"train/{k}": v for k, v in train_metrics.items()},
                    **{f"val/{k}": v for k, v in val_metrics.items()},
                    "epoch": epoch,
                }
            )

        save_checkpoint(
            save_dir / "last.pt",
            model,
            optimizer,
            epoch,
            best_val_loss,
            args,
            dataset_metadata,
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(
                save_dir / "best.pt",
                model,
                optimizer,
                epoch,
                best_val_loss,
                args,
                dataset_metadata,
            )
            print(f"saved best checkpoint: {save_dir / 'best.pt'}")

        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(
                save_dir / f"epoch_{epoch:04d}.pt",
                model,
                optimizer,
                epoch,
                best_val_loss,
                args,
                dataset_metadata,
            )

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
