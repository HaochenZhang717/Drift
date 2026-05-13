import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from data_provider.data_provider import get_test, get_train
from img_transformations import DelayEmbedder
from models.soft_vqvae import SoftVQVAE

try:
    import wandb
except ImportError:
    wandb = None


def get_args():
    parser = argparse.ArgumentParser(
        description="Train a CNN SoftVQ-VAE time-series tokenizer on benchmark datasets."
    )

    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--datasets_dir", type=str, required=True)
    parser.add_argument("--rel_path", type=str, default=None)
    parser.add_argument("--rel_path_train", type=str, default=None)
    parser.add_argument("--rel_path_valid", type=str, default=None)
    parser.add_argument("--ts_seq_len", type=int, default=256)
    parser.add_argument("--window_stride", type=int, default=None)
    parser.add_argument("--ts_stride", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--train_split", type=str, default="train", choices=["train"])
    parser.add_argument("--val_split", type=str, default="test", choices=["test"])

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--recon_weight", type=float, default=1.0)
    parser.add_argument("--ts_recon_weight", type=float, default=1.0)
    parser.add_argument("--kl_weight", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--drop_last", action="store_true")
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)

    parser.add_argument("--delay", type=int, default=12)
    parser.add_argument("--embedding", type=int, default=12)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--code_dim", type=int, default=32)
    parser.add_argument("--num_codes", type=int, default=512)
    parser.add_argument("--ch_mult", type=parse_int_tuple, default=(1, 2, 4))
    parser.add_argument("--attn_resolutions", type=parse_int_tuple, default=())
    parser.add_argument("--attn_type", type=str, default="vanilla", choices=["vanilla", "linear", "none"])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--learnable_temperature", action="store_true")
    parser.add_argument("--l2_norm", action="store_true")
    parser.add_argument("--tanh_out", action="store_true")
    parser.add_argument("--one_channel", action="store_true")
    parser.add_argument("--ts_loss_type", type=str, default="l2", choices=["l1", "l2"])

    parser.add_argument("--save_dir", type=str, default="./fid_vae_ckpts/soft_vqvae_benchmark")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--wandb_log_images_every", type=int, default=0)

    args = parser.parse_args()
    if args.rel_path is None and not (args.rel_path_train and args.rel_path_valid):
        parser.error("Provide --rel_path, or provide both --rel_path_train and --rel_path_valid.")
    if args.rel_path is None:
        args.rel_path = args.rel_path_train
    return args


def parse_int_tuple(value: str | tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(value, tuple):
        return value
    values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _extract_series(item):
    if isinstance(item, dict):
        x = item["target"] if "target" in item else next(iter(item.values()))
    elif isinstance(item, (list, tuple)):
        x = item[0]
    else:
        x = item

    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)

    x = x.astype(np.float32)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(f"Expected sample with shape (T, C), got {x.shape}")
    return x


def _fit_minmax_stats(base_dataset, one_channel=False):
    data_min = None
    data_max = None
    for idx in range(len(base_dataset)):
        series = _extract_series(base_dataset[idx])
        tensor = torch.from_numpy(series)
        if one_channel:
            tensor = tensor[:, :1]
        sample_min = tensor.amin(dim=0)
        sample_max = tensor.amax(dim=0)
        data_min = sample_min if data_min is None else torch.minimum(data_min, sample_min)
        data_max = sample_max if data_max is None else torch.maximum(data_max, sample_max)
    if data_min is None or data_max is None:
        raise ValueError("Cannot fit min-max statistics on an empty dataset.")
    return data_min.to(torch.float32), data_max.to(torch.float32)


class BenchmarkTensorDataset(Dataset):
    """Expose benchmark samples as normalized tensors shaped (C, T)."""

    def __init__(self, base_dataset, data_min, data_max, one_channel=False):
        self.base_dataset = base_dataset
        self.data_min = data_min
        self.data_max = data_max
        self.denom = torch.clamp(self.data_max - self.data_min, min=1e-6)
        self.one_channel = one_channel

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        series = _extract_series(self.base_dataset[idx])
        if self.one_channel:
            series = series[:, :1]
        tensor = torch.from_numpy(series).to(torch.float32)
        tensor = torch.clamp((tensor - self.data_min) / self.denom, 0.0, 1.0)
        tensor = tensor * 2.0 - 1.0
        tensor = tensor.permute(1, 0).contiguous()
        return (tensor,)


def _make_dataset_config(args, flag):
    stride = args.window_stride
    if stride is None:
        stride = args.ts_stride
    if stride is None:
        stride = args.stride

    config = {
        "name": args.dataset_name,
        "data": args.data,
        "datasets_dir": args.datasets_dir,
        "rel_path": args.rel_path,
        "path": args.rel_path,
        "seq_len": args.ts_seq_len,
        "flag": flag,
    }
    if args.rel_path_train is not None:
        config["rel_path_train"] = args.rel_path_train
    if args.rel_path_valid is not None:
        config["rel_path_valid"] = args.rel_path_valid
    if stride is not None:
        config["window_stride"] = stride
        config["ts_stride"] = stride
        config["stride"] = stride
    return config


def load_benchmark_datasets(args):
    train_base = get_train(_make_dataset_config(args, args.train_split))
    val_base = get_test(_make_dataset_config(args, args.val_split))
    data_min, data_max = _fit_minmax_stats(train_base, one_channel=args.one_channel)

    train_dataset = BenchmarkTensorDataset(train_base, data_min, data_max, args.one_channel)
    val_dataset = BenchmarkTensorDataset(val_base, data_min, data_max, args.one_channel)
    sample = train_dataset[0][0]
    print(
        f"Loaded benchmark dataset | data={args.data} | rel_path={args.rel_path} | "
        f"train={len(train_dataset)} | val={len(val_dataset)} | sample={tuple(sample.shape)}",
        flush=True,
    )
    return train_dataset, val_dataset, data_min, data_max


def series_batch_to_images(
    series: torch.Tensor,
    embedder: DelayEmbedder,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    # Dataset tensors are (B, C, T); DelayEmbedder expects (B, T, C).
    series_btc = series.permute(0, 2, 1).contiguous()
    images, pad_mask = embedder.ts_to_img(series_btc, pad=True, mask=0, return_pad_mask=True)
    return images, pad_mask


def _delay_num_cols(seq_len: int, delay: int, embedding: int) -> int:
    col = 0
    while (col * delay + embedding) <= seq_len:
        col += 1
    if col < embedding and col * delay != seq_len and col * delay + embedding > seq_len:
        col += 1
    return max(col, 1)


def images_to_series_average_overlap(
    images: torch.Tensor,
    embedder: DelayEmbedder,
    seq_len: int,
) -> torch.Tensor:
    """
    Inverse delay embedding with overlap averaging.
    images: (B, C, H, W) padded square delay-images
    returns: (B, C, T)
    """
    bsz, channels, emb, _ = images.shape
    cols = _delay_num_cols(seq_len=seq_len, delay=embedder.delay, embedding=emb)
    img_non_square = images[:, :, :, :cols]

    series_sum = torch.zeros((bsz, channels, seq_len), dtype=images.dtype, device=images.device)
    series_cnt = torch.zeros_like(series_sum)

    for i in range(cols - 1):
        start = i * embedder.delay
        end = start + emb
        patch = img_non_square[:, :, :, i]
        series_sum[:, :, start:end] += patch
        series_cnt[:, :, start:end] += 1.0

    start = (cols - 1) * embedder.delay
    rem = max(seq_len - start, 1)
    patch_last = img_non_square[:, :, :rem, cols - 1]
    series_sum[:, :, start:] += patch_last
    series_cnt[:, :, start:] += 1.0

    return series_sum / torch.clamp(series_cnt, min=1.0)


def make_image_embedder(args, seq_len: int, device: torch.device) -> DelayEmbedder:
    return DelayEmbedder(device=device, seq_len=seq_len, delay=args.delay, embedding=args.embedding)


def infer_image_shape(args, sample: torch.Tensor, device: torch.device) -> tuple[int, int]:
    embedder = make_image_embedder(args, sample.shape[-1], device)
    image, _ = series_batch_to_images(sample.unsqueeze(0).to(device), embedder)
    if image.shape[-1] != image.shape[-2]:
        raise ValueError(f"Delay image must be square after padding, got {tuple(image.shape)}")
    return int(image.shape[1]), int(image.shape[-1])


def validate_image_shape_args(resolution: int, ch_mult: tuple[int, ...]) -> None:
    latent_downsample = 2 ** (len(ch_mult) - 1)
    if resolution % latent_downsample != 0:
        raise ValueError(
            f"Delay image resolution ({resolution}) must be divisible by backbone downsample "
            f"factor ({latent_downsample}) from --ch_mult={ch_mult}."
        )


def _update_used_codes(mask: torch.Tensor, indices: torch.Tensor) -> None:
    unique = torch.unique(indices.detach().cpu())
    mask[unique] = True


def run_epoch(
    model,
    dataloader,
    optimizer,
    device,
    args,
    training: bool,
    embedder: DelayEmbedder,
):
    model.train(training)
    totals = {
        "loss": 0.0,
        "recon_loss": 0.0,
        "image_recon_loss": 0.0,
        "ts_recon_loss": 0.0,
        "kl_loss": 0.0,
        "perplexity": 0.0,
        "assignment_entropy": 0.0,
        "batch_entropy": 0.0,
        "temperature": 0.0,
    }
    used_mask = torch.zeros(int(model.num_codes), dtype=torch.bool)
    n_batches = 0
    max_batches = args.max_train_batches if training else args.max_val_batches

    desc = "Train" if training else "Val"
    pbar = tqdm(dataloader, desc=desc, file=sys.stdout, dynamic_ncols=True)
    for batch in pbar:
        series = batch[0].to(device, non_blocking=True)
        seq_len = int(series.shape[-1])
        x, pad_mask = series_batch_to_images(series, embedder)
        if pad_mask is None:
            raise RuntimeError("pad_mask is required for masked reconstruction loss.")

        with torch.set_grad_enabled(training):
            out = model(x)
            image_loss_dict = model.loss_function(
                x,
                out["recon"],
                out["kl_loss"],
                recon_weight=args.recon_weight,
                kl_weight=0.0,
                mask=pad_mask,
            )
            recon_series = images_to_series_average_overlap(out["recon"], embedder, seq_len=seq_len)
            if args.ts_loss_type == "l1":
                ts_recon = torch.mean(torch.abs(recon_series - series))
            else:
                ts_recon = torch.mean((recon_series - series) ** 2)

            image_recon = image_loss_dict["recon_loss"]
            kl_loss = image_loss_dict["kl_loss"]
            total_recon = image_recon + args.ts_recon_weight * ts_recon
            loss = total_recon + args.kl_weight * kl_loss
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

        _update_used_codes(used_mask, out["indices"])
        totals["loss"] += float(loss.item())
        totals["recon_loss"] += float(total_recon.item())
        totals["image_recon_loss"] += float(image_recon.item())
        totals["ts_recon_loss"] += float(ts_recon.item())
        totals["kl_loss"] += float(kl_loss.item())
        totals["perplexity"] += float(out["perplexity"].item())
        totals["assignment_entropy"] += float(out["assignment_entropy"].item())
        totals["batch_entropy"] += float(out["batch_entropy"].item())
        totals["temperature"] += float(out["temperature"].item())
        n_batches += 1

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            img_rec=f"{image_recon.item():.4f}",
            ts_rec=f"{ts_recon.item():.4f}",
            kl=f"{kl_loss.item():.4f}",
            ppx=f"{out['perplexity'].item():.2f}",
            used=int(used_mask.sum().item()),
        )

        if max_batches is not None and n_batches >= max_batches:
            break

    metrics = {key: value / max(n_batches, 1) for key, value in totals.items()}
    metrics["used_codes"] = float(used_mask.sum().item())
    metrics["used_code_ratio"] = metrics["used_codes"] / float(model.num_codes)
    return metrics


def save_checkpoint(path: Path, model, optimizer, epoch, best_val_loss, args, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "args": vars(args),
            "metadata": metadata,
        },
        path,
    )


def log_metrics(prefix: str, metrics: dict[str, float], epoch: int) -> None:
    pieces = [f"{key}: {value:.6f}" for key, value in metrics.items()]
    print(f"{prefix} epoch {epoch}: " + " | ".join(pieces), flush=True)


@torch.no_grad()
def collect_val_preview(
    model: SoftVQVAE,
    val_loader: DataLoader,
    device: torch.device,
    embedder: DelayEmbedder,
):
    model.eval()
    batch = next(iter(val_loader))
    series = batch[0].to(device, non_blocking=True)
    x, pad_mask = series_batch_to_images(series, embedder)
    out = model(x)
    return x.detach().cpu(), out["recon"].detach().cpu(), None if pad_mask is None else pad_mask.detach().cpu()


def train(args):
    set_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    train_dataset, val_dataset, data_min, data_max = load_benchmark_datasets(args)

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

    sample = train_dataset[0][0]
    channels, seq_len = sample.shape
    image_channels, image_resolution = infer_image_shape(args, sample, device)
    if image_channels != channels:
        raise ValueError(
            f"Delay image channels ({image_channels}) should match time-series channels ({channels})."
        )
    validate_image_shape_args(image_resolution, args.ch_mult)
    embedder = make_image_embedder(args, seq_len, device)

    model = SoftVQVAE(
        input_dim=channels,
        output_dim=channels,
        resolution=image_resolution,
        hidden_size=args.hidden_size,
        num_res_blocks=args.num_layers,
        code_dim=args.code_dim,
        num_codes=args.num_codes,
        ch_mult=args.ch_mult,
        attn_resolutions=args.attn_resolutions,
        attn_type=args.attn_type,
        dropout=args.dropout,
        temperature=args.temperature,
        learnable_temperature=args.learnable_temperature,
        l2_norm=args.l2_norm,
        tanh_out=args.tanh_out,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    save_dir = Path(args.save_dir) / (f"{args.dataset_name}_one_channel" if args.one_channel else args.dataset_name)
    save_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataset_name": args.dataset_name,
        "data": args.data,
        "rel_path": args.rel_path,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "input_channels": channels,
        "input_seq_len": seq_len,
        "image_channels": image_channels,
        "image_resolution": image_resolution,
        "delay": args.delay,
        "embedding": args.embedding,
        "data_min": data_min.tolist(),
        "data_max": data_max.tolist(),
    }
    np.savez(
        save_dir / "dataset_metadata.npz",
        dataset_name=np.array(args.dataset_name),
        data=np.array(args.data),
        rel_path=np.array(args.rel_path),
        ts_seq_len=np.array(seq_len, dtype=np.int64),
        image_resolution=np.array(image_resolution, dtype=np.int64),
        delay=np.array(args.delay, dtype=np.int64),
        embedding=np.array(args.embedding, dtype=np.int64),
        code_dim=np.array(args.code_dim, dtype=np.int64),
        num_codes=np.array(args.num_codes, dtype=np.int64),
        latent_downsample=np.array(model.latent_downsample, dtype=np.int64),
        ch_mult=np.array(args.ch_mult, dtype=np.int64),
        train_size=np.array(len(train_dataset), dtype=np.int64),
        val_size=np.array(len(val_dataset), dtype=np.int64),
        data_min=data_min.numpy(),
        data_max=data_max.numpy(),
        extra_scale=np.array(False, dtype=bool),
    )

    if wandb is None:
        raise ImportError("wandb is not installed. Please install wandb to run this training script.")
    run_name = f"{args.dataset_name}_len{seq_len}"
    wb = wandb.init(
        project="soft-vqvae",
        name=run_name,
        config={**vars(args), **metadata, "save_dir": str(save_dir)},
        dir=str(save_dir),
    )

    print(f"device: {device}", flush=True)
    print(f"delay image shape: ({image_channels}, {image_resolution}, {image_resolution})", flush=True)
    print(f"latent shape: ({args.code_dim}, {model.latent_resolution}, {model.latent_resolution})", flush=True)
    print(f"model params: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    print(f"save_dir: {save_dir}", flush=True)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        print(f"\n===== Epoch {epoch} =====", flush=True)
        train_metrics = run_epoch(model, train_loader, optimizer, device, args, training=True, embedder=embedder)
        val_metrics = run_epoch(model, val_loader, None, device, args, training=False, embedder=embedder)
        log_metrics("train", train_metrics, epoch)
        log_metrics("val", val_metrics, epoch)

        wb.log(
            {
                **{f"train/{key}": value for key, value in train_metrics.items()},
                **{f"val/{key}": value for key, value in val_metrics.items()},
                "val/best_loss": min(best_val_loss, val_metrics["loss"]),
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
            },
            step=epoch,
        )
        if args.wandb_log_images_every > 0 and epoch % args.wandb_log_images_every == 0:
            x_vis, recon_vis, mask_vis = collect_val_preview(model, val_loader, device, embedder)
            log_payload = {
                "viz/input_image": wandb.Image(x_vis[0, 0].numpy()),
                "viz/recon_image": wandb.Image(recon_vis[0, 0].numpy()),
            }
            if mask_vis is not None:
                log_payload["viz/pad_mask"] = wandb.Image(mask_vis[0, 0].numpy())
            wb.log(log_payload, step=epoch)

        save_checkpoint(save_dir / "last.pt", model, optimizer, epoch, best_val_loss, args, metadata)
        torch.save(model.state_dict(), save_dir / "last_state_dict.pt")
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(save_dir / "best.pt", model, optimizer, epoch, best_val_loss, args, metadata)
            torch.save(model.state_dict(), save_dir / "best_state_dict.pt")
            print(f"Saved BEST model: {save_dir / 'best.pt'}", flush=True)

        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(
                save_dir / f"epoch_{epoch:04d}.pt",
                model,
                optimizer,
                epoch,
                best_val_loss,
                args,
                metadata,
            )

    wb.finish()


if __name__ == "__main__":
    train(get_args())
