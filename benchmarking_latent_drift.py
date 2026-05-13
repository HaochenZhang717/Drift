"""
Train a latent-space drifting model with frozen SoftVQ-VAE.

Pipeline:
1) time series -> delay image
2) delay image -> frozen SoftVQ-VAE encoder+quantizer -> continuous latent
3) drift model generates in latent space
4) drifting loss is computed in latent space
5) generation: latent -> frozen decoder -> image -> time series
"""
import argparse
import math
import random
import time
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from data_provider.data_provider import get_test, get_train
from drifting import compute_V
from img_transformations import DelayEmbedder
from models.soft_vqvae import SoftVQVAE
from models.unconditional_model import DriftDiT_models
from utils.utils_drift import EMA, WarmupLRScheduler, save_checkpoint, set_seed


def parse_temperatures(value: str) -> list[float]:
    vals = [float(x) for x in value.split(",") if x.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("temperatures must contain at least one value")
    return vals


def parse_float_list(value: str) -> list[float]:
    vals = [float(x) for x in value.split(",") if x.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("value must contain at least one float")
    return vals


def parse_int_tuple(value: str | tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(value, tuple):
        return value
    vals = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return tuple(vals)


def _extract_ts_sample(sample: Any) -> torch.Tensor:
    if isinstance(sample, (list, tuple)):
        sample = sample[0]
    if not torch.is_tensor(sample):
        sample = torch.as_tensor(sample, dtype=torch.float32)
    sample = sample.to(torch.float32)
    if sample.ndim == 1:
        sample = sample.unsqueeze(-1)
    if sample.ndim != 2:
        raise ValueError(f"Expected time-series sample shape (T, C), got {tuple(sample.shape)}")
    return sample


def _fit_minmax_stats(base_dataset: Dataset, one_channel: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    data_min = None
    data_max = None
    for idx in range(len(base_dataset)):
        sample = _extract_ts_sample(base_dataset[idx])
        if one_channel:
            sample = sample[:, :1]
        sample_min = sample.amin(dim=0)
        sample_max = sample.amax(dim=0)
        data_min = sample_min if data_min is None else torch.minimum(data_min, sample_min)
        data_max = sample_max if data_max is None else torch.maximum(data_max, sample_max)
    if data_min is None or data_max is None:
        raise ValueError("Cannot fit min-max statistics on an empty dataset.")
    return data_min, data_max


class MinMaxNormalizedTimeSeriesDataset(Dataset):
    def __init__(self, base_dataset: Dataset, data_min: torch.Tensor, data_max: torch.Tensor, one_channel: bool = False):
        self.base_dataset = base_dataset
        self.one_channel = one_channel
        self.data_min = data_min.to(torch.float32)
        self.data_max = data_max.to(torch.float32)
        self.denom = torch.clamp(self.data_max - self.data_min, min=1e-6)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        sample = _extract_ts_sample(self.base_dataset[idx])
        if self.one_channel:
            sample = sample[:, :1]
        sample = torch.clamp((sample - self.data_min) / self.denom, 0.0, 1.0)
        sample = sample * 2.0 - 1.0
        return sample  # (T, C)


def delay_embedding_num_cols(seq_len: int, delay: int, embedding: int) -> int:
    col = 0
    while (col * delay + embedding) <= seq_len:
        col += 1
    if col < embedding and col * delay != seq_len and col * delay + embedding > seq_len:
        col += 1
    return max(col, 1)


def delay_images_to_series(images: torch.Tensor, config: dict, average_overlap: bool) -> torch.Tensor:
    """
    Convert padded delay images back to time series using explicit reconstruction.
    Returns shape: (B, T, C)
    """
    bsz, channels, emb, _ = images.shape
    seq_len = int(config["ts_seq_len"])
    delay = int(config["ts_delay"])
    cols = delay_embedding_num_cols(seq_len, delay, emb)

    img_non_square = images[:, :, :, :cols]
    series_sum = torch.zeros((bsz, channels, seq_len), dtype=images.dtype, device=images.device)
    series_cnt = torch.zeros_like(series_sum)
    series_overwrite = torch.zeros_like(series_sum)

    for i in range(cols - 1):
        start = i * delay
        end = start + emb
        patch = img_non_square[:, :, :, i]
        series_sum[:, :, start:end] += patch
        series_cnt[:, :, start:end] += 1.0
        series_overwrite[:, :, start:end] = patch

    # last column partial write
    start = (cols - 1) * delay
    rem = seq_len - start
    rem = max(rem, 1)
    patch_last = img_non_square[:, :, :rem, cols - 1]
    series_sum[:, :, start:] += patch_last
    series_cnt[:, :, start:] += 1.0
    series_overwrite[:, :, start:] = patch_last

    if average_overlap:
        series = series_sum / torch.clamp(series_cnt, min=1.0)
    else:
        series = series_overwrite

    return series.permute(0, 2, 1).contiguous()


def save_time_series_grid(series: torch.Tensor, save_path: str, ncol: int = 8):
    if series.ndim == 2:
        series = series.unsqueeze(-1)
    num_samples, seq_len, _ = series.shape
    nrow = math.ceil(num_samples / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.0, nrow * 1.5), sharex=True, sharey=True)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    t = torch.arange(seq_len).cpu().numpy()
    data = series.detach().cpu().numpy()
    for i, ax in enumerate(axes):
        if i < num_samples:
            ax.plot(t, data[i, :, 0], linewidth=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis("off")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.3)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


class LatentQueue:
    def __init__(self, capacity: int, latent_shape: tuple[int, int, int]):
        self.capacity = capacity
        self.latent_shape = latent_shape
        self.queue = torch.empty((0, *latent_shape), dtype=torch.float32)

    def add(self, x: torch.Tensor):
        x = x.detach().cpu()
        self.queue = torch.cat([self.queue, x], dim=0)
        if self.queue.shape[0] > self.capacity:
            self.queue = self.queue[-self.capacity:]

    def is_ready(self, n: int) -> bool:
        return self.queue.shape[0] >= n

    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        idx = torch.randint(0, self.queue.shape[0], (n,))
        return self.queue[idx].to(device)


def build_dataset_config(config: dict, flag: str) -> dict:
    stride = config.get("window_stride")
    if stride is None:
        stride = config.get("ts_stride")
    if stride is None:
        stride = config.get("stride")
    dataset_config = {
        "name": config["dataset_name"],
        "data": config["data"],
        "datasets_dir": config["datasets_dir"],
        "rel_path": config["rel_path"],
        "path": config["rel_path"],
        "seq_len": config["ts_seq_len"],
        "flag": flag,
    }
    if config.get("rel_path_train") is not None:
        dataset_config["rel_path_train"] = config["rel_path_train"]
    if config.get("rel_path_valid") is not None:
        dataset_config["rel_path_valid"] = config["rel_path_valid"]
    if stride is not None:
        dataset_config["window_stride"] = int(stride)
        dataset_config["ts_stride"] = int(stride)
        dataset_config["stride"] = int(stride)
    return dataset_config


def load_frozen_softvqvae(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "args" not in ckpt or "model_state_dict" not in ckpt:
        raise ValueError("SoftVQ-VAE checkpoint must contain 'args' and 'model_state_dict'.")
    vae_args = ckpt["args"]
    model = SoftVQVAE(
        input_dim=1 if vae_args.get("one_channel", False) else 1,  # benchmark scripts are one-channel by default
        output_dim=1 if vae_args.get("one_channel", False) else 1,
        resolution=int(vae_args["embedding"]),
        hidden_size=int(vae_args["hidden_size"]),
        num_res_blocks=int(vae_args["num_layers"]),
        code_dim=int(vae_args["code_dim"]),
        num_codes=int(vae_args["num_codes"]),
        ch_mult=parse_int_tuple(vae_args["ch_mult"]),
        attn_resolutions=parse_int_tuple(vae_args.get("attn_resolutions", ())),
        dropout=float(vae_args.get("dropout", 0.0)),
        temperature=float(vae_args.get("temperature", 0.07)),
        learnable_temperature=bool(vae_args.get("learnable_temperature", False)),
        l2_norm=bool(vae_args.get("l2_norm", False)),
        attn_type=vae_args.get("attn_type", "vanilla"),
        tanh_out=bool(vae_args.get("tanh_out", False)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, vae_args


@torch.no_grad()
def ts_to_latent(batch_ts: torch.Tensor, embedder: DelayEmbedder, softvq: SoftVQVAE) -> torch.Tensor:
    # batch_ts: (B, T, C)
    images, _ = embedder.ts_to_img(batch_ts, pad=True, mask=0, return_pad_mask=True)
    z_e, q = softvq.encode(images)
    _ = z_e
    return q["z_q"]


def compute_latent_drift_loss(z_gen: torch.Tensor, z_pos: torch.Tensor, temperatures: list[float]) -> tuple[torch.Tensor, dict]:
    feat_gen = z_gen.flatten(start_dim=1)
    feat_pos = z_pos.flatten(start_dim=1)
    feat_neg = feat_gen

    V_total = torch.zeros_like(feat_gen)
    v_norm_sum = 0.0
    for tau in temperatures:
        V_tau = compute_V(feat_gen, feat_pos, feat_neg, tau, mask_self=True)
        v_norm = torch.sqrt(torch.mean(V_tau ** 2) + 1e-8)
        V_total = V_total + V_tau / (v_norm + 1e-8)
        v_norm_sum += float(v_norm.item())

    target = (feat_gen + V_total).detach()
    loss = F.mse_loss(feat_gen, target)
    info = {
        "loss": float(loss.item()),
        "drift_norm": float(torch.sqrt(torch.mean(V_total ** 2) + 1e-8).item()),
        "v_norm": v_norm_sum,
    }
    return loss, info


@torch.no_grad()
def sample_and_save(
    model: nn.Module,
    softvq: SoftVQVAE,
    config: dict,
    device: torch.device,
    save_path: str,
    num_samples: int = 80,
):
    model.eval()
    latent = torch.randn(
        num_samples,
        int(config["latent_channels"]),
        int(config["latent_img_size"]),
        int(config["latent_img_size"]),
        device=device,
    )
    z_gen = model(latent)
    img = softvq.decode(z_gen)
    ts = delay_images_to_series(img, config, average_overlap=bool(config.get("decode_average_overlap", False)))
    save_time_series_grid(ts, save_path, ncol=8)


def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    config = {
        "model": args.model,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "ema_decay": args.ema_decay,
        "warmup_steps": args.warmup_steps,
        "epochs": args.epochs,
        "queue_size": args.queue_size,
        "batch_n_pos": args.batch_n_pos,
        "batch_n_neg": args.batch_n_neg,
        "temperatures": args.temperatures,
        "ts_seq_len": args.ts_seq_len,
        "ts_delay": args.ts_delay,
        "ts_embedding": args.ts_embedding,
        "window_stride": args.window_stride,
        "ts_stride": args.ts_stride,
        "stride": args.stride,
        "dataset_name": args.dataset_name,
        "data": args.data,
        "datasets_dir": args.datasets_dir,
        "rel_path": args.rel_path,
        "rel_path_train": args.rel_path_train,
        "rel_path_valid": args.rel_path_valid,
        "one_channel": args.one_channel,
        "decode_average_overlap": args.decode_average_overlap,
        "softvq_ckpt_path": args.softvq_ckpt_path,
    }
    return config


def train(
    config: dict,
    output_dir: str,
    seed: int,
    num_workers: int,
    log_interval: int,
    sample_interval: int,
    batch_size: int,
    max_train_batches: int | None = None,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(output_dir) / config["dataset_name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds_raw = get_train(build_dataset_config(config, "train"))
    test_ds_raw = get_test(build_dataset_config(config, "test"))
    data_min, data_max = _fit_minmax_stats(train_ds_raw, one_channel=bool(config.get("one_channel", False)))
    train_ds = MinMaxNormalizedTimeSeriesDataset(train_ds_raw, data_min, data_max, one_channel=bool(config.get("one_channel", False)))
    test_ds = MinMaxNormalizedTimeSeriesDataset(test_ds_raw, data_min, data_max, one_channel=bool(config.get("one_channel", False)))
    print(f"Dataset sizes | train: {len(train_ds)} | test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)
    _ = test_loader

    softvq, vae_args = load_frozen_softvqvae(config["softvq_ckpt_path"], device)
    print(f"Loaded frozen SoftVQ-VAE from {config['softvq_ckpt_path']}")

    embedder = DelayEmbedder(
        device=device,
        seq_len=config["ts_seq_len"],
        delay=config["ts_delay"],
        embedding=config["ts_embedding"],
    )

    with torch.no_grad():
        sample = next(iter(train_loader))  # (B, T, C)
        lat = ts_to_latent(sample.to(device), embedder, softvq)
    latent_shape = lat.shape[1:]  # (C, H, W)
    latent_img_size = latent_shape[-1]
    latent_channels = latent_shape[0]
    config["latent_img_size"] = latent_img_size
    config["latent_channels"] = latent_channels
    print(f"Latent shape: (C={latent_channels}, H={latent_img_size}, W={latent_img_size})")

    model = DriftDiT_models[config["model"]](img_size=latent_img_size, in_channels=latent_channels).to(device)
    ema_decays = config["ema_decays"]
    ema_trackers = []
    for decay in ema_decays:
        ema_trackers.append((decay, EMA(model, decay=decay)))
    primary_ema = ema_trackers[0][1]
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = WarmupLRScheduler(optimizer, warmup_steps=config["warmup_steps"], base_lr=config["lr"])
    queue = LatentQueue(capacity=config["queue_size"], latent_shape=(latent_channels, latent_img_size, latent_img_size))

    global_step = 0
    for epoch in range(config["epochs"]):
        start = time.time()
        loss_sum = 0.0
        drift_sum = 0.0
        n_batches = 0

        for batch_idx, batch_ts in enumerate(train_loader):
            batch_ts = batch_ts.to(device, non_blocking=True)  # (B, T, C)
            with torch.no_grad():
                z_real = ts_to_latent(batch_ts, embedder, softvq)
            queue.add(z_real)

            if not queue.is_ready(config["batch_n_pos"]):
                continue

            z_noise = torch.randn(
                config["batch_n_neg"],
                latent_channels,
                latent_img_size,
                latent_img_size,
                device=device,
            )
            z_gen = model(z_noise)
            z_pos = queue.sample(config["batch_n_pos"], device)
            loss, info = compute_latent_drift_loss(z_gen, z_pos, config["temperatures"])

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()
            scheduler.step()
            for _, ema in ema_trackers:
                ema.update(model)

            info["grad_norm"] = float(grad_norm.item())
            loss_sum += info["loss"]
            drift_sum += info["drift_norm"]
            n_batches += 1
            global_step += 1

            if global_step % log_interval == 0:
                print(
                    f"Epoch {epoch+1}/{config['epochs']} | Step {global_step} | "
                    f"Loss: {info['loss']:.4f} | Drift: {info['drift_norm']:.4f} | "
                    f"|V|: {info['v_norm']:.4f} | Grad: {info['grad_norm']:.4f} | "
                    f"LR: {scheduler.get_lr():.6f}"
                )

            if max_train_batches is not None and (batch_idx + 1) >= max_train_batches:
                break

        epoch_time = time.time() - start
        avg_loss = loss_sum / max(n_batches, 1)
        avg_drift = drift_sum / max(n_batches, 1)
        print(f"\nEpoch {epoch+1} completed in {epoch_time:.1f}s | Avg Loss: {avg_loss:.4f} | Avg Drift: {avg_drift:.4f}\n")


        if epoch > 0 and epoch % 100 == 0:
            ckpt_path = output_dir / f"checkpoint_epoch{epoch+1}.pt"
            save_checkpoint(
                str(ckpt_path),
                model,
                primary_ema,
                optimizer,
                scheduler,
                epoch,
                global_step,
                config,
            )
            for decay, ema in ema_trackers:
                ema_name = str(decay).replace(".", "p")
                ema_path = output_dir / f"checkpoint_epoch{epoch+1}_ema_{ema_name}.pt"
                torch.save(
                    {
                        "ema_decay": decay,
                        "model_state_dict": ema.shadow.state_dict(),
                        "epoch": epoch,
                        "step": global_step,
                        "config": config,
                    },
                    ema_path,
                )

        # if (epoch + 1) % sample_interval == 0:
        #     for decay, ema in ema_trackers:
        #         ema_name = str(decay).replace(".", "p")
        #         sample_path = output_dir / f"samples_epoch{epoch+1}_ema_{ema_name}.png"
        #         sample_and_save(ema.shadow, softvq, config, device, str(sample_path), num_samples=80)
        #         print(f"Saved samples to {sample_path}")

    final_path = output_dir / "checkpoint_final.pt"
    save_checkpoint(
        str(final_path),
        model,
        primary_ema,
        optimizer,
        scheduler,
        config["epochs"] - 1,
        global_step,
        config,
    )
    for decay, ema in ema_trackers:
        ema_name = str(decay).replace(".", "p")
        final_ema_path = output_dir / f"checkpoint_final_ema_{ema_name}.pt"
        torch.save(
            {
                "ema_decay": decay,
                "model_state_dict": ema.shadow.state_dict(),
                "epoch": config["epochs"] - 1,
                "step": global_step,
                "config": config,
            },
            final_ema_path,
        )
    print(f"Training complete. Final checkpoint: {final_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark latent drift training with frozen SoftVQ-VAE.")
    parser.add_argument("--output_dir", type=str, default="./outputs/latent_drift")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--sample_interval", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_train_batches", type=int, default=None)

    parser.add_argument("--model", type=str, default="DriftDiT-Tiny", choices=sorted(DriftDiT_models.keys()))
    parser.add_argument("--batch_n_pos", type=int, default=256)
    parser.add_argument("--batch_n_neg", type=int, default=256)
    parser.add_argument("--temperatures", type=parse_temperatures, default=[0.02, 0.05, 0.2])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument(
        "--ema_decays",
        type=parse_float_list,
        default=None,
        help="Comma-separated EMA decays, e.g. 0.999,0.995,0.99. If set, overrides --ema_decay.",
    )
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--queue_size", type=int, default=2048)

    parser.add_argument("--ts_seq_len", type=int, default=512)
    parser.add_argument("--ts_delay", type=int, default=8)
    parser.add_argument("--ts_embedding", type=int, default=64)
    parser.add_argument("--window_stride", type=int, default=None)
    parser.add_argument("--ts_stride", type=int, default=1)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--one_channel", action="store_true")
    parser.add_argument("--decode_average_overlap", action="store_true")

    parser.add_argument("--softvq_ckpt_path", type=str, required=True)

    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--datasets_dir", type=str, required=True)
    parser.add_argument("--rel_path", type=str, default=None)
    parser.add_argument("--rel_path_train", type=str, default=None)
    parser.add_argument("--rel_path_valid", type=str, default=None)

    args = parser.parse_args()
    if args.rel_path is None and not (args.rel_path_train and args.rel_path_valid):
        parser.error("Provide --rel_path, or provide both --rel_path_train and --rel_path_valid.")
    if args.rel_path is None:
        args.rel_path = args.rel_path_train

    config = build_config(args)
    if args.ema_decays is not None:
        config["ema_decays"] = args.ema_decays
    else:
        config["ema_decays"] = [args.ema_decay]
    train(
        config=config,
        output_dir=args.output_dir,
        seed=args.seed,
        num_workers=args.num_workers,
        log_interval=args.log_interval,
        sample_interval=args.sample_interval,
        batch_size=args.batch_size,
        max_train_batches=args.max_train_batches,
    )


if __name__ == "__main__":
    main()
