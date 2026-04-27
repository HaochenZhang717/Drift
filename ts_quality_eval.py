from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from img_transformations import DelayEmbedder


def delay_embedding_num_cols(config: dict) -> int:
    """Return the number of non-padded columns in a delay-embedding image."""
    seq_len = config["ts_seq_len"]
    delay = config["ts_delay"]
    embedding = config["ts_embedding"]

    col = 0
    while (col * delay + embedding) <= seq_len and col < embedding:
        col += 1

    if (
        col < embedding
        and col * delay != seq_len
        and col * delay + embedding > seq_len
    ):
        col += 1

    return max(1, col)


def delay_images_to_series(
    images: torch.Tensor,
    config: dict,
    device: torch.device,
) -> torch.Tensor:
    """Invert delay-embedded images back to time-series tensors."""
    seq_len = config["ts_seq_len"]
    delay = config["ts_delay"]
    embedding = config["ts_embedding"]
    images = images.to(device)
    embedder = DelayEmbedder(device=device, seq_len=seq_len, delay=delay, embedding=embedding)
    embedder.img_shape = (
        images.shape[0],
        images.shape[1],
        embedding,
        delay_embedding_num_cols(config),
    )
    return embedder.img_to_ts(images)


@torch.no_grad()
def generate_time_series_samples(
    model: nn.Module,
    config: dict,
    device: torch.device,
    num_samples: int,
    batch_size: int = 256,
) -> torch.Tensor:
    """Generate time-series samples from a delay-embedding image generator."""
    model.eval()
    all_series = []

    for start in range(0, num_samples, batch_size):
        current_batch = min(batch_size, num_samples - start)
        noise = torch.randn(
            current_batch,
            config["in_channels"],
            config["img_size"],
            config["img_size"],
            device=device,
        )
        samples = model(noise)
        all_series.append(delay_images_to_series(samples, config, device).detach().cpu())

    return torch.cat(all_series, dim=0)


@torch.no_grad()
def collect_real_time_series(
    dataset: Dataset,
    config: dict,
    device: torch.device,
    num_samples: Optional[int] = None,
    batch_size: int = 256,
    num_workers: int = 0,
) -> torch.Tensor:
    """Collect real samples and invert their delay images back to time series."""
    if num_samples is None:
        num_samples = len(dataset)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    all_series = []
    num_collected = 0

    for batch in loader:
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        needed = num_samples - num_collected
        images = images[:needed]
        all_series.append(delay_images_to_series(images, config, device).detach().cpu())
        num_collected += images.shape[0]
        if num_collected >= num_samples:
            break

    if not all_series:
        raise ValueError("No real time-series samples were collected for metric evaluation.")

    return torch.cat(all_series, dim=0)[:num_samples]


def parse_metric_names(eval_metrics: str) -> list:
    metric_aliases = {
        "disc": "disc",
        "discriminative": "disc",
        "pred": "pred",
        "predictive": "pred",
        "contextfid": "contextFID",
        "context_fid": "contextFID",
        "vaefid": "vaeFID",
        "vae_fid": "vaeFID",
    }
    return [
        metric_aliases.get(metric.strip().lower(), metric.strip())
        for metric in eval_metrics.split(",")
        if metric.strip()
    ]


def evaluate_time_series_metrics(
    model: nn.Module,
    test_dataset: Dataset,
    config: dict,
    device: torch.device,
    eval_metrics: list,
    metric_iteration: int,
    num_workers: int,
    num_samples: Optional[int] = None,
    base_path: Optional[str] = None,
    vae_ckpt_root: Optional[str] = None,
    output_dir: Optional[Path] = None,
    step: Optional[int] = None,
) -> Dict[str, Any]:
    """Run copied time-series metrics on real test sequences and generated sequences."""
    eval_size = len(test_dataset) if num_samples is None else min(num_samples, len(test_dataset))
    real_sig = collect_real_time_series(
        test_dataset,
        config,
        device,
        num_samples=eval_size,
        num_workers=num_workers,
    ).numpy().astype(np.float32)
    gen_sig = generate_time_series_samples(
        model,
        config,
        device,
        num_samples=eval_size,
    ).numpy().astype(np.float32)

    if output_dir is not None and step is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / f"real_test_step{step}.npy", real_sig)
        np.save(output_dir / f"generated_step{step}.npy", gen_sig)

    results: Dict[str, Any] = {
        "metric/eval_num_samples": int(eval_size),
    }

    if "disc" in eval_metrics:
        from metrics.discriminative_torch import discriminative_score_metrics

        disc_scores = [
            discriminative_score_metrics(real_sig, gen_sig, device)
            for _ in range(metric_iteration)
        ]
        results["metric/disc_mean"] = float(np.round(np.mean(disc_scores), 4))
        results["metric/disc_std"] = float(np.round(np.std(disc_scores), 4))

    if "pred" in eval_metrics:
        from metrics.predictive_metrics_pytorch import predictive_score_metrics

        pred_scores = [
            predictive_score_metrics(real_sig, gen_sig, device=device)
            for _ in range(metric_iteration)
        ]
        results["metric/pred_mean"] = float(np.round(np.nanmean(pred_scores), 4))
        results["metric/pred_std"] = float(np.round(np.nanstd(pred_scores), 4))

    if "contextFID" in eval_metrics:
        from metrics.context_fid import Context_FID

        results["metric/context_fid"] = float(
            Context_FID(real_sig, gen_sig, config["dataset"], device, base_path)
        )

    if "vaeFID" in eval_metrics:
        from metrics.vae_fid import VAE_FID

        results["metric/vae_fid"] = float(
            VAE_FID(real_sig, gen_sig, config["dataset"], device, vae_ckpt_root=vae_ckpt_root)
        )

    return results
