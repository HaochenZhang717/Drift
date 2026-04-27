"""
Training script for Drifting Models with unconditional generation.
Includes a synthetic sine-wave time-series dataset that is transformed to images
via delay embedding for DiT training.
"""
import argparse
import math
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


from utils.utils_dataset import get_dataset

from torchvision import datasets, transforms
from model import DriftDiT_Tiny, DriftDiT_Small, DriftDiT_models
from drifting import (
    compute_V,
    normalize_features,
    normalize_drift,
)
from feature_encoder import create_feature_encoder, pretrain_mae
from feature_extractors.ts2vec.models.encoder import TSEncoder
from utils.utils_drift import (
    EMA,
    WarmupLRScheduler,
    SampleQueue,
    save_checkpoint,
    load_checkpoint,
    save_image_grid,
    count_parameters,
    set_seed,
)
import pandas as pd
from ts_quality_eval import (
    collect_real_time_series,
    delay_images_to_series,
    evaluate_time_series_metrics,
    parse_metric_names,
)

try:
    import wandb
except ImportError:
    wandb = None



# Default hyperparameters
TS_SINE_CONFIG = {
    "model": "DriftDiT-Tiny",
    "img_size": 16,  # Must match delay embedding size
    "in_channels": 1,
    "num_classes": 1,
    "batch_nc": 1,
    "batch_n_pos": 320,
    "batch_n_neg": 320,
    "temperatures": [0.02, 0.05, 0.2],
    "lr": 2e-4,
    "weight_decay": 0.01,
    "grad_clip": 2.0,
    "ema_decay": 0.999,
    "warmup_steps": 1000,
    "epochs": 100,
    "alpha_min": 1.0,
    "alpha_max": 1.0,
    "use_feature_encoder": False,
    "loss_domain": "time_series",
    "queue_size": 1280,
    "label_dropout": 0.0,
    # Sine time-series synthesis + delay embedding params
    "ts_num_samples_train": 60000,
    "ts_num_samples_test": 10000,
    "ts_seq_len": 128,  # Gives exactly 32 delay columns for delay=1, embedding=32
    "ts_delay": 8,
    "ts_embedding": 16,
    "ts_components_min": 1,
    "ts_components_max": 3,
    "ts_freq_min": 1.0,
    "ts_freq_max": 6.0,
    "ts_amp_min": 0.2,
    "ts_amp_max": 1.0,
    "ts_noise_std": 0.03,
}

TS_GLUCOSE_CONFIG = {
    "model": "DriftDiT-Tiny",
    "img_size": 12,  # Must match delay embedding size
    "in_channels": 1,
    "num_classes": 1,
    "batch_nc": 1,
    "batch_n_pos": 320,
    "batch_n_neg": 320,
    "temperatures": [0.02, 0.05, 0.2],
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "ema_decay": 0.999,
    "warmup_steps": 1000,
    "epochs": 100,
    "alpha_min": 1.0,
    "alpha_max": 1.0,
    "use_feature_encoder": False,
    "loss_domain": "time_series",
    "queue_size": 1280,
    "label_dropout": 0.0,
    # Sine time-series synthesis + delay embedding params
    "ts_seq_len": 128,  # Gives exactly 32 delay columns for delay=1, embedding=32
    "ts_delay": 12,
    "ts_embedding": 12,
    "ts_stride": 128,
}





def sample_batch(
    queue: SampleQueue,
    num_classes: int,
    n_pos: int,
    device: torch.device,
) -> tuple:
    """Sample a batch of positive samples from the queue."""
    x_pos_list = []
    labels_list = []

    for c in range(num_classes):
        x_c = queue.sample(c, n_pos, device)
        x_pos_list.append(x_c)
        labels_list.append(torch.full((n_pos,), c, device=device, dtype=torch.long))

    x_pos = torch.cat(x_pos_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    return x_pos, labels


def _strip_module_prefix(state_dict: dict) -> dict:
    if not state_dict:
        return state_dict
    if not all(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k[len("module."):]: v for k, v in state_dict.items()}


def load_ts_feature_encoder_from_ckpt(
    ckpt_path: str,
    device: torch.device,
) -> nn.Module:
    """
    Load a pretrained TS encoder checkpoint and return a frozen TSEncoder.
    Supports checkpoints saved by train_full_series_ts2vec_glucose.py.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    metadata = ckpt.get("metadata", {})
    required = ["input_dims", "output_dims", "hidden_dims", "depth", "mask_prob"]
    missing = [k for k in required if k not in metadata]
    if missing:
        raise ValueError(
            f"Missing fields {missing} in TS encoder checkpoint metadata at {ckpt_path}"
        )

    encoder = TSEncoder(
        input_dims=metadata["input_dims"],
        output_dims=metadata["output_dims"],
        hidden_dims=metadata["hidden_dims"],
        depth=metadata["depth"],
        mask_prob=metadata["mask_prob"],
    ).to(device)

    state_dict = ckpt.get("model_state_dict")
    if state_dict is None:
        raise ValueError(f"Checkpoint at {ckpt_path} does not contain model_state_dict")
    state_dict = _strip_module_prefix(state_dict)
    encoder.load_state_dict(state_dict, strict=True)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    return encoder


def compute_drifting_loss(
    x_gen: torch.Tensor,
    x_pos: torch.Tensor,
    feature_encoder: Optional[nn.Module],
    temperatures: list,
    ts_loss_config: Optional[dict] = None,
) -> tuple:
    """
    Compute class-conditional drifting loss with multi-scale features.

    Following paper Section A.5: compute drifting loss at each scale, then sum.

    Args:
        x_gen: Generated samples (B, C, H, W)
        labels_gen: Labels for generated samples (B,)
        x_pos: Positive (real) samples (B_pos, C, H, W)
        labels_pos: Labels for positive samples (B_pos,)
        feature_encoder: Feature encoder (returns List[Tensor] for multi-scale)
        temperatures: List of temperatures for V computation
        use_pixel_space: Whether to use pixel space directly
        spatial_mask: Optional mask with 1.0 on valid regions and 0.0 on padding
        ts_loss_config: If provided, convert delay-embedding images back to
            time-series and compute the loss in that domain.

    Returns:
        loss: Scalar loss
        info: Dict with metrics
    """
    device = x_gen.device
    # Two primary branches:
    # 1) no feature encoder -> convert image to time series and use flattened
    #    sequence vectors;
    # 2) with feature encoder -> convert image to time series, then extract
    #    deterministic full-series features from the encoder.

    rep_gen = delay_images_to_series(x_gen, ts_loss_config, device)
    rep_pos = delay_images_to_series(x_pos, ts_loss_config, device)

    if feature_encoder is None:
        feat_gen_list = [rep_gen.flatten(start_dim=1)]
        feat_pos_list = [rep_pos.flatten(start_dim=1)]
    else:
        feat_gen_seq = feature_encoder(rep_gen, mask="all_true")
        with torch.no_grad():
            feat_pos_seq = feature_encoder(rep_pos, mask="all_true")

        if isinstance(feature_encoder, TSEncoder):
            feat_gen = F.max_pool1d(
                feat_gen_seq.transpose(1, 2),
                kernel_size=feat_gen_seq.size(1),
            ).squeeze(-1)
            feat_gen = F.normalize(feat_gen, p=2, dim=1)

            feat_pos = F.max_pool1d(
                feat_pos_seq.transpose(1, 2),
                kernel_size=feat_pos_seq.size(1),
            ).squeeze(-1)
            feat_pos = F.normalize(feat_pos, p=2, dim=1)
        else:
            raise ValueError("feature_encoder must be an instance of [TSEncoder,]")

        feat_gen_list = [feat_gen]
        feat_pos_list = [feat_pos]

    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    total_drift_norm = 0.0

    # Compute loss at each scale
    for scale_idx, (feat_gen, feat_pos) in enumerate(zip(feat_gen_list, feat_pos_list)):
        # Negatives: generated samples (following Algorithm 1: y_neg = x)
        feat_neg = feat_gen
        # Compute V with multiple temperatures
        V_total = torch.zeros_like(feat_gen)
        for tau in temperatures:
            V_tau = compute_V(
                feat_gen,
                feat_pos,
                feat_neg,
                tau,
                mask_self=True,  # y_neg = x, so mask self
            )
            # Normalize each V before summing
            v_norm = torch.sqrt(torch.mean(V_tau ** 2) + 1e-8)
            V_tau = V_tau / (v_norm + 1e-8)
            V_total = V_total + V_tau

        # Loss: MSE(phi(x), stopgrad(phi(x) + V))
        target = (feat_gen + V_total).detach()
        loss_scale = F.mse_loss(feat_gen, target)

        total_loss = total_loss + loss_scale
        total_drift_norm += (V_total ** 2).mean().item() ** 0.5

    info = {
        "loss": total_loss.item(),
        "drift_norm": total_drift_norm,
    }

    return total_loss, info


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    queue: SampleQueue,
    config: dict,
    device: torch.device,
    feature_encoder: Optional[nn.Module] = None,
) -> dict:
    """
    Single training step (Algorithm 1).

    1. Sample class labels and CFG alpha
    2. Generate samples from noise
    3. Sample positive samples from queue
    4. Compute drifting field and loss
    5. Update model
    """
    model.train()
    num_classes = config["num_classes"]
    n_pos = config["batch_n_pos"]
    n_neg = config["batch_n_neg"]
    temperatures = config["temperatures"]
    ts_loss_config = (
        config
        if config.get("loss_domain") == "time_series"
        else None
    )

    # Total batch size
    batch_size = n_neg

    # Sample noise
    noise = torch.randn(
        batch_size,
        config["in_channels"],
        config["img_size"],
        config["img_size"],
        device=device,
    )

    # Generate samples
    x_gen = model(noise) # (n_class*n_neg, 1, 32, 32)

    # Sample positive samples from queue
    x_pos, labels_pos = sample_batch(queue, num_classes, n_pos, device)
    # x_pos.shape == (n_class*n_pos, 1, 32, 32)

    # Compute drifting loss
    loss, info = compute_drifting_loss(
        x_gen,
        x_pos,
        feature_encoder,
        temperatures,
        ts_loss_config=ts_loss_config,
    )

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Gradient clipping
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), config["grad_clip"]
    )
    info["grad_norm"] = grad_norm.item()

    # Optimizer step
    optimizer.step()

    return info


def fill_queue(
    queue: SampleQueue,
    dataloader: DataLoader,
    device: torch.device,
    min_samples: int = 64,
):
    """Fill the sample queue with real data."""
    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            x, labels = batch[0], torch.zeros(batch[0].shape[0], dtype=torch.long)
        else:
            x, labels = batch, torch.zeros(batch.shape[0], dtype=torch.long)

        queue.add(x, labels)

        if queue.is_ready(min_samples):
            break


def prefix_metric_results(results: Dict[str, Any], split: str) -> Dict[str, Any]:
    """Prefix metric keys with the evaluated data split."""
    return {
        key.replace("metric/", f"metric/{split}/", 1): value
        for key, value in results.items()
    }


def train(
    dataset: str = "sine",
    output_dir: str = "./outputs",
    data_root: str = "./data",
    download: bool = True,
    resume: Optional[str] = None,
    seed: int = 42,
    num_workers: int = 1,
    log_interval: int = 1,
    save_interval: int = 10,
    sample_interval: int = 10,
    eval_step_interval: int = 500,
    eval_metrics: str = "disc",
    eval_num_samples: Optional[int] = None,
    metric_iteration: int = 1,
    wandb_enabled: bool = False,
    wandb_project: str = "drifting-model-ts",
    wandb_run_name: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_mode: Optional[str] = None,
    metrics_base_path: Optional[str] = None,
    vae_ckpt_root: Optional[str] = None,
    ts_feature_encoder_ckpt: Optional[str] = None,
):
    """Main training function."""
    set_seed(seed)

    # Get config
    dataset_name = dataset.lower()
    if dataset_name in ["sine", "ts", "timeseries", "synthetic_sine"]:
        config = TS_SINE_CONFIG.copy()
    elif dataset_name == "glucose":
        config = TS_GLUCOSE_CONFIG.copy()
    else:
        raise ValueError(f"Unsupported dataset for this script: {dataset}")
    config["dataset"] = dataset
    metric_names = parse_metric_names(eval_metrics)
    metric_iteration = max(1, metric_iteration)
    config["eval_metrics"] = metric_names
    config["eval_step_interval"] = eval_step_interval
    config["eval_num_samples"] = eval_num_samples
    config["metric_iteration"] = metric_iteration
    config["ts_feature_encoder_ckpt"] = ts_feature_encoder_ckpt

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(output_dir) / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = None
    if wandb_enabled:
        if wandb is None:
            print("wandb is not installed; continuing without wandb logging.")
        else:
            wandb_run = wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=wandb_run_name,
                mode=wandb_mode,
                config=config,
                dir=str(output_dir),
            )

    # Load dataset
    train_dataset, test_dataset = get_dataset(
        dataset,
        config=config,
        root=data_root,
        download=download,
        seed=seed,
    )
    print(f"Dataset sizes | train: {len(train_dataset)} | test: {len(test_dataset)}")
    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Create model
    model_fn = DriftDiT_models[config["model"]]
    model = model_fn(
        img_size=config["img_size"],
        in_channels=config["in_channels"],
    ).to(device)

    print(f"Model: {config['model']}, Parameters: {count_parameters(model):,}")

    # Create EMA
    ema = EMA(model, decay=config["ema_decay"])

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        betas=(0.9, 0.95),
        weight_decay=config["weight_decay"],
    )

    # Create scheduler
    steps_per_epoch = len(train_loader)
    scheduler = WarmupLRScheduler(
        optimizer,
        warmup_steps=config["warmup_steps"],
        base_lr=config["lr"],
    )

    # Create sample queue
    queue = SampleQueue(
        num_classes=config["num_classes"],
        queue_size=config["queue_size"],
        sample_shape=(config["in_channels"], config["img_size"], config["img_size"]),
    )

    # Feature encoder
    feature_encoder = None
    if ts_feature_encoder_ckpt is not None:
        print(f"Loading time-series feature encoder from {ts_feature_encoder_ckpt}")
        feature_encoder = load_ts_feature_encoder_from_ckpt(
            ts_feature_encoder_ckpt,
            device,
        )
        config["use_feature_encoder"] = True
        print("Using pretrained TS feature encoder for drifting loss.")

    # elif config["use_feature_encoder"]:
    #     print("Creating feature encoder...")
    #     feature_encoder = create_feature_encoder(
    #         dataset=dataset,
    #         feature_dim=512,
    #         multi_scale=True,
    #         use_pretrained=True,  # Use ImageNet-pretrained ResNet
    #     ).to(device)
    #
    #     # For pretrained ResNet, no need for MAE pre-training
    #     # The ImageNet features work well for natural images
    #     print("Using ImageNet-pretrained ResNet encoder")
    #
    #     feature_encoder.eval()
    #     for param in feature_encoder.parameters():
    #         param.requires_grad = False

    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    if resume:
        checkpoint = load_checkpoint(resume, model, ema, optimizer, scheduler)
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["step"]
        print(f"Resumed from epoch {start_epoch}, step {global_step}")

    # Training loop
    print(f"\nStarting training for {config['epochs']} epochs...")
    for epoch in range(start_epoch, config["epochs"]):
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_drift_norm = 0.0
        num_batches = 0

        # Fill queue at start of each epoch
        fill_queue(queue, train_loader, device, min_samples=64)

        for batch_idx, batch in enumerate(train_loader):
            if isinstance(batch, (list, tuple)):
                x_real = batch[0].to(device)
                labels_real = torch.zeros(x_real.shape[0], dtype=torch.long, device=device)

            else:
                x_real = batch.to(device)
                labels_real = torch.zeros(x_real.shape[0], dtype=torch.long, device=device)

            # Add to queue
            queue.add(x_real.cpu(), labels_real.cpu())

            # Skip if queue not ready
            if not queue.is_ready(config["batch_n_pos"]):
                continue

            # Training step
            info = train_step(
                model,
                optimizer,
                queue,
                config,
                device,
                feature_encoder,
            )

            # Update EMA and scheduler
            ema.update(model)
            scheduler.step()

            # Accumulate metrics
            epoch_loss += info["loss"]
            epoch_drift_norm += info["drift_norm"]
            num_batches += 1
            global_step += 1

            # Logging
            if global_step % log_interval == 0:
                lr = scheduler.get_lr()
                print(
                    f"Epoch {epoch+1}/{config['epochs']} | "
                    f"Step {global_step} | "
                    f"Loss: {info['loss']:.4f} | "
                    f"Drift: {info['drift_norm']:.4f} | "
                    f"Grad: {info['grad_norm']:.4f} | "
                    f"LR: {lr:.6f}"
                )
                if wandb_run is not None:
                    wandb.log(
                        {
                            "train/loss": info["loss"],
                            "train/drift_norm": info["drift_norm"],
                            "train/grad_norm": info["grad_norm"],
                            "train/lr": lr,
                            "train/epoch": epoch + 1,
                        },
                        step=global_step,
                    )

            # Generate samples every 500 steps for quick visualization
            if global_step % 500 == 0:
                sample_path = output_dir / f"samples_step{global_step}.png"
                real_sample_path = output_dir / f"real_samples_step{global_step}.png"
                generate_samples(
                    ema.shadow,
                    config,
                    device,
                    str(sample_path),
                    num_samples=80,
                )
                save_real_time_series_samples(
                    train_dataset,
                    config,
                    device,
                    str(real_sample_path),
                    num_samples=80,
                    num_workers=num_workers,
                )
                print(f"Saved samples to {sample_path}")
                print(f"Saved real samples to {real_sample_path}")
                if wandb_run is not None:
                    wandb.log(
                        {
                            "samples/step": wandb.Image(str(sample_path)),
                            "samples/real_step": wandb.Image(str(real_sample_path)),
                        },
                        step=global_step,
                    )

            if (
                metric_names
                and eval_step_interval > 0
                and global_step % eval_step_interval == 0
            ):
                try:
                    metric_results = {}
                    for split, eval_dataset in (
                        ("train", train_dataset),
                        ("test", test_dataset),
                    ):
                        split_output_dir = output_dir / "metric_samples" / split
                        split_results = evaluate_time_series_metrics(
                            ema.shadow,
                            eval_dataset,
                            config,
                            device,
                            eval_metrics=metric_names,
                            num_samples=eval_num_samples,
                            metric_iteration=metric_iteration,
                            num_workers=num_workers,
                            base_path=metrics_base_path,
                            vae_ckpt_root=vae_ckpt_root,
                            output_dir=split_output_dir,
                            step=global_step,
                        )
                        metric_results.update(
                            prefix_metric_results(split_results, split)
                        )
                    metric_str = " | ".join(
                        f"{key}: {value:.4f}" for key, value in metric_results.items()
                    )
                    print(f"Metrics step {global_step} | {metric_str}")
                    if wandb_run is not None:
                        wandb.log(metric_results, step=global_step)
                except Exception as exc:
                    print(f"Metric evaluation failed at step {global_step}: {exc}")
                    if wandb_run is not None:
                        wandb.log(
                            {"metric/eval_failed": 1, "metric/eval_error": str(exc)},
                            step=global_step,
                        )

        # Epoch summary
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / max(num_batches, 1)
        avg_drift = epoch_drift_norm / max(num_batches, 1)
        print(
            f"\nEpoch {epoch+1} completed in {epoch_time:.1f}s | "
            f"Avg Loss: {avg_loss:.4f} | "
            f"Avg Drift Norm: {avg_drift:.4f}\n"
        )
        if wandb_run is not None:
            wandb.log(
                {
                    "epoch/loss": avg_loss,
                    "epoch/drift_norm": avg_drift,
                    "epoch/time_sec": epoch_time,
                },
                step=global_step,
            )

        # Save checkpoint
        if (epoch + 1) % save_interval == 0:
            ckpt_path = output_dir / f"checkpoint_epoch{epoch+1}.pt"
            save_checkpoint(
                str(ckpt_path),
                model,
                ema,
                optimizer,
                scheduler,
                epoch,
                global_step,
                config,
            )
            print(f"Saved checkpoint to {ckpt_path}")

        # Generate samples
        if (epoch + 1) % sample_interval == 0:
            sample_path = output_dir / f"samples_epoch{epoch+1}.png"
            real_sample_path = output_dir / f"real_samples_epoch{epoch+1}.png"
            generate_samples(
                ema.shadow,
                config,
                device,
                str(sample_path),
                num_samples=80,
            )
            save_real_time_series_samples(
                train_dataset,
                config,
                device,
                str(real_sample_path),
                num_samples=80,
                num_workers=num_workers,
            )
            print(f"Saved samples to {sample_path}")
            print(f"Saved real samples to {real_sample_path}")
            if wandb_run is not None:
                wandb.log(
                    {
                        "samples/epoch": wandb.Image(str(sample_path)),
                        "samples/real_epoch": wandb.Image(str(real_sample_path)),
                    },
                    step=global_step,
                )

    # Final checkpoint
    final_path = output_dir / "checkpoint_final.pt"
    save_checkpoint(
        str(final_path),
        model,
        ema,
        optimizer,
        scheduler,
        config["epochs"] - 1,
        global_step,
        config,
    )
    print(f"Training complete! Final checkpoint saved to {final_path}")
    if wandb_run is not None:
        wandb.finish()


def save_time_series_grid(
    series: torch.Tensor,
    save_path: str,
    ncol: int = 8,
):
    """Save time-series samples as a grid of line plots."""
    if series.ndim == 2:
        series = series.unsqueeze(-1)

    num_samples, seq_len, channels = series.shape
    ncol = max(1, ncol)
    nrow = math.ceil(num_samples / ncol)

    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(ncol * 2.0, nrow * 1.5),
        sharex=True,
        sharey=True,
    )
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    t = torch.arange(seq_len).cpu().numpy()
    series_np = series.detach().cpu().numpy()

    for i, ax in enumerate(axes):
        if i < num_samples:
            ax.plot(t, series_np[i, :, 0], linewidth=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis("off")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.3)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def save_real_time_series_samples(
    dataset: Dataset,
    config: dict,
    device: torch.device,
    save_path: str,
    num_samples: int = 80,
    num_workers: int = 0,
):
    """Collect real delay-embedded samples and save them as time-series plots."""
    series = collect_real_time_series(
        dataset,
        config,
        device,
        num_samples=min(num_samples, len(dataset)),
        num_workers=num_workers,
    )
    save_time_series_grid(series, save_path, ncol=8)


@torch.no_grad()
def generate_samples(
    model: nn.Module,
    config: dict,
    device: torch.device,
    save_path: str,
    num_samples: int = 80,
    alpha: float = 1.5,
):
    """Generate samples and save visualization."""
    model.eval()

    in_channels = config["in_channels"]
    img_size = config["img_size"]

    noise = torch.randn(num_samples, in_channels, img_size, img_size, device=device)
    labels = torch.zeros(num_samples, device=device, dtype=torch.long)
    alpha_tensor = torch.full((num_samples,), alpha, device=device)
    samples = model(noise, labels, alpha_tensor)

    series = delay_images_to_series(samples, config, device)
    save_time_series_grid(series, save_path, ncol=8)  # 10 x 8 for 80 samples

    return samples


def main():
    parser = argparse.ArgumentParser(description="Train Drifting Models")
    parser.add_argument(
        "--dataset",
        type=str,
        default="glucose",
        choices=["sine", "glucose"],
        help="Dataset to train on",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/glucose_unconditional_debug",
        help="Output directory",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./AI-READI",
        help="Dataset root directory",
    )
    parser.add_argument(
        "--no_download",
        action="store_true",
        help="Disable automatic dataset download",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="Logging interval (steps)",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=10,
        help="Checkpoint save interval (epochs)",
    )
    parser.add_argument(
        "--sample_interval",
        type=int,
        default=10,
        help="Sample generation interval (epochs)",
    )
    parser.add_argument(
        "--eval_step_interval",
        "--eval_interval",
        dest="eval_step_interval",
        type=int,
        default=500,
        help="Metric evaluation interval in training steps. Set <= 0 to disable.",
    )
    parser.add_argument(
        "--eval_metrics",
        type=str,
        default="disc",
        help="Comma-separated metrics to run: disc,pred,contextFID,vaeFID",
    )
    parser.add_argument(
        "--eval_num_samples",
        type=int,
        default=None,
        help="Number of real/generated samples for metrics. Defaults to the full test set.",
    )
    parser.add_argument(
        "--metric_iteration",
        type=int,
        default=10,
        help="Number of repeated metric runs for mean/std metrics",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="drifting-model-ts",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Weights & Biases run name",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Weights & Biases entity",
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default=None,
        choices=[None, "online", "offline", "disabled"],
        help="Weights & Biases mode",
    )
    parser.add_argument(
        "--metrics_base_path",
        type=str,
        default=None,
        help="Base path for cached metric checkpoints/representations",
    )
    parser.add_argument(
        "--vae_ckpt_root",
        type=str,
        default=None,
        help="Checkpoint root for vaeFID",
    )
    parser.add_argument(
        "--ts_feature_encoder_ckpt",
        type=str,
        default=None,
        help="Path to pretrained TS encoder checkpoint (full_series_ts2vec_glucose.pt).",
    )
    parser.add_argument(
        "--glucose_stride",
        type=int,
        default=TS_GLUCOSE_CONFIG["ts_stride"],
        help="Sliding-window stride for the Glucose dataset",
    )

    args = parser.parse_args()
    TS_GLUCOSE_CONFIG["ts_stride"] = args.glucose_stride

    train(
        dataset=args.dataset,
        output_dir=args.output_dir,
        data_root=args.data_root,
        download=not args.no_download,
        resume=args.resume,
        seed=args.seed,
        num_workers=args.num_workers,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        sample_interval=args.sample_interval,
        eval_step_interval=args.eval_step_interval,
        eval_metrics=args.eval_metrics,
        eval_num_samples=args.eval_num_samples,
        metric_iteration=args.metric_iteration,
        wandb_enabled=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode,
        metrics_base_path=args.metrics_base_path,
        vae_ckpt_root=args.vae_ckpt_root,
        ts_feature_encoder_ckpt=args.ts_feature_encoder_ckpt,
    )


if __name__ == "__main__":
    main()
