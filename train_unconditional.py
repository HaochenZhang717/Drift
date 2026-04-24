"""
Training script for Drifting Models on MNIST and CIFAR-10.
Implements Algorithm 1 from the paper with class-conditional generation.
"""

import argparse
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import datasets, transforms

from model import DriftDiT_Tiny, DriftDiT_Small, DriftDiT_models
from img_transformations import DelayEmbedder
from drifting import (
    compute_V,
    normalize_features,
    normalize_drift,
)
from feature_encoder import create_feature_encoder, pretrain_mae
from utils import (
    EMA,
    WarmupLRScheduler,
    SampleQueue,
    save_checkpoint,
    load_checkpoint,
    save_image_grid,
    count_parameters,
    set_seed,
)


# Default hyperparameters
MNIST_CONFIG = {
    "model": "DriftDiT-Tiny",
    "img_size": 32,
    "in_channels": 1,
    "num_classes": 1,
    "batch_nc": 1,  # Number of classes per batch
    "batch_n_pos": 320,  # Positive samples per class
    "batch_n_neg": 320,  # Negative samples per class
    "temperatures": [0.02, 0.05, 0.2],
    "lr": 2e-4,
    "weight_decay": 0.01,
    "grad_clip": 2.0,
    "ema_decay": 0.999,
    "warmup_steps": 1000,
    "epochs": 100,
    "alpha_min": 1.0,
    "alpha_max": 1.0,
    "use_feature_encoder": False,  # Pixel space for MNIST
    "queue_size": 1280,
    "label_dropout": 0.1,
}

CIFAR10_CONFIG = {
    "model": "DriftDiT-Small",
    "img_size": 32,
    "in_channels": 3,
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
    "use_feature_encoder": True,
    "queue_size": 1280,
    "label_dropout": 0.1,
}

GLUCOSE_CONFIG = {
    "model": "DriftDiT-Tiny",
    "img_size": 16,
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
    "queue_size": 1280,
    "label_dropout": 0.0,
    "window_size": 128,
    "window_stride": 32,
    "delay": 8,
    "embedding": 16,
}


class GlucoseParquetDataset(Dataset):
    """Windowed glucose time series dataset backed by parquet files."""

    def __init__(
        self,
        parquet_path: str,
        window_size: int,
        window_stride: int,
        delay: int,
        embedding: int,
        value_min: Optional[float] = None,
        value_max: Optional[float] = None,
    ):
        self.parquet_path = parquet_path
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if window_stride <= 0:
            raise ValueError("window_stride must be positive")
        if delay <= 0:
            raise ValueError("delay must be positive")
        if embedding <= 0:
            raise ValueError("embedding must be positive")

        self.window_size = window_size
        self.window_stride = window_stride
        self.delay = delay
        self.embedding = embedding
        self.embedder = DelayEmbedder(
            device=torch.device("cpu"),
            seq_len=window_size,
            delay=delay,
            embedding=embedding,
        )

        df = pd.read_parquet(parquet_path, columns=["glucose"])
        self.sequences = []
        self.window_index = []

        if value_min is None or value_max is None:
            value_min, value_max = self._compute_min_max(df["glucose"])
        self.value_min = float(value_min)
        self.value_max = float(value_max)

        for row in df.itertuples(index=False):
            seq = self._normalize_sequence(row.glucose)
            if seq.numel() == 0:
                continue

            seq_idx = len(self.sequences)
            self.sequences.append(seq)
            for start in self._window_starts(len(seq)):
                self.window_index.append((seq_idx, start))

        if not self.window_index:
            raise ValueError(f"No glucose windows could be constructed from {parquet_path}")

    @staticmethod
    def _compute_min_max(series: pd.Series) -> tuple[float, float]:
        mins = []
        maxs = []
        for values in series:
            arr = np.asarray(values, dtype=np.float32)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue
            mins.append(float(arr.min()))
            maxs.append(float(arr.max()))

        if not mins:
            raise ValueError("Could not compute glucose min/max from parquet data")

        return min(mins), max(maxs)

    def _normalize_sequence(self, values: np.ndarray) -> torch.Tensor:
        arr = np.asarray(values, dtype=np.float32)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return torch.empty(0, dtype=torch.float32)

        scale = max(self.value_max - self.value_min, 1e-6)
        arr = 2.0 * ((arr - self.value_min) / scale) - 1.0
        arr = np.clip(arr, -1.0, 1.0)
        return torch.from_numpy(arr)

    def _window_starts(self, seq_len: int) -> list[int]:
        if seq_len <= self.window_size:
            return [0]

        starts = list(range(0, seq_len - self.window_size + 1, self.window_stride))
        last_start = seq_len - self.window_size
        if starts[-1] != last_start:
            starts.append(last_start)
        return starts

    def __len__(self) -> int:
        return len(self.window_index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq_idx, start = self.window_index[idx]
        seq = self.sequences[seq_idx]
        window = seq[start:start + self.window_size]
        if window.shape[0] < self.window_size:
            pad_len = self.window_size - window.shape[0]
            pad_value = float(window[-1].item()) if window.numel() > 0 else 0.0
            window = F.pad(window, (0, pad_len), value=pad_value)

        ts = window.view(1, self.window_size, 1)
        img = self.embedder.ts_to_img(ts, pad=True, mask=0.0).squeeze(0)
        label = torch.tensor(0, dtype=torch.long)
        return img, label


def get_dataset(
    name: str,
    root: str = "./data",
    download: bool = True,
    config: Optional[dict] = None,
) -> tuple:
    """Get dataset and transforms."""
    name = name.lower()
    config = config or {}

    if name == "mnist":
        # MNIST data will be at {root}/mnist/MNIST/raw/
        mnist_root = os.path.join(root, "mnist")
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # [-1, 1]
        ])
        train_dataset = datasets.MNIST(mnist_root, train=True, download=download, transform=transform)
        test_dataset = datasets.MNIST(mnist_root, train=False, download=download, transform=transform)
    elif name in ["cifar10", "cifar"]:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        train_dataset = datasets.CIFAR10(root, train=True, download=download, transform=transform)
        test_dataset = datasets.CIFAR10(root, train=False, download=download, transform=test_transform)
    elif name == "glucose":
        train_path = os.path.join(root, "glucose_train.parquet")
        valid_path = os.path.join(root, "glucose_valid.parquet")
        test_path = os.path.join(root, "glucose_test.parquet")

        train_dataset = GlucoseParquetDataset(
            parquet_path=train_path,
            window_size=config["window_size"],
            window_stride=config["window_stride"],
            delay=config["delay"],
            embedding=config["embedding"],
        )
        valid_dataset = GlucoseParquetDataset(
            parquet_path=valid_path,
            window_size=config["window_size"],
            window_stride=config["window_stride"],
            delay=config["delay"],
            embedding=config["embedding"],
            value_min=train_dataset.value_min,
            value_max=train_dataset.value_max,
        )
        heldout_test_dataset = GlucoseParquetDataset(
            parquet_path=test_path,
            window_size=config["window_size"],
            window_stride=config["window_stride"],
            delay=config["delay"],
            embedding=config["embedding"],
            value_min=train_dataset.value_min,
            value_max=train_dataset.value_max,
        )
        test_dataset = ConcatDataset([valid_dataset, heldout_test_dataset])
    else:
        raise ValueError(f"Unknown dataset: {name}")

    return train_dataset, test_dataset


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


def compute_drifting_loss(
    x_gen: torch.Tensor,
    labels_gen: torch.Tensor,
    x_pos: torch.Tensor,
    labels_pos: torch.Tensor,
    feature_encoder: Optional[nn.Module],
    temperatures: list,
    use_pixel_space: bool = False,
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

    Returns:
        loss: Scalar loss
        info: Dict with metrics
    """
    device = x_gen.device
    num_classes = labels_gen.max().item() + 1

    # Extract features
    if use_pixel_space or feature_encoder is None:
        # Pixel space: single scale
        feat_gen_list = [x_gen.flatten(start_dim=1)]
        feat_pos_list = [x_pos.flatten(start_dim=1)]
    else:
        # Multi-scale feature maps from pretrained encoder
        feat_gen_maps = feature_encoder(x_gen)  # List of (B, C, H, W)
        with torch.no_grad():
            feat_pos_maps = feature_encoder(x_pos)

        # Global average pool each scale to get vectors
        feat_gen_list = [F.adaptive_avg_pool2d(f, 1).flatten(1) for f in feat_gen_maps]
        feat_pos_list = [F.adaptive_avg_pool2d(f, 1).flatten(1) for f in feat_pos_maps]

    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    total_drift_norm = 0.0
    num_losses = 0

    # Compute loss per class
    for c in range(num_classes):
        mask_gen = labels_gen == c
        mask_pos = labels_pos == c

        if not mask_gen.any() or not mask_pos.any():
            continue

        # Compute loss at each scale
        for scale_idx, (feat_gen, feat_pos) in enumerate(zip(feat_gen_list, feat_pos_list)):
            feat_gen_c = feat_gen[mask_gen]
            feat_pos_c = feat_pos[mask_pos]

            # Negatives: generated samples from current class (following Algorithm 1: y_neg = x)
            feat_neg_c = feat_gen_c

            # Simple L2 normalization (projects to unit sphere)
            feat_gen_c_norm = F.normalize(feat_gen_c, p=2, dim=1)
            feat_pos_c_norm = F.normalize(feat_pos_c, p=2, dim=1)
            feat_neg_c_norm = F.normalize(feat_neg_c, p=2, dim=1)

            # Compute V with multiple temperatures
            V_total = torch.zeros_like(feat_gen_c_norm)
            for tau in temperatures:
                V_tau = compute_V(
                    feat_gen_c_norm,
                    feat_pos_c_norm,
                    feat_neg_c_norm,
                    tau,
                    mask_self=True,  # y_neg = x, so mask self
                )
                # Normalize each V before summing
                v_norm = torch.sqrt(torch.mean(V_tau ** 2) + 1e-8)
                V_tau = V_tau / (v_norm + 1e-8)
                V_total = V_total + V_tau

            # Loss: MSE(phi(x), stopgrad(phi(x) + V))
            target = (feat_gen_c_norm + V_total).detach()
            loss_scale = F.mse_loss(feat_gen_c_norm, target)

            total_loss = total_loss + loss_scale
            total_drift_norm += (V_total ** 2).mean().item() ** 0.5
            num_losses += 1

    if num_losses == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), {"loss": 0.0, "drift_norm": 0.0}

    loss = total_loss / num_losses
    info = {
        "loss": loss.item(),
        "drift_norm": total_drift_norm / num_losses,
    }

    return loss, info


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
    alpha_min = config["alpha_min"]
    alpha_max = config["alpha_max"]
    temperatures = config["temperatures"]
    use_pixel = not config["use_feature_encoder"]

    # Total batch size
    batch_size = num_classes * n_neg

    # Sample class labels (repeat each class n_neg times)
    labels = torch.arange(num_classes, device=device).repeat_interleave(n_neg)

    # Sample CFG alpha ~ Uniform(alpha_min, alpha_max)
    alpha = torch.empty(batch_size, device=device).uniform_(alpha_min, alpha_max)

    # Sample noise
    noise = torch.randn(
        batch_size,
        config["in_channels"],
        config["img_size"],
        config["img_size"],
        device=device,
    )

    # Generate samples
    x_gen = model(noise, labels, alpha) # (n_class*n_neg, 1, 32, 32)

    # Sample positive samples from queue
    x_pos, labels_pos = sample_batch(queue, num_classes, n_pos, device)
    # x_pos.shape == (n_class*n_pos, 1, 32, 32)

    # Compute drifting loss
    loss, info = compute_drifting_loss(
        x_gen,
        labels,
        x_pos,
        labels_pos,
        feature_encoder,
        temperatures,
        use_pixel_space=use_pixel,
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


def train(
    dataset: str = "mnist",
    output_dir: str = "./outputs_unconditional",
    data_root: str = "./data",
    download: bool = True,
    resume: Optional[str] = None,
    seed: int = 42,
    num_workers: int = 4,
    log_interval: int = 1,
    save_interval: int = 10,
    sample_interval: int = 10,
):
    """Main training function."""
    set_seed(seed)

    # Get config
    dataset_name = dataset.lower()
    if dataset_name == "mnist":
        config = MNIST_CONFIG.copy()
    elif dataset_name in ["cifar10", "cifar"]:
        config = CIFAR10_CONFIG.copy()
    elif dataset_name == "glucose":
        config = GLUCOSE_CONFIG.copy()
        config["img_size"] = config["embedding"]
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    config["dataset"] = dataset

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(output_dir) / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    train_dataset, test_dataset = get_dataset(
        dataset,
        root=data_root,
        download=download,
        config=config,
    )
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
        num_classes=config["num_classes"],
        label_dropout=config["label_dropout"],
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

    # Feature encoder (for CIFAR)
    feature_encoder = None
    if config["use_feature_encoder"]:
        print("Creating feature encoder...")
        feature_encoder = create_feature_encoder(
            dataset=dataset,
            feature_dim=512,
            multi_scale=True,
            use_pretrained=True,  # Use ImageNet-pretrained ResNet
        ).to(device)

        # For pretrained ResNet, no need for MAE pre-training
        # The ImageNet features work well for natural images
        print("Using ImageNet-pretrained ResNet encoder")

        feature_encoder.eval()
        for param in feature_encoder.parameters():
            param.requires_grad = False

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

            # Generate samples every 500 steps for quick visualization
            if global_step % 500 == 0:
                sample_path = output_dir / f"samples_step{global_step}.png"
                generate_samples(
                    ema.shadow,
                    config,
                    device,
                    str(sample_path),
                    num_samples=80,
                )
                print(f"Saved samples to {sample_path}")

        # Epoch summary
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / max(num_batches, 1)
        avg_drift = epoch_drift_norm / max(num_batches, 1)
        print(
            f"\nEpoch {epoch+1} completed in {epoch_time:.1f}s | "
            f"Avg Loss: {avg_loss:.4f} | "
            f"Avg Drift Norm: {avg_drift:.4f}\n"
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
            generate_samples(
                ema.shadow,
                config,
                device,
                str(sample_path),
                num_samples=80,
            )
            print(f"Saved samples to {sample_path}")

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


@torch.no_grad()
def generate_samples(
    model: nn.Module,
    config: dict,
    device: torch.device,
    save_path: str,
    num_samples: int = 80,
    alpha: float = 1.5,
):
    """Generate samples for visualization."""
    model.eval()

    in_channels = config["in_channels"]
    img_size = config["img_size"]

    noise = torch.randn(num_samples, in_channels, img_size, img_size, device=device)
    labels = torch.zeros(num_samples, device=device, dtype=torch.long)
    alpha_tensor = torch.full((num_samples,), alpha, device=device)
    samples = model(noise, labels, alpha_tensor)
    samples = samples.clamp(-1, 1)

    # 10 rows x 8 cols => nrow=8 for 80 samples
    save_image_grid(samples, save_path, nrow=8)


def main():
    parser = argparse.ArgumentParser(description="Train Drifting Models")
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        help="Dataset to train on",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
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
        default=4,
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
        "--glucose_window_size",
        type=int,
        default=GLUCOSE_CONFIG["window_size"],
        help="Sliding window length for the Glucose dataset",
    )
    parser.add_argument(
        "--glucose_window_stride",
        type=int,
        default=GLUCOSE_CONFIG["window_stride"],
        help="Sliding window stride for the Glucose dataset",
    )
    parser.add_argument(
        "--glucose_delay",
        type=int,
        default=GLUCOSE_CONFIG["delay"],
        help="Delay parameter for Glucose delay embedding",
    )
    parser.add_argument(
        "--glucose_embedding",
        type=int,
        default=GLUCOSE_CONFIG["embedding"],
        help="Embedding size for Glucose delay embedding",
    )

    args = parser.parse_args()

    GLUCOSE_CONFIG["window_size"] = args.glucose_window_size
    GLUCOSE_CONFIG["window_stride"] = args.glucose_window_stride
    GLUCOSE_CONFIG["delay"] = args.glucose_delay
    GLUCOSE_CONFIG["embedding"] = args.glucose_embedding
    GLUCOSE_CONFIG["img_size"] = args.glucose_embedding

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
    )


if __name__ == "__main__":
    main()
