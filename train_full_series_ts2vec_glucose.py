"""
Train a glucose time-series encoder directly on full-series features.

This reuses the TS2Vec TSEncoder architecture, but unlike standard TS2Vec it
max-pools the per-timestep representations during training and applies an
instance contrastive loss to the resulting full-series vectors.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from feature_extractors.ts2vec.models.encoder import TSEncoder
from feature_extractors.ts2vec.utils import init_dl_program
from train_ts2vec_glucose import compute_min_max, load_glucose_windows

try:
    import wandb
except ImportError:
    wandb = None


def full_series_pool(repr_seq: torch.Tensor) -> torch.Tensor:
    """Max-pool [B, T, D] timestamp features into [B, D]."""
    return F.max_pool1d(
        repr_seq.transpose(1, 2),
        kernel_size=repr_seq.size(1),
    ).squeeze(-1)


def full_series_contrastive_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    """Symmetric NT-Xent loss for two full-series views of each sample."""
    if z1.size(0) <= 1:
        return z1.new_tensor(0.0)

    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    z = F.normalize(z, p=2, dim=1)

    logits = torch.matmul(z, z.t()) / temperature
    logits = logits.masked_fill(
        torch.eye(2 * batch_size, dtype=torch.bool, device=z.device),
        float("-inf"),
    )

    targets = torch.arange(2 * batch_size, device=z.device)
    targets = torch.where(targets < batch_size, targets + batch_size, targets - batch_size)
    return F.cross_entropy(logits, targets)


def random_crop_batch(
    x: torch.Tensor,
    min_ratio: float,
    max_ratio: float,
) -> torch.Tensor:
    """Randomly crop each batch to one shared temporal length."""
    if min_ratio >= 1.0 and max_ratio >= 1.0:
        return x

    seq_len = x.size(1)
    min_len = max(2, int(round(seq_len * min_ratio)))
    max_len = max(min_len, int(round(seq_len * max_ratio)))
    max_len = min(max_len, seq_len)
    crop_len = int(torch.randint(min_len, max_len + 1, (1,)).item())
    start = int(torch.randint(0, seq_len - crop_len + 1, (1,)).item())
    return x[:, start : start + crop_len]


def make_contrastive_views(
    x: torch.Tensor,
    crop_min_ratio: float,
    crop_max_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    x1 = random_crop_batch(x, crop_min_ratio, crop_max_ratio)
    x2 = random_crop_batch(x, crop_min_ratio, crop_max_ratio)
    return x1.clone(), x2.clone()


def run_epoch(
    encoder: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    temperature: float,
    crop_min_ratio: float,
    crop_max_ratio: float,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> float:
    is_train = optimizer is not None
    # Validation loss should measure the same augmented contrastive task as
    # training, so keep train-mode masking/dropout but disable gradients.
    encoder.train()

    total_loss = 0.0
    total_steps = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for (x,) in loader:
            x = x.to(device)
            x1, x2 = make_contrastive_views(x, crop_min_ratio, crop_max_ratio)

            z1 = full_series_pool(encoder(x1))
            z2 = full_series_pool(encoder(x2))
            loss = full_series_contrastive_loss(z1, z2, temperature=temperature)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_steps += 1

    return total_loss / max(total_steps, 1)


@torch.no_grad()
def encode_full_series(
    encoder: torch.nn.Module,
    data: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Encode numpy data [N, T, C] into full-series features [N, D]."""
    was_training = encoder.training
    encoder.eval()

    loader = DataLoader(
        TensorDataset(torch.from_numpy(data).float()),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    features = []
    for (x,) in loader:
        x = x.to(device)
        repr_seq = encoder(x, mask="all_true")
        features.append(full_series_pool(repr_seq).cpu())

    encoder.train(was_training)
    return torch.cat(features, dim=0).numpy()


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train TS2Vec-style encoder with full-series contrastive loss on glucose"
    )
    parser.add_argument("--data_root", type=str, default="./AI-READI")
    parser.add_argument("--train_file", type=str, default="glucose_train.parquet")
    parser.add_argument("--valid_file", type=str, default="glucose_valid.parquet")
    parser.add_argument("--column", type=str, default="glucose")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./feature_extractors/checkpoints/full_series_ts2vec_glucose",
    )
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--normalize", type=str, default="minmax", choices=["minmax", "none"])
    parser.add_argument("--max_train_windows", type=int, default=None)
    parser.add_argument("--max_valid_windows", type=int, default=4096)

    parser.add_argument("--output_dims", type=int, default=320)
    parser.add_argument("--hidden_dims", type=int, default=64)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--mask_prob",
        type=float,
        default=0.8,
        help="Probability of keeping each timestamp for binomial masking.",
    )
    parser.add_argument(
        "--crop_min_ratio",
        type=float,
        default=1.0,
        help="Use 1.0 to train on full-length views; lower values add random crop augmentation.",
    )

    parser.add_argument("--crop_max_ratio", type=float, default=1.0)

    parser.add_argument(
        "--val_repeats",
        type=int,
        default=1,
        help="Number of repeated validation passes to average over random masks/crops.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_threads", type=int, default=None)
    parser.add_argument("--save_full_series_features", action="store_true")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb_project", type=str, default="drifting-model-ts")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default=None,
        choices=[None, "online", "offline", "disabled"],
        help="Weights & Biases mode.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_root / args.train_file
    valid_path = data_root / args.valid_file
    device = init_dl_program(args.device, seed=args.seed, max_threads=args.num_threads)

    value_min = None
    value_max = None
    if args.normalize == "minmax":
        value_min, value_max = compute_min_max(str(train_path), args.column)

    print(f"Loading train windows from {train_path}")
    train_data = load_glucose_windows(
        str(train_path),
        column=args.column,
        seq_len=args.seq_len,
        stride=args.stride,
        value_min=value_min,
        value_max=value_max,
        normalize=args.normalize,
        max_windows=args.max_train_windows,
    )
    print(f"Train data: {train_data.shape}")

    print(f"Loading validation windows from {valid_path}")
    valid_data = load_glucose_windows(
        str(valid_path),
        column=args.column,
        seq_len=args.seq_len,
        stride=args.stride,
        value_min=value_min,
        value_max=value_max,
        normalize=args.normalize,
        max_windows=args.max_valid_windows,
    )
    print(f"Valid data: {valid_data.shape}")

    encoder = TSEncoder(
        input_dims=train_data.shape[-1],
        output_dims=args.output_dims,
        hidden_dims=args.hidden_dims,
        depth=args.depth,
        mask_prob=args.mask_prob,
    ).to(device)

    optimizer = torch.optim.AdamW(
        encoder.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_data).float()),
        batch_size=min(args.batch_size, len(train_data)),
        shuffle=True,
        drop_last=True,
    )

    valid_loader = DataLoader(
        TensorDataset(torch.from_numpy(valid_data).float()),
        batch_size=min(args.batch_size, len(valid_data)),
        shuffle=False,
        drop_last=True,
    )

    metadata = {
        "training_objective": "full_series_nt_xent",
        "architecture": "TSEncoder",
        "pooling": "max over time during training and encoding",
        "data_root": str(data_root),
        "train_path": str(train_path),
        "valid_path": str(valid_path) if valid_path.exists() else None,
        "column": args.column,
        "seq_len": args.seq_len,
        "stride": args.stride,
        "normalize": args.normalize,
        "value_min": value_min,
        "value_max": value_max,
        "input_dims": train_data.shape[-1],
        "output_dims": args.output_dims,
        "hidden_dims": args.hidden_dims,
        "depth": args.depth,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "iters": args.iters,
        "temperature": args.temperature,
        "mask_prob": args.mask_prob,
        "crop_min_ratio": args.crop_min_ratio,
        "crop_max_ratio": args.crop_max_ratio,
        "val_repeats": args.val_repeats,
        "seed": args.seed,
        "device": str(device),
        "full_series_feature_shape": ["N", args.output_dims],
    }
    save_json(output_dir / "metadata.json", metadata)

    wandb_run = None
    if args.wandb:
        if wandb is None:
            print("wandb is not installed; continuing without wandb logging.")
        else:
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name,
                mode=args.wandb_mode,
                config=metadata,
                dir=str(output_dir),
            )

    loss_log = []
    val_loss_log = []
    best_val_loss = float("inf")
    best_epoch = None
    global_step = 0
    print("Training full-series contrastive encoder...")
    for epoch in range(args.epochs):
        encoder.train()
        epoch_loss = 0.0
        epoch_steps = 0

        for (x,) in train_loader:
            if args.iters is not None and global_step >= args.iters:
                break

            x = x.to(device)
            x1, x2 = make_contrastive_views(
                x,
                args.crop_min_ratio,
                args.crop_max_ratio,
            )

            z1 = full_series_pool(encoder(x1))
            z2 = full_series_pool(encoder(x2))
            loss = full_series_contrastive_loss(z1, z2, temperature=args.temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1

        avg_loss = epoch_loss / max(epoch_steps, 1)
        loss_log.append(avg_loss)


        val_losses = [
            run_epoch(
                encoder,
                valid_loader,
                device=device,
                temperature=args.temperature,
                crop_min_ratio=args.crop_min_ratio,
                crop_max_ratio=args.crop_max_ratio,
                optimizer=None,
            )
            for _ in range(max(1, args.val_repeats))
        ]
        val_loss = float(np.mean(val_losses))
        val_loss_log.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_path = output_dir / "best_full_series_ts2vec_glucose.pt"
            torch.save(
                {
                    "model_state_dict": encoder.state_dict(),
                    "metadata": metadata,
                    "loss_log": loss_log,
                    "val_loss_log": val_loss_log,
                    "best_val_loss": best_val_loss,
                    "best_epoch": best_epoch,
                    "global_step": global_step,
                },
                best_path,
            )
            print(f"Saved best validation checkpoint to {best_path}")

        print(
            f"Epoch #{epoch}: loss={avg_loss:.6f} | "
            f"val_loss={val_loss:.6f} | best_val_loss={best_val_loss:.6f}"
        )
        if wandb_run is not None:
            wandb.log(
                {
                    "train/loss": avg_loss,
                    "val/loss": val_loss,
                    "val/best_loss": best_val_loss,
                    "val/best_epoch": best_epoch,
                    "train/epoch": epoch,
                    "train/global_step": global_step,
                },
                step=global_step,
            )

        if args.iters is not None and global_step >= args.iters:
            break

    np.save(output_dir / "loss_log.npy", np.asarray(loss_log, dtype=np.float32))
    if val_loss_log:
        np.save(output_dir / "val_loss_log.npy", np.asarray(val_loss_log, dtype=np.float32))

    ckpt_path = output_dir / "full_series_ts2vec_glucose.pt"
    torch.save(
        {
            "model_state_dict": encoder.state_dict(),
            "metadata": metadata,
            "loss_log": loss_log,
            "val_loss_log": val_loss_log,
            "best_val_loss": best_val_loss if best_epoch is not None else None,
            "best_epoch": best_epoch,
            "global_step": global_step,
        },
        ckpt_path,
    )
    print(f"Saved checkpoint to {ckpt_path}")

    if args.save_full_series_features and valid_data is not None:
        print("Encoding validation full-series features...")
        features = encode_full_series(
            encoder,
            valid_data,
            device=device,
            batch_size=args.batch_size,
        )
        np.save(output_dir / "valid_full_series_features.npy", features.astype(np.float32))
        print(f"Saved full-series validation features: {features.shape}")
        if wandb_run is not None:
            wandb.log(
                {"features/valid_full_series_count": int(features.shape[0])},
                step=global_step,
            )

    if wandb_run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
