"""
Train the multi-scale masked autoencoder on glucose windows.

This script mirrors the glucose data/loading/checkpoint conventions used by
train_full_series_ts2vec_glucose.py, but trains MultiScaleTimeSeriesMAE with
masked patch reconstruction.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from multi_scale_mae import MultiScaleTimeSeriesMAE

import wandb


def compute_min_max(parquet_path: str, column: str) -> Tuple[float, float]:
    df = pd.read_parquet(parquet_path, columns=[column])
    mins = []
    maxs = []
    for values in df[column]:
        arr = np.asarray(values, dtype=np.float32)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        mins.append(float(arr.min()))
        maxs.append(float(arr.max()))
    if not mins:
        raise ValueError(f"Could not compute min/max from {parquet_path}:{column}")
    return min(mins), max(maxs)


def load_glucose_windows(
    parquet_path: str,
    column: str,
    seq_len: int,
    stride: int,
    value_min: Optional[float] = None,
    value_max: Optional[float] = None,
    normalize: str = "minmax",
    max_windows: Optional[int] = None,
) -> np.ndarray:
    """Load glucose windows as a float32 array of shape (N, seq_len, 1)."""
    df = pd.read_parquet(parquet_path, columns=[column])
    windows = []

    if normalize == "minmax" and (value_min is None or value_max is None):
        value_min, value_max = compute_min_max(parquet_path, column)

    for values in df[column]:
        arr = np.asarray(values, dtype=np.float32)
        arr = arr[np.isfinite(arr)]
        if arr.size < seq_len:
            continue

        for start in range(0, arr.size - seq_len + 1, stride):
            window = arr[start : start + seq_len].astype(np.float32, copy=True)

            if normalize == "minmax":
                scale = max(float(value_max) - float(value_min), 1e-6)
                window = 2.0 * (window - float(value_min)) / scale - 1.0
                window = np.clip(window, -1.0, 1.0)
            elif normalize == "none":
                pass
            else:
                raise ValueError(f"Unknown normalize mode: {normalize}")

            windows.append(window[:, None])
            if max_windows is not None and len(windows) >= max_windows:
                return np.stack(windows, axis=0).astype(np.float32)

    if not windows:
        raise ValueError(f"No windows of length {seq_len} found in {parquet_path}")

    return np.stack(windows, axis=0).astype(np.float32)


def init_dl_program(
        device_name,
        seed=None,
        use_cudnn=True,
        deterministic=False,
        benchmark=False,
        use_tf32=False,
        max_threads=None
):
    import torch
    if max_threads is not None:
        torch.set_num_threads(max_threads)  # intraop
        if torch.get_num_interop_threads() != max_threads:
            torch.set_num_interop_threads(max_threads)  # interop
        try:
            import mkl
        except:
            pass
        else:
            mkl.set_num_threads(max_threads)

    if seed is not None:
        random.seed(seed)
        seed += 1
        np.random.seed(seed)
        seed += 1
        torch.manual_seed(seed)

    if isinstance(device_name, (str, int)):
        device_name = [device_name]

    devices = []
    for t in reversed(device_name):
        t_device = torch.device(t)
        devices.append(t_device)
        if t_device.type == 'cuda':
            assert torch.cuda.is_available()
            torch.cuda.set_device(t_device)
            if seed is not None:
                seed += 1
                torch.cuda.manual_seed(seed)
    devices.reverse()
    torch.backends.cudnn.enabled = use_cudnn
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark

    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = use_tf32
        torch.backends.cuda.matmul.allow_tf32 = use_tf32

    return devices if len(devices) > 1 else devices[0]


def parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_float_tuple(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_epoch(
    model: MultiScaleTimeSeriesMAE,
    loader: DataLoader,
    device: torch.device,
    do_mask: bool = True,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_steps = 0
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for (x,) in loader:
            x = x.to(device)
            loss = model(x, do_mask=do_mask)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_steps += 1

    return total_loss / max(total_steps, 1)


@torch.no_grad()
def encode_full_series(
    model: MultiScaleTimeSeriesMAE,
    data: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Encode [N, T, C] windows into concatenated pooled fused-scale features."""
    was_training = model.training
    model.eval()

    loader = DataLoader(
        TensorDataset(torch.from_numpy(data).float()),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    features = []
    for (x,) in loader:
        x = x.to(device)
        z_list = [model._encode_scale(x, idx, do_mask=False)[0] for idx in range(3)]
        fused = model._bridge(z_list)
        pooled = [z.max(dim=1).values for z in fused]
        features.append(torch.cat(pooled, dim=1).cpu())

    model.train(was_training)
    return torch.cat(features, dim=0).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train multi-scale MAE on glucose data")
    parser.add_argument("--data_root", type=str, default="./AI-READI")
    parser.add_argument("--train_file", type=str, default="glucose_train.parquet")
    parser.add_argument("--valid_file", type=str, default="glucose_valid.parquet")
    parser.add_argument("--column", type=str, default="glucose")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./feature_extractors/checkpoints/multi_scale_mae_glucose",
    )
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--normalize", type=str, default="minmax", choices=["minmax", "none"])
    parser.add_argument("--max_train_windows", type=int, default=None)
    parser.add_argument("--max_valid_windows", type=int, default=4096)

    parser.add_argument("--patch_sizes", type=str, default="8,16,32")
    parser.add_argument(
        "--strides",
        type=str,
        default=None,
        help="Comma-separated patch strides. Defaults to patch_size // 2.",
    )
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=None,
        help="Fused token dimension after the multi-scale bridge. Defaults to embed_dim.",
    )
    parser.add_argument("--encoder_depth", type=int, default=2)
    parser.add_argument("--bridge_depth", type=int, default=2)
    parser.add_argument("--decoder_depth", type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)
    parser.add_argument("--mask_ratio", type=float, default=0.25)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--loss_weights", type=str, default="1.0,1.0,1.0")

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--val_interval", type=int, default=10)
    parser.add_argument("--val_repeats", type=int, default=1)
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
        choices=["online", "offline", "disabled"],
        help="Weights & Biases mode.",
    )
    args = parser.parse_args()

    patch_sizes = parse_int_tuple(args.patch_sizes)
    strides = parse_int_tuple(args.strides) if args.strides is not None else None
    loss_weights = parse_float_tuple(args.loss_weights)
    if len(patch_sizes) != 3:
        raise ValueError("--patch_sizes must contain exactly three comma-separated integers")
    if strides is not None and len(strides) != 3:
        raise ValueError("--strides must contain exactly three comma-separated integers")
    if len(loss_weights) != 3:
        raise ValueError("--loss_weights must contain exactly three comma-separated numbers")

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

    valid_data = None
    if valid_path.exists() and args.max_valid_windows != 0:
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

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_data).float()),
        batch_size=min(args.batch_size, len(train_data)),
        shuffle=True,
        drop_last=True,
    )
    valid_loader = None
    if valid_data is not None:
        valid_loader = DataLoader(
            TensorDataset(torch.from_numpy(valid_data).float()),
            batch_size=min(args.batch_size, len(valid_data)),
            shuffle=False,
            drop_last=False,
        )

    model = MultiScaleTimeSeriesMAE(
        input_dims=train_data.shape[-1],
        seq_len=args.seq_len,
        patch_sizes=patch_sizes,
        strides=strides,
        embed_dim=args.embed_dim,
        latent_dim=args.latent_dim,
        encoder_depth=args.encoder_depth,
        bridge_depth=args.bridge_depth,
        decoder_depth=args.decoder_depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        mask_ratio=args.mask_ratio,
        dropout=args.dropout,
        loss_weights=loss_weights,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    metadata = {
        "training_objective": "multi_scale_masked_patch_reconstruction",
        "architecture": "MultiScaleTimeSeriesMAE",
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
        "patch_sizes": list(patch_sizes),
        "strides": list(model.strides),
        "embed_dim": args.embed_dim,
        "latent_dim": model.latent_dim,
        "encoder_depth": args.encoder_depth,
        "bridge_depth": args.bridge_depth,
        "decoder_depth": args.decoder_depth,
        "num_heads": args.num_heads,
        "mlp_ratio": args.mlp_ratio,
        "mask_ratio": args.mask_ratio,
        "dropout": args.dropout,
        "loss_weights": list(loss_weights),
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "iters": args.iters,
        "val_interval": args.val_interval,
        "val_repeats": args.val_repeats,
        "max_train_windows": args.max_train_windows,
        "max_valid_windows": args.max_valid_windows,
        "seed": args.seed,
        "device": str(device),
        "full_series_feature": {
            "pooling": "max over tokens per fused scale, concatenated across scales",
            "shape": ["N", model.latent_dim * 3],
        },
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

    print("Training multi-scale MAE...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        for (x,) in train_loader:
            if args.iters is not None and global_step >= args.iters:
                break

            x = x.to(device)
            loss = model(x, do_mask=True)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1

        avg_loss = epoch_loss / max(epoch_steps, 1)
        loss_log.append(avg_loss)
        print(f"Epoch #{epoch}: Train loss={avg_loss:.6f}")

        if wandb_run is not None:
            wandb.log(
                {
                    "train/loss": avg_loss,
                    "train/epoch": epoch,
                    "train/global_step": global_step,
                },
                step=global_step,
            )

        should_validate = (
            valid_loader is not None
            and args.val_interval > 0
            and epoch % args.val_interval == 0
            and epoch != 0
        )
        if should_validate:
            val_losses = [
                run_epoch(
                    model,
                    valid_loader,
                    device=device,
                    do_mask=True,
                    optimizer=None,
                )
                for _ in range(max(1, args.val_repeats))
            ]
            val_loss = float(np.mean(val_losses))
            val_loss_log.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_path = output_dir / "best_multi_scale_mae_glucose.pt"
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
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
                f"Epoch #{epoch}: Train loss={avg_loss:.6f} | "
                f"val_loss={val_loss:.6f} | best_val_loss={best_val_loss:.6f}"
            )
            if wandb_run is not None:
                wandb.log(
                    {
                        "val/loss": val_loss,
                        "val/best_loss": best_val_loss,
                        "val/best_epoch": best_epoch,
                        "val/global_step": global_step,
                    },
                    step=global_step,
                )

        if args.iters is not None and global_step >= args.iters:
            break

    ckpt_path = output_dir / "multi_scale_mae_glucose.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
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
            model,
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
