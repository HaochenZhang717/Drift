"""
Train a TS2Vec feature encoder on glucose windows.

The encoder is trained with the standard TS2Vec hierarchical contrastive
objective. Full-series features are produced by max-pooling the trained
per-timestep representations with encoding_window="full_series"; this is the
representation intended for the downstream drifting loss.
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch

from feature_extractors.ts2vec.ts2vec import TS2Vec
from feature_extractors.ts2vec.utils import init_dl_program


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


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TS2Vec on glucose data")
    parser.add_argument("--data_root", type=str, default="./AI-READI")
    parser.add_argument("--train_file", type=str, default="glucose_train.parquet")
    parser.add_argument("--valid_file", type=str, default="glucose_valid.parquet")
    parser.add_argument("--column", type=str, default="glucose")
    parser.add_argument("--output_dir", type=str, default="./feature_extractors/checkpoints/ts2vec_glucose")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--normalize", type=str, default="minmax", choices=["minmax", "none"])
    parser.add_argument("--max_train_windows", type=int, default=None)
    parser.add_argument("--max_valid_windows", type=int, default=4096)

    parser.add_argument("--output_dims", type=int, default=320)
    parser.add_argument("--hidden_dims", type=int, default=64)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--max_train_length", type=int, default=None)
    parser.add_argument("--temporal_unit", type=int, default=0)
    parser.add_argument(
        "--mask_prob",
        type=float,
        default=0.5,
        help="Probability of keeping each timestamp for TS2Vec binomial masking.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_threads", type=int, default=None)
    parser.add_argument("--save_full_series_features", action="store_true")
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

    metadata = {
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
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "iters": args.iters,
        "max_train_length": args.max_train_length,
        "temporal_unit": args.temporal_unit,
        "mask_prob": args.mask_prob,
        "seed": args.seed,
        "device": str(device),
        "full_series_feature": {
            "encoding_window": "full_series",
            "pooling": "max over time in TS2Vec.encode",
            "shape": ["N", args.output_dims],
        },
    }
    save_json(output_dir / "metadata.json", metadata)

    model = TS2Vec(
        input_dims=train_data.shape[-1],
        output_dims=args.output_dims,
        hidden_dims=args.hidden_dims,
        depth=args.depth,
        device=device,
        lr=args.lr,
        batch_size=args.batch_size,
        max_train_length=args.max_train_length,
        temporal_unit=args.temporal_unit,
        mask_prob=args.mask_prob,
    )

    print("Training TS2Vec...")
    loss_log = model.fit(
        train_data,
        n_epochs=args.epochs if args.iters is None else None,
        n_iters=args.iters,
        verbose=True,
    )
    np.save(output_dir / "loss_log.npy", np.asarray(loss_log, dtype=np.float32))

    ckpt_path = output_dir / "ts2vec_glucose.pt"
    torch.save(
        {
            "averaged_model_state_dict": model.net.state_dict(),
            "raw_model_state_dict": model._net.state_dict(),
            "metadata": metadata,
            "loss_log": loss_log,
        },
        ckpt_path,
    )
    model.save(output_dir / "ts2vec_glucose_averaged_state_dict.pt")
    print(f"Saved checkpoint to {ckpt_path}")

    if args.save_full_series_features and valid_data is not None:
        print("Encoding validation full-series features...")
        features = model.encode(
            valid_data,
            encoding_window="full_series",
            batch_size=args.batch_size,
        )
        np.save(output_dir / "valid_full_series_features.npy", features.astype(np.float32))
        print(f"Saved full-series validation features: {features.shape}")


if __name__ == "__main__":
    main()
