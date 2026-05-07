#!/usr/bin/env python3
import argparse
import os
import sys

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Instantiate mujoco dataset from main project data_provider and print basic stats."
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=24,
        help="Sequence length passed to data_provider.datasets.mujoco.Mujoco",
    )
    parser.add_argument(
        "--datasets-dir",
        type=str,
        default="/Users/zhc/Documents/PhD/projects/ImagenFew/data",
        help="Root datasets directory used by the data provider",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="./TSG/mujoco0.0",
        help="Dataset relative path under datasets-dir",
    )
    return parser.parse_args()


def summarize_tensor(x: torch.Tensor):
    x = x.detach().cpu()
    shape = tuple(x.shape)
    ndim = x.ndim
    num_samples = shape[0] if ndim >= 1 else 0
    seq_len = shape[1] if ndim >= 2 else None
    num_channels = shape[2] if ndim >= 3 else None

    numel = x.numel()
    nan_count = int(torch.isnan(x).sum().item()) if x.is_floating_point() else 0
    nan_ratio = float(nan_count / numel) if numel > 0 else 0.0

    finite_vals = x
    if x.is_floating_point():
        finite_mask = torch.isfinite(x)
        finite_vals = x[finite_mask]

    if numel == 0 or finite_vals.numel() == 0:
        min_v = None
        max_v = None
        mean_v = None
        std_v = None
    else:
        min_v = float(finite_vals.min().item())
        max_v = float(finite_vals.max().item())
        mean_v = float(finite_vals.mean().item())
        std_v = float(finite_vals.std(unbiased=False).item())

    stats = {
        "shape": shape,
        "ndim": ndim,
        "num_samples": int(num_samples),
        "seq_len": int(seq_len) if seq_len is not None else None,
        "num_channels": int(num_channels) if num_channels is not None else None,
        "dtype": str(x.dtype),
        "min": min_v,
        "max": max_v,
        "mean": mean_v,
        "std": std_v,
        "nan_count": nan_count,
        "nan_ratio": nan_ratio,
    }
    return stats


def main():
    args = parse_args()
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from data_provider.datasets.mujoco import Mujoco

    full_path = os.path.abspath(os.path.join(args.datasets_dir, args.path))
    if not os.path.exists(full_path):
        print("Dataset path does not exist, cannot instantiate mujoco dataset yet.")
        print(f"datasets_dir : {os.path.abspath(args.datasets_dir)}")
        print(f"path         : {args.path}")
        print(f"full_path    : {full_path}")
        print("")
        print("Please pass a valid existing path, for example:")
        print("python scripts/inspect_mujoco_dataset.py --datasets-dir <root> --path <mujoco_rel_path>")
        sys.exit(2)

    dataset = Mujoco(
        seq_len=args.seq_len,
        path=args.path,
        datasets_dir=args.datasets_dir,
    )

    stats = summarize_tensor(dataset)

    print("=== Mujoco Dataset Stats ===")
    print(f"datasets_dir : {os.path.abspath(args.datasets_dir)}")
    print(f"path         : {args.path}")
    print(f"full_path    : {full_path}")
    print(f"shape        : {stats['shape']}")
    print(f"ndim         : {stats['ndim']}")
    print(f"samples      : {stats['num_samples']}")
    print(f"seq_len      : {stats['seq_len']}")
    print(f"channels     : {stats['num_channels']}")
    print(f"dtype        : {stats['dtype']}")
    print(f"min          : {stats['min']}")
    print(f"max          : {stats['max']}")
    print(f"mean         : {stats['mean']}")
    print(f"std          : {stats['std']}")
    print(f"nan_count    : {stats['nan_count']}")
    print(f"nan_ratio    : {stats['nan_ratio']:.6f}")


if __name__ == "__main__":
    main()
