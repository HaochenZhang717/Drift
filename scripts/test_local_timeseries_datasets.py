import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data_provider.data_provider import data_provider
from data_provider.datasets import ErcotData, HouseholdData


def _to_tensor(sample):
    if torch.is_tensor(sample):
        return sample.detach().cpu()
    return torch.as_tensor(sample)


def _assert_sample_ok(name, sample, seq_len):
    sample = _to_tensor(sample)
    if sample.ndim != 2:
        raise AssertionError(f"{name} sample should have shape (T, C), got {tuple(sample.shape)}")
    if sample.shape[0] != seq_len:
        raise AssertionError(f"{name} sample length should be {seq_len}, got {sample.shape[0]}")
    if sample.shape[1] <= 0:
        raise AssertionError(f"{name} should have at least one channel.")
    if not torch.isfinite(sample).all():
        raise AssertionError(f"{name} sample contains NaN or infinite values.")
    return sample


def _assert_scaled_range(name, dataset, tolerance=1e-5):
    data = _to_tensor(dataset.data)
    data_min = float(data.min())
    data_max = float(data.max())
    if data_min < -1.0 - tolerance or data_max > 1.0 + tolerance:
        raise AssertionError(
            f"{name} data should be scaled to [-1, 1], got min={data_min:.6f}, max={data_max:.6f}"
        )
    return data_min, data_max


def _plot_samples(name, train_sample, valid_sample, output_dir, max_channels):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    train_np = _to_tensor(train_sample).numpy()
    valid_np = _to_tensor(valid_sample).numpy()

    channels = min(train_np.shape[1], max_channels)
    fig, axes = plt.subplots(
        channels,
        1,
        figsize=(11, max(2.6, 2.0 * channels)),
        sharex=True,
        constrained_layout=True,
    )
    if channels == 1:
        axes = [axes]

    x_train = np.arange(train_np.shape[0])
    x_valid = np.arange(valid_np.shape[0])
    for channel_idx, ax in enumerate(axes):
        ax.plot(x_train, train_np[:, channel_idx], label="train", color="#2563eb", linewidth=1.6)
        ax.plot(x_valid, valid_np[:, channel_idx], label="valid", color="#dc2626", linewidth=1.4, alpha=0.85)
        ax.set_ylabel(f"ch {channel_idx}")
        ax.grid(True, alpha=0.25)
        if channel_idx == 0:
            ax.legend(loc="upper right")

    axes[-1].set_xlabel("time index within sample window")
    fig.suptitle(f"{name}: first train and valid windows")
    output_path = output_dir / f"{name}_train_valid_preview.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"[PLOT] {name} preview saved to {output_path}")


def _test_dataset_class(name, dataset_cls, datasets_dir, rel_path, seq_len, visualize=False, output_dir=None, max_plot_channels=6):
    train = dataset_cls(
        datasets_dir=datasets_dir,
        rel_path=rel_path,
        flag="train",
        seq_len=seq_len,
    )
    valid = dataset_cls(
        datasets_dir=datasets_dir,
        rel_path=rel_path,
        flag="val",
        seq_len=seq_len,
    )

    if len(train) <= 0:
        raise AssertionError(f"{name} train split is empty.")
    if len(valid) <= 0:
        raise AssertionError(f"{name} valid split is empty.")

    train_sample = _assert_sample_ok(f"{name} train", train[0], seq_len)
    valid_sample = _assert_sample_ok(f"{name} valid", valid[0], seq_len)
    train_min, train_max = _assert_scaled_range(f"{name} train", train)
    valid_min, valid_max = _assert_scaled_range(f"{name} valid", valid)
    if train_sample.shape[1] != valid_sample.shape[1]:
        raise AssertionError(
            f"{name} train/valid channel mismatch: {train_sample.shape[1]} vs {valid_sample.shape[1]}"
        )

    print(
        f"[OK] {name} class loader | train={len(train)} valid={len(valid)} "
        f"sample_shape={tuple(train_sample.shape)} "
        f"train_range=({train_min:.3f}, {train_max:.3f}) "
        f"valid_range=({valid_min:.3f}, {valid_max:.3f})"
    )
    if visualize:
        _plot_samples(name, train_sample, valid_sample, output_dir, max_plot_channels)


def _test_provider(name, data_name, datasets_dir, rel_path, seq_len, batch_size):
    args = SimpleNamespace(
        datasets=[{"name": name, "data": data_name, "rel_path": rel_path}],
        train_on_datasets=[name],
        seq_len=seq_len,
        datasets_dir=datasets_dir,
        batch_size=batch_size,
        num_workers=0,
        finetune=False,
        input_channels=None,
    )
    loader, _, trainsets, metadatas = data_provider(args)
    validset, class_idx = loader.gen_dataloader(name)

    if name not in trainsets:
        raise AssertionError(f"{name} missing from provider trainsets.")
    if len(trainsets[name]) <= 0:
        raise AssertionError(f"{name} provider train split is empty.")
    if len(validset) <= 0:
        raise AssertionError(f"{name} provider valid split is empty.")

    train_sample = _assert_sample_ok(f"{name} provider train", trainsets[name][0], seq_len)
    valid_sample = _assert_sample_ok(f"{name} provider valid", validset[0], seq_len)
    if train_sample.shape[1] != metadatas[name]["channels"]:
        raise AssertionError(
            f"{name} metadata channels={metadatas[name]['channels']} "
            f"does not match sample channels={train_sample.shape[1]}."
        )
    if valid_sample.shape[1] != train_sample.shape[1]:
        raise AssertionError(f"{name} provider train/valid channel mismatch.")

    print(
        f"[OK] {name} data_provider | train={len(trainsets[name])} valid={len(validset)} "
        f"channels={metadatas[name]['channels']} class_idx={class_idx}"
    )


def main():
    parser = argparse.ArgumentParser(description="Smoke-test ErcotData and HouseholdData loaders.")
    parser.add_argument(
        "--datasets-dir",
        default="/Users/zhc/Documents/Time_Series_Datasets",
        help="Directory containing the local time-series datasets.",
    )
    parser.add_argument("--household-rel-path", default="HouseHold_6.csv")
    parser.add_argument("--ercot-rel-path", default="ERCOT_merged.csv")
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save train/valid preview plots for datasets that load successfully.",
    )
    parser.add_argument(
        "--plot-output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "local_timeseries_dataset_tests",
        help="Directory for visualization PNG files.",
    )
    parser.add_argument("--max-plot-channels", type=int, default=6)
    parser.add_argument(
        "--skip-ercot",
        action="store_true",
        help="Skip ERCOT checks when ERCOT CSV data is not available yet.",
    )
    parser.add_argument(
        "--require-ercot",
        action="store_true",
        help="Fail if ERCOT checks cannot run because ERCOT CSV data is unavailable.",
    )
    args = parser.parse_args()

    _test_dataset_class(
        "HouseholdData",
        HouseholdData,
        args.datasets_dir,
        args.household_rel_path,
        args.seq_len,
        visualize=args.visualize,
        output_dir=args.plot_output_dir,
        max_plot_channels=args.max_plot_channels,
    )
    _test_provider(
        "HouseholdData",
        "HouseholdData",
        args.datasets_dir,
        args.household_rel_path,
        args.seq_len,
        args.batch_size,
    )

    if args.skip_ercot:
        print("[SKIP] ErcotData checks skipped.")
        return

    try:
        _test_dataset_class(
            "ErcotData",
            ErcotData,
            args.datasets_dir,
            args.ercot_rel_path,
            args.seq_len,
            visualize=args.visualize,
            output_dir=args.plot_output_dir,
            max_plot_channels=args.max_plot_channels,
        )
        _test_provider(
            "ErcotData",
            "ErcotData",
            args.datasets_dir,
            args.ercot_rel_path,
            args.seq_len,
            args.batch_size,
        )
    except FileNotFoundError as exc:
        if args.require_ercot:
            raise
        print(f"[SKIP] ErcotData checks skipped: {exc}")


if __name__ == "__main__":
    main()
