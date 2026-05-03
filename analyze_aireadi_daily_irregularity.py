#!/usr/bin/env python3
"""Inspect per-modality time-series irregularity in AIREADI daily windows."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.utils_dataset import AIREADIModalityImputationDataset, AIREADI_MODALITY_SPECS


def _safe_float(x: float) -> float:
    if x is None:
        return float("nan")
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize_irregularity(
    dataset: AIREADIModalityImputationDataset,
    modalities: list[str],
) -> pd.DataFrame:
    total_windows = len(dataset.window_index)

    per_mod_counts: dict[str, list[int]] = defaultdict(list)
    per_mod_gaps_min: dict[str, list[float]] = defaultdict(list)
    per_mod_window_gap_cv: dict[str, list[float]] = defaultdict(list)

    for patient_id, window_ref in dataset.window_index:
        window_start_ns, window_end_ns = dataset._window_bounds(patient_id, window_ref)

        for modality in modalities:
            indices = dataset._event_indices_in_window(
                patient_id=patient_id,
                modality=modality,
                window_start_ns=window_start_ns,
                window_end_ns=window_end_ns,
            )
            count = int(indices.size)
            per_mod_counts[modality].append(count)

            if count <= 1:
                continue

            item = dataset.patient_data[patient_id][modality]
            ts = np.sort(item["time_local"][indices].astype(np.int64))
            dt_min = np.diff(ts).astype(np.float64) / 60e9
            if dt_min.size == 0:
                continue

            valid = np.isfinite(dt_min) & (dt_min >= 0)
            dt_min = dt_min[valid]
            if dt_min.size == 0:
                continue

            per_mod_gaps_min[modality].extend(dt_min.tolist())

            m = float(np.mean(dt_min))
            s = float(np.std(dt_min))
            if m > 0:
                per_mod_window_gap_cv[modality].append(s / m)

    rows: list[dict[str, Any]] = []
    for modality in modalities:
        counts = per_mod_counts[modality]
        observed_windows = int(sum(c > 0 for c in counts))
        coverage = observed_windows / total_windows if total_windows > 0 else float("nan")
        missing_window_ratio = 1.0 - coverage if np.isfinite(coverage) else float("nan")

        gaps = per_mod_gaps_min[modality]
        cvals = per_mod_window_gap_cv[modality]

        rows.append(
            {
                "modality": modality,
                "windows_total": int(total_windows),
                "windows_with_events": observed_windows,
                "window_coverage": _safe_float(coverage),
                "missing_window_ratio": _safe_float(missing_window_ratio),
                "event_count_mean": _safe_float(np.mean(counts) if counts else np.nan),
                "event_count_median": _safe_float(np.median(counts) if counts else np.nan),
                "event_count_p90": _percentile([float(x) for x in counts], 90),
                "event_count_max": _safe_float(np.max(counts) if counts else np.nan),
                "gap_minutes_mean": _safe_float(np.mean(gaps) if gaps else np.nan),
                "gap_minutes_median": _safe_float(np.median(gaps) if gaps else np.nan),
                "gap_minutes_p90": _percentile(gaps, 90),
                "gap_minutes_p99": _percentile(gaps, 99),
                "gap_cv_mean": _safe_float(np.mean(cvals) if cvals else np.nan),
                "gap_cv_median": _safe_float(np.median(cvals) if cvals else np.nan),
            }
        )

    df = pd.DataFrame(rows)
    return df.sort_values("missing_window_ratio", ascending=False).reset_index(drop=True)


def collect_daily_event_counts(
    dataset: AIREADIModalityImputationDataset,
    modalities: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for patient_id, window_ref in dataset.window_index:
        window_start_ns, window_end_ns = dataset._window_bounds(patient_id, window_ref)
        day = pd.to_datetime(window_start_ns).date().isoformat()
        for modality in modalities:
            indices = dataset._event_indices_in_window(
                patient_id=patient_id,
                modality=modality,
                window_start_ns=window_start_ns,
                window_end_ns=window_end_ns,
            )
            rows.append(
                {
                    "patient_id": patient_id,
                    "day": day,
                    "modality": modality,
                    "event_count": int(indices.size),
                }
            )
    return pd.DataFrame(rows)


def plot_modality_histograms(
    count_df: pd.DataFrame,
    out_dir: Path,
    bins: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for modality, sub in count_df.groupby("modality", sort=True):
        values = sub["event_count"].to_numpy(dtype=np.int64)
        vmax = int(values.max()) if values.size else 1
        hist_bins = min(max(10, bins), max(10, vmax + 1))

        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        ax.hist(values, bins=hist_bins, color="#4C78A8", edgecolor="white")
        ax.set_title(f"{modality}: Daily Event Count Distribution")
        ax.set_xlabel("Event Count per Day")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / f"{modality}_daily_event_count_hist.png", dpi=160)
        plt.close(fig)


def test_daily_calorie_alignment(
    dataset: AIREADIModalityImputationDataset,
    sample_limit: int = 200,
    expected_fill_value: float = 0.0,
    plot_samples: int = 50,
    plot_out_dir: Path | None = None,
    shuffle: bool = True,
    seed: int = 42,
    require_observed_for_plot: bool = True,
) -> dict[str, int]:
    aligned_specs = [
        ("calorie", "calorie_aligned_to_glucose"),
        ("heart_rate", "heart_rate_aligned_to_glucose"),
        ("respiratory_rate", "respiratory_rate_aligned_to_glucose"),
        ("physical_activity", "physical_activity_aligned_to_glucose"),
    ]
    stats = {
        "checked_samples": 0,
        "missing_alignment_key": 0,
        "shape_mismatch": 0,
        "fill_value_mismatch": 0,
        "ok": 0,
    }

    plotted_by_key = {aligned_key: 0 for _, aligned_key in aligned_specs}
    if plot_out_dir is not None:
        plot_out_dir.mkdir(parents=True, exist_ok=True)

    n = min(len(dataset), max(0, int(sample_limit)))
    indices = np.arange(len(dataset), dtype=np.int64)
    if shuffle:
        rng = np.random.default_rng(int(seed))
        rng.shuffle(indices)
    indices = indices[:n]

    for sample_pos, idx in enumerate(indices):
        item = dataset[int(idx)]
        stats["checked_samples"] += 1
        modalities = item.get("modalities", {})
        sample_ok = True
        for raw_key, aligned_key in aligned_specs:
            raw_data = modalities.get(raw_key)
            if raw_data is None:
                continue
            aligned = modalities.get(aligned_key)
            if aligned is None:
                stats["missing_alignment_key"] += 1
                sample_ok = False
                continue

            values = aligned["values"].detach().cpu().numpy().reshape(-1)
            mask = aligned["mask"].detach().cpu().numpy().reshape(-1)
            if values.shape != mask.shape:
                stats["shape_mismatch"] += 1
                sample_ok = False
                continue

            missing_positions = mask <= 0.0
            if np.any(missing_positions):
                missing_vals = values[missing_positions]
                if not np.allclose(missing_vals, expected_fill_value):
                    stats["fill_value_mismatch"] += 1
                    sample_ok = False
                    continue

            if plotted_by_key[aligned_key] < max(0, int(plot_samples)):
                aligned_t_ns = aligned["time_local"].detach().cpu().numpy().reshape(-1)
                aligned_t = pd.to_datetime(aligned_t_ns)
                aligned_v = values

                raw_t_ns = raw_data["time_local"].detach().cpu().numpy().reshape(-1)
                raw_t = pd.to_datetime(raw_t_ns)
                raw_v = raw_data["values"].detach().cpu().numpy().reshape(-1)
                has_observed = bool(np.any(mask > 0.0))
                if require_observed_for_plot and (not has_observed):
                    continue

                fig, ax = plt.subplots(figsize=(10, 4.8))
                if raw_v.size > 0:
                    ax.plot(raw_t, raw_v, "o-", markersize=3, linewidth=1.0, alpha=0.8, label=f"raw_{raw_key}")
                if aligned_v.size > 0:
                    ax.plot(
                        aligned_t,
                        aligned_v,
                        "o-",
                        markersize=3,
                        linewidth=1.0,
                        alpha=0.9,
                        label=aligned_key,
                    )
                    missing_aligned = mask <= 0.0
                    if np.any(missing_aligned):
                        ax.scatter(
                            aligned_t[missing_aligned],
                            aligned_v[missing_aligned],
                            s=18,
                            marker="x",
                            label=f"aligned_missing({expected_fill_value:g})",
                        )

                ax.set_title(
                    f"Alignment Check ({raw_key}) | sample_idx={int(idx)} | draw={sample_pos} | "
                    f"patient={item.get('patient_id', '')}"
                )
                ax.set_xlabel("Time")
                ax.set_ylabel("Value")
                ax.grid(True, alpha=0.25)
                ax.legend(loc="best")
                fig.tight_layout()
                fig.savefig(plot_out_dir / f"{raw_key}_alignment_compare_sample_{int(idx)}.png", dpi=160)
                plt.close(fig)
                plotted_by_key[aligned_key] += 1

        if sample_ok:
            stats["ok"] += 1

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect irregularity of each time-series modality under AIREADI daily windows.",
    )
    parser.add_argument("--root", type=str, required=True, help="Directory containing *_<split>.parquet files.")
    parser.add_argument("--participants_tsv_path", type=str, default=None)
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--anchor_modality", type=str, default="glucose")
    parser.add_argument("--target_modality", type=str, default="glucose")
    parser.add_argument("--window_size", type=int, default=288)
    parser.add_argument("--daily_min_events", type=int, default=288)
    parser.add_argument("--max_anchor_gap_minutes", type=float, default=10.0)
    parser.add_argument("--max_window_span_hours", type=float, default=24.0)
    parser.add_argument("--anchor_sampling_minutes", type=float, default=5.0)
    parser.add_argument("--anchor_sampling_tolerance_seconds", type=float, default=2.0)
    parser.add_argument("--raw_values", action="store_true", help="Keep raw values (no normalization).")
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=list(AIREADI_MODALITY_SPECS.keys()),
        help="Subset of modalities to analyze.",
    )
    parser.add_argument("--csv_out", type=str, default="", help="Optional output CSV path.")
    parser.add_argument(
        "--daily_count_csv_out",
        type=str,
        default="",
        help="Optional CSV path for per-day event counts (patient_id/day/modality/event_count).",
    )
    parser.add_argument(
        "--hist_out_dir",
        type=str,
        default="",
        help="Output directory for per-modality daily event-count histograms.",
    )
    parser.add_argument("--hist_bins", type=int, default=50, help="Max bins for histogram.")
    parser.add_argument(
        "--test_calorie_alignment",
        action="store_true",
        help="Run sanity checks for daily modality->glucose alignment outputs in dataset['modalities'].",
    )
    parser.add_argument("--test_samples", type=int, default=200, help="How many samples to check.")
    parser.add_argument("--test_plot_samples", type=int, default=50, help="How many checked samples to plot.")
    parser.add_argument("--test_shuffle", action="store_true", help="Shuffle samples before testing/plotting.")
    parser.add_argument("--test_seed", type=int, default=42, help="Random seed for shuffled sampling.")
    parser.add_argument(
        "--test_require_observed_for_plot",
        action="store_true",
        help="Only plot samples with at least one aligned observed point (mask>0).",
    )
    parser.add_argument(
        "--test_plot_out_dir",
        type=str,
        default="",
        help="Output directory for alignment comparison plots.",
    )
    args = parser.parse_args()

    unknown = sorted(set(args.modalities) - set(AIREADI_MODALITY_SPECS.keys()))
    if unknown:
        raise ValueError(f"Unknown modalities: {unknown}")

    min_events = {args.anchor_modality: args.daily_min_events}
    max_events = {args.anchor_modality: args.window_size}

    dataset = AIREADIModalityImputationDataset(
        root=args.root,
        split=args.split,
        modalities=args.modalities,
        anchor_modality=args.anchor_modality,
        target_modality=args.target_modality,
        window_size=args.window_size,
        window_stride=args.window_size,
        window_mode="daily",
        daily_min_events=args.daily_min_events,
        max_events_per_modality=max_events,
        min_events_per_modality=min_events,
        normalize=not args.raw_values,
        max_anchor_gap_minutes=args.max_anchor_gap_minutes,
        max_window_span_hours=args.max_window_span_hours,
        anchor_sampling_minutes=args.anchor_sampling_minutes,
        anchor_sampling_tolerance_seconds=args.anchor_sampling_tolerance_seconds,
        participants_tsv_path=args.participants_tsv_path,
        include_last_window=True,
        require_complete=False,
        pad=False,
        include_clinical_static=False,
        include_participant_metadata=args.participants_tsv_path is not None,
        include_study_group=args.participants_tsv_path is not None,
        include_clinical_site=False,
        return_dict=True,
    )

    summary = summarize_irregularity(dataset, args.modalities)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print("\nAIREADI daily-window irregularity summary:\n")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6g}"))

    daily_counts = collect_daily_event_counts(dataset, args.modalities)
    if args.daily_count_csv_out:
        count_out = Path(args.daily_count_csv_out)
        count_out.parent.mkdir(parents=True, exist_ok=True)
        daily_counts.to_csv(count_out, index=False)
        print(f"\nSaved daily event-count CSV: {count_out}")

    if args.hist_out_dir:
        hist_out_dir = Path(args.hist_out_dir)
    else:
        hist_out_dir = Path("outputs") / "aireadi_daily_event_count_hists"
    plot_modality_histograms(daily_counts, hist_out_dir, bins=max(5, int(args.hist_bins)))
    print(f"\nSaved per-modality histograms under: {hist_out_dir}")

    if args.csv_out:
        out_path = Path(args.csv_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_path, index=False)
        print(f"\nSaved CSV: {out_path}")

    if args.test_calorie_alignment:
        print("\nRunning daily alignment checks...")
        test_plot_out_dir = (
            Path(args.test_plot_out_dir)
            if args.test_plot_out_dir
            else Path("outputs") / "aireadi_daily_calorie_alignment_checks"
        )
        test_stats = test_daily_calorie_alignment(
            dataset=dataset,
            sample_limit=args.test_samples,
            expected_fill_value=0.0,
            plot_samples=args.test_plot_samples,
            plot_out_dir=test_plot_out_dir,
            shuffle=args.test_shuffle,
            seed=args.test_seed,
            require_observed_for_plot=args.test_require_observed_for_plot,
        )
        print(
            "Alignment test stats: "
            f"checked={test_stats['checked_samples']} | "
            f"ok={test_stats['ok']} | "
            f"missing_key={test_stats['missing_alignment_key']} | "
            f"shape_mismatch={test_stats['shape_mismatch']} | "
            f"fill_value_mismatch={test_stats['fill_value_mismatch']}"
        )
        print(f"Saved alignment comparison plots under: {test_plot_out_dir}")


if __name__ == "__main__":
    main()
