#!/usr/bin/env python3
"""Inspect per-modality time-series irregularity in AIREADI daily windows."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

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
        return_dict=False,
    )

    summary = summarize_irregularity(dataset, args.modalities)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print("\nAIREADI daily-window irregularity summary:\n")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6g}"))

    if args.csv_out:
        out_path = Path(args.csv_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_path, index=False)
        print(f"\nSaved CSV: {out_path}")


if __name__ == "__main__":
    main()
