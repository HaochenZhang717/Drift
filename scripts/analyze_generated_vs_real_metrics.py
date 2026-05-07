#!/usr/bin/env python3
"""Analyze generated-vs-real classifier metrics across MM checkpoint sweeps.

Inputs:
- outputs/eval_results/mm_ckpt_sweep/*.json
- outputs/eval_results/generated_vs_real_classifier_gap.json

Outputs (in outputs/eval_results/analysis_generated_vs_real):
- mm_sweep_metrics.csv
- baseline_metrics.csv
- summary_table.tex
- checkpoint_rankings.tex
- acc_macro_f1_vs_epoch.png
- gap_vs_epoch.png
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class Metrics:
    real_acc_mean: float
    real_acc_std: float
    gen_acc_mean: float
    gen_acc_std: float
    real_f1_mean: float
    real_f1_std: float
    gen_f1_mean: float
    gen_f1_std: float
    gap_acc_mean: float
    gap_acc_std: float
    gap_f1_mean: float
    gap_f1_std: float


def _get_mean_std(agg: Dict, key: str) -> Tuple[float, float]:
    v = agg[key]
    return float(v["mean"]), float(v["std"])


def parse_metrics(obj: Dict) -> Metrics:
    agg = obj["aggregate"]
    real_acc_mean, real_acc_std = _get_mean_std(agg, "real.acc")
    gen_acc_mean, gen_acc_std = _get_mean_std(agg, "generated.acc")
    real_f1_mean, real_f1_std = _get_mean_std(agg, "real.macro_f1")
    gen_f1_mean, gen_f1_std = _get_mean_std(agg, "generated.macro_f1")
    gap_acc_mean, gap_acc_std = _get_mean_std(agg, "gap.delta_acc_gen_minus_real")
    gap_f1_mean, gap_f1_std = _get_mean_std(agg, "gap.delta_macro_f1_gen_minus_real")
    return Metrics(
        real_acc_mean,
        real_acc_std,
        gen_acc_mean,
        gen_acc_std,
        real_f1_mean,
        real_f1_std,
        gen_f1_mean,
        gen_f1_std,
        gap_acc_mean,
        gap_acc_std,
        gap_f1_mean,
        gap_f1_std,
    )


def parse_epoch_ema(filename: str) -> Tuple[int, int]:
    m = re.search(r"epoch(\d+)_ema(\d+)", filename)
    if not m:
        raise ValueError(f"Cannot parse epoch/ema from filename: {filename}")
    return int(m.group(1)), int(m.group(2))


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_mm_df(mm_dir: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    for p in sorted(mm_dir.glob("*.json")):
        epoch, ema = parse_epoch_ema(p.name)
        metrics = parse_metrics(load_json(p))
        rows.append(
            {
                "source": p.name,
                "epoch": epoch,
                "ema": ema,
                "real_acc_mean": metrics.real_acc_mean,
                "real_acc_std": metrics.real_acc_std,
                "gen_acc_mean": metrics.gen_acc_mean,
                "gen_acc_std": metrics.gen_acc_std,
                "real_f1_mean": metrics.real_f1_mean,
                "real_f1_std": metrics.real_f1_std,
                "gen_f1_mean": metrics.gen_f1_mean,
                "gen_f1_std": metrics.gen_f1_std,
                "gap_acc_mean": metrics.gap_acc_mean,
                "gap_acc_std": metrics.gap_acc_std,
                "gap_f1_mean": metrics.gap_f1_mean,
                "gap_f1_std": metrics.gap_f1_std,
            }
        )
    df = pd.DataFrame(rows).sort_values(["ema", "epoch"]).reset_index(drop=True)
    return df


def build_baseline_df(path: Path) -> pd.DataFrame:
    metrics = parse_metrics(load_json(path))
    return pd.DataFrame(
        [
            {
                "name": "generated_vs_real_baseline",
                "real_acc_mean": metrics.real_acc_mean,
                "real_acc_std": metrics.real_acc_std,
                "gen_acc_mean": metrics.gen_acc_mean,
                "gen_acc_std": metrics.gen_acc_std,
                "real_f1_mean": metrics.real_f1_mean,
                "real_f1_std": metrics.real_f1_std,
                "gen_f1_mean": metrics.gen_f1_mean,
                "gen_f1_std": metrics.gen_f1_std,
                "gap_acc_mean": metrics.gap_acc_mean,
                "gap_acc_std": metrics.gap_acc_std,
                "gap_f1_mean": metrics.gap_f1_mean,
                "gap_f1_std": metrics.gap_f1_std,
            }
        ]
    )


def plot_acc_f1(mm_df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ema, g in mm_df.groupby("ema"):
        g = g.sort_values("epoch")
        axes[0, 0].errorbar(g["epoch"], g["real_acc_mean"], yerr=g["real_acc_std"], marker="o", capsize=3, label=f"ema{ema}")
        axes[0, 1].errorbar(g["epoch"], g["gen_acc_mean"], yerr=g["gen_acc_std"], marker="o", capsize=3, label=f"ema{ema}")
        axes[1, 0].errorbar(g["epoch"], g["real_f1_mean"], yerr=g["real_f1_std"], marker="o", capsize=3, label=f"ema{ema}")
        axes[1, 1].errorbar(g["epoch"], g["gen_f1_mean"], yerr=g["gen_f1_std"], marker="o", capsize=3, label=f"ema{ema}")

    axes[0, 0].set_title("Real Accuracy vs Epoch")
    axes[0, 1].set_title("Generated Accuracy vs Epoch")
    axes[1, 0].set_title("Real Macro-F1 vs Epoch")
    axes[1, 1].set_title("Generated Macro-F1 vs Epoch")

    for ax in axes.ravel():
        ax.grid(alpha=0.3)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")

    axes[0, 0].legend(title="Checkpoint")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_gaps(mm_df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    for ema, g in mm_df.groupby("ema"):
        g = g.sort_values("epoch")
        axes[0].errorbar(g["epoch"], g["gap_acc_mean"], yerr=g["gap_acc_std"], marker="o", capsize=3, label=f"ema{ema}")
        axes[1].errorbar(g["epoch"], g["gap_f1_mean"], yerr=g["gap_f1_std"], marker="o", capsize=3, label=f"ema{ema}")

    axes[0].axhline(0, color="black", linewidth=1, linestyle="--")
    axes[1].axhline(0, color="black", linewidth=1, linestyle="--")
    axes[0].set_title("Accuracy Gap (Generated - Real)")
    axes[1].set_title("Macro-F1 Gap (Generated - Real)")

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Delta")

    axes[0].legend(title="Checkpoint")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _fmt_pm(mean: float, std: float, nd: int = 4) -> str:
    return f"{mean:.{nd}f} $\\pm$ {std:.{nd}f}"


def write_summary_latex(mm_df: pd.DataFrame, base_df: pd.DataFrame, out: Path) -> None:
    best_acc_idx = mm_df["gen_acc_mean"].idxmax()
    best_f1_idx = mm_df["gen_f1_mean"].idxmax()
    best_gap_acc_idx = mm_df["gap_acc_mean"].idxmax()
    best_gap_f1_idx = mm_df["gap_f1_mean"].idxmax()

    picks = [
        ("Baseline", None, None, base_df.iloc[0]),
        ("Best Generated Acc (MM)", int(mm_df.loc[best_acc_idx, "epoch"]), int(mm_df.loc[best_acc_idx, "ema"]), mm_df.loc[best_acc_idx]),
        ("Best Generated Macro-F1 (MM)", int(mm_df.loc[best_f1_idx, "epoch"]), int(mm_df.loc[best_f1_idx, "ema"]), mm_df.loc[best_f1_idx]),
        ("Largest Acc Gap (MM)", int(mm_df.loc[best_gap_acc_idx, "epoch"]), int(mm_df.loc[best_gap_acc_idx, "ema"]), mm_df.loc[best_gap_acc_idx]),
        ("Largest Macro-F1 Gap (MM)", int(mm_df.loc[best_gap_f1_idx, "epoch"]), int(mm_df.loc[best_gap_f1_idx, "ema"]), mm_df.loc[best_gap_f1_idx]),
    ]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Accuracy and Macro-F1 comparison between baseline and selected MM checkpoints.}")
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append(r"Setting & Epoch & EMA & Generated Acc & Generated Macro-F1 & Gap (Acc / F1) \\")
    lines.append(r"\midrule")

    for label, ep, ema, row in picks:
        ep_str = "-" if ep is None else str(ep)
        ema_str = "-" if ema is None else str(ema)
        gacc = _fmt_pm(float(row["gen_acc_mean"]), float(row["gen_acc_std"]))
        gf1 = _fmt_pm(float(row["gen_f1_mean"]), float(row["gen_f1_std"]))
        gap = f"{_fmt_pm(float(row['gap_acc_mean']), float(row['gap_acc_std']))} / {_fmt_pm(float(row['gap_f1_mean']), float(row['gap_f1_std']))}"
        lines.append(f"{label} & {ep_str} & {ema_str} & {gacc} & {gf1} & {gap} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\label{tab:mm_baseline_summary}")
    lines.append(r"\end{table}")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_ranking_latex(mm_df: pd.DataFrame, out: Path) -> None:
    rank = mm_df.sort_values(["gen_acc_mean", "gen_f1_mean"], ascending=False).copy()
    rank = rank[["epoch", "ema", "gen_acc_mean", "gen_acc_std", "gen_f1_mean", "gen_f1_std", "gap_acc_mean", "gap_f1_mean"]]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{MM checkpoint ranking by generated accuracy (tie-broken by generated macro-F1).}")
    lines.append(r"\begin{tabular}{cccccc}")
    lines.append(r"\toprule")
    lines.append(r"Epoch & EMA & Generated Acc & Generated Macro-F1 & Acc Gap & F1 Gap \\")
    lines.append(r"\midrule")

    for _, row in rank.iterrows():
        lines.append(
            f"{int(row['epoch'])} & {int(row['ema'])} & "
            f"{_fmt_pm(float(row['gen_acc_mean']), float(row['gen_acc_std']))} & "
            f"{_fmt_pm(float(row['gen_f1_mean']), float(row['gen_f1_std']))} & "
            f"{float(row['gap_acc_mean']):.4f} & {float(row['gap_f1_mean']):.4f} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\label{tab:mm_ckpt_ranking}")
    lines.append(r"\end{table}")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mm-dir", default="outputs/eval_results/mm_ckpt_sweep", type=Path)
    parser.add_argument("--baseline", default="outputs/eval_results/generated_vs_real_classifier_gap.json", type=Path)
    parser.add_argument("--out-dir", default="outputs/eval_results/analysis_generated_vs_real", type=Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    mm_df = build_mm_df(args.mm_dir)
    base_df = build_baseline_df(args.baseline)

    mm_csv = args.out_dir / "mm_sweep_metrics.csv"
    base_csv = args.out_dir / "baseline_metrics.csv"
    mm_df.to_csv(mm_csv, index=False)
    base_df.to_csv(base_csv, index=False)

    plot_acc_f1(mm_df, args.out_dir / "acc_macro_f1_vs_epoch.png")
    plot_gaps(mm_df, args.out_dir / "gap_vs_epoch.png")

    write_summary_latex(mm_df, base_df, args.out_dir / "summary_table.tex")
    write_ranking_latex(mm_df, args.out_dir / "checkpoint_rankings.tex")

    best_acc = mm_df.loc[mm_df["gen_acc_mean"].idxmax()]
    best_f1 = mm_df.loc[mm_df["gen_f1_mean"].idxmax()]

    print("Wrote analysis to:", args.out_dir)
    print("Best generated accuracy checkpoint:", f"epoch={int(best_acc['epoch'])}, ema={int(best_acc['ema'])}, acc={best_acc['gen_acc_mean']:.4f}")
    print("Best generated macro-F1 checkpoint:", f"epoch={int(best_f1['epoch'])}, ema={int(best_f1['ema'])}, f1={best_f1['gen_f1_mean']:.4f}")


if __name__ == "__main__":
    main()
