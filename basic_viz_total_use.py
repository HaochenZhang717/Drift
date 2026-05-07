import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CSV_PATH = Path('/Users/zhc/Downloads/Total_use.csv')
OUT_DIR = Path('visualizations_total_use')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    df = pd.read_csv(CSV_PATH)

    # Remove exported index column if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    # Keep numeric columns only
    num_df = df.select_dtypes(include=[np.number]).copy()

    if num_df.empty:
        raise ValueError('No numeric columns found for visualization.')

    print(f'Data shape (numeric): {num_df.shape}')

    # 1) Global value distribution (sampled for speed)
    flat = num_df.to_numpy().ravel()
    sample_size = min(200_000, flat.size)
    rng = np.random.default_rng(42)
    if flat.size > sample_size:
        flat = rng.choice(flat, size=sample_size, replace=False)

    plt.figure(figsize=(8, 5))
    plt.hist(flat, bins=80, color='#4C78A8', alpha=0.9)
    plt.title('Global Distribution of Values (Sampled)')
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(OUT_DIR / '01_global_distribution.png', dpi=180)
    plt.close()

    # 2) Row-level mean trend (use row index as x)
    row_mean = num_df.mean(axis=1)
    plt.figure(figsize=(10, 4))
    plt.plot(row_mean.values, color='#F58518', linewidth=1)
    plt.title('Row-wise Mean Trend')
    plt.xlabel('Row index')
    plt.ylabel('Mean across columns')
    plt.tight_layout()
    plt.savefig(OUT_DIR / '02_row_mean_trend.png', dpi=180)
    plt.close()

    # 3) First 6 columns trend (for quick comparison)
    first_cols = list(num_df.columns[:6])
    plt.figure(figsize=(10, 5))
    for c in first_cols:
        plt.plot(num_df[c].values, linewidth=0.8, label=str(c), alpha=0.85)
    plt.title('Trend of First 6 Numeric Columns')
    plt.xlabel('Row index')
    plt.ylabel('Value')
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / '03_first6_columns_trend.png', dpi=180)
    plt.close()

    # 4) Boxplot of first 20 columns to check spread/outliers
    box_cols = list(num_df.columns[:20])
    plt.figure(figsize=(12, 5))
    plt.boxplot([num_df[c].values for c in box_cols], labels=box_cols, showfliers=False)
    plt.title('Boxplot of First 20 Numeric Columns (No Outliers)')
    plt.xlabel('Column')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUT_DIR / '04_boxplot_first20.png', dpi=180)
    plt.close()

    # 5) Correlation heatmap for first 25 columns
    corr_cols = list(num_df.columns[:25])
    corr = num_df[corr_cols].corr()
    plt.figure(figsize=(10, 8))
    im = plt.imshow(corr.values, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title('Correlation Heatmap (First 25 Columns)')
    plt.xticks(range(len(corr_cols)), corr_cols, rotation=90, fontsize=7)
    plt.yticks(range(len(corr_cols)), corr_cols, fontsize=7)
    plt.tight_layout()
    plt.savefig(OUT_DIR / '05_corr_heatmap_first25.png', dpi=180)
    plt.close()

    # Basic summary export
    summary = num_df.describe().T
    summary.to_csv(OUT_DIR / 'summary_stats.csv')

    print('Saved visualizations to:', OUT_DIR.resolve())
    for p in sorted(OUT_DIR.glob('*.png')):
        print('-', p.name)
    print('Saved summary:', (OUT_DIR / 'summary_stats.csv').name)


if __name__ == '__main__':
    main()
