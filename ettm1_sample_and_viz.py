import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Read ETTm1 samples and visualize channels.")
    parser.add_argument(
        "--data",
        type=str,
        default="/Users/zhc/Documents/Time_Series_Datasets/ETTm1.csv",
        help="Path to ETTm1.csv",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="/Users/zhc/Documents/PhD/projects/drifting-model/ettm1_visualization.png",
        help="Output path for visualization image",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5,
        help="How many random rows to print",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    out_path = Path(args.out)

    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])

    print("Dataset shape:", df.shape)
    print("Columns:", list(df.columns))

    print("\nFirst rows:")
    print(df.head(args.n_samples).to_string(index=False))

    print("\nRandom rows:")
    print(df.sample(args.n_samples, random_state=42).sort_values("date").to_string(index=False))

    channels_meaning = {
        "HUFL": "High Useful Load",
        "HULL": "High Useless Load",
        "MUFL": "Middle Useful Load",
        "MULL": "Middle Useless Load",
        "LUFL": "Low Useful Load",
        "LULL": "Low Useless Load",
        "OT": "Oil Temperature (target variable in many forecasting setups)",
    }

    print("\nChannel meaning:")
    for k, v in channels_meaning.items():
        print(f"- {k}: {v}")

    # Plot one week to make seasonal patterns easy to see (15-min data => 96 points/day).
    week_points = 96 * 7
    view = df.iloc[:week_points].copy()

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    load_cols = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]
    for c in load_cols:
        axes[0].plot(view["date"], view[c], label=c, linewidth=1)
    axes[0].set_title("ETTm1 Load Channels (First 7 Days)")
    axes[0].set_ylabel("Load")
    axes[0].legend(ncol=3, fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(view["date"], view["OT"], color="black", linewidth=1.4, label="OT")
    axes[1].set_title("ETTm1 Oil Temperature OT (First 7 Days)")
    axes[1].set_ylabel("Temperature")
    axes[1].set_xlabel("Date")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    print(f"\nSaved visualization to: {out_path}")


if __name__ == "__main__":
    main()
