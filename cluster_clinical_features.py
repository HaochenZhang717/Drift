import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from test_aireadi_dataset import DATASET_KWARGS
from utils.utils_dataset import (
    AIREADIModalityImputationDataset,
    build_aireadi_clinical_features,
)


DEFAULT_FEATURES = [
    "age_years",
    "bmi",
    "hba1c",
    "lab_glucose",
    "waist_cm",
    "triglycerides",
    "hdl_cholesterol",
    "ldl_cholesterol",
    "systolic_bp",
    "diastolic_bp",
]


def make_feature_frame(clinical_root: str, patient_ids: list[str], features: list[str]) -> pd.DataFrame:
    feature_by_patient, mask_by_patient, feature_names, _ = build_aireadi_clinical_features(
        clinical_root=clinical_root,
        patient_ids=patient_ids,
        normalize=False,
    )
    missing_features = [name for name in features if name not in feature_names]
    if missing_features:
        raise ValueError(f"Unknown clinical features: {missing_features}")

    indices = [feature_names.index(name) for name in features]
    rows = []
    for patient_id in patient_ids:
        values = feature_by_patient[patient_id]
        mask = mask_by_patient[patient_id]
        if all(float(mask[idx]) > 0 for idx in indices):
            row = {"patient_id": patient_id}
            row.update({name: float(values[idx]) for name, idx in zip(features, indices)})
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No patients have all selected clinical features observed.")
    return df


def cluster_feature_frame(df: pd.DataFrame, features: list[str], n_clusters: int, seed: int) -> tuple[pd.DataFrame, np.ndarray, float]:
    x_raw = df[features].to_numpy(dtype=np.float32)
    x = StandardScaler().fit_transform(x_raw)
    model = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20)
    clusters = model.fit_predict(x)
    silhouette = silhouette_score(x, clusters) if n_clusters > 1 and len(df) > n_clusters else float("nan")

    pca_xy = PCA(n_components=2, random_state=seed).fit_transform(x)
    out = df.copy()
    out["cluster"] = clusters
    out["pca_1"] = pca_xy[:, 0]
    out["pca_2"] = pca_xy[:, 1]
    return out, x, silhouette


def plot_pca_clusters(clustered: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    scatter = ax.scatter(
        clustered["pca_1"],
        clustered["pca_2"],
        c=clustered["cluster"],
        cmap="tab10",
        s=18,
        alpha=0.7,
        edgecolors="none",
    )
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title(f"Clinical Feature Clusters (n={len(clustered)})")
    ax.grid(True, alpha=0.25)
    legend = ax.legend(*scatter.legend_elements(), title="cluster", loc="best")
    ax.add_artist(legend)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_age_bmi_clusters(clustered: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    scatter = ax.scatter(
        clustered["age_years"],
        clustered["bmi"],
        c=clustered["cluster"],
        cmap="tab10",
        s=18,
        alpha=0.7,
        edgecolors="none",
    )
    ax.set_xlabel("Age (years, raw)")
    ax.set_ylabel("BMI (raw)")
    ax.set_title("Raw Age vs BMI Colored by Clinical Cluster")
    ax.grid(True, alpha=0.25)
    legend = ax.legend(*scatter.legend_elements(), title="cluster", loc="best")
    ax.add_artist(legend)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def print_cluster_summary(clustered: pd.DataFrame, features: list[str], silhouette: float) -> None:
    print(f"patients used: {len(clustered)}")
    print(f"clusters: {clustered['cluster'].nunique()}")
    print(f"silhouette: {silhouette:.4f}")
    print("\ncluster sizes")
    print(clustered["cluster"].value_counts().sort_index().to_string())
    print("\ncluster raw feature means")
    print(clustered.groupby("cluster")[features].mean().round(3).to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster patient-level AI-READI clinical features for conditioning."
    )
    parser.add_argument("--data-root", default="/Users/zhc/Downloads/AI-READI-processed")
    parser.add_argument("--clinical-root", default="/Users/zhc/Downloads/clinical_data")
    parser.add_argument("--split", default="train", choices=["train", "valid", "test"])
    parser.add_argument("--n-clusters", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="outputs/clinical_clusters")
    parser.add_argument(
        "--features",
        nargs="+",
        default=DEFAULT_FEATURES,
        help="Clinical feature names to use. Patients missing any selected feature are dropped.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_kwargs = dict(DATASET_KWARGS)
    dataset_kwargs["clinical_root"] = None
    dataset = AIREADIModalityImputationDataset(
        root=args.data_root,
        split=args.split,
        **dataset_kwargs,
    )
    patients = dataset.patient_ids
    feature_df = make_feature_frame(args.clinical_root, patients, args.features)
    clustered, _, silhouette = cluster_feature_frame(
        feature_df,
        args.features,
        n_clusters=args.n_clusters,
        seed=args.seed,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{args.split}_clinical_clusters.csv"
    pca_path = out_dir / f"{args.split}_clinical_clusters_pca.png"
    age_bmi_path = out_dir / f"{args.split}_age_bmi_by_cluster.png"
    clustered.to_csv(csv_path, index=False)
    plot_pca_clusters(clustered, pca_path)
    plot_age_bmi_clusters(clustered, age_bmi_path)

    print_cluster_summary(clustered, args.features, silhouette)
    print(f"\nsaved: {csv_path.resolve()}")
    print(f"saved: {pca_path.resolve()}")
    print(f"saved: {age_bmi_path.resolve()}")


if __name__ == "__main__":
    main()
