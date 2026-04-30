import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.utils_dataset import (
    AIREADI_MODALITY_SPECS,
    AIREADIModalityImputationDataset,
    analyze_aireadi_missingness,
    build_aireadi_clinical_features,
    get_dataset,
)


DATA_ROOT = Path(os.environ.get("AIREADI_ROOT", "/Users/zhc/Downloads/AI-READI-processed"))
CLINICAL_ROOT = Path(os.environ.get("AIREADI_CLINICAL_ROOT", "/Users/zhc/Downloads/clinical_data"))
PARTICIPANTS_TSV = Path(os.environ.get("AIREADI_PARTICIPANTS_TSV", "/Users/zhc/Downloads/AI-READI/participants.tsv"))
VIS_PATH = Path(os.environ.get("AIREADI_VIS_PATH", "outputs/aireadi_multimodal_window.png"))
AGE_BMI_VIS_PATH = Path(os.environ.get("AIREADI_AGE_BMI_VIS_PATH", "outputs/aireadi_age_bmi_raw.png"))

CFG = {
    "ts_seq_len": 128,
    "ts_stride": 512,
    "modalities": [
        "glucose",
        "calorie",
        "heart_rate",
        "respiratory_rate",
        "oxygen_saturation",
        "stress",
        "physical_activity",
        "sleep",
    ],
    "anchor_modality": "glucose",
    "target_modality": "glucose",
    "max_anchor_gap_minutes": 10.0,
    "max_window_span_hours": 12.0,
    "anchor_sampling_minutes": 5.0,
    "anchor_sampling_tolerance_seconds": 2.0,
    "clinical_root": str(CLINICAL_ROOT),
    "participants_tsv_path": str(PARTICIPANTS_TSV),
    "max_events_per_modality": {
        "glucose": 128,
        "calorie": 512,
        "heart_rate": 1024,
        "respiratory_rate": 2048,
        "oxygen_saturation": 1024,
        "stress": 2048,
        "physical_activity": 1024,
        "sleep": 256,
    },
}

DATASET_KWARGS = {
    "modalities": CFG["modalities"],
    "anchor_modality": CFG["anchor_modality"],
    "target_modality": CFG["target_modality"],
    "window_size": CFG["ts_seq_len"],
    "window_stride": CFG["ts_stride"],
    "max_events_per_modality": CFG["max_events_per_modality"],
    "max_anchor_gap_minutes": CFG["max_anchor_gap_minutes"],
    "max_window_span_hours": CFG["max_window_span_hours"],
    "anchor_sampling_minutes": CFG["anchor_sampling_minutes"],
    "anchor_sampling_tolerance_seconds": CFG["anchor_sampling_tolerance_seconds"],
    "clinical_root": CFG["clinical_root"],
    "participants_tsv_path": CFG["participants_tsv_path"],
}


def _require_data_root():
    assert DATA_ROOT.exists(), (
        f"AI-READI data root does not exist: {DATA_ROOT}. "
        "Set AIREADI_ROOT=/path/to/AI-READI-processed if needed."
    )
    assert CLINICAL_ROOT.exists(), (
        f"AI-READI clinical root does not exist: {CLINICAL_ROOT}. "
        "Set AIREADI_CLINICAL_ROOT=/path/to/clinical_data if needed."
    )
    assert PARTICIPANTS_TSV.exists(), (
        f"AI-READI participants.tsv does not exist: {PARTICIPANTS_TSV}. "
        "Set AIREADI_PARTICIPANTS_TSV=/path/to/participants.tsv if needed."
    )


def _assert_modality_item(name, data, max_events):
    assert data["values"].shape == (max_events, 1), name
    assert data["time_local"].shape == (max_events,), name
    assert data["time_end_local"].shape == (max_events,), name
    assert data["relative_time_hours"].shape == (max_events, 1), name
    assert data["mask"].shape == (max_events, 1), name
    assert data["length"].ndim == 0, name
    assert data["raw_length"].ndim == 0, name
    assert data["truncated"].ndim == 0, name
    assert data["present"].ndim == 0, name

    assert torch.isfinite(data["values"]).all(), name
    assert torch.isfinite(data["relative_time_hours"]).all(), name

    length = int(data["length"].item())
    raw_length = int(data["raw_length"].item())
    assert 0 <= length <= max_events, name
    assert raw_length >= length, name
    assert bool(data["present"]) == (raw_length > 0), name

    mask = data["mask"].bool().squeeze(-1)
    assert int(mask.sum().item()) == length, name
    if length > 0:
        times = data["time_local"][mask]
        end_times = data["time_end_local"][mask]
        rel_hours = data["relative_time_hours"][mask]
        assert torch.all(times[1:] >= times[:-1]), name
        assert torch.all(end_times >= times), name
        assert rel_hours.min() >= -1e-4, name


def test_missingness_summary():
    _require_data_root()
    summary = analyze_aireadi_missingness(str(DATA_ROOT))
    print("\nAI-READI missingness / sampling summary")
    print(summary.to_string(index=False))

    expected_modalities = set(AIREADI_MODALITY_SPECS)
    assert expected_modalities.issubset(set(summary["modality"]))
    assert {"train", "valid", "test"}.issubset(set(summary["split"]))
    assert (summary["rows"] > 0).all()
    assert (summary["patients_with_data"] >= 0).all()
    assert (summary["missing_row_pct"] >= 0).all()


def test_dataset_single_item_and_batch():
    _require_data_root()
    dataset = AIREADIModalityImputationDataset(root=str(DATA_ROOT), split="train", **DATASET_KWARGS)
    assert len(dataset) > 0

    item = dataset[0]
    assert item["patient_id"]
    assert item["anchor_modality"] == "glucose"
    assert item["target_modality"] == "glucose"
    assert int(item["window_end_time_local"]) >= int(item["window_start_time_local"])
    assert dataset.clinical_feature_names
    assert item["clinical_static"].shape == (len(dataset.clinical_feature_names),)
    assert item["clinical_mask"].shape == (len(dataset.clinical_feature_names),)
    assert torch.isfinite(item["clinical_static"]).all()
    assert item["clinical_mask"].sum() > 0
    assert int(item["study_group_label"]) in {0, 1, 2, 3}
    assert item["study_group_one_hot"].shape == (4,)
    assert item["study_group_one_hot"].sum() == 1
    assert int(item["clinical_site_code"]) in {1, 4, 7}
    assert int(item["clinical_site_label"]) in {0, 1, 2}
    assert item["clinical_site_one_hot"].shape == (3,)
    assert item["clinical_site_one_hot"].sum() == 1

    assert item["target"].shape == (CFG["max_events_per_modality"]["glucose"], 1)
    assert set(item["modalities"]) == set(CFG["modalities"])

    for name, data in item["modalities"].items():
        _assert_modality_item(name, data, CFG["max_events_per_modality"][name])

    glucose = item["modalities"]["glucose"]
    assert int(glucose["length"].item()) == CFG["ts_seq_len"]
    assert torch.all(glucose["mask"].bool())
    assert glucose["values"].min() >= -1.0
    assert glucose["values"].max() <= 1.0
    glucose_times = glucose["time_local"][glucose["mask"].bool().squeeze(-1)]
    glucose_gaps_seconds = (glucose_times[1:] - glucose_times[:-1]).double() / 1e9
    assert torch.all(
        torch.abs(glucose_gaps_seconds - CFG["anchor_sampling_minutes"] * 60)
        <= CFG["anchor_sampling_tolerance_seconds"]
    )

    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    assert batch["target"].shape == (4, CFG["max_events_per_modality"]["glucose"], 1)
    assert batch["study_group_label"].shape == (4,)
    assert batch["study_group_one_hot"].shape == (4, 4)
    assert batch["clinical_site_one_hot"].shape == (4, 3)
    for name in CFG["modalities"]:
        assert batch["modalities"][name]["values"].shape == (
            4,
            CFG["max_events_per_modality"][name],
            1,
        )
        assert batch["modalities"][name]["mask"].shape == (
            4,
            CFG["max_events_per_modality"][name],
            1,
        )


def test_get_dataset_entrypoint():
    _require_data_root()
    train_dataset, test_dataset = get_dataset("aireadi_imputation", CFG, root=str(DATA_ROOT))
    assert len(train_dataset) > 0
    assert len(test_dataset) > 0
    assert train_dataset.value_ranges
    assert test_dataset.value_ranges == train_dataset.value_ranges
    assert train_dataset.clinical_feature_names == test_dataset.clinical_feature_names
    assert train_dataset.clinical_feature_stats == test_dataset.clinical_feature_stats


def test_glucose_study_group_only():
    _require_data_root()
    cfg = {
        "modalities": ["glucose"],
        "anchor_modality": "glucose",
        "target_modality": "glucose",
        "ts_seq_len": 128,
        "ts_stride": 512,
        "max_events_per_modality": {"glucose": 128},
        "anchor_sampling_minutes": 5.0,
        "anchor_sampling_tolerance_seconds": 2.0,
        "participants_tsv_path": str(PARTICIPANTS_TSV),
        "include_clinical_static": False,
        "include_study_group": True,
        "include_clinical_site": False,
    }
    dataset = AIREADIModalityImputationDataset(
        root=str(DATA_ROOT),
        split="train",
        modalities=cfg["modalities"],
        anchor_modality=cfg["anchor_modality"],
        target_modality=cfg["target_modality"],
        window_size=cfg["ts_seq_len"],
        window_stride=cfg["ts_stride"],
        max_events_per_modality=cfg["max_events_per_modality"],
        anchor_sampling_minutes=cfg["anchor_sampling_minutes"],
        anchor_sampling_tolerance_seconds=cfg["anchor_sampling_tolerance_seconds"],
        participants_tsv_path=cfg["participants_tsv_path"],
        include_clinical_static=cfg["include_clinical_static"],
        include_study_group=cfg["include_study_group"],
        include_clinical_site=cfg["include_clinical_site"],
    )
    item = dataset[0]
    assert set(item["modalities"]) == {"glucose"}
    assert "study_group_label" in item
    assert item["study_group_one_hot"].shape == (4,)
    assert "clinical_static" not in item
    assert "clinical_site_one_hot" not in item


def _sample_score(dataset, idx):
    item = dataset[idx]
    score = 0
    total_events = 0
    for name in CFG["modalities"]:
        if name == CFG["target_modality"]:
            continue
        raw_length = int(item["modalities"][name]["raw_length"])
        score += int(raw_length > 0)
        total_events += raw_length
    return score, total_events


def inspect_samples(dataset, n=5, search_first=512):
    print(f"\nDataset windows: {len(dataset)}")
    print("Value ranges learned from train split:")
    for name, value_range in dataset.value_ranges.items():
        print(f"  {name:18s} {value_range}")
    if dataset.clinical_feature_names:
        print("\nClinical features:")
        print(f"  n_features={len(dataset.clinical_feature_names)}")
        print("  " + ", ".join(dataset.clinical_feature_names[:12]) + " ...")
    print("\nSample windows")
    scored = [
        (_sample_score(dataset, idx), idx)
        for idx in range(min(search_first, len(dataset)))
    ]
    sample_indices = [
        idx
        for _, idx in sorted(scored, key=lambda item: item[0], reverse=True)[:n]
    ]
    for idx in sample_indices:
        item = dataset[idx]
        start = int(item["window_start_time_local"])
        end = int(item["window_end_time_local"])
        span_hours = (end - start) / 3.6e12
        print(f"\n[{idx}] patient={item['patient_id']} span_hours={span_hours:.2f}")
        for name in CFG["modalities"]:
            data = item["modalities"][name]
            print(
                f"  {name:18s} length={int(data['length']):4d} "
                f"raw={int(data['raw_length']):4d} "
                f"present={bool(data['present'])} "
                f"truncated={bool(data['truncated'])}"
            )


def pick_multimodal_window(dataset, search_first=2048):
    best_idx = 0
    best_score = (-1, -1)
    for idx in range(min(search_first, len(dataset))):
        score = _sample_score(dataset, idx)
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx


def plot_multimodal_window(dataset, output_path=VIS_PATH):
    idx = pick_multimodal_window(dataset)
    item = dataset[idx]

    fig, axes = plt.subplots(
        len(CFG["modalities"]),
        1,
        figsize=(12, 2.2 * len(CFG["modalities"])),
        sharex=True,
        constrained_layout=True,
    )
    if len(CFG["modalities"]) == 1:
        axes = [axes]

    for ax, name in zip(axes, CFG["modalities"]):
        data = item["modalities"][name]
        mask = data["mask"].bool().squeeze(-1)
        x = data["relative_time_hours"][mask].squeeze(-1).cpu().numpy()
        y = data["values"][mask].squeeze(-1).cpu().numpy()

        if len(x) == 0:
            ax.text(0.5, 0.5, "missing in this window", ha="center", va="center")
            ax.set_xlim(0, 10.7)
        elif name == "sleep":
            ax.step(x, y, where="post", linewidth=1.4)
            ax.scatter(x, y, s=10)
        else:
            ax.plot(x, y, linewidth=1.1)
            ax.scatter(x, y, s=6)

        ax.set_ylabel(name)
        ax.grid(True, alpha=0.25)
        ax.set_title(
            f"{name}: {int(data['raw_length'])} samples"
            + (" (truncated)" if bool(data["truncated"]) else ""),
            loc="left",
            fontsize=10,
        )

    span_hours = (
        int(item["window_end_time_local"]) - int(item["window_start_time_local"])
    ) / 3.6e12
    axes[-1].set_xlabel("Hours from window start")
    fig.suptitle(
        f"AI-READI raw-value window {idx} | patient={item['patient_id']} | span={span_hours:.2f} h",
        fontsize=13,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"\nSaved multimodal window visualization to: {output_path.resolve()}")
    return output_path


def plot_raw_age_bmi(dataset, output_path=AGE_BMI_VIS_PATH):
    patient_ids = dataset.patient_ids
    clinical_features, clinical_masks, feature_names, _ = build_aireadi_clinical_features(
        clinical_root=str(CLINICAL_ROOT),
        patient_ids=patient_ids,
        normalize=False,
    )
    age_idx = feature_names.index("age_years")
    bmi_idx = feature_names.index("bmi")

    ages = []
    bmis = []
    plotted_patient_ids = []
    for patient_id in patient_ids:
        feature = clinical_features[patient_id]
        mask = clinical_masks[patient_id]
        if mask[age_idx] > 0 and mask[bmi_idx] > 0:
            ages.append(float(feature[age_idx]))
            bmis.append(float(feature[bmi_idx]))
            plotted_patient_ids.append(patient_id)

    if not ages:
        raise ValueError("No patients with both raw age and BMI clinical features")

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    ax.scatter(ages, bmis, s=18, alpha=0.55, edgecolors="none")
    ax.set_xlabel("Age (years, raw)")
    ax.set_ylabel("BMI (raw)")
    ax.set_title(f"AI-READI Clinical Features Per Patient: Age vs BMI (n={len(ages)})")
    ax.grid(True, alpha=0.25)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"\nSaved raw clinical age/BMI visualization to: {output_path.resolve()}")
    print(f"Plotted {len(plotted_patient_ids)} patients with both age and BMI.")
    return output_path


def main():
    _require_data_root()
    test_missingness_summary()
    test_dataset_single_item_and_batch()
    test_get_dataset_entrypoint()
    test_glucose_study_group_only()

    dataset = AIREADIModalityImputationDataset(root=str(DATA_ROOT), split="train", **DATASET_KWARGS)
    inspect_samples(dataset)
    raw_dataset = AIREADIModalityImputationDataset(
        root=str(DATA_ROOT),
        split="train",
        normalize=False,
        **DATASET_KWARGS,
    )
    plot_multimodal_window(raw_dataset)
    plot_raw_age_bmi(raw_dataset)
    print("\nAll AI-READI dataset checks passed.")


if __name__ == "__main__":
    main()
