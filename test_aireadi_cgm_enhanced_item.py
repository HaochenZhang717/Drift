import os
from pathlib import Path
from pprint import pprint

import torch
from torch.utils.data import DataLoader

from utils.utils_dataset import AIREADIModalityImputationDataset


DATA_ROOT = Path(
    os.environ.get(
        "AIREADI_ROOT",
        "/Users/zhc/Documents/Time_Series_Datasets/AI-READI-processed",
    )
)
PARTICIPANTS_TSV = Path(
    os.environ.get(
        "AIREADI_PARTICIPANTS_TSV",
        "/Users/zhc/Documents/Time_Series_Datasets/AI-READI-processed/participants.tsv",
    )
)
CGM_FEATURE_CSV = Path(
    os.environ.get(
        "AIREADI_CGM_FEATURE_CSV",
        "/Users/zhc/Documents/Time_Series_Datasets/AI-READI-processed/cgm_enhanced_features_clean.csv",
    )
)

SELECTED_MODALITIES = ["glucose", "heart_rate", "calorie", "physical_activity", "respiratory_rate"]
TS_SEQ_LEN = 288
WINDOW_MODE = "daily"
DAILY_MIN_EVENTS = 288
MAX_EVENTS_PER_MODALITY = {m: TS_SEQ_LEN for m in SELECTED_MODALITIES}
MIN_EVENTS_PER_MODALITY = {"glucose": DAILY_MIN_EVENTS}


def _build_dataset(split: str = "train") -> AIREADIModalityImputationDataset:
    return AIREADIModalityImputationDataset(
        root=str(DATA_ROOT),
        split=split,
        modalities=SELECTED_MODALITIES,
        anchor_modality="glucose",
        target_modality="glucose",
        window_size=TS_SEQ_LEN,
        window_stride=TS_SEQ_LEN,
        window_mode=WINDOW_MODE,
        daily_min_events=DAILY_MIN_EVENTS,
        max_events_per_modality=MAX_EVENTS_PER_MODALITY,
        min_events_per_modality=MIN_EVENTS_PER_MODALITY,
        normalize=True,
        max_anchor_gap_minutes=10.0,
        max_window_span_hours=24.0,
        anchor_sampling_minutes=5.0,
        anchor_sampling_tolerance_seconds=2.0,
        participants_tsv_path=str(PARTICIPANTS_TSV),
        include_clinical_static=False,
        include_participant_metadata=True,
        include_study_group=True,
        include_clinical_site=False,
        cgm_enhanced_features_path=str(CGM_FEATURE_CSV),
        include_cgm_enhanced_features=True,
        pad=True,
        return_dict=True,
    )


def test_aireadi_item_with_cgm_enhanced_features():
    assert DATA_ROOT.exists(), f"Missing AIREADI_ROOT: {DATA_ROOT}"
    assert PARTICIPANTS_TSV.exists(), f"Missing AIREADI_PARTICIPANTS_TSV: {PARTICIPANTS_TSV}"
    assert CGM_FEATURE_CSV.exists(), f"Missing AIREADI_CGM_FEATURE_CSV: {CGM_FEATURE_CSV}"

    dataset = _build_dataset(split="train")
    assert len(dataset) > 0

    item = dataset[0]
    assert "cgm_enhanced_numeric_values" in item
    assert "cgm_enhanced_numeric_mask" in item
    assert "cgm_enhanced_binary_values" in item
    assert "cgm_enhanced_binary_mask" in item
    assert "cgm_enhanced_medication_codes" in item
    assert "cgm_enhanced_medication_mask" in item
    assert "cgm_enhanced_medication_raw" in item
    assert "cgm_enhanced_medication_vocab" in item

    assert item["cgm_enhanced_numeric_values"].shape == (6,)
    assert item["cgm_enhanced_numeric_mask"].shape == (6,)
    assert item["cgm_enhanced_binary_values"].shape == (6,)
    assert item["cgm_enhanced_binary_mask"].shape == (6,)
    assert item["cgm_enhanced_medication_codes"].shape == (6,)
    assert item["cgm_enhanced_medication_mask"].shape == (6,)

    assert torch.isfinite(item["cgm_enhanced_numeric_values"]).all()
    assert torch.isfinite(item["cgm_enhanced_binary_values"]).all()
    assert set(torch.unique(item["cgm_enhanced_numeric_mask"]).tolist()).issubset({0.0, 1.0})
    assert set(torch.unique(item["cgm_enhanced_binary_mask"]).tolist()).issubset({0.0, 1.0})
    assert set(torch.unique(item["cgm_enhanced_medication_mask"]).tolist()).issubset({0.0, 1.0})

    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    assert batch["cgm_enhanced_numeric_values"].shape == (4, 6)
    assert batch["cgm_enhanced_numeric_mask"].shape == (4, 6)
    assert batch["cgm_enhanced_binary_values"].shape == (4, 6)
    assert batch["cgm_enhanced_binary_mask"].shape == (4, 6)
    assert batch["cgm_enhanced_medication_codes"].shape == (4, 6)
    assert batch["cgm_enhanced_medication_mask"].shape == (4, 6)


def test_print_first_item_preview():
    assert DATA_ROOT.exists(), f"Missing AIREADI_ROOT: {DATA_ROOT}"
    assert PARTICIPANTS_TSV.exists(), f"Missing AIREADI_PARTICIPANTS_TSV: {PARTICIPANTS_TSV}"
    assert CGM_FEATURE_CSV.exists(), f"Missing AIREADI_CGM_FEATURE_CSV: {CGM_FEATURE_CSV}"

    dataset = _build_dataset(split="train")
    item = dataset[0]

    print("\n=== AIREADI item preview ===")
    print(f"dataset_len: {len(dataset)}")
    print(f"patient_id: {item['patient_id']}")
    print(f"window_start_ns: {int(item['window_start_time_local'])}")
    print(f"window_end_ns:   {int(item['window_end_time_local'])}")
    print("\nmodality keys:", list(item["modalities"].keys()))

    print("\n[cgm_enhanced_numeric_values]")
    print(item["cgm_enhanced_numeric_feature_names"])
    print(item["cgm_enhanced_numeric_values"].tolist())
    print(item["cgm_enhanced_numeric_mask"].tolist())

    print("\n[cgm_enhanced_binary_values]")
    print(item["cgm_enhanced_binary_feature_names"])
    print(item["cgm_enhanced_binary_values"].tolist())
    print(item["cgm_enhanced_binary_mask"].tolist())

    print("\n[cgm_enhanced_medication_codes]")
    print(item["cgm_enhanced_medication_feature_names"])
    print(item["cgm_enhanced_medication_codes"].tolist())
    print(item["cgm_enhanced_medication_mask"].tolist())

    print("\n[cgm_enhanced_medication_raw]")
    pprint(item["cgm_enhanced_medication_raw"])

    vocab_sizes = {k: len(v) for k, v in item["cgm_enhanced_medication_vocab"].items()}
    print("\n[cgm_enhanced_medication_vocab_sizes]")
    pprint(vocab_sizes)


if __name__ == "__main__":
    test_aireadi_item_with_cgm_enhanced_features()
    test_print_first_item_preview()
