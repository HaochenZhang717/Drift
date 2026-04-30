import argparse
import math
import os
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional
import matplotlib
matplotlib.use("Agg")
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from img_transformations import DelayEmbedder

def normalize_ts_to_unit_range(ts: torch.Tensor) -> torch.Tensor:
    """
    Normalize a 1D time series to [-1, 1] with per-series min-max scaling.
    """
    ts_min = ts.min()
    ts_max = ts.max()
    scale = ts_max - ts_min
    if scale < 1e-6:
        return torch.zeros_like(ts)
    return 2.0 * (ts - ts_min) / scale - 1.0


class SyntheticSineDataset(Dataset):
    """Synthetic sine-wave time series, transformed to images with DelayEmbedder."""

    def __init__(
        self,
        num_samples: int,
        embedder: DelayEmbedder,
        seq_len: int,
        components_min: int,
        components_max: int,
        freq_min: float,
        freq_max: float,
        amp_min: float,
        amp_max: float,
        noise_std: float,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.embedder = embedder
        self.seq_len = seq_len
        self.components_min = components_min
        self.components_max = components_max
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.amp_min = amp_min
        self.amp_max = amp_max
        self.noise_std = noise_std
        self.seed = seed

    def __len__(self) -> int:
        return self.num_samples

    def _make_signal(self, idx: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(self.seed + idx)
        t = torch.linspace(0.0, 1.0, self.seq_len, dtype=torch.float32)
        signal = torch.zeros_like(t)

        n_components = int(
            torch.randint(
                low=self.components_min,
                high=self.components_max + 1,
                size=(1,),
                generator=g,
            ).item()
        )

        for _ in range(n_components):
            amp = torch.empty(1).uniform_(self.amp_min, self.amp_max, generator=g).item()
            freq = torch.empty(1).uniform_(self.freq_min, self.freq_max, generator=g).item()
            phase = torch.empty(1).uniform_(0.0, 2.0 * math.pi, generator=g).item()
            signal = signal + amp * torch.sin(2.0 * math.pi * freq * t + phase)

        if self.noise_std > 0:
            signal = signal + torch.randn(self.seq_len, generator=g) * self.noise_std
        signal = normalize_ts_to_unit_range(signal)
        return signal.unsqueeze(-1)  # (seq_len, 1)

    def __getitem__(self, idx: int) -> tuple:
        ts = self._make_signal(idx)
        img = self.embedder.ts_to_img(ts.unsqueeze(0), pad=True, mask=0.0).squeeze(0)
        label = torch.tensor(0, dtype=torch.long)
        return img, label


class GlucoseSlidingWindowDataset(Dataset):

    def __init__(

        self,

        parquet_path,

        embedder,

        seq_len=128,

        stride=1,

        normalize=True,

        value_min=None,

        value_max=None,

    ):

        self.df = pd.read_parquet(parquet_path)

        self.embedder = embedder

        self.seq_len = seq_len

        self.stride = stride

        self.normalize = normalize

        self.value_min = value_min

        self.value_max = value_max

        if self.normalize and (self.value_min is None or self.value_max is None):
            self.value_min, self.value_max = _compute_sequence_min_max(self.df["glucose"])

        self.windows = []  # (row_idx, start_idx)

        # 预计算所有 window 索引（关键）

        for row_idx, ts in enumerate(self.df["glucose"]):

            L = len(ts)

            if L < seq_len:

                continue

            for start in range(0, L - seq_len + 1, stride):

                self.windows.append((row_idx, start))

    def __len__(self):

        return len(self.windows)

    @staticmethod
    def _compute_min_max(series):
        return _compute_sequence_min_max(series)

    def _process_ts(self, ts):

        arr = np.asarray(ts, dtype=np.float32)

        if self.normalize:
            arr = _normalize_array(arr, self.value_min, self.value_max)

        ts = torch.from_numpy(arr.astype(np.float32))

        return ts.unsqueeze(-1)  # (T,1)

    def __getitem__(self, idx):

        row_idx, start = self.windows[idx]

        full_ts = self.df["glucose"].iloc[row_idx]

        window = full_ts[start : start + self.seq_len]

        ts = self._process_ts(window)

        img = self.embedder.ts_to_img(

            ts.unsqueeze(0),  # (1,T,1)

            pad=True,

            mask=0.0,

        ).squeeze(0)

        label = torch.tensor(0, dtype=torch.long)

        return img, label


def _first_patient_id(values) -> Optional[str]:
    if values is None:
        return None

    if isinstance(values, str):
        token = values
    else:
        arr = np.asarray(values, dtype=object).reshape(-1)
        token = None
        for value in arr:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                continue
            token = str(value)
            if token:
                break
        if token is None:
            return None

    token = token.strip()
    if not token:
        return None
    if token.isdigit():
        token = f"AIREADI-{int(token):04d}"
    return token


def _compute_sequence_min_max(series) -> tuple[float, float]:
    mins = []
    maxs = []
    for values in series:
        arr = np.asarray(values, dtype=np.float32)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        mins.append(float(arr.min()))
        maxs.append(float(arr.max()))

    if not mins:
        raise ValueError("Could not compute min/max from parquet data")

    return min(mins), max(maxs)


def _normalize_array(values: np.ndarray, value_min: float, value_max: float) -> np.ndarray:
    scale = max(float(value_max) - float(value_min), 1e-6)
    values = 2.0 * (values.astype(np.float32) - float(value_min)) / scale - 1.0
    return np.clip(values, -1.0, 1.0)


@dataclass(frozen=True)
class AIREADIModalitySpec:
    file_prefix: str
    value_column: str
    time_column: str
    end_time_column: Optional[str] = None
    missing_column: Optional[str] = "is_missing"
    categorical: bool = False


AIREADI_MODALITY_SPECS: Dict[str, AIREADIModalitySpec] = {
    "glucose": AIREADIModalitySpec(
        file_prefix="glucose",
        value_column="glucose",
        time_column="time_local",
        missing_column=None,
    ),
    "calorie": AIREADIModalitySpec(
        file_prefix="calorie",
        value_column="calorie",
        time_column="time_local",
    ),
    "heart_rate": AIREADIModalitySpec(
        file_prefix="heart_rate",
        value_column="heart_rate",
        time_column="time_local",
    ),
    "respiratory_rate": AIREADIModalitySpec(
        file_prefix="respiratory_rate",
        value_column="respiratory_rate",
        time_column="time_local",
    ),
    "oxygen_saturation": AIREADIModalitySpec(
        file_prefix="oxygen_saturation",
        value_column="spo2",
        time_column="time_local",
    ),
    "stress": AIREADIModalitySpec(
        file_prefix="stress",
        value_column="stress",
        time_column="time_local",
    ),
    "physical_activity": AIREADIModalitySpec(
        file_prefix="physical_activity",
        value_column="steps",
        time_column="time_start_local",
        end_time_column="time_end_local",
    ),
    "sleep": AIREADIModalitySpec(
        file_prefix="sleep",
        value_column="sleep_stage",
        time_column="time_start_local",
        end_time_column="time_end_local",
        categorical=True,
    ),
}


def _as_object_array(values) -> np.ndarray:
    if values is None:
        return np.asarray([], dtype=object)
    if isinstance(values, np.ndarray):
        return values.reshape(-1)
    if isinstance(values, (list, tuple, pd.Series)):
        return np.asarray(values, dtype=object).reshape(-1)
    return np.asarray([values], dtype=object)


def _datetime_ns(values) -> np.ndarray:
    raw = _as_object_array(values)
    if raw.size == 0:
        return np.asarray([], dtype=np.int64)
    times = pd.to_datetime(raw, errors="coerce")
    return pd.DatetimeIndex(times).astype("datetime64[ns]").asi8.astype(np.int64)


def _series_from_row_for_modality(
    row,
    spec: AIREADIModalitySpec,
    categorical_vocab: Optional[Dict[str, int]] = None,
) -> Optional[Dict[str, np.ndarray]]:
    patient_id = _first_patient_id(getattr(row, "patient_id"))
    if patient_id is None:
        return None

    start_ns = _datetime_ns(getattr(row, spec.time_column, None))
    if spec.end_time_column is None:
        end_ns = start_ns.copy()
    else:
        end_ns = _datetime_ns(getattr(row, spec.end_time_column, None))

    values_raw = _as_object_array(getattr(row, spec.value_column))
    n = min(len(values_raw), len(start_ns), len(end_ns))
    if n == 0:
        return None

    start_ns = start_ns[:n]
    end_ns = end_ns[:n]
    values_raw = values_raw[:n]
    valid_time = (start_ns != np.iinfo(np.int64).min) & (end_ns != np.iinfo(np.int64).min)

    if spec.categorical:
        if categorical_vocab is None:
            categorical_vocab = {}
        encoded = np.zeros(n, dtype=np.float32)
        valid_value = np.zeros(n, dtype=bool)
        for idx, value in enumerate(values_raw):
            if value is None:
                continue
            token = str(value).strip()
            if not token:
                continue
            if token not in categorical_vocab:
                categorical_vocab[token] = len(categorical_vocab) + 1
            encoded[idx] = float(categorical_vocab[token])
            valid_value[idx] = True
        values = encoded
    else:
        values = pd.to_numeric(pd.Series(values_raw), errors="coerce").to_numpy(dtype=np.float32)
        valid_value = np.isfinite(values)

    valid = valid_time & valid_value
    if not valid.any():
        return None

    start_ns = start_ns[valid]
    end_ns = end_ns[valid]
    values = values[valid].astype(np.float32)
    order = np.argsort(start_ns, kind="mergesort")
    start_ns = start_ns[order]
    end_ns = end_ns[order]
    values = values[order]

    return {
        "patient_id": patient_id,
        "value": values,
        "time_local": start_ns.astype(np.int64),
        "time_end_local": end_ns.astype(np.int64),
    }


def _build_aireadi_patient_index(
    df: pd.DataFrame,
    spec: AIREADIModalitySpec,
    categorical_vocab: Optional[Dict[str, int]] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    rows_by_patient: Dict[str, list] = {}
    for row in df.itertuples(index=False):
        item = _series_from_row_for_modality(row, spec, categorical_vocab)
        if item is None:
            continue
        rows_by_patient.setdefault(item["patient_id"], []).append(item)

    by_patient: Dict[str, Dict[str, np.ndarray]] = {}
    for patient_id, items in rows_by_patient.items():
        values = np.concatenate([item["value"] for item in items]).astype(np.float32)
        times = np.concatenate([item["time_local"] for item in items]).astype(np.int64)
        end_times = np.concatenate([item["time_end_local"] for item in items]).astype(np.int64)
        order = np.argsort(times, kind="mergesort")
        by_patient[patient_id] = {
            "value": values[order],
            "time_local": times[order],
            "time_end_local": end_times[order],
        }
    return by_patient


def analyze_aireadi_missingness(
    root: str,
    splits: tuple[str, ...] = ("train", "valid", "test"),
    modalities: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Summarize row-level missingness and sampling density for processed AI-READI parquet files."""
    root_path = Path(root)
    modalities = modalities or list(AIREADI_MODALITY_SPECS.keys())
    rows = []

    for split in splits:
        for modality in modalities:
            spec = AIREADI_MODALITY_SPECS[modality]
            path = root_path / f"{spec.file_prefix}_{split}.parquet"
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            patient_ids = df["patient_id"].map(_first_patient_id)

            if spec.missing_column and spec.missing_column in df.columns:
                missing_rows = int(df[spec.missing_column].sum())
            else:
                missing_rows = 0

            valid_counts = []
            interval_minutes = []
            for row in df.itertuples(index=False):
                item = _series_from_row_for_modality(row, spec, categorical_vocab={})
                n_events = 0 if item is None else len(item["value"])
                valid_counts.append(n_events)
                if item is not None and len(item["time_local"]) > 1:
                    diffs = np.diff(np.sort(item["time_local"])).astype(np.float64) / 60e9
                    diffs = diffs[np.isfinite(diffs) & (diffs >= 0)]
                    if diffs.size:
                        interval_minutes.extend(diffs[:2000])

            valid_counts = np.asarray(valid_counts, dtype=np.int64)
            nonempty = valid_counts > 0
            rows.append(
                {
                    "split": split,
                    "modality": modality,
                    "rows": int(len(df)),
                    "missing_rows": missing_rows,
                    "missing_row_pct": 100.0 * missing_rows / max(len(df), 1),
                    "patients_total": int(patient_ids.nunique()),
                    "patients_with_data": int(patient_ids[nonempty].nunique()),
                    "valid_events": int(valid_counts.sum()),
                    "events_per_nonempty_row_median": float(np.median(valid_counts[nonempty]))
                    if nonempty.any()
                    else 0.0,
                    "median_interval_min": float(np.median(interval_minutes))
                    if interval_minutes
                    else np.nan,
                }
            )

    return pd.DataFrame(rows)


AIREADI_CLINICAL_MEASUREMENT_FEATURES: Dict[str, tuple[str, ...]] = {
    "systolic_bp": ("bp1_sysbp_vsorres", "bp2_sysbp_vsorres"),
    "diastolic_bp": ("bp1_diabp_vsorres", "bp2_diabp_vsorres"),
    "pulse": ("pulse_vsorres", "pulse_vsorres_2"),
    "weight_kg": ("weight_vsorres",),
    "height_cm": ("height_vsorres",),
    "bmi": ("bmi_vsorres",),
    "waist_cm": ("waist_vsorres",),
    "whr": ("whr_vsorres",),
    "lab_glucose": ("import_glucose",),
    "hba1c": ("import_hba1c",),
    "hdl_cholesterol": ("import_hdl_cholesterol",),
    "ldl_cholesterol": ("import_ldl_cholesterol",),
    "total_cholesterol": ("import_total_cholesterol",),
    "triglycerides": ("import_triglycerides",),
    "creatinine": ("import_creatinine",),
    "albumin": ("import_albumin",),
    "urine_creatinine": ("import_urine_creatinine",),
    "urine_albumin": ("import_urine_albumin",),
    "va_letter_photopic_od": ("viaodplog",),
    "va_letter_photopic_os": ("viaosplog",),
    "va_letter_mesopic_od": ("viaodmlog",),
    "va_letter_mesopic_os": ("viaosmlog",),
    "logmar_photopic_od": ("viaodpscore",),
    "logmar_photopic_os": ("viaospscore",),
    "logmar_mesopic_od": ("viaodmscore",),
    "logmar_mesopic_os": ("viaosmscore",),
}


AIREADI_CLINICAL_CONDITION_FEATURES: Dict[str, tuple[str, ...]] = {
    "condition_elevated_a1c": ("mh_a1c",),
    "condition_hypertension": ("mhoccur_hbp",),
    "condition_high_cholesterol": ("mhoccur_clsh",),
    "condition_type2_diabetes": ("mhterm_dm2",),
    "condition_type1_diabetes": ("mhterm_dm1",),
    "condition_prediabetes": ("mhterm_predm",),
    "condition_obesity": ("mhoccur_obs",),
    "condition_kidney": ("mhoccur_rnl",),
    "condition_diabetic_retinopathy": ("mhoccur_pdr",),
    "condition_dry_eye": ("mhoccur_ded",),
    "condition_cataracts": ("mhoccur_crt",),
}


def _clinical_patient_id(values) -> Optional[str]:
    if values is None or pd.isna(values):
        return None
    try:
        token = str(int(values))
    except (TypeError, ValueError):
        token = str(values).strip()
    if not token:
        return None
    token = token.replace("AIREADI-", "")
    if token.isdigit():
        return f"AIREADI-{int(token):04d}"
    return token


AI_READI_STUDY_GROUPS = [
    "healthy",
    "pre_diabetes_lifestyle_controlled",
    "oral_medication_and_or_non_insulin_injectable_medication_controlled",
    "insulin_dependent",
]


AI_READI_CLINICAL_SITES = ["UW", "UCSD", "UAB"]


def _site_from_patient_id(patient_id: str) -> tuple[int, int, torch.Tensor]:
    """Return compact label, original site code, and one-hot clinical site from AIREADI patient id."""
    token = str(patient_id).replace("AIREADI-", "")
    if not token.isdigit():
        label = -1
        code = -1
    else:
        code = int(token) // 1000
        label = {1: 0, 4: 1, 7: 2}.get(code, -1)

    one_hot = torch.zeros(3, dtype=torch.float32)
    if 0 <= label < 3:
        one_hot[label] = 1.0
    return label, code, one_hot


def _load_aireadi_participant_metadata(
    participants_tsv_path: Optional[str],
) -> Dict[str, Dict[str, Any]]:
    if participants_tsv_path is None:
        return {}

    path = Path(participants_tsv_path)
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path, sep="\t")
    metadata: Dict[str, Dict[str, Any]] = {}
    for row in df.itertuples(index=False):
        patient_id = _clinical_patient_id(getattr(row, "person_id"))
        if patient_id is None:
            continue

        study_group = str(getattr(row, "study_group", "")).strip()
        study_group_label = (
            AI_READI_STUDY_GROUPS.index(study_group)
            if study_group in AI_READI_STUDY_GROUPS
            else -1
        )
        study_group_one_hot = torch.zeros(len(AI_READI_STUDY_GROUPS), dtype=torch.float32)
        if study_group_label >= 0:
            study_group_one_hot[study_group_label] = 1.0

        clinical_site = str(getattr(row, "clinical_site", "")).strip()
        clinical_site_label = (
            AI_READI_CLINICAL_SITES.index(clinical_site)
            if clinical_site in AI_READI_CLINICAL_SITES
            else -1
        )
        clinical_site_one_hot = torch.zeros(len(AI_READI_CLINICAL_SITES), dtype=torch.float32)
        if clinical_site_label >= 0:
            clinical_site_one_hot[clinical_site_label] = 1.0

        metadata[patient_id] = {
            "study_group": study_group,
            "study_group_label": study_group_label,
            "study_group_one_hot": study_group_one_hot,
            "clinical_site": clinical_site,
            "clinical_site_label": clinical_site_label,
            "clinical_site_one_hot": clinical_site_one_hot,
            "recommended_split": str(getattr(row, "recommended_split", "")).strip(),
        }
    return metadata


def _source_starts_with(series: pd.Series, prefixes: tuple[str, ...]) -> pd.Series:
    source = series.fillna("").astype(str).str.lower()
    return source.apply(lambda value: any(value.startswith(prefix.lower()) for prefix in prefixes))


def build_aireadi_clinical_features(
    clinical_root: str,
    patient_ids: Optional[list[str]] = None,
    normalize: bool = True,
    feature_stats: Optional[Dict[str, Any]] = None,
    age_reference_year: int = 2026,
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], list[str], Dict[str, Any]]:
    """
    Build patient-level clinical covariates from AI-READI OMOP tables.

    Continuous features are z-scored with train-set statistics when
    ``normalize=True`` and missing continuous values are imputed to 0 after
    normalization. The returned mask marks which clinical features were observed.
    """
    root = Path(clinical_root)
    if not root.exists():
        raise FileNotFoundError(root)

    patient_ids = sorted(set(patient_ids or []))
    feature_names = (
        ["age_years"]
        + list(AIREADI_CLINICAL_MEASUREMENT_FEATURES.keys())
        + list(AIREADI_CLINICAL_CONDITION_FEATURES.keys())
    )
    continuous_names = ["age_years"] + list(AIREADI_CLINICAL_MEASUREMENT_FEATURES.keys())
    condition_names = list(AIREADI_CLINICAL_CONDITION_FEATURES.keys())
    features = pd.DataFrame(index=patient_ids, columns=feature_names, dtype=np.float32)

    person_path = root / "person.csv"
    if person_path.exists():
        person = pd.read_csv(person_path, usecols=["person_id", "year_of_birth"])
        person["patient_id"] = person["person_id"].map(_clinical_patient_id)
        person["age_years"] = age_reference_year - pd.to_numeric(
            person["year_of_birth"],
            errors="coerce",
        )
        age = person.dropna(subset=["patient_id"]).set_index("patient_id")["age_years"]
        common = features.index.intersection(age.index)
        features.loc[common, "age_years"] = age.loc[common].astype(np.float32).to_numpy()

    measurement_path = root / "measurement.csv"
    if measurement_path.exists():
        measurement = pd.read_csv(
            measurement_path,
            usecols=["person_id", "measurement_source_value", "value_as_number"],
            low_memory=False,
        )
        measurement["patient_id"] = measurement["person_id"].map(_clinical_patient_id)
        measurement["value_as_number"] = pd.to_numeric(
            measurement["value_as_number"],
            errors="coerce",
        )
        measurement = measurement.dropna(subset=["patient_id", "value_as_number"])
        for feature_name, prefixes in AIREADI_CLINICAL_MEASUREMENT_FEATURES.items():
            selected = measurement[_source_starts_with(measurement["measurement_source_value"], prefixes)]
            if selected.empty:
                continue
            values = selected.groupby("patient_id")["value_as_number"].mean()
            common = features.index.intersection(values.index)
            features.loc[common, feature_name] = values.loc[common].astype(np.float32).to_numpy()

    for feature_name in condition_names:
        features[feature_name] = features[feature_name].fillna(0.0)

    condition_path = root / "condition_occurrence.csv"
    if condition_path.exists():
        condition = pd.read_csv(
            condition_path,
            usecols=["person_id", "condition_source_value"],
            low_memory=False,
        )
        condition["patient_id"] = condition["person_id"].map(_clinical_patient_id)
        condition = condition.dropna(subset=["patient_id"])
        for feature_name, prefixes in AIREADI_CLINICAL_CONDITION_FEATURES.items():
            selected = condition[_source_starts_with(condition["condition_source_value"], prefixes)]
            positive_ids = selected["patient_id"].unique()
            positive_ids = features.index.intersection(positive_ids)
            features.loc[positive_ids, feature_name] = 1.0

    observed_mask = features.notna().astype(np.float32)
    if feature_stats is None:
        feature_stats = {"feature_names": feature_names, "continuous": {}}
        for name in continuous_names:
            values = pd.to_numeric(features[name], errors="coerce").dropna().to_numpy(dtype=np.float32)
            if values.size == 0:
                mean, std = 0.0, 1.0
            else:
                mean = float(values.mean())
                std = float(values.std())
                if std < 1e-6:
                    std = 1.0
            feature_stats["continuous"][name] = {"mean": mean, "std": std}
    elif feature_stats.get("feature_names") != feature_names:
        raise ValueError("clinical feature_stats do not match the current clinical feature set")

    if normalize:
        for name in continuous_names:
            stats = feature_stats["continuous"][name]
            features[name] = (features[name].astype(np.float32) - stats["mean"]) / stats["std"]

    features = features.fillna(0.0).astype(np.float32)
    feature_by_patient = {
        patient_id: torch.from_numpy(features.loc[patient_id].to_numpy(dtype=np.float32))
        for patient_id in features.index
    }
    mask_by_patient = {
        patient_id: torch.from_numpy(observed_mask.loc[patient_id].to_numpy(dtype=np.float32))
        for patient_id in observed_mask.index
    }
    return feature_by_patient, mask_by_patient, feature_names, feature_stats


class AIREADIModalityImputationDataset(Dataset):
    """
    Windowed multi-modal AI-READI dataset for modality imputation.

    Windows are anchored on one modality, usually glucose. Every returned item
    contains the observed values and local timestamps for all requested
    modalities that fall inside the anchor window's local-time span.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        modalities: Optional[list[str]] = None,
        anchor_modality: str = "glucose",
        target_modality: Optional[str] = "glucose",
        window_size: int = 128,
        window_stride: int = 128,
        window_mode: str = "sliding",
        daily_min_events: Optional[int] = None,
        max_events_per_modality: Optional[Dict[str, int]] = None,
        normalize: bool = True,
        value_ranges: Optional[Dict[str, tuple[float, float]]] = None,
        categorical_vocabs: Optional[Dict[str, Dict[str, int]]] = None,
        include_last_window: bool = True,
        require_complete: bool = False,
        min_events_per_modality: Optional[Dict[str, int]] = None,
        max_anchor_gap_minutes: Optional[float] = None,
        max_window_span_hours: Optional[float] = None,
        anchor_sampling_minutes: Optional[float] = None,
        anchor_sampling_tolerance_seconds: float = 2.0,
        clinical_root: Optional[str] = None,
        normalize_clinical: bool = True,
        clinical_feature_stats: Optional[Dict[str, Any]] = None,
        clinical_age_reference_year: int = 2026,
        participants_tsv_path: Optional[str] = None,
        include_clinical_static: bool = True,
        include_participant_metadata: bool = True,
        include_study_group: bool = True,
        include_clinical_site: bool = True,
        pad: bool = True,
        return_dict: bool = True,
    ):
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if window_stride <= 0:
            raise ValueError("window_stride must be positive")
        if window_mode not in {"sliding", "daily"}:
            raise ValueError("window_mode must be one of {'sliding', 'daily'}")

        self.root = Path(root)
        self.split = split
        self.modalities = modalities or list(AIREADI_MODALITY_SPECS.keys())
        unknown = sorted(set(self.modalities) - set(AIREADI_MODALITY_SPECS))
        if unknown:
            raise ValueError(f"Unknown AI-READI modalities: {unknown}")
        if anchor_modality not in self.modalities:
            raise ValueError("anchor_modality must be included in modalities")
        if target_modality is not None and target_modality not in self.modalities:
            raise ValueError("target_modality must be included in modalities")

        self.anchor_modality = anchor_modality
        self.target_modality = target_modality
        self.window_size = int(window_size)
        self.window_stride = int(window_stride)
        self.window_mode = window_mode
        self.daily_min_events = int(daily_min_events) if daily_min_events is not None else None
        self.normalize = bool(normalize)
        self.include_last_window = bool(include_last_window)
        self.require_complete = bool(require_complete)
        self.max_anchor_gap_minutes = max_anchor_gap_minutes
        self.max_window_span_hours = max_window_span_hours
        self.anchor_sampling_minutes = anchor_sampling_minutes
        self.anchor_sampling_tolerance_seconds = float(anchor_sampling_tolerance_seconds)
        self.clinical_root = clinical_root
        self.normalize_clinical = bool(normalize_clinical)
        self.participants_tsv_path = participants_tsv_path
        self.include_clinical_static = bool(include_clinical_static)
        self.include_participant_metadata = bool(include_participant_metadata)
        self.include_study_group = bool(include_study_group)
        self.include_clinical_site = bool(include_clinical_site)
        self.pad = bool(pad)
        self.return_dict = bool(return_dict)
        self.max_events_per_modality = max_events_per_modality or {}
        self.min_events_per_modality = min_events_per_modality or {}
        self.categorical_vocabs = categorical_vocabs or {}
        self.value_ranges = {} if value_ranges is None else dict(value_ranges)

        self.patient_data: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
        self.modality_stats: Dict[str, Dict[str, Any]] = {}

        for modality in self.modalities:
            spec = AIREADI_MODALITY_SPECS[modality]
            path = self.root / f"{spec.file_prefix}_{split}.parquet"
            if not path.exists():
                raise FileNotFoundError(path)
            df = pd.read_parquet(path)
            vocab = self.categorical_vocabs.setdefault(modality, {}) if spec.categorical else None
            by_patient = _build_aireadi_patient_index(df, spec, categorical_vocab=vocab)

            if self.normalize and not spec.categorical and modality not in self.value_ranges:
                all_values = [
                    item["value"][np.isfinite(item["value"])]
                    for item in by_patient.values()
                    if item["value"].size > 0
                ]
                if all_values:
                    values = np.concatenate(all_values)
                    self.value_ranges[modality] = (float(values.min()), float(values.max()))

            for patient_id, item in by_patient.items():
                values = item["value"].astype(np.float32)
                if self.normalize and not spec.categorical and modality in self.value_ranges:
                    values = _normalize_array(values, *self.value_ranges[modality])
                self.patient_data.setdefault(patient_id, {})[modality] = {
                    "value": values.astype(np.float32),
                    "time_local": item["time_local"].astype(np.int64),
                    "time_end_local": item["time_end_local"].astype(np.int64),
                }

            self.modality_stats[modality] = {
                "patients_with_data": len(by_patient),
                "events": int(sum(len(item["value"]) for item in by_patient.values())),
                "value_range": self.value_ranges.get(modality),
                "categorical_vocab": self.categorical_vocabs.get(modality),
            }

        self.patient_ids = sorted(
            patient_id
            for patient_id, modalities_by_patient in self.patient_data.items()
            if anchor_modality in modalities_by_patient
        )
        self.participant_metadata = (
            _load_aireadi_participant_metadata(participants_tsv_path)
            if self.include_participant_metadata
            else {}
        )
        self.clinical_features: Dict[str, torch.Tensor] = {}
        self.clinical_masks: Dict[str, torch.Tensor] = {}
        self.clinical_feature_names: list[str] = []
        self.clinical_feature_stats: Optional[Dict[str, Any]] = clinical_feature_stats
        if self.include_clinical_static and clinical_root is not None:
            (
                self.clinical_features,
                self.clinical_masks,
                self.clinical_feature_names,
                self.clinical_feature_stats,
            ) = build_aireadi_clinical_features(
                clinical_root=clinical_root,
                patient_ids=self.patient_ids,
                normalize=self.normalize_clinical,
                feature_stats=clinical_feature_stats,
                age_reference_year=clinical_age_reference_year,
            )

        self.window_index: list[tuple[str, Any]] = []
        for patient_id in self.patient_ids:
            if self.window_mode == "daily":
                window_refs = self._daily_window_refs(patient_id)
            else:
                anchor_len = len(self.patient_data[patient_id][anchor_modality]["value"])
                window_refs = self._window_starts(anchor_len)
            for window_ref in window_refs:
                if self._window_is_usable(patient_id, window_ref):
                    self.window_index.append((patient_id, window_ref))

        if not self.window_index:
            raise ValueError(
                "No AI-READI modality-imputation windows could be constructed "
                f"from {self.root} split={split}"
            )

    def _window_starts(self, seq_len: int) -> list[int]:
        if seq_len < self.window_size:
            return []
        starts = list(range(0, seq_len - self.window_size + 1, self.window_stride))
        last_start = seq_len - self.window_size
        if self.include_last_window and starts[-1] != last_start:
            starts.append(last_start)
        return starts

    def _daily_window_refs(self, patient_id: str) -> list[tuple[int, int, int]]:
        anchor_time = self.patient_data[patient_id][self.anchor_modality]["time_local"]
        if anchor_time.size == 0:
            return []

        timestamps = pd.to_datetime(anchor_time)
        day_starts = pd.DatetimeIndex(timestamps).normalize().unique()
        min_events = self.daily_min_events
        if min_events is None:
            min_events = self.window_size if self.require_complete else 1

        refs = []
        one_day_ns = int(24 * 60 * 60 * 1e9)
        for day_start in day_starts:
            day_start_ns = int(pd.Timestamp(day_start).value)
            day_end_ns = day_start_ns + one_day_ns - 1
            in_day = (anchor_time >= day_start_ns) & (anchor_time <= day_end_ns)
            indices = np.flatnonzero(in_day)
            if indices.size < min_events:
                continue
            refs.append((int(indices[0]), day_start_ns, day_end_ns))
        return refs

    def _event_indices_in_window(
        self,
        patient_id: str,
        modality: str,
        window_start_ns: int,
        window_end_ns: int,
    ) -> np.ndarray:
        item = self.patient_data.get(patient_id, {}).get(modality)
        if item is None:
            return np.asarray([], dtype=np.int64)
        event_time = item["time_local"]
        event_end_time = item["time_end_local"]
        overlaps = (event_time <= window_end_ns) & (event_end_time >= window_start_ns)
        return np.flatnonzero(overlaps).astype(np.int64)

    def _window_bounds(self, patient_id: str, start: Any) -> tuple[int, int]:
        if isinstance(start, tuple):
            return int(start[1]), int(start[2])
        anchor_time = self.patient_data[patient_id][self.anchor_modality]["time_local"]
        window_time = anchor_time[start:start + self.window_size]
        return int(window_time[0]), int(window_time[-1])

    def _anchor_times_for_window(self, patient_id: str, start: Any) -> np.ndarray:
        anchor_time = self.patient_data[patient_id][self.anchor_modality]["time_local"]
        if isinstance(start, tuple):
            window_start_ns, window_end_ns = self._window_bounds(patient_id, start)
            in_window = (anchor_time >= window_start_ns) & (anchor_time <= window_end_ns)
            return anchor_time[in_window]
        return anchor_time[start:start + self.window_size]

    def _window_is_usable(self, patient_id: str, start: Any) -> bool:
        window_start_ns, window_end_ns = self._window_bounds(patient_id, start)
        window_time = self._anchor_times_for_window(patient_id, start)
        if self.max_window_span_hours is not None:
            span_hours = (window_end_ns - window_start_ns) / 3.6e12
            if span_hours > float(self.max_window_span_hours):
                return False
        if self.max_anchor_gap_minutes is not None:
            if len(window_time) > 1:
                max_gap_minutes = np.diff(window_time).max() / 60e9
                if max_gap_minutes > float(self.max_anchor_gap_minutes):
                    return False
        if self.anchor_sampling_minutes is not None and len(window_time) > 1:
            expected_gap_ns = float(self.anchor_sampling_minutes) * 60e9
            tolerance_ns = self.anchor_sampling_tolerance_seconds * 1e9
            observed_gaps = np.diff(window_time).astype(np.float64)
            if not np.all(np.abs(observed_gaps - expected_gap_ns) <= tolerance_ns):
                return False
        for modality in self.modalities:
            min_events = self.min_events_per_modality.get(
                modality,
                1 if self.require_complete else 0,
            )
            if min_events <= 0:
                continue
            count = len(self._event_indices_in_window(patient_id, modality, window_start_ns, window_end_ns))
            if count < min_events:
                return False
        return True

    def _pack_modality(
        self,
        patient_id: str,
        modality: str,
        window_start_ns: int,
        window_end_ns: int,
    ) -> Dict[str, torch.Tensor]:
        item = self.patient_data.get(patient_id, {}).get(modality)
        if item is None:
            indices = np.asarray([], dtype=np.int64)
            values = np.asarray([], dtype=np.float32)
            times = np.asarray([], dtype=np.int64)
            end_times = np.asarray([], dtype=np.int64)
        else:
            indices = self._event_indices_in_window(patient_id, modality, window_start_ns, window_end_ns)
            values = item["value"][indices].astype(np.float32)
            times = item["time_local"][indices].astype(np.int64)
            end_times = item["time_end_local"][indices].astype(np.int64)

        raw_length = len(values)
        truncated = False
        if self.pad:
            max_events = int(self.max_events_per_modality.get(modality, self.window_size))
            if raw_length > max_events:
                values = values[:max_events]
                times = times[:max_events]
                end_times = end_times[:max_events]
                truncated = True
            length = len(values)
            packed_values = torch.zeros(max_events, 1, dtype=torch.float32)
            packed_times = torch.zeros(max_events, dtype=torch.long)
            packed_end_times = torch.zeros(max_events, dtype=torch.long)
            packed_relative_hours = torch.zeros(max_events, 1, dtype=torch.float32)
            mask = torch.zeros(max_events, 1, dtype=torch.float32)
            if length:
                packed_values[:length, 0] = torch.from_numpy(values)
                packed_times[:length] = torch.from_numpy(times)
                packed_end_times[:length] = torch.from_numpy(end_times)
                relative_hours = (times.astype(np.float64) - float(window_start_ns)) / 3.6e12
                packed_relative_hours[:length, 0] = torch.from_numpy(relative_hours.astype(np.float32))
                mask[:length, 0] = 1.0
        else:
            length = raw_length
            packed_values = torch.from_numpy(values).view(-1, 1)
            packed_times = torch.from_numpy(times).long()
            packed_end_times = torch.from_numpy(end_times).long()
            relative_hours = (times.astype(np.float64) - float(window_start_ns)) / 3.6e12
            packed_relative_hours = torch.from_numpy(relative_hours.astype(np.float32)).view(-1, 1)
            mask = torch.ones(length, 1, dtype=torch.float32)

        return {
            "values": packed_values,
            "time_local": packed_times,
            "time_end_local": packed_end_times,
            "relative_time_hours": packed_relative_hours,
            "mask": mask,
            "length": torch.tensor(length, dtype=torch.long),
            "raw_length": torch.tensor(raw_length, dtype=torch.long),
            "truncated": torch.tensor(truncated),
            "present": torch.tensor(raw_length > 0),
        }

    def __len__(self) -> int:
        return len(self.window_index)

    def __getitem__(self, idx: int):
        patient_id, start = self.window_index[idx]
        site_label, site_code, site_one_hot = _site_from_patient_id(patient_id)
        participant_meta = self.participant_metadata.get(patient_id, {})
        study_group_label = participant_meta.get("study_group_label", -1)
        study_group_one_hot = participant_meta.get(
            "study_group_one_hot",
            torch.zeros(len(AI_READI_STUDY_GROUPS), dtype=torch.float32),
        )
        clinical_site_label = participant_meta.get("clinical_site_label", site_label)
        clinical_site_one_hot = participant_meta.get("clinical_site_one_hot", site_one_hot)
        window_start_ns, window_end_ns = self._window_bounds(patient_id, start)
        modalities = {
            modality: self._pack_modality(patient_id, modality, window_start_ns, window_end_ns)
            for modality in self.modalities
        }
        target = modalities[self.target_modality]["values"] if self.target_modality else torch.empty(0)
        condition_modalities = [m for m in self.modalities if m != self.target_modality]

        if not self.return_dict:
            return target, modalities

        result = {
            "target": target,
            "modalities": modalities,
            "condition_modalities": condition_modalities,
            "patient_id": patient_id,
            "anchor_modality": self.anchor_modality,
            "target_modality": self.target_modality,
            "window_start_time_local": torch.tensor(window_start_ns, dtype=torch.long),
            "window_end_time_local": torch.tensor(window_end_ns, dtype=torch.long),
            "anchor_start_index": torch.tensor(start[0] if isinstance(start, tuple) else start, dtype=torch.long),
            "label": torch.tensor(0, dtype=torch.long),
        }

        if self.include_clinical_static:
            result["clinical_static"] = self.clinical_features.get(
                patient_id,
                torch.empty(0, dtype=torch.float32),
            )
            result["clinical_mask"] = self.clinical_masks.get(
                patient_id,
                torch.empty(0, dtype=torch.float32),
            )
        if self.include_study_group:
            result["study_group_label"] = torch.tensor(study_group_label, dtype=torch.long)
            result["study_group_one_hot"] = study_group_one_hot
            result["study_group"] = participant_meta.get("study_group", "")
        if self.include_clinical_site:
            result["clinical_site_label"] = torch.tensor(clinical_site_label, dtype=torch.long)
            result["clinical_site_code"] = torch.tensor(site_code, dtype=torch.long)
            result["clinical_site_one_hot"] = clinical_site_one_hot
            result["clinical_site"] = participant_meta.get("clinical_site", "")

        return result


def _time_series_from_row(
    row,
    value_column: str,
    time_column: str = "time_local",
) -> Optional[pd.Series]:
    values = np.asarray(getattr(row, value_column), dtype=np.float32)
    times_raw = getattr(row, time_column)
    if times_raw is None:
        return None
    times = pd.to_datetime(np.asarray(times_raw), errors="coerce")

    n = min(len(values), len(times))
    if n == 0:
        return None

    values = values[:n]
    times = pd.DatetimeIndex(times[:n])
    valid = np.isfinite(values) & ~times.isna()
    if not valid.any():
        return None

    series = pd.Series(values[valid], index=times[valid])
    series = series.groupby(level=0).mean().sort_index()
    return series


class GlucoseCalorieImputationDataset(Dataset):
    """
    Paired modality-imputation dataset.

    Glucose is the fixed-length target modality. Windows are cut from each
    patient's local-time glucose series, then activity/calorie events from the
    same patient are selected when their local timestamps fall inside the glucose
    window's local-time span.
    """

    def __init__(
        self,
        glucose_parquet_path: str,
        calorie_parquet_path: str,
        window_size: int,
        window_stride: int,
        delay: int,
        embedding: int,
        glucose_min: Optional[float] = None,
        glucose_max: Optional[float] = None,
        calorie_min: Optional[float] = None,
        calorie_max: Optional[float] = None,
        condition_as_image: bool = False,
        return_dict: bool = True,
        include_last_window: bool = True,
        max_activity_events: Optional[int] = None,
        require_activity: bool = True,
        min_activity_events: Optional[int] = None,
    ):
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if window_stride <= 0:
            raise ValueError("window_stride must be positive")
        if delay <= 0:
            raise ValueError("delay must be positive")
        if embedding <= 0:
            raise ValueError("embedding must be positive")

        self.window_size = window_size
        self.window_stride = window_stride
        self.condition_as_image = condition_as_image
        self.return_dict = return_dict
        self.include_last_window = include_last_window
        self.max_activity_events = max_activity_events or window_size
        self.require_activity = require_activity
        if min_activity_events is None:
            min_activity_events = 1 if require_activity else 0
        if min_activity_events < 0:
            raise ValueError("min_activity_events must be non-negative")
        self.min_activity_events = int(min_activity_events)
        self.embedder = DelayEmbedder(
            device=torch.device("cpu"),
            seq_len=window_size,
            delay=delay,
            embedding=embedding,
        )

        glucose_df = pd.read_parquet(
            glucose_parquet_path,
            columns=["glucose", "patient_id", "time_local"],
        )
        calorie_df = pd.read_parquet(
            calorie_parquet_path,
            columns=["calorie", "patient_id", "time_local"],
        )

        if glucose_min is None or glucose_max is None:
            glucose_min, glucose_max = _compute_sequence_min_max(glucose_df["glucose"])
        if calorie_min is None or calorie_max is None:
            calorie_min, calorie_max = _compute_sequence_min_max(calorie_df["calorie"])
        self.glucose_min = float(glucose_min)
        self.glucose_max = float(glucose_max)
        self.calorie_min = float(calorie_min)
        self.calorie_max = float(calorie_max)

        glucose_by_patient = self._build_patient_index(glucose_df, "glucose")
        calorie_by_patient = self._build_patient_index(calorie_df, "calorie")
        self.sequences = []
        self.window_index = []

        for patient_id, glucose_series in glucose_by_patient.items():
            if len(glucose_series) < self.window_size:
                continue
            calorie_series = calorie_by_patient.get(
                patient_id,
                pd.Series(dtype=np.float32, index=pd.DatetimeIndex([])),
            )
            if len(calorie_series) == 0:
                continue

            glucose_values = _normalize_array(
                glucose_series.to_numpy(dtype=np.float32),
                self.glucose_min,
                self.glucose_max,
            )
            activity_values = _normalize_array(
                calorie_series.to_numpy(dtype=np.float32),
                self.calorie_min,
                self.calorie_max,
            )
            glucose_time_local = pd.DatetimeIndex(glucose_series.index).asi8.astype(np.int64)
            activity_time_local = pd.DatetimeIndex(calorie_series.index).asi8.astype(np.int64)

            seq_idx = len(self.sequences)
            self.sequences.append(
                {
                    "glucose": torch.from_numpy(glucose_values),
                    "glucose_time_local": torch.from_numpy(glucose_time_local),
                    "activity_calorie": torch.from_numpy(activity_values),
                    "activity_time_local": torch.from_numpy(activity_time_local),
                    "patient_id": patient_id,
                }
            )
            for start in self._window_starts(len(glucose_values)):
                if self._activity_count_in_window(seq_idx, start) < self.min_activity_events:
                    continue
                self.window_index.append((seq_idx, start))

        if not self.window_index:
            raise ValueError(
                "No paired glucose/calorie windows could be constructed from "
                f"{glucose_parquet_path} and {calorie_parquet_path}"
            )

    @staticmethod
    def _build_patient_index(df: pd.DataFrame, value_column: str) -> Dict[str, pd.Series]:
        rows_by_patient: Dict[str, list] = {}
        for row in df.itertuples(index=False):
            patient_id = _first_patient_id(row.patient_id)
            if patient_id is None:
                continue
            series = _time_series_from_row(row, value_column)
            if series is None:
                continue
            rows_by_patient.setdefault(patient_id, []).append(series)

        by_patient = {}
        for patient_id, series_list in rows_by_patient.items():
            series = pd.concat(series_list).groupby(level=0).mean().sort_index()
            if len(series) > 0:
                by_patient[patient_id] = series
        return by_patient

    def _window_starts(self, seq_len: int) -> list[int]:
        if seq_len < self.window_size:
            return []

        starts = list(range(0, seq_len - self.window_size + 1, self.window_stride))
        last_start = seq_len - self.window_size
        if self.include_last_window and starts[-1] != last_start:
            starts.append(last_start)
        return starts

    def _activity_slice(self, seq_idx: int, start: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq = self.sequences[seq_idx]
        glucose_time = seq["glucose_time_local"][start:start + self.window_size]
        window_start = glucose_time[0]
        window_end = glucose_time[-1]
        activity_time = seq["activity_time_local"]
        in_window = (activity_time >= window_start) & (activity_time <= window_end)
        return torch.nonzero(in_window, as_tuple=False).flatten(), glucose_time

    def _has_activity_in_window(self, seq_idx: int, start: int) -> bool:
        activity_idx, _ = self._activity_slice(seq_idx, start)
        return activity_idx.numel() > 0

    def _activity_count_in_window(self, seq_idx: int, start: int) -> int:
        activity_idx, _ = self._activity_slice(seq_idx, start)
        return int(activity_idx.numel())

    def _pad_activity(
        self,
        activity_calorie: torch.Tensor,
        activity_time_local: torch.Tensor,
        glucose_time_local: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if activity_calorie.numel() > self.max_activity_events:
            n_target = min(self.max_activity_events, glucose_time_local.numel())
            source_time = activity_time_local.cpu().numpy().astype(np.float64)
            source_values = activity_calorie.cpu().numpy().astype(np.float32)
            target_time = glucose_time_local[:n_target].cpu().numpy().astype(np.float64)
            interpolated = np.interp(target_time, source_time, source_values).astype(np.float32)

            calorie = torch.zeros(self.max_activity_events, 1, dtype=torch.float32)
            time_local = torch.zeros(self.max_activity_events, dtype=torch.long)
            mask = torch.zeros(self.max_activity_events, 1, dtype=torch.float32)
            calorie[:n_target, 0] = torch.from_numpy(interpolated)
            time_local[:n_target] = glucose_time_local[:n_target].long()
            mask[:n_target, 0] = 1.0
            return (
                calorie,
                time_local,
                mask,
                torch.tensor(n_target, dtype=torch.long),
                torch.tensor(True),
            )

        n_events = activity_calorie.numel()
        calorie = torch.zeros(self.max_activity_events, 1, dtype=torch.float32)
        time_local = torch.zeros(self.max_activity_events, dtype=torch.long)
        mask = torch.zeros(self.max_activity_events, 1, dtype=torch.float32)

        if n_events > 0:
            calorie[:n_events, 0] = activity_calorie[:n_events].float()
            time_local[:n_events] = activity_time_local[:n_events].long()
            mask[:n_events, 0] = 1.0

        return (
            calorie,
            time_local,
            mask,
            torch.tensor(n_events, dtype=torch.long),
            torch.tensor(False),
        )

    def __len__(self) -> int:
        return len(self.window_index)

    def __getitem__(self, idx: int):
        seq_idx, start = self.window_index[idx]
        seq = self.sequences[seq_idx]
        end = start + self.window_size

        glucose = seq["glucose"][start:end].view(self.window_size, 1)
        glucose_time_local = seq["glucose_time_local"][start:end]
        activity_idx, _ = self._activity_slice(seq_idx, start)
        activity_calorie_raw = seq["activity_calorie"][activity_idx]
        activity_time_local_raw = seq["activity_time_local"][activity_idx]
        raw_activity_length = torch.tensor(activity_calorie_raw.numel(), dtype=torch.long)
        (
            activity_calorie,
            activity_time_local,
            activity_mask,
            activity_length,
            activity_interpolated,
        ) = self._pad_activity(
            activity_calorie_raw,
            activity_time_local_raw,
            glucose_time_local,
        )


        glucose_img = self.embedder.ts_to_img(
            glucose.unsqueeze(0),
            pad=True,
            mask=0.0,
        ).squeeze(0)

        condition = activity_calorie
        if self.condition_as_image:
            condition = self.embedder.ts_to_img(
                activity_calorie[:self.window_size].unsqueeze(0),
                pad=True,
                mask=0.0,
            ).squeeze(0)

        if not self.return_dict:
            return glucose_img, condition

        return {
            "target": glucose_img,
            "condition": condition,
            "condition_mask": activity_mask,
            "glucose": glucose,
            "glucose_time_local": glucose_time_local,
            "activity_calorie": activity_calorie,
            "activity_time_local": activity_time_local,
            "activity_mask": activity_mask,
            "activity_length": activity_length,
            "raw_activity_length": raw_activity_length,
            "activity_interpolated": activity_interpolated,
            "patient_id": seq["patient_id"],
            "label": torch.tensor(0, dtype=torch.long),
        }


def get_dataset(
    name: str,
    config: dict,
    root: str = "./data",
    download: bool = True,
    seed: int = 42,
) -> tuple:
    """Get dataset and transforms."""
    name = name.lower()
    if name in ["sine", "ts", "timeseries", "synthetic_sine"]:
        embedder = DelayEmbedder(
            device=torch.device("cpu"),
            seq_len=config["ts_seq_len"],
            delay=config["ts_delay"],
            embedding=config["ts_embedding"],
        )
        train_dataset = SyntheticSineDataset(
            num_samples=config["ts_num_samples_train"],
            embedder=embedder,
            seq_len=config["ts_seq_len"],
            components_min=config["ts_components_min"],
            components_max=config["ts_components_max"],
            freq_min=config["ts_freq_min"],
            freq_max=config["ts_freq_max"],
            amp_min=config["ts_amp_min"],
            amp_max=config["ts_amp_max"],
            noise_std=config["ts_noise_std"],
            seed=seed,
        )
        test_dataset = SyntheticSineDataset(
            num_samples=config["ts_num_samples_test"],
            embedder=embedder,
            seq_len=config["ts_seq_len"],
            components_min=config["ts_components_min"],
            components_max=config["ts_components_max"],
            freq_min=config["ts_freq_min"],
            freq_max=config["ts_freq_max"],
            amp_min=config["ts_amp_min"],
            amp_max=config["ts_amp_max"],
            noise_std=config["ts_noise_std"],
            seed=seed + 1_000_000,
        )
    elif name == "glucose":
        stride = config.get("ts_stride", 128)
        embedder = DelayEmbedder(
            device=torch.device("cpu"),
            seq_len=config["ts_seq_len"],
            delay=config["ts_delay"],
            embedding=config["ts_embedding"],
        )

        train_dataset = GlucoseSlidingWindowDataset(
            parquet_path=os.path.join(root, "glucose_train.parquet"),
            embedder=embedder,
            seq_len=config["ts_seq_len"],
            stride=stride,
        )

        test_dataset = GlucoseSlidingWindowDataset(
            parquet_path=os.path.join(root, "glucose_test.parquet"),
            embedder=embedder,
            seq_len=config["ts_seq_len"],
            stride=stride,
            value_min=train_dataset.value_min,
            value_max=train_dataset.value_max,
        )

        print("{} train and {} test datasets".format(len(train_dataset), len(test_dataset)))

    elif name in ["glucose_imputation", "glucose_calorie_imputation"]:
        stride = config.get("ts_stride", config.get("window_stride", 128))
        seq_len = config.get("ts_seq_len", config.get("window_size", 128))
        delay = config.get("ts_delay", config.get("delay", 12))
        embedding = config.get("ts_embedding", config.get("embedding", 12))
        condition_as_image = config.get("condition_as_image", False)
        return_dict = config.get("return_dict", True)
        max_activity_events = config.get("max_activity_events", seq_len)
        require_activity = config.get("require_activity", True)
        min_activity_events = config.get(
            "min_activity_events",
            1 if require_activity else 0,
        )

        train_dataset = GlucoseCalorieImputationDataset(
            glucose_parquet_path=os.path.join(root, "glucose_train.parquet"),
            calorie_parquet_path=os.path.join(root, "calorie_train.parquet"),
            window_size=seq_len,
            window_stride=stride,
            delay=delay,
            embedding=embedding,
            condition_as_image=condition_as_image,
            return_dict=return_dict,
            max_activity_events=max_activity_events,
            require_activity=require_activity,
            min_activity_events=min_activity_events,
        )

        test_dataset = GlucoseCalorieImputationDataset(
            glucose_parquet_path=os.path.join(root, "glucose_test.parquet"),
            calorie_parquet_path=os.path.join(root, "calorie_test.parquet"),
            window_size=seq_len,
            window_stride=stride,
            delay=delay,
            embedding=embedding,
            glucose_min=train_dataset.glucose_min,
            glucose_max=train_dataset.glucose_max,
            calorie_min=train_dataset.calorie_min,
            calorie_max=train_dataset.calorie_max,
            condition_as_image=condition_as_image,
            return_dict=return_dict,
            max_activity_events=max_activity_events,
            require_activity=require_activity,
            min_activity_events=min_activity_events,
        )

        print("{} train and {} test paired datasets".format(len(train_dataset), len(test_dataset)))

    elif name in ["aireadi_imputation", "aireadi_multimodal_imputation"]:
        modalities = config.get("modalities")
        anchor_modality = config.get("anchor_modality", "glucose")
        target_modality = config.get("target_modality", "glucose")
        seq_len = config.get("ts_seq_len", config.get("window_size", 128))
        stride = config.get("ts_stride", config.get("window_stride", seq_len))
        window_mode = config.get("window_mode", "sliding")
        daily_min_events = config.get("daily_min_events")
        max_events_per_modality = config.get("max_events_per_modality")
        require_complete = config.get("require_complete", False)
        min_events_per_modality = config.get("min_events_per_modality")
        max_anchor_gap_minutes = config.get("max_anchor_gap_minutes")
        max_window_span_hours = config.get("max_window_span_hours")
        anchor_sampling_minutes = config.get("anchor_sampling_minutes")
        anchor_sampling_tolerance_seconds = config.get("anchor_sampling_tolerance_seconds", 2.0)
        clinical_root = config.get("clinical_root")
        normalize_clinical = config.get("normalize_clinical", True)
        clinical_age_reference_year = config.get("clinical_age_reference_year", 2026)
        participants_tsv_path = config.get("participants_tsv_path")
        include_clinical_static = config.get("include_clinical_static", True)
        include_participant_metadata = config.get("include_participant_metadata", True)
        include_study_group = config.get("include_study_group", True)
        include_clinical_site = config.get("include_clinical_site", True)
        pad = config.get("pad", True)
        return_dict = config.get("return_dict", True)

        train_dataset = AIREADIModalityImputationDataset(
            root=root,
            split="train",
            modalities=modalities,
            anchor_modality=anchor_modality,
            target_modality=target_modality,
            window_size=seq_len,
            window_stride=stride,
            window_mode=window_mode,
            daily_min_events=daily_min_events,
            max_events_per_modality=max_events_per_modality,
            require_complete=require_complete,
            min_events_per_modality=min_events_per_modality,
            max_anchor_gap_minutes=max_anchor_gap_minutes,
            max_window_span_hours=max_window_span_hours,
            anchor_sampling_minutes=anchor_sampling_minutes,
            anchor_sampling_tolerance_seconds=anchor_sampling_tolerance_seconds,
            clinical_root=clinical_root,
            normalize_clinical=normalize_clinical,
            clinical_age_reference_year=clinical_age_reference_year,
            participants_tsv_path=participants_tsv_path,
            include_clinical_static=include_clinical_static,
            include_participant_metadata=include_participant_metadata,
            include_study_group=include_study_group,
            include_clinical_site=include_clinical_site,
            pad=pad,
            return_dict=return_dict,
        )

        test_dataset = AIREADIModalityImputationDataset(
            root=root,
            split="test",
            modalities=modalities,
            anchor_modality=anchor_modality,
            target_modality=target_modality,
            window_size=seq_len,
            window_stride=stride,
            window_mode=window_mode,
            daily_min_events=daily_min_events,
            max_events_per_modality=max_events_per_modality,
            value_ranges=train_dataset.value_ranges,
            categorical_vocabs=train_dataset.categorical_vocabs,
            require_complete=require_complete,
            min_events_per_modality=min_events_per_modality,
            max_anchor_gap_minutes=max_anchor_gap_minutes,
            max_window_span_hours=max_window_span_hours,
            anchor_sampling_minutes=anchor_sampling_minutes,
            anchor_sampling_tolerance_seconds=anchor_sampling_tolerance_seconds,
            clinical_root=clinical_root,
            normalize_clinical=normalize_clinical,
            clinical_feature_stats=train_dataset.clinical_feature_stats,
            clinical_age_reference_year=clinical_age_reference_year,
            participants_tsv_path=participants_tsv_path,
            include_clinical_static=include_clinical_static,
            include_participant_metadata=include_participant_metadata,
            include_study_group=include_study_group,
            include_clinical_site=include_clinical_site,
            pad=pad,
            return_dict=return_dict,
        )

        print("{} train and {} test AI-READI multimodal windows".format(len(train_dataset), len(test_dataset)))

    elif name == "mnist":
        # MNIST data will be at {root}/mnist/MNIST/raw/
        mnist_root = os.path.join(root, "mnist")
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # [-1, 1]
        ])
        train_dataset = datasets.MNIST(mnist_root, train=True, download=download, transform=transform)
        test_dataset = datasets.MNIST(mnist_root, train=False, download=download, transform=transform)
    elif name in ["cifar10", "cifar"]:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        train_dataset = datasets.CIFAR10(root, train=True, download=download, transform=transform)
        test_dataset = datasets.CIFAR10(root, train=False, download=download, transform=test_transform)
    else:
        raise ValueError(f"Unknown dataset: {name}. Use one of: sine, mnist, cifar10")

    return train_dataset, test_dataset
