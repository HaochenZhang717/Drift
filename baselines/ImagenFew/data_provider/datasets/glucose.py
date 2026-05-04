import os
import numpy as np
import pandas as pd
import torch


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
        raise ValueError("Could not compute min/max from glucose parquet data")
    return min(mins), max(maxs)


def _normalize_array(values: np.ndarray, value_min: float, value_max: float) -> np.ndarray:
    scale = max(float(value_max) - float(value_min), 1e-6)
    values = 2.0 * (values.astype(np.float32) - float(value_min)) / scale - 1.0
    return np.clip(values, -1.0, 1.0)


def Glucose(seq_len, datasets_dir, rel_path="glucose_train.parquet", stride=1, **kwargs):
    parquet_path = os.path.join(datasets_dir, rel_path)
    flag = kwargs.get("flag", "unknown")
    print(
        f"[GlucoseDataset] flag={flag} parquet_path={parquet_path} "
        f"seq_len={seq_len} stride={stride}"
    )
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(
            f"Glucose parquet not found: {parquet_path}. "
            "Expected a parquet with a 'glucose' column containing sequences."
        )

    df = pd.read_parquet(parquet_path)
    if "glucose" not in df.columns:
        raise ValueError(f"Expected column 'glucose' in {parquet_path}")

    value_min, value_max = _compute_sequence_min_max(df["glucose"])
    windows = []
    for series in df["glucose"]:
        arr = np.asarray(series, dtype=np.float32)
        if arr.size < seq_len:
            continue
        arr = _normalize_array(arr, value_min, value_max)
        for start in range(0, arr.size - seq_len + 1, int(stride)):
            windows.append(arr[start:start + seq_len])

    if not windows:
        raise ValueError(
            f"No glucose sliding windows produced from {parquet_path} "
            f"with seq_len={seq_len}, stride={stride}"
        )

    tensor = torch.from_numpy(np.asarray(windows, dtype=np.float32)).unsqueeze(-1)
    return tensor
