import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _resolve_glucose_path(root_path, rel_path, flag):
    if rel_path:
        path = rel_path
        if "{split}" in path:
            split_name = "valid" if flag in {"val", "valid"} else flag
            path = path.format(split=split_name)
        path = path if os.path.isabs(path) else os.path.join(root_path, path)
        if os.path.isdir(path):
            return os.path.join(path, _split_filename(flag))
        return path
    return os.path.join(root_path, _split_filename(flag))


def _split_filename(flag):
    if flag == "train":
        return "glucose_train.parquet"
    if flag in {"val", "valid"}:
        return "glucose_valid.parquet"
    if flag == "test":
        return "glucose_test.parquet"
    raise ValueError(f"Unsupported split '{flag}'. Expected train, val/valid, or test.")


def _compute_sequence_min_max(series):
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
        raise ValueError("Could not compute min/max from glucose parquet data.")
    return min(mins), max(maxs)


def _normalize_array(values, value_min, value_max):
    scale = max(float(value_max) - float(value_min), 1e-6)
    values = 2.0 * (values.astype(np.float32) - float(value_min)) / scale - 1.0
    return np.clip(values, -1.0, 1.0)


def GlucoseSliding(**config):
    return GlucoseSlidingWindowTensorDataset(
        root_path=config["datasets_dir"],
        rel_path=config.get("rel_path", None),
        flag=config["flag"],
        seq_len=config["seq_len"],
        stride=config.get("window_stride", config.get("ts_stride", config.get("stride", 1))),
        normalize=config.get("normalize", True),
        column=config.get("column", "glucose"),
    )


class GlucoseSlidingWindowTensorDataset(Dataset):
    """Sliding glucose windows from parquet, returned as time-series tensors (T, 1)."""

    def __init__(
        self,
        root_path,
        rel_path=None,
        flag="train",
        seq_len=128,
        stride=1,
        normalize=True,
        column="glucose",
    ):
        super().__init__()
        self.root_path = root_path
        self.rel_path = rel_path
        self.flag = flag
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.normalize = normalize
        self.column = column
        self.parquet_path = _resolve_glucose_path(root_path, rel_path, flag)

        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive.")
        if self.stride <= 0:
            raise ValueError("stride must be positive.")
        if not os.path.exists(self.parquet_path):
            raise FileNotFoundError(f"Glucose parquet not found: {self.parquet_path}")

        self.df = pd.read_parquet(self.parquet_path)
        if self.column not in self.df.columns:
            raise ValueError(f"Expected column '{self.column}' in {self.parquet_path}")

        self.value_min = None
        self.value_max = None
        if self.normalize:
            train_path = _resolve_glucose_path(root_path, rel_path, "train")
            train_df = pd.read_parquet(train_path)
            if self.column not in train_df.columns:
                raise ValueError(f"Expected column '{self.column}' in {train_path}")
            self.value_min, self.value_max = _compute_sequence_min_max(train_df[self.column])

        self.windows = []
        for row_idx, series in enumerate(self.df[self.column]):
            if len(series) < self.seq_len:
                continue
            for start in range(0, len(series) - self.seq_len + 1, self.stride):
                self.windows.append((row_idx, start))

        if not self.windows:
            raise ValueError(
                f"No glucose windows produced from {self.parquet_path} "
                f"with seq_len={self.seq_len}, stride={self.stride}."
            )

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        row_idx, start = self.windows[idx]
        series = self.df[self.column].iloc[row_idx]
        window = np.asarray(series[start:start + self.seq_len], dtype=np.float32)
        if self.normalize:
            window = _normalize_array(window, self.value_min, self.value_max)
        return torch.from_numpy(window.astype(np.float32)).unsqueeze(-1)

    def inverse_transform(self, data):
        if not self.normalize or self.value_min is None or self.value_max is None:
            return data
        if torch.is_tensor(data):
            data_np = data.detach().cpu().numpy()
        else:
            data_np = np.asarray(data)
        restored = ((data_np + 1.0) / 2.0) * (self.value_max - self.value_min) + self.value_min
        if torch.is_tensor(data):
            return torch.from_numpy(restored).to(data.dtype)
        return restored
