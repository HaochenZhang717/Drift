import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _resolve_path(root_path, rel_path):
    path = Path(rel_path)
    if not path.is_absolute():
        path = Path(root_path) / rel_path
    return path


def _extract_year(path):
    match = re.search(r"(19|20)\d{2}", path.stem)
    return int(match.group()) if match else None


def _split_borders(num_rows, seq_len, flag):
    num_train = int(num_rows * 0.8)

    if flag == "train":
        return 0, num_train, 0, num_train
    if flag in {"val", "valid", "test"}:
        return num_train, num_rows, 0, num_train
    raise ValueError(f"Unsupported split '{flag}'. Expected train, val/valid, or test.")


class _CSVWindowDataset(Dataset):
    def __init__(
        self,
        root_path,
        rel_path,
        flag="train",
        seq_len=24,
        scale=True,
        stride=1,
        value_cols=None,
        drop_cols=None,
    ):
        super().__init__()
        self.root_path = root_path
        self.rel_path = rel_path
        self.flag = flag
        self.seq_len = int(seq_len)
        self.scale = scale
        self.stride = int(stride)
        self.value_cols = value_cols
        self.drop_cols = set(drop_cols or [])
        self.data_min = None
        self.data_max = None

        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive.")
        if self.stride <= 0:
            raise ValueError("stride must be positive.")

        full_data = self._read_data()
        self._prepare_data(full_data)

    def _read_data(self):
        raise NotImplementedError

    def _select_numeric_data(self, df):
        if self.value_cols:
            missing = [col for col in self.value_cols if col not in df.columns]
            if missing:
                raise ValueError(f"Missing requested value columns in {self.rel_path}: {missing}")
            data = df[self.value_cols]
        else:
            data = df.drop(columns=[col for col in self.drop_cols if col in df.columns])
            data = data.select_dtypes(include=[np.number])

        if data.empty:
            raise ValueError(f"No numeric value columns found in {self.rel_path}.")

        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.interpolate(limit_direction="both").ffill().bfill()
        data = data.dropna(axis=1, how="all")
        if data.empty:
            raise ValueError(f"All numeric value columns are empty after cleaning {self.rel_path}.")
        return data.to_numpy(dtype=np.float32)

    def _prepare_data(self, full_data):
        border1, border2, train_border1, train_border2 = _split_borders(
            len(full_data), self.seq_len, self.flag
        )

        train_data = full_data[train_border1:train_border2]
        split_data = full_data[border1:border2]
        if len(split_data) < self.seq_len:
            raise ValueError(
                f"Split '{self.flag}' for {self.rel_path} has {len(split_data)} rows, "
                f"which is shorter than seq_len={self.seq_len}."
            )

        if self.scale:
            self.data_min = train_data.min(axis=0, keepdims=True)
            self.data_max = train_data.max(axis=0, keepdims=True)
            denom = self.data_max - self.data_min
            denom = np.where(denom == 0, 1.0, denom)
            split_data = 2.0 * ((split_data - self.data_min) / denom) - 1.0
            if self.flag in {"val", "valid", "test"}:
                split_data = np.clip(split_data, -1.0, 1.0)
            split_data = split_data.astype(np.float32)

        self.data = torch.from_numpy(split_data.astype(np.float32))
        self.window_starts = list(range(0, len(self.data) - self.seq_len + 1, self.stride))

    def __len__(self):
        return len(self.window_starts)

    def __getitem__(self, index):
        start = self.window_starts[index]
        return self.data[start : start + self.seq_len]

    def inverse_transform(self, data):
        if self.data_min is None or self.data_max is None:
            return data

        original_shape = data.shape
        if torch.is_tensor(data):
            data_np = data.detach().cpu().numpy()
        else:
            data_np = np.asarray(data)

        flat = data_np.reshape(-1, original_shape[-1])
        restored = ((flat + 1.0) / 2.0) * (self.data_max - self.data_min) + self.data_min
        restored = restored.reshape(original_shape)
        if torch.is_tensor(data):
            return torch.from_numpy(restored).to(data.dtype)
        return restored


class ErcotData(_CSVWindowDataset):
    def __init__(self, **config):
        rel_path = config.get("rel_path", "ERCOT_merged.csv")
        super().__init__(
            root_path=config["datasets_dir"],
            rel_path=rel_path,
            flag=config["flag"],
            seq_len=config["seq_len"],
            scale=config.get("scale", True),
            stride=config.get("window_stride", 1),
            value_cols=config.get("value_cols"),
            drop_cols=config.get("drop_cols", ["source_year"]),
        )

    def _read_data(self):
        path = _resolve_path(self.root_path, self.rel_path)
        if path.is_file():
            df = pd.read_csv(path)
            return self._select_numeric_data(df)

        if not path.is_dir():
            raise FileNotFoundError(
                f"ERCOT data path not found: {path}. Provide rel_path for a merged CSV or a directory of ERCOT CSVs."
            )

        merged_path = path / "ERCOT_merged.csv"
        if merged_path.exists():
            df = pd.read_csv(merged_path)
            return self._select_numeric_data(df)

        csv_files = [csv_path for csv_path in path.glob("*.csv") if _extract_year(csv_path) is not None]
        csv_files = sorted(csv_files, key=_extract_year)
        if not csv_files:
            raise FileNotFoundError(
                f"No ERCOT CSV files with a year in the filename were found in {path}."
            )

        frames = []
        for csv_path in csv_files:
            df = pd.read_csv(csv_path)
            if df.shape[1] < 2:
                raise ValueError(f"{csv_path} has fewer than two columns; expected time plus values.")
            frames.append(df)
        return self._select_numeric_data(pd.concat(frames, ignore_index=True))


class HouseholdData(_CSVWindowDataset):
    def __init__(self, **config):
        rel_path = config.get("rel_path", "HouseHold_6.csv")
        super().__init__(
            root_path=config["datasets_dir"],
            rel_path=rel_path,
            flag=config["flag"],
            seq_len=config["seq_len"],
            scale=config.get("scale", True),
            stride=config.get("window_stride", 1),
            value_cols=config.get("value_cols"),
            drop_cols=config.get("drop_cols"),
        )

    def _read_data(self):
        path = _resolve_path(self.root_path, self.rel_path)
        if not path.exists():
            raise FileNotFoundError(f"Household data file not found: {path}")
        df = pd.read_csv(path)
        return self._select_numeric_data(df)
