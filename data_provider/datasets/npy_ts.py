import os

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


def NpyTimeSeries(**config):
    return Dataset_NpyTimeSeries(
        root_path=config["datasets_dir"],
        rel_path=config["rel_path"],
        flag=config["flag"],
        scale=config.get("scale", True),
    )


class Dataset_NpyTimeSeries(Dataset):
    SPLIT_MAP = {
        "train": "train",
        "val": "valid",
        "valid": "valid",
        "test": "test",
    }

    def __init__(self, root_path, rel_path, flag="train", scale=True):
        if flag not in self.SPLIT_MAP:
            raise ValueError(f"Unsupported split '{flag}'. Expected one of {sorted(self.SPLIT_MAP)}")

        self.root_path = root_path
        self.rel_path = rel_path
        self.flag = flag
        self.scale = scale
        self.dataset_dir = os.path.join(root_path, rel_path)
        self.scaler = None

        split = self.SPLIT_MAP[flag]
        train_data = self._load_array("train_ts.npy")
        current_data = self._load_array(f"{split}_ts.npy")
        self.data = self._scale_data(train_data, current_data)

    def _load_array(self, filename):
        path = os.path.join(self.dataset_dir, filename)
        data = np.load(path, allow_pickle=True)
        if data.dtype == object:
            data = np.stack(data)
        data = np.asarray(data, dtype=np.float32)
        if data.ndim == 2:
            data = data[..., None]
        if data.ndim != 3:
            raise ValueError(f"Expected {path} to have shape (N, T, C) or (N, T), got {data.shape}")
        return torch.from_numpy(data).to(torch.float32)

    def _scale_mode(self):
        if self.scale is True:
            return "standard"
        if self.scale is False or self.scale is None:
            return "none"
        return str(self.scale).lower()

    def _scale_data(self, train_data, current_data):
        mode = self._scale_mode()
        if mode in {"none", "false", "0"}:
            return current_data

        num_features = train_data.shape[-1]
        train_np = train_data.reshape(-1, num_features).cpu().numpy()
        current_np = current_data.reshape(-1, num_features).cpu().numpy()

        if mode in {"standard", "standardize", "zscore", "true", "1"}:
            self.scaler = StandardScaler()
            self.scaler.fit(train_np)
            scaled = self.scaler.transform(current_np)
        elif mode in {"minmax", "minmax11", "minus_one_one", "-1,1", "[-1,1]"}:
            data_min = train_np.min(axis=0, keepdims=True)
            data_max = train_np.max(axis=0, keepdims=True)
            denom = data_max - data_min
            denom = np.where(denom == 0, 1.0, denom)
            scaled = 2.0 * ((current_np - data_min) / denom) - 1.0
            self.scaler = {
                "mode": "minmax",
                "min": data_min.astype(np.float32),
                "max": data_max.astype(np.float32),
            }
        else:
            raise ValueError(
                f"Unsupported scale mode '{self.scale}'. "
                "Use false, true/standard, or minmax for [-1, 1] scaling."
            )

        return torch.from_numpy(scaled.reshape(current_data.shape)).to(torch.float32)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def inverse_transform(self, data):
        original_shape = data.shape
        num_features = original_shape[-1]

        if torch.is_tensor(data):
            data_np = data.detach().cpu().numpy()
        else:
            data_np = np.asarray(data)

        flat = data_np.reshape(-1, num_features)
        mode = self._scale_mode()
        if mode in {"standard", "standardize", "zscore", "true", "1"}:
            restored = self.scaler.inverse_transform(flat).reshape(original_shape)
        elif mode in {"minmax", "minmax11", "minus_one_one", "-1,1", "[-1,1]"}:
            data_min = self.scaler["min"]
            data_max = self.scaler["max"]
            restored = ((flat + 1.0) / 2.0) * (data_max - data_min) + data_min
            restored = restored.reshape(original_shape)
        else:
            restored = data_np
        if torch.is_tensor(data):
            return torch.from_numpy(restored).to(data.dtype)
        return restored
