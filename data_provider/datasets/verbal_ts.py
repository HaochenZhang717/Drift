import json
import os

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


def VerbalTS(**config):
    return Dataset_VerbalTS(
        root_path=config["datasets_dir"],
        rel_path=config["rel_path"],
        flag=config["flag"],
        scale=config.get("scale", True),
    )


class Dataset_VerbalTS(Dataset):
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
        self.scaler = StandardScaler()

        split = self.SPLIT_MAP[flag]
        train_data = self._load_array("train_ts.npy")
        current_data = self._load_array(f"{split}_ts.npy")
        self.data = self._scale_data(train_data, current_data)
        self.attrs_idx = self._load_optional_array(f"{split}_attrs_idx.npy")
        self.text_caps = self._load_optional_array(f"{split}_text_caps.npy")
        self.meta = self._load_meta()

    def _load_array(self, filename):
        path = os.path.join(self.dataset_dir, filename)
        data = np.load(path, allow_pickle=True)
        if data.dtype == object:
            data = np.stack(data)
        return torch.from_numpy(data).to(torch.float32)

    def _load_optional_array(self, filename):
        path = os.path.join(self.dataset_dir, filename)
        if not os.path.exists(path):
            return None
        return np.load(path, allow_pickle=True)

    def _scale_data(self, train_data, current_data):
        if not self.scale:
            return current_data

        num_features = train_data.shape[-1]
        self.scaler.fit(train_data.reshape(-1, num_features).cpu().numpy())

        current_np = current_data.reshape(-1, num_features).cpu().numpy()
        scaled = self.scaler.transform(current_np)
        return torch.from_numpy(scaled.reshape(current_data.shape)).to(torch.float32)

    def _load_meta(self):
        path = os.path.join(self.dataset_dir, "meta.json")
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

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

        restored = self.scaler.inverse_transform(data_np.reshape(-1, num_features)).reshape(original_shape)
        if torch.is_tensor(data):
            return torch.from_numpy(restored).to(data.dtype)
        return restored
