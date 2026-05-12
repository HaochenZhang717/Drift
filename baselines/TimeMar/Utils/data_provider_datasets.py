import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_provider.data_provider import data_dict  # noqa: E402


def _as_numpy(dataset):
    samples = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        if isinstance(item, (tuple, list)):
            item = item[0]
        if torch.is_tensor(item):
            item = item.detach().cpu().numpy()
        samples.append(np.asarray(item, dtype=np.float32))
    if not samples:
        raise ValueError("Dataset produced no samples.")
    return np.stack(samples).astype(np.float32)


def _to_zero_one(x):
    return np.clip((x + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)


class DataProviderWindowDataset(Dataset):
    """Adapter that exposes project data_provider datasets to TimeMar.

    TimeMar trains on tensors in [-1, 1] with shape [N, T, C]. Its AR stage also
    expects normalized [0, 1] numpy files under output/samples; this adapter writes
    those files when instantiated so train_ar.py can run unchanged.
    """

    def __init__(
        self,
        name,
        data,
        datasets_dir,
        window,
        split="train",
        save2npy=True,
        output_dir="./output",
        **kwargs,
    ):
        super().__init__()
        if split == "val":
            split = "valid"
        if split not in {"train", "valid", "test"}:
            raise ValueError(f"Unsupported split '{split}'. Expected train/valid/test.")

        self.name = name
        self.data = data
        self.datasets_dir = datasets_dir
        self.window = int(window)
        self.split = split
        self.output_dir = output_dir

        backend = data_dict.get(data)
        if backend is None:
            raise ValueError(f"Unknown data_provider backend: {data}")

        provider_config = dict(kwargs)
        provider_config.update(
            {
                "name": name,
                "data": data,
                "datasets_dir": datasets_dir,
                "seq_len": self.window,
                "flag": "val" if split == "valid" else split,
            }
        )
        self.provider_dataset = backend(**provider_config)
        self.samples = _as_numpy(self.provider_dataset)

        if self.samples.ndim != 3:
            raise ValueError(
                f"Expected data_provider dataset to return [N, T, C], got {self.samples.shape}."
            )
        if self.samples.shape[1] != self.window:
            raise ValueError(
                f"Expected window length {self.window}, got {self.samples.shape[1]}."
            )

        if save2npy:
            self._save_timemar_arrays()

    def _save_timemar_arrays(self):
        samples_dir = os.path.join(self.output_dir, "samples")
        os.makedirs(samples_dir, exist_ok=True)

        ground_truth = self.samples
        inverse_transform = getattr(self.provider_dataset, "inverse_transform", None)
        if inverse_transform is not None:
            try:
                restored = inverse_transform(torch.from_numpy(self.samples))
                if torch.is_tensor(restored):
                    restored = restored.detach().cpu().numpy()
                ground_truth = np.asarray(restored, dtype=np.float32)
            except Exception:
                ground_truth = self.samples

        suffix = "valid" if self.split == "valid" else self.split
        np.save(
            os.path.join(samples_dir, f"{self.name}_ground_truth_{self.window}_{suffix}.npy"),
            ground_truth.astype(np.float32),
        )
        np.save(
            os.path.join(samples_dir, f"{self.name}_norm_truth_{self.window}_{suffix}.npy"),
            _to_zero_one(self.samples),
        )

    def __getitem__(self, idx):
        return torch.from_numpy(self.samples[idx]).float()

    def __len__(self):
        return int(self.samples.shape[0])
