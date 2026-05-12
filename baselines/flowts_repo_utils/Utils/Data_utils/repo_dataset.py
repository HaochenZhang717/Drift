import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class RepoWindowDataset(Dataset):
    def __init__(
        self,
        name,
        datasets_dir,
        data_name,
        window=64,
        period="train",
        output_dir="./OUTPUT",
        rel_path=None,
        rel_path_train=None,
        rel_path_valid=None,
        rel_path_test=None,
        stride=1,
        window_stride=None,
        ts_stride=None,
        save2npy=True,
        **_,
    ):
        super().__init__()
        self.name = name
        self.window = int(window)
        self.period = period
        self.auto_norm = True
        self.var_num = None
        self.save2npy = save2npy
        self.output_dir = output_dir

        split_map = {"train": "train", "valid": "valid", "val": "valid", "test": "test"}
        if period not in split_map:
            raise ValueError(f"Unsupported period '{period}'. Expected one of {sorted(split_map)}")
        split_flag = split_map[period]

        stride_val = window_stride if window_stride is not None else (ts_stride if ts_stride is not None else stride)

        rel_for_split = rel_path
        if split_flag == "train" and rel_path_train:
            rel_for_split = rel_path_train
        elif split_flag == "valid" and rel_path_valid:
            rel_for_split = rel_path_valid
        elif split_flag == "test" and rel_path_test:
            rel_for_split = rel_path_test

        dataset = self._build_repo_dataset(
            data_name=data_name,
            datasets_dir=datasets_dir,
            seq_len=self.window,
            flag=split_flag,
            rel_path=rel_for_split,
            stride=stride_val,
        )

        self.samples = self._to_tensor(dataset)
        self.sample_num = self.samples.shape[0]
        self.var_num = self.samples.shape[-1]
        self._maybe_dump_arrays()

    def _build_repo_dataset(self, data_name, datasets_dir, seq_len, flag, rel_path, stride):
        from data_provider import datasets as repo_datasets

        if not hasattr(repo_datasets, data_name):
            raise ValueError(f"Unsupported data_name '{data_name}' in data_provider.datasets.")

        builder = getattr(repo_datasets, data_name)
        cfg = {
            "datasets_dir": datasets_dir,
            "seq_len": seq_len,
            "flag": flag,
            "stride": stride,
        }
        if rel_path:
            cfg["rel_path"] = rel_path
        return builder(**cfg)

    def _to_tensor(self, dataset):
        items = []
        for idx in range(len(dataset)):
            x = dataset[idx]
            if isinstance(x, (tuple, list)):
                x = x[0]
            if torch.is_tensor(x):
                x = x.detach().cpu().to(torch.float32)
            else:
                x = torch.tensor(x, dtype=torch.float32)
            if x.ndim == 1:
                x = x.unsqueeze(-1)
            items.append(x)
        if not items:
            raise ValueError("Dataset produced zero samples.")
        return torch.stack(items, dim=0)

    def _maybe_dump_arrays(self):
        if not self.save2npy:
            return
        split_tag = "train" if self.period == "train" else "test"
        out_dir = Path(self.output_dir) / "samples"
        out_dir.mkdir(parents=True, exist_ok=True)
        arr = self.samples.numpy()
        np.save(out_dir / f"{self.name}_norm_truth_{self.window}_{split_tag}.npy", arr)
        np.save(out_dir / f"{self.name}_ground_truth_{self.window}_{split_tag}.npy", arr)

    def __getitem__(self, ind):
        return self.samples[ind]

    def __len__(self):
        return self.sample_num
