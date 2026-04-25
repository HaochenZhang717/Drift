import argparse
import math
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional
import matplotlib
matplotlib.use("Agg")
import pandas as pd
import matplotlib.pyplot as plt
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

    ):

        self.df = pd.read_parquet(parquet_path)

        self.embedder = embedder

        self.seq_len = seq_len

        self.stride = stride

        self.normalize = normalize

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

    def _process_ts(self, ts):

        ts = torch.tensor(ts, dtype=torch.float32)

        if self.normalize:
            ts = normalize_ts_to_unit_range(ts)

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
        embedder = DelayEmbedder(
            device=torch.device("cpu"),
            seq_len=config["ts_seq_len"],
            delay=config["ts_delay"],
            embedding=config["ts_embedding"],
        )

        train_dataset = GlucoseSlidingWindowDataset(
            parquet_path="./AI-READI/glucose_train.parquet",
            embedder=embedder,
            seq_len=config["ts_seq_len"],
            stride=128,
        )

        test_dataset = GlucoseSlidingWindowDataset(
            parquet_path="./AI-READI/glucose_test.parquet",
            embedder=embedder,
            seq_len=config["ts_seq_len"],
            stride=128,
        )

        print("{} train and {} test datasets".format(len(train_dataset), len(test_dataset)))

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


