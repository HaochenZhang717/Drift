import numpy as np
import torchaudio.transforms as transforms
import os
import sys
import torch
import functools
import torch.utils.data as Data
import pandas as pd
from sklearn.preprocessing import StandardScaler

from torch.utils.data.distributed import DistributedSampler

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def MinMaxScaler(data, return_scalers=False):
    """Min Max normalizer.

    Args:
      - data: original data

    Returns:
      - norm_data: normalized data
    """
    min = np.min(data, 0)
    max = np.max(data, 0)
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-7)
    if return_scalers:
        return norm_data, min, max
    return norm_data


def MinMaxArgs(data, min, max):
    """
    Args:
        data: given data
        min: given min value
        max: given max value

    Returns:
        min-max scaled data by given min and max
    """
    numerator = data - min
    denominator = max - min
    norm_data = numerator / (denominator + 1e-7)
    return norm_data


import torch

def sine_data_generation(no: int, seq_len: int, dim: int) -> torch.Tensor:
    """
    Generate synthetic sine wave data using PyTorch.

    Args:
        no (int): Number of samples.
        seq_len (int): Sequence length.
        dim (int): Number of features.

    Returns:
        torch.Tensor: Generated data of shape (no, seq_len, dim), normalized to [0, 1].
    """
    # Random frequencies and phases for each sample and feature
    freq = torch.rand(no, dim) * 0.1  # (no, dim)
    phase = torch.rand(no, dim) * 0.1  # (no, dim)

    # Time indices
    t = torch.arange(seq_len).float()  # (seq_len,)

    # Expand to match dimensions for broadcasting
    t = t.view(1, seq_len, 1)  # (1, seq_len, 1)
    freq = freq.view(no, 1, dim)  # (no, 1, dim)
    phase = phase.view(no, 1, dim)  # (no, 1, dim)

    # Generate sine waves
    data = torch.sin(freq * t + phase)  # (no, seq_len, dim)

    # Normalize from [-1, 1] to [0, 1]
    data = (data + 1) * 0.5

    return data





AIREADI_SIGNAL_TO_SPLIT_FILE = {
    "glucose": {
        "train": "glucose_train.parquet",
        "val": "glucose_valid.parquet",
        "valid": "glucose_valid.parquet",
        "test": "glucose_test.parquet",
    },
    "calorie": {
        "train": "calorie_train.parquet",
        "val": "calorie_valid.parquet",
        "valid": "calorie_valid.parquet",
        "test": "calorie_test.parquet",
    },
}


def resolve_aireadi_signal_path(root_path, signal="glucose", rel_path="AI-READI-processed", split="train"):
    signal_to_file = AIREADI_SIGNAL_TO_SPLIT_FILE.get(signal)
    if signal_to_file is None:
        raise ValueError(f"Unsupported AI-READI signal {signal}. Expected one of {sorted(AIREADI_SIGNAL_TO_SPLIT_FILE)}")
    if split not in signal_to_file:
        raise ValueError(f"Unsupported split {split}. Expected one of {sorted(signal_to_file)}")
    return os.path.join(root_path, rel_path, signal_to_file[split])


def read_aireadi_signal_frame(root_path, signal="glucose", rel_path="AI-READI-processed", split="train"):
    return pd.read_parquet(resolve_aireadi_signal_path(root_path, signal=signal, rel_path=rel_path, split=split))


def fit_aireadi_signal_scaler(root_path, signal="glucose", rel_path="AI-READI-processed", drop_nan=True):
    scaler = StandardScaler()
    train_df = read_aireadi_signal_frame(root_path, signal=signal, rel_path=rel_path, split="train")
    all_values = []
    for values in train_df[signal]:
        arr = np.asarray(values, dtype=np.float32)
        if drop_nan:
            arr = arr[np.isfinite(arr)]
        if arr.size > 0:
            all_values.append(arr)

    if not all_values:
        raise ValueError(f"No valid {signal} values found in AI-READI train split.")

    scaler.fit(np.concatenate(all_values, axis=0).reshape(-1, 1))
    return scaler


def normalize_aireadi_signal_values(values, scaler=None, drop_nan=True):
    arr = np.asarray(values, dtype=np.float32)
    if drop_nan:
        arr = arr[np.isfinite(arr)]
    if scaler is not None and arr.size > 0:
        arr = scaler.transform(arr.reshape(-1, 1)).reshape(-1).astype(np.float32)
    return arr


def load_aireadi_windows(
    root_path,
    signal="glucose",
    rel_path="AI-READI-processed",
    split="train",
    seq_len=24,
    scale=True,
    stride=1,
    min_seq_len=None,
    drop_nan=True,
):
    seq_len = int(seq_len)
    stride = int(stride)
    min_seq_len = int(min_seq_len) if min_seq_len is not None else seq_len

    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if min_seq_len < seq_len:
        min_seq_len = seq_len

    scaler = fit_aireadi_signal_scaler(
        root_path, signal=signal, rel_path=rel_path, drop_nan=drop_nan
    ) if scale else None
    df = read_aireadi_signal_frame(root_path, signal=signal, rel_path=rel_path, split=split)

    samples = []
    sample_index = []
    for row_idx, row in enumerate(df.itertuples(index=False)):
        values = normalize_aireadi_signal_values(getattr(row, signal), scaler=scaler, drop_nan=drop_nan)
        if values.size < min_seq_len:
            continue

        patient_id = getattr(row, "patient_id", None)
        max_start = values.shape[0] - seq_len
        for start in range(0, max_start + 1, stride):
            end = start + seq_len
            samples.append(torch.from_numpy(values[start:end]).unsqueeze(-1))
            sample_index.append({
                "row_idx": row_idx,
                "patient_id": patient_id[start] if isinstance(patient_id, np.ndarray) else patient_id,
                "start": start,
                "end": end,
            })

    if not samples:
        raise ValueError(
            f"No windows were created for signal={signal}, split={split}. "
            f"Check seq_len={seq_len} and stride={stride}."
        )

    return samples, sample_index, scaler


def resolve_aireadi_glucose_path(root_path, rel_path="AI-READI-processed", split="train"):
    return resolve_aireadi_signal_path(root_path, signal="glucose", rel_path=rel_path, split=split)


def read_aireadi_glucose_frame(root_path, rel_path="AI-READI-processed", split="train"):
    return read_aireadi_signal_frame(root_path, signal="glucose", rel_path=rel_path, split=split)


def fit_aireadi_glucose_scaler(root_path, rel_path="AI-READI-processed", drop_nan=True):
    return fit_aireadi_signal_scaler(root_path, signal="glucose", rel_path=rel_path, drop_nan=drop_nan)


def normalize_aireadi_glucose_values(values, scaler=None, drop_nan=True):
    return normalize_aireadi_signal_values(values, scaler=scaler, drop_nan=drop_nan)


def load_aireadi_glucose_windows(
    root_path,
    rel_path="AI-READI-processed",
    split="train",
    seq_len=24,
    scale=True,
    stride=1,
    min_seq_len=None,
    drop_nan=True,
):
    return load_aireadi_windows(
        root_path=root_path,
        signal="glucose",
        rel_path=rel_path,
        split=split,
        seq_len=seq_len,
        scale=scale,
        stride=stride,
        min_seq_len=min_seq_len,
        drop_nan=drop_nan,
    )


def real_data_loading(data_name, seq_len, root_path):
    """Load and preprocess real-world data.

    Args:
      - data_name: stock or energy
      - seq_len: sequence length

    Returns:
      - data: preprocessed data.
    """
    assert data_name in ['stock', 'energy', 'metro', 'AirQuality']

    if data_name == 'stock':
        ori_data = np.loadtxt(os.path.join(root_path, 'TSG/stocks/stock_data.csv'), delimiter=",", skiprows=1)
    elif data_name == 'energy':
        ori_data = np.loadtxt(os.path.join(root_path, 'TSG/energy/energy_data.csv'), delimiter=",", skiprows=1)
    elif data_name == 'AirQuality':
        ori_data = np.loadtxt(os.path.join(root_path, 'TSG/air_quality/AirQualityUCI.csv'), delimiter=",", skiprows=1, usecols=range(2,15))
    elif data_name == 'metro':
        ori_data = np.loadtxt(os.path.join(root_path, 'TSG/metro_data.csv'), delimiter=",", skiprows=1)

    # Flip the data to make chronological data
    ori_data = ori_data[::-1]
    # Normalize the data
    ori_data = MinMaxScaler(ori_data)

    # Preprocess the data
    temp_data = []
    # Cut data by sequence length
    for i in range(0, len(ori_data) - seq_len):
        _x = ori_data[i:i + seq_len]
        temp_data.append(_x)

    # Mix the data (to make it similar to i.i.d)
    idx = np.random.permutation(len(temp_data))
    data = []
    for i in range(len(temp_data)):
        data.append(temp_data[idx[i]])

    return temp_data


class CombinedShortRangeDataset(Data.Dataset):
    def __init__(self, datasets, max_channels):
        self.datasets = datasets
        self.max_channels = max_channels
        self.data = []
        self.class_labels = []
        self.class_indices = {'sine': 0, 'stock': 1, 'energy': 2, 'mujoco': 3}
        for i, dataset in enumerate(self.datasets):
            current_channels = dataset.size(-1)
            if current_channels < max_channels:
                padding = torch.zeros(*dataset.size()[:-1], max_channels - current_channels, dtype=dataset.dtype, device=dataset.device)
                dataset = torch.cat((dataset, padding), dim=-1)
            self.data.append(dataset)
            self.class_labels.append(torch.full((len(dataset),), i))
        self.data = torch.cat(self.data)
        self.class_labels = torch.cat(self.class_labels)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.class_labels[idx]
    def gen_dataloader(self, args):
        """
        Args:
            args: arguments
        Returns:
            dataloader for the specified dataset along with its corresponding class index
        """
        dataset = self.datasets[self.class_indices[args.dataset]]
        dataset = Data.TensorDataset(dataset)
        if args.test_batch_size is None:
            args.test_batch_size = args.batch_size
        return Data.DataLoader(dataset=dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers)



def gen_multidataset_loader(args):
    datasets = []
    metadata = []
    # sine:
    args.dataset_size = 10000
    ori_data = sine_data_generation(args.dataset_size, args.seq_len, args.input_channels)
    ori_data = torch.Tensor(np.array(ori_data))
    datasets.append(ori_data)
    metadata.append({'channels': args.input_channels})
    # stock:
    ori_data = real_data_loading('stock', args.seq_len)
    ori_data = torch.Tensor(np.array(ori_data))
    datasets.append(ori_data)
    metadata.append({'channels': 6})
    # energy:
    ori_data = real_data_loading('energy', args.seq_len)
    ori_data = torch.Tensor(np.array(ori_data))
    datasets.append(ori_data)
    metadata.append({'channels': 28})
    # mujoco:
    mujoco_dataset = torch.Tensor(MujocoDataset(args.seq_len, 'mujoco', args.path, 0.0).original_sample)
    datasets.append(mujoco_dataset)
    metadata.append({'channels': 14})
    # create a combined dataset
    max_channels = functools.reduce(lambda acc, x: max(acc, x.size(-1)), datasets, -1)
    combined_dataset = CombinedShortRangeDataset(datasets, max_channels)
    train_loader = Data.DataLoader(dataset=combined_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = Data.DataLoader(dataset=combined_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return train_loader, test_loader, combined_dataset, metadata


def normalize(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-7)
    return norm_data


def stft_transform(data, args):
    data = torch.permute(data, (0, 2, 1))  # we permute to match requirements of torchaudio.transforms.Spectrogram
    n_fft = args.n_fft
    hop_length = args.hop_length
    spec = transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, center=True, power=None)
    transformed_data = spec(data)
    real, min_real, max_real = MinMaxScaler(transformed_data.real.numpy(), True)
    real = (real - 0.5) * 2
    imag, min_imag, max_imag = MinMaxScaler(transformed_data.imag.numpy(), True)
    imag = (imag - 0.5) * 2
    # saving min and max values, we will need them for inverse transform
    args.min_real, args.max_real = torch.Tensor(min_real), torch.Tensor(max_real)
    args.min_imag, args.max_imag = torch.Tensor(min_imag), torch.Tensor(max_imag)
    return torch.Tensor(real), torch.tensor(imag)


def load_data(dir):
    tensors = {}
    for filename in os.listdir(dir):
        if filename.endswith('.pt'):
            tensor_name = filename.split('.')[0]
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            tensor_value = torch.load(str(dir / filename), map_location=device)
            tensors[tensor_name] = tensor_value
    return tensors


def save_data(dir, **tensors):
    for tensor_name, tensor_value in tensors.items():
        torch.save(tensor_value, str(dir / tensor_name) + '.pt')


class MujocoDataset(torch.utils.data.Dataset):
    def __init__(self, seq_len, data_name, path, missing_rate=0.0):
        # import pdb;pdb.set_trace()
        import pathlib
        here = pathlib.Path(__file__).resolve().parent.parent
        base_loc = here / 'data'
        loc = pathlib.Path(path)
        if os.path.exists(loc):
            tensors = load_data(loc)
            self.samples = tensors['data']
            self.original_sample = tensors['original_data']
            self.original_sample = self.original_sample.cpu().numpy()
            self.samples = self.samples.cpu().numpy()
            self.size = len(self.samples)
        else:
            if not os.path.exists(base_loc):
                os.mkdir(base_loc)
            if not os.path.exists(loc):
                os.mkdir(loc)
            loc = here / 'data' / data_name
            tensors = load_data(loc)
            time = tensors['train_X'][:, :, :1].cpu().numpy()
            data = tensors['train_X'][:, :, 1:].reshape(-1, 14).cpu().numpy()

            self.original_sample = []
            norm_data = normalize(data)
            norm_data = norm_data.reshape(4620, seq_len, 14)
            idx = torch.randperm(len(norm_data))

            for i in range(len(norm_data)):
                self.original_sample.append(norm_data[idx[i]].copy())
            self.X_mean = np.mean(np.array(self.original_sample), axis=0).reshape(1,
                                                                                  np.array(self.original_sample).shape[
                                                                                      1],
                                                                                  np.array(self.original_sample).shape[
                                                                                      2])
            generator = torch.Generator().manual_seed(56789)
            for i in range(len(norm_data)):
                removed_points = torch.randperm(norm_data[i].shape[0], generator=generator)[
                                 :int(norm_data[i].shape[0] * missing_rate)].sort().values
                norm_data[i][removed_points] = float('nan')
            norm_data = np.concatenate((norm_data, time), axis=2)
            self.samples = []
            for i in range(len(norm_data)):
                self.samples.append(norm_data[idx[i]])

            self.samples = np.array(self.samples)

            norm_data_tensor = torch.Tensor(self.samples[:, :, :-1]).float().cuda()

            time = torch.FloatTensor(list(range(norm_data_tensor.size(1)))).cuda()
            self.last = torch.Tensor(self.samples[:, :, -1][:, -1]).float()
            self.original_sample = torch.tensor(self.original_sample)
            self.samples = torch.tensor(self.samples)
            loc = here / 'data' / (data_name + str(missing_rate))
            save_data(loc, data=self.samples,
                      original_data=self.original_sample
                      )
            self.original_sample = np.array(self.original_sample)
            self.samples = np.array(self.samples)
            self.size = len(self.samples)

    def __getitem__(self, index):
        return self.original_sample[index], self.samples[index]

    def __len__(self):
        return len(self.samples)
