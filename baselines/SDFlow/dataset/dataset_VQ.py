import sys
import importlib.util
from pathlib import Path

import torch
from torch.utils import data

import pandas as pd
from scipy import io
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DATA_PROVIDER_DATASETS = {
    'GlucoseSliding',
    'glucose',
    'ErcotData',
    'ercot',
    'HouseholdData',
    'household',
}


DATA_PROVIDER_ALIASES = {
    'glucose': 'GlucoseSliding',
    'ercot': 'ErcotData',
    'household': 'HouseholdData',
}


def _load_repo_module(module_name, relative_path):
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_local_timeseries = _load_repo_module(
    'sdflow_local_timeseries',
    'data_provider/datasets/local_timeseries.py',
)
_glucose_sliding = _load_repo_module(
    'sdflow_glucose_sliding',
    'data_provider/datasets/glucose_sliding.py',
)


DATA_PROVIDER_FACTORIES = {
    'GlucoseSliding': _glucose_sliding.GlucoseSliding,
    'ErcotData': _local_timeseries.ErcotData,
    'HouseholdData': _local_timeseries.HouseholdData,
}


class TSDataset(data.Dataset):
    def __init__(self, dataset_name, window_size = 192, unit_length = 4, dataset_type='train'):
        self.window_size = window_size
        self.unit_length = unit_length
        self.dataset_name = dataset_name

        if dataset_name == 'stock':
            csv_path = './dataset/stock_data.csv'
        elif dataset_name == 'energy':
            csv_path = './dataset/energy_data.csv'
        elif dataset_name == 'etth':
            csv_path = './dataset/ETTh.csv'
        elif dataset_name == 'fmri':
            csv_path = './dataset/sim4.mat'

        if dataset_name in ['stock','energy']:
            data = pd.read_csv(csv_path).values.astype(float)
        elif dataset_name == 'fmri':
            data = io.loadmat(csv_path)['ts']
        else:
            data = pd.read_csv(csv_path).values[:,1:].astype(float)

        num_train = int(len(data) * 1.0)# less than 1 for prediction tasks
        num_test = int(len(data) * 0.0)
        num_vali = len(data) - num_train - num_test

        border1s = [0, num_train, len(data) - num_test]
        border2s = [num_train, num_train + num_vali, len(data)]
        train_data = data[border1s[0]:border2s[0]]

        self.mean = train_data.mean(0)
        self.std = train_data.std(0)
        self.min = train_data.min(0)
        self.max = train_data.max(0)
        data = (data - self.min) / (self.max - self.min)


        if dataset_type == 'train':
            self.data = data[border1s[0]:border2s[0]]
        elif dataset_type == 'val':
            self.data = data[border1s[1]:border2s[1]]
        elif dataset_type == 'test':
            self.data = data[border1s[2]:border2s[2]]
        print("{} data: Length is {}, Number of nodes is {}".format(dataset_type, self.data.shape[0], self.data.shape[1]))

    def inv_transform(self, data):
        return data * self.std + self.mean

    def norm_transform(self, data):
        return (data - self.mean) / self.std
    def inv_minmax_transform(self, data):
        return data * (self.max - self.min) + self.min



    def __len__(self):
        return (len(self.data) - self.window_size) + 1

    def __getitem__(self, item):
        data = self.data[item:item+self.window_size]

        return data

def DATALoader(dataset_name,
               batch_size,
               num_workers = 8,
               window_size = 192,
               unit_length = 4,
               dataset_type='train',
               datasets_dir=None,
               rel_path=None,
               rel_path_train=None,
               rel_path_valid=None,
               rel_path_test=None,
               stride=1,
               window_stride=None,
               ts_stride=None,
               value_cols=None,
               drop_cols=None,
               scale=True,
               normalize=True,
               column='glucose'):

    if dataset_name == 'sine':
        data_dir = "./dataset/sine_ground_truth_24_train.npy"
        trainSet = np.load(data_dir)
    elif dataset_name == 'mujoco':
        data_dir = "./dataset/mujoco_norm_truth_24_train.npy"
        trainSet = np.load(data_dir)
    elif dataset_name in DATA_PROVIDER_DATASETS:
        trainSet = build_data_provider_dataset(
            dataset_name=dataset_name,
            dataset_type=dataset_type,
            datasets_dir=datasets_dir,
            window_size=window_size,
            rel_path=rel_path,
            rel_path_train=rel_path_train,
            rel_path_valid=rel_path_valid,
            rel_path_test=rel_path_test,
            stride=stride,
            window_stride=window_stride,
            ts_stride=ts_stride,
            value_cols=value_cols,
            drop_cols=drop_cols,
            scale=scale,
            normalize=normalize,
            column=column,
        )
        num_workers = 0
    else:
        trainSet = TSDataset(dataset_name, window_size=window_size, unit_length=unit_length, dataset_type=dataset_type)


    train_loader = torch.utils.data.DataLoader(trainSet,
                                              batch_size,
                                              shuffle=True,
                                              num_workers=num_workers,
                                              drop_last = False)
    
    return train_loader


def build_data_provider_dataset(
    dataset_name,
    dataset_type,
    datasets_dir,
    window_size,
    rel_path=None,
    rel_path_train=None,
    rel_path_valid=None,
    rel_path_test=None,
    stride=1,
    window_stride=None,
    ts_stride=None,
    value_cols=None,
    drop_cols=None,
    scale=True,
    normalize=True,
    column='glucose',
):
    if datasets_dir is None:
        raise ValueError(f"--datasets-dir is required for SDFlow dataname '{dataset_name}'.")

    canonical_name = DATA_PROVIDER_ALIASES.get(dataset_name, dataset_name)
    data_factory = DATA_PROVIDER_FACTORIES[canonical_name]
    flag = 'val' if dataset_type in {'val', 'valid'} else dataset_type

    split_rel_path = rel_path
    if flag == 'train' and rel_path_train:
        split_rel_path = rel_path_train
    elif flag in {'val', 'valid'} and rel_path_valid:
        split_rel_path = rel_path_valid
    elif flag == 'test' and rel_path_test:
        split_rel_path = rel_path_test

    config = {
        'name': canonical_name,
        'data': canonical_name,
        'datasets_dir': datasets_dir,
        'rel_path': split_rel_path,
        'flag': flag,
        'seq_len': window_size,
        'stride': stride,
        'window_stride': window_stride if window_stride is not None else stride,
        'ts_stride': ts_stride if ts_stride is not None else stride,
        'value_cols': value_cols,
        'drop_cols': drop_cols,
        'scale': scale,
        'normalize': normalize,
        'column': column,
    }
    return data_factory(**config)


def infer_num_channels(loader):
    sample = next(iter(loader))
    if isinstance(sample, (list, tuple)):
        sample = sample[0]
    if sample.ndim != 3:
        raise ValueError(f"Expected SDFlow dataset batch with shape (B, T, C), got {tuple(sample.shape)}.")
    return int(sample.shape[-1])


def cycle(iterable):
    while True:
        for x in iterable:
            yield x
