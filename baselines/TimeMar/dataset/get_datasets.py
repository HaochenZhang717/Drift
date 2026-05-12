import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
import torch
import importlib
from torch.utils.data import ConcatDataset

def normalize_to_neg_one_to_one(x):
    return x * 2 - 1

def instantiate_from_config(config):
    if config is None:
        return None
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")
    module, cls = config["target"].rsplit(".", 1)
    cls = getattr(importlib.import_module(module, package=None), cls)
    return cls(**config.get("params", dict()))


def build_dataset(config):
    dataloader_config = config.dataloader
    batch_size = config.batch_size
    dataloader_config.params.output_dir = config.output_dir

    # 新增：如果是你自己的 presplit npy dataset 或 data_provider adapter，
    # 就分别实例化 train / valid，避免沿用作者原本的“最后一个 batch 当验证集”逻辑。
    if dataloader_config.target in {
        "Utils.npy_datasets.PreSplitNPYDataset",
        "Utils.data_provider_datasets.DataProviderWindowDataset",
    }:
        train_cfg = {
            "target": dataloader_config.target,
            "params": dict(dataloader_config.params)
        }
        val_cfg = {
            "target": dataloader_config.target,
            "params": dict(dataloader_config.params)
        }

        train_cfg["params"]["split"] = "train"
        val_cfg["params"]["split"] = "valid"

        train_dataset = instantiate_from_config(train_cfg)
        val_dataset = instantiate_from_config(val_cfg)
        return train_dataset, val_dataset

    # 原始逻辑保留
    dataset = instantiate_from_config(dataloader_config)

    total_size = len(dataset)
    train_size = total_size - batch_size
    val_size = batch_size
    val_indices = list(range(train_size, train_size + val_size))
    val_dataset = Subset(dataset, val_indices)

    return dataset, val_dataset


def build_dataloader(config, args=None):
    dataloader_config = config.dataloader
    batch_size = config.batch_size

    try:
        dim = config.dataloader.params.dim
    except Exception:
        print("no dim in config")
        dim = 0

    if dim == 5:
        if config.dataloader.params.window == 24:
            file_path = f'../output/samples/Sines_ground_truth_24_train.npy'
            train_dataset = normalize_to_neg_one_to_one(np.load(file_path).astype(np.float32))
            total_size = len(train_dataset)
            train_size = total_size - batch_size
            val_size = batch_size
            val_indices = list(range(train_size, train_size + val_size))
            val_dataset = Subset(train_dataset, val_indices)
            print("data load from:", file_path)

    elif dim == 14:
        file_path = f'../output/samples/Mujoco_norm_truth_24_train.npy'
        train_dataset = normalize_to_neg_one_to_one(np.load(file_path).astype(np.float32))
        val_dataset = normalize_to_neg_one_to_one(np.load(file_path).astype(np.float32))
        print("data load from:", file_path)

    else:
        print("load data from config")
        train_dataset, val_dataset = build_dataset(config)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=False,
        persistent_workers=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )

    return train_loader, val_loader


def build_dataloader_var(config, data, args=None, window=24):
    if data == "Sines":
        file_path = f'../output/samples/{data}_ground_truth_{window}_train.npy'
    elif data == "Mujoco":
        file_path = f'../output/samples/{data}_norm_truth_{window}_train.npy'
    else:
        window = config['dataloader']['params']['window']
        file_path = f'./output/samples/{data}_norm_truth_{window}_train.npy'

    train_dataset = normalize_to_neg_one_to_one(np.load(file_path).astype(np.float32))
    ori_dataset = normalize_to_neg_one_to_one(np.load(file_path).astype(np.float32))

    repeat_times = 10
    train_dataset = [train_dataset for _ in range(repeat_times)]
    train_dataset = ConcatDataset(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        config.batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=False
    )

    val_loader = torch.utils.data.DataLoader(
        ori_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, val_loader

def build_dataloader_var(config, data, args=None, window=24):
    if data=="Sines":
        file_path=f'../output/samples/{data}_ground_truth_{window}_train.npy'
    elif data=="Mujoco":
        file_path=f'../output/samples/{data}_norm_truth_{window}_train.npy'
    else:
        window = config['dataloader']['params']['window']
        file_path=f'./output/samples/{data}_norm_truth_{window}_train.npy'

    train_dataset = normalize_to_neg_one_to_one(np.load(file_path).astype(np.float32))
    ori_dataset = normalize_to_neg_one_to_one(np.load(file_path).astype(np.float32))
    repeat_times=10
    from torch.utils.data import ConcatDataset
    train_dataset = [train_dataset for _ in range(repeat_times)]
    train_dataset = ConcatDataset(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                              config.batch_size,
                                              shuffle=True,
                                              num_workers=8,
                                              drop_last = False)
    val_loader = torch.utils.data.DataLoader(
        ori_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False
    )
    return train_loader, val_loader
