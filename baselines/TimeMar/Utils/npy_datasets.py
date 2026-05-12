import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler


def normalize_to_neg_one_to_one(x):
    return x * 2.0 - 1.0


class PreSplitNPYDataset(Dataset):
    def __init__(
        self,
        name,
        data_root,
        window,
        split="train",            # train / valid / test
        save2npy=True,
        neg_one_to_one=True,
        seed=123,
        output_dir="../output",
        **kwargs
    ):
        super().__init__()
        assert split in ["train", "valid", "test"]

        self.name = name
        self.data_root = data_root
        self.window = window
        self.split = split
        self.save2npy = save2npy
        self.neg_one_to_one = neg_one_to_one
        self.output_dir = output_dir

        train = np.load(os.path.join(data_root, "train_ts.npy")).astype(np.float32)
        valid = np.load(os.path.join(data_root, "valid_ts.npy")).astype(np.float32)
        test  = np.load(os.path.join(data_root, "test_ts.npy")).astype(np.float32)

        assert train.ndim == 3 and valid.ndim == 3 and test.ndim == 3
        assert train.shape[1] == window, f"train window mismatch: {train.shape[1]} != {window}"
        assert valid.shape[1] == window, f"valid window mismatch: {valid.shape[1]} != {window}"
        assert test.shape[1] == window, f"test window mismatch: {test.shape[1]} != {window}"

        self.feature_dim = train.shape[-1]

        # 用 train split 拟合 MinMaxScaler
        self.scaler = MinMaxScaler()
        self.scaler.fit(train.reshape(-1, self.feature_dim))

        def to_01(x):
            return self.scaler.transform(x.reshape(-1, self.feature_dim)).reshape(x.shape)

        def to_model_space(x):
            x01 = to_01(x)
            if self.neg_one_to_one:
                x01 = normalize_to_neg_one_to_one(x01)
            return x01.astype(np.float32)

        self.train_raw = train
        self.valid_raw = valid
        self.test_raw = test

        self.train = to_model_space(train)
        self.valid = to_model_space(valid)
        self.test = to_model_space(test)

        # 为 train_ar.py / FID 保存作者原本期望的文件
        if self.save2npy:
            save_dir = os.path.join(output_dir, "samples")
            os.makedirs(save_dir, exist_ok=True)

            np.save(os.path.join(save_dir, f"{name}_ground_truth_{window}_train.npy"), train)
            np.save(os.path.join(save_dir, f"{name}_ground_truth_{window}_valid.npy"), valid)
            np.save(os.path.join(save_dir, f"{name}_ground_truth_{window}_test.npy"), test)

            # 注意：这里存的是 [0,1] 范围，和作者原始 TimeMAR 代码保持一致
            np.save(os.path.join(save_dir, f"{name}_norm_truth_{window}_train.npy"), to_01(train).astype(np.float32))
            np.save(os.path.join(save_dir, f"{name}_norm_truth_{window}_valid.npy"), to_01(valid).astype(np.float32))
            np.save(os.path.join(save_dir, f"{name}_norm_truth_{window}_test.npy"),  to_01(test).astype(np.float32))

        split_map = {
            "train": self.train,
            "valid": self.valid,
            "test": self.test,
        }
        self.samples = split_map[split]

    def __getitem__(self, idx):
        return torch.from_numpy(self.samples[idx]).float()

    def __len__(self):
        return len(self.samples)