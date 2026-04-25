import numpy as np
from tqdm import tqdm
from utils.utils_dataset import get_dataset
import torch


def save_glucose_ts_as_npy(dataset, save_path):
    """
    Save raw time-series windows (not images) to .npy
    Output shape: (N, T, 1)
    """
    all_ts = []

    for idx in tqdm(range(len(dataset)), desc=f"Saving {save_path}"):
        row_idx, start = dataset.windows[idx]
        full_ts = dataset.df["glucose"].iloc[row_idx]
        window = full_ts[start : start + dataset.seq_len]

        ts = torch.tensor(window, dtype=torch.float32)

        # if dataset.normalize:
        #     ts = normalize_ts_to_unit_range(ts)

        ts = ts.unsqueeze(-1)  # (T,1)
        all_ts.append(ts.numpy())

    all_ts = np.stack(all_ts, axis=0)  # (N, T, 1)

    np.save(save_path, all_ts)
    print(f"Saved {all_ts.shape} to {save_path}")



if __name__ == "__main__":
    config = {
        "ts_seq_len": 128,
        "ts_delay": 8,
        "ts_embedding": 16,
    }

    train_dataset, test_dataset = get_dataset(
        "glucose",
        config=config,
        root="./AI-READI"
    )

    save_glucose_ts_as_npy(train_dataset, "./data/glucose/train_ts.npy")
    save_glucose_ts_as_npy(test_dataset, "./data/glucose/test_ts.npy")