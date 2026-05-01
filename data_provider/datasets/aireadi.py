import torch
from torch.utils.data import Dataset

from utils.utils_data import load_aireadi_windows


def AIREADIGlucose(**config):
    config = dict(config)
    config["signal"] = "glucose"
    return AIREADI(**config)


def AIREADICalorie(**config):
    config = dict(config)
    config["signal"] = "calorie"
    return AIREADI(**config)


def AIREADI(**config):
    signal = config.get("signal", config.get("target_col", "glucose"))
    signal = signal.lower()
    if signal not in {"glucose", "calorie"}:
        raise ValueError(f"Unsupported AI-READI signal {signal}. Expected 'glucose' or 'calorie'.")
    return AIREADIGlucoseDataset(
        root_path=config["datasets_dir"],
        rel_path=config.get("rel_path", "AI-READI-processed"),
        flag=config["flag"],
        seq_len=config["seq_len"],
        signal=signal,
        scale=config.get("scale", True),
        stride=config.get("window_stride", 1),
        min_seq_len=config.get("min_seq_len", None),
        drop_nan=config.get("drop_nan", True),
        return_metadata=config.get("return_metadata", False),
    )


class AIREADIGlucoseDataset(Dataset):
    def __init__(
        self,
        root_path,
        rel_path="AI-READI-processed",
        flag="train",
        seq_len=24,
        signal="glucose",
        scale=True,
        stride=1,
        min_seq_len=None,
        drop_nan=True,
        return_metadata=False,
    ):
        super().__init__()
        self.root_path = root_path
        self.rel_path = rel_path
        self.flag = flag
        self.seq_len = int(seq_len)
        self.signal = signal.lower()
        self.scale = scale
        self.stride = int(stride)
        self.min_seq_len = int(min_seq_len) if min_seq_len is not None else self.seq_len
        self.drop_nan = drop_nan
        self.return_metadata = return_metadata

        self.samples = []
        self.sample_index = []
        self.scaler = None
        self._build_dataset()

    def _build_dataset(self):
        self.samples, self.sample_index, self.scaler = load_aireadi_windows(
            root_path=self.root_path,
            signal=self.signal,
            rel_path=self.rel_path,
            split=self.flag,
            seq_len=self.seq_len,
            scale=self.scale,
            stride=self.stride,
            min_seq_len=self.min_seq_len,
            drop_nan=self.drop_nan,
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        if self.return_metadata:
            return sample, self.sample_index[index]
        return sample
