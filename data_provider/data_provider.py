from data_provider.datasets import ETTh, ETTm, Custom, UEA, GLUONTS, Sine, Stock, Energy, Mujoco, PSM, MSL, SMAP, SMD, SWAT, AirQuality, AIREADI, AIREADICalorie, AIREADIGlucose, VerbalTS, NpyTimeSeries
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from .multi_dataloader_iter import MultiDataloaderIter
import functools
import torch
import os
import numpy as np


class VerbalTSTextDataset(Dataset):
    """Dataset wrapper for (time_series_tensor, raw_text_string)."""

    def __init__(self, primary_tensor, texts):
        if not torch.is_tensor(primary_tensor):
            raise TypeError(f"Expected primary_tensor to be torch.Tensor, got {type(primary_tensor)}")
        if len(primary_tensor) != len(texts):
            raise ValueError(
                f"Primary tensor count ({len(primary_tensor)}) does not match text count ({len(texts)})."
            )
        self.primary_tensor = primary_tensor
        self.texts = list(texts)

    def __len__(self):
        return int(self.primary_tensor.shape[0])

    def __getitem__(self, idx):
        return self.primary_tensor[idx], self.texts[idx]

    def subset(self, n):
        return VerbalTSTextDataset(self.primary_tensor[:n], self.texts[:n])


def _get_primary_tensor(dataset):
    if isinstance(dataset, VerbalTSTextDataset):
        return dataset.primary_tensor
    return dataset.tensors[0] if isinstance(dataset, TensorDataset) else dataset


def _replace_primary_tensor(dataset, primary_tensor):
    if isinstance(dataset, VerbalTSTextDataset):
        return VerbalTSTextDataset(primary_tensor, dataset.texts)
    if isinstance(dataset, TensorDataset):
        return TensorDataset(primary_tensor, *dataset.tensors[1:])
    return primary_tensor


def _load_context_embeddings(path):
    embeds = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(embeds, dict):
        if "embeddings" in embeds:
            embeds = embeds["embeddings"]
        else:
            raise ValueError(f"Expected embedding dict at {path} to contain 'embeddings'.")
    if not torch.is_tensor(embeds):
        raise TypeError(f"Expected torch.Tensor embeddings at {path}, got {type(embeds)}")
    if embeds.ndim not in (2, 3):
        raise ValueError(f"Expected embeddings with shape (N, D) or (N, L, D), got {tuple(embeds.shape)} at {path}")
    return embeds.to(torch.float32)


def _should_load_verbalts_context(args, config):
    handler = str(getattr(args, "handler", ""))
    explicit = bool(getattr(args, "load_long_clip_context", False))
    return config.get("data") == "verbal_ts" and (
        explicit
        or "ImagenFewCrossAttention" in handler
        or "ImagenTimeVectorCond" in handler
    )


def _should_load_verbalts_tokens(args, config):
    handler = str(getattr(args, "handler", ""))
    return config.get("data") == "verbal_ts" and "models.DiT" in handler


def _attach_verbalts_context(args, config, split, dataset):
    if not _should_load_verbalts_context(args, config):
        return dataset
    args.use_precomputed_context = True

    dataset_dir = os.path.join(config["datasets_dir"], config["rel_path"])
    suffix = getattr(args, "verbalts_context_suffix", None) or config.get("context_suffix")
    candidate_paths = []
    if suffix:
        candidate_paths.append(os.path.join(dataset_dir, f"{split}_embeds_{suffix}.pt"))
    candidate_paths.extend([
        os.path.join(dataset_dir, f"{split}_embeds_long_clip.pt"),
        os.path.join(dataset_dir, f"{split}_embeds_qwen3_4b.pt"),
    ])
    embeds_path = next((path for path in candidate_paths if os.path.exists(path)), None)
    if embeds_path is None:
        raise FileNotFoundError(
            f"Expected context embeddings for split '{split}' in {dataset_dir}. "
            f"Tried: {candidate_paths}"
        )

    embeds = _load_context_embeddings(embeds_path)
    index_order = getattr(args, "_verbalts_index_order", {}).get((config["name"], split))
    if index_order is not None:
        embeds = embeds[torch.as_tensor(index_order, dtype=torch.long)]
    primary_tensor = _get_primary_tensor(dataset)
    if len(primary_tensor) != len(embeds):
        raise ValueError(
            f"Split '{split}' time-series count ({len(primary_tensor)}) does not match "
            f"embedding count ({len(embeds)}) for dataset {config['name']}."
        )

    if getattr(args, "context_dim", None) is None:
        args.context_dim = int(embeds.shape[-1])

    return TensorDataset(primary_tensor, embeds)


def _load_caption_texts(dataset_dir, split, npy_name):
    caps_path = os.path.join(dataset_dir, f"{split}_{npy_name}.npy")
    if not os.path.exists(caps_path):
        fallback_path = os.path.join(dataset_dir, f"{split}_text_caps.npy")
        if os.path.exists(fallback_path):
            caps_path = fallback_path
        else:
            raise FileNotFoundError(f"Caption file not found: {caps_path}")
    caps = np.load(caps_path, allow_pickle=True)
    texts = []
    for cap in caps:
        if isinstance(cap, np.ndarray) and cap.ndim > 0:
            texts.append(str(cap[0]))
        elif isinstance(cap, (list, tuple)):
            texts.append(str(cap[0]))
        else:
            texts.append(str(cap))
    return texts


def _attach_verbalts_tokens(args, config, split, dataset):
    if not _should_load_verbalts_tokens(args, config):
        return dataset

    dataset_dir = os.path.join(config["datasets_dir"], config["rel_path"])
    npy_name = getattr(args, "verbalts_text_npy_name", "my_text_caps")

    texts = _load_caption_texts(dataset_dir, split, npy_name)

    index_order = getattr(args, "_verbalts_index_order", {}).get((config["name"], split))
    if index_order is not None:
        texts = [texts[int(i)] for i in index_order]

    primary_tensor = _get_primary_tensor(dataset)
    if len(primary_tensor) != len(texts):
        raise ValueError(
            f"Split '{split}' time-series count ({len(primary_tensor)}) does not match "
            f"caption count ({len(texts)}) for dataset {config['name']}."
        )

    return VerbalTSTextDataset(primary_tensor, texts)

data_dict = {
    'ETTh1': ETTh,
    'ETTh2': ETTh,
    'ETTm1': ETTm,
    'ETTm2': ETTm,
    'custom': Custom,
    'PSM': PSM,
    'MSL': MSL,
    'SMAP': SMAP,
    'SMD': SMD,
    'SWAT': SWAT,
    'UEA': UEA,
    'gluonts': GLUONTS,
    'sine': Sine,
    'stock': Stock,
    'energy': Energy,
    'mujoco': Mujoco,
    'AirQuality': AirQuality,
    'AIREADI': AIREADI,
    'AIREADICalorie': AIREADICalorie,
    'AIREADIGlucose': AIREADIGlucose,
    'verbal_ts': VerbalTS,
    'npy_ts': NpyTimeSeries,
}

def random_permute(trainset, testset):
    perm_train = torch.randperm(len(trainset), generator=torch.Generator().manual_seed(0)).numpy()
    perm_test = torch.randperm(len(testset), generator=torch.Generator().manual_seed(0)).numpy()

    return Subset(trainset, perm_train), Subset(testset, perm_test)

def random_subset(dataset, subset_p=None, subset_n=None):
    num_samples = len(dataset)
    if subset_n is not None:
        num_samples = subset_n
    if subset_p is not None:
        num_samples = int(num_samples * subset_p)
    indices = torch.arange(0, num_samples, dtype=int) % len(dataset)
    return Subset(dataset, indices)


def _has_effective_subset(subset_p=None, subset_n=None, dataset_len=None):
    if subset_n is not None:
        return dataset_len is None or int(subset_n) < int(dataset_len)
    if subset_p is not None:
        return float(subset_p) < 1.0
    return False


def _resolve_subset_indices(dataset):
    if isinstance(dataset, Subset):
        indices = np.asarray(dataset.indices)
        parent_indices = _resolve_subset_indices(dataset.dataset)
        if parent_indices is None:
            return indices
        return np.asarray(parent_indices)[indices]
    return None

def data_provider(args):

    trainsets = {}
    testsets  = {}
    trainloaders = {}
    testloaders  = {}
    samplers = {}
    metadatas  = {}
    args._verbalts_index_order = {}

    datasets = [dataset for dataset in args.datasets if dataset['name'] in args.train_on_datasets]

    for config in datasets:
        metadata = {}
        config['seq_len'] = args.seq_len
        config['datasets_dir'] = args.datasets_dir
        trainset, testset = get_train(config), get_test(config)
        subset_p = getattr(args,'subset_p', None)
        subset_n = getattr(args,'subset_n', None)

        # Randomly permute train/testsets
        trainset, testset = random_permute(trainset, testset)
        if _has_effective_subset(subset_p, subset_n, len(trainset)) and (not 'subset_n' in config.keys()):
            trainset, testset = random_subset(trainset, subset_p, subset_n), trainset
        args._verbalts_index_order[(config["name"], "train")] = _resolve_subset_indices(trainset)
        args._verbalts_index_order[(config["name"], "test")] = _resolve_subset_indices(testset)
        trainset, testset = dataset_to_tensor(trainset, args), dataset_to_tensor(testset, args)
        if _should_load_verbalts_tokens(args, config):
            trainset = _attach_verbalts_tokens(args, config, "train", trainset)
            testset = _attach_verbalts_tokens(args, config, "test", testset)
        else:
            trainset = _attach_verbalts_context(args, config, "train", trainset)
            testset = _attach_verbalts_context(args, config, "test", testset)

        caption_embeddings_path = getattr(args, "caption_embeddings_path", None)
        if caption_embeddings_path and config["name"] in args.train_on_datasets:
            caption_embeddings = torch.load(caption_embeddings_path, map_location="cpu", weights_only=False)
            if not torch.is_tensor(caption_embeddings):
                raise TypeError(
                    f"caption_embeddings_path must point to a tensor, got {type(caption_embeddings)}"
                )
            if caption_embeddings.ndim != 3:
                raise ValueError(
                    f"Expected caption embeddings with shape (N, C, D), got {tuple(caption_embeddings.shape)}"
                )
            train_primary = _get_primary_tensor(trainset)
            if caption_embeddings.shape[0] != train_primary.shape[0]:
                raise ValueError(
                    f"Caption embedding count ({caption_embeddings.shape[0]}) does not match "
                    f"train set size ({train_primary.shape[0]}) for dataset {config['name']}."
                )
            trainset = TensorDataset(train_primary, caption_embeddings.to(torch.float32))

        if args.finetune:
            train_seq_tensor = _get_primary_tensor(trainset)
            assert train_seq_tensor.size(1) == args.seq_len, f"{config['name']} Does not output proper sequence length"
        else:
            train_seq_tensor = _get_primary_tensor(trainset)
            if train_seq_tensor.size(1) != args.seq_len:
                train_seq_tensor = torch.nn.functional.pad(train_seq_tensor, (0, 0, 0, args.seq_len - train_seq_tensor.size(1)))
                trainset = _replace_primary_tensor(trainset, train_seq_tensor)
        print(f"{config['name']} Contains: {len(trainset)} train datapoints; {len(testset)} test datapoints;")

        metadata['name'] = config['name']
        metadata['channels'] = _get_primary_tensor(trainset).size(-1)
        if isinstance(trainset, TensorDataset) and len(trainset.tensors) > 1:
            metadata['context_dim'] = int(trainset.tensors[1].shape[-1])

        if args.input_channels is not None:
            train_primary = _get_primary_tensor(trainset)
            if train_primary.size(2) < args.input_channels:
                train_primary = torch.nn.functional.pad(
                    train_primary,
                    (0, args.input_channels - train_primary.size(2), 0, 0),
                )
                trainset = _replace_primary_tensor(trainset, train_primary)
            test_primary = _get_primary_tensor(testset)
            if test_primary.size(2) < args.input_channels:
                test_primary = torch.nn.functional.pad(
                    test_primary,
                    (0, args.input_channels - test_primary.size(2), 0, 0),
                )
                testset = _replace_primary_tensor(testset, test_primary)

        if config['name'] in args.train_on_datasets:
            trainsets[config['name']] = trainset
            if getattr(args, 'ddp', False):
                samplers[config['name']] = DistributedSampler(trainsets[config['name']])
            else:
                samplers[config['name']] = None
            trainloaders[config['name']] = (DataLoader(dataset=trainsets[config['name']], batch_size=args.batch_size, num_workers=args.num_workers, sampler=samplers[config['name']]), metadata)
        testsets[config['name']] = testset
        metadatas[config['name']] = metadata

    args.input_channels = functools.reduce(lambda acc, metadata: max(acc, metadata['channels']), metadatas.values(), args.input_channels if args.input_channels is not None else 1)
    dataset_loader = MultiDataloaderIter(trainloaders, testsets)
    return dataset_loader, samplers, trainsets, metadatas


def get_train(config):
    Data = data_dict[config['data']]
    if Data is None:
        raise ImportError(f"Dataset backend '{config['data']}' is unavailable because its optional dependency is not installed.")
    config['flag'] = 'train'
    if 'subset_n' in config.keys():
        return Subset(Data(**config), torch.arange(config['subset_n']))
    return Data(**config)

def get_test(config):
    Data = data_dict[config['data']]
    if Data is None:
        raise ImportError(f"Dataset backend '{config['data']}' is unavailable because its optional dependency is not installed.")
    config['flag'] = 'test'
    return Data(**config)

def dataset_to_tensor(dataset, args):
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    tensor = []
    for i, item in enumerate(loader):
        if type(item) is list:
            adjusted_item = item[0][:, :args.seq_len]
            tensor.append(adjusted_item)
        else:
            adjusted_item = item[:, :args.seq_len]
            tensor.append(adjusted_item)
    dataset = torch.concat(tensor, dim=0)
    return dataset
