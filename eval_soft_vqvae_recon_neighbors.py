import argparse
import math
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data_provider.data_provider import get_test, get_train
from img_transformations import DelayEmbedder
from models.soft_vqvae import SoftVQVAE


def parse_int_tuple(value):
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(int(v) for v in value)
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        return tuple(int(p) for p in parts)
    raise ValueError(f"Cannot parse tuple from value: {value}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _extract_series(item):
    if isinstance(item, dict):
        x = item["target"] if "target" in item else next(iter(item.values()))
    elif isinstance(item, (list, tuple)):
        x = item[0]
    else:
        x = item

    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)

    x = x.astype(np.float32)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(f"Expected sample with shape (T, C), got {x.shape}")
    return x


def _fit_minmax_stats(base_dataset, one_channel=False):
    data_min = None
    data_max = None
    for idx in range(len(base_dataset)):
        series = _extract_series(base_dataset[idx])
        tensor = torch.from_numpy(series)
        if one_channel:
            tensor = tensor[:, :1]
        sample_min = tensor.amin(dim=0)
        sample_max = tensor.amax(dim=0)
        data_min = sample_min if data_min is None else torch.minimum(data_min, sample_min)
        data_max = sample_max if data_max is None else torch.maximum(data_max, sample_max)
    if data_min is None or data_max is None:
        raise ValueError("Cannot fit min-max statistics on an empty dataset.")
    return data_min.to(torch.float32), data_max.to(torch.float32)


class BenchmarkTensorDataset(Dataset):
    def __init__(self, base_dataset, data_min, data_max, one_channel=False):
        self.base_dataset = base_dataset
        self.data_min = data_min
        self.data_max = data_max
        self.denom = torch.clamp(self.data_max - self.data_min, min=1e-6)
        self.one_channel = one_channel

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        series = _extract_series(self.base_dataset[idx])
        if self.one_channel:
            series = series[:, :1]
        tensor = torch.from_numpy(series).to(torch.float32)
        tensor = torch.clamp((tensor - self.data_min) / self.denom, 0.0, 1.0)
        tensor = tensor * 2.0 - 1.0
        tensor = tensor.permute(1, 0).contiguous()  # (C, T)
        return tensor


def make_dataset_config(args_dict, flag):
    stride = args_dict.get("window_stride")
    if stride is None:
        stride = args_dict.get("ts_stride")
    if stride is None:
        stride = args_dict.get("stride")

    config = {
        "name": args_dict["dataset_name"],
        "data": args_dict["data"],
        "datasets_dir": args_dict["datasets_dir"],
        "rel_path": args_dict["rel_path"],
        "path": args_dict["rel_path"],
        "seq_len": int(args_dict["ts_seq_len"]),
        "flag": flag,
    }
    if args_dict.get("rel_path_train") is not None:
        config["rel_path_train"] = args_dict["rel_path_train"]
    if args_dict.get("rel_path_valid") is not None:
        config["rel_path_valid"] = args_dict["rel_path_valid"]
    if stride is not None:
        config["window_stride"] = stride
        config["ts_stride"] = stride
        config["stride"] = stride
    return config


def load_datasets(args_dict):
    train_base = get_train(make_dataset_config(args_dict, args_dict.get("train_split", "train")))
    val_base = get_test(make_dataset_config(args_dict, args_dict.get("val_split", "test")))

    one_channel = bool(args_dict.get("one_channel", False))
    data_min, data_max = _fit_minmax_stats(train_base, one_channel=one_channel)

    train_dataset = BenchmarkTensorDataset(train_base, data_min, data_max, one_channel=one_channel)
    val_dataset = BenchmarkTensorDataset(val_base, data_min, data_max, one_channel=one_channel)
    return train_dataset, val_dataset


def build_model(args_dict, resolution, channels, device):
    model = SoftVQVAE(
        input_dim=channels,
        output_dim=channels,
        resolution=resolution,
        hidden_size=int(args_dict["hidden_size"]),
        num_res_blocks=int(args_dict["num_layers"]),
        code_dim=int(args_dict["code_dim"]),
        num_codes=int(args_dict["num_codes"]),
        ch_mult=parse_int_tuple(args_dict["ch_mult"]),
        attn_resolutions=parse_int_tuple(args_dict.get("attn_resolutions", ())),
        dropout=float(args_dict.get("dropout", 0.0)),
        temperature=float(args_dict.get("temperature", 0.07)),
        learnable_temperature=bool(args_dict.get("learnable_temperature", False)),
        l2_norm=bool(args_dict.get("l2_norm", False)),
        attn_type=args_dict.get("attn_type", "vanilla"),
        tanh_out=bool(args_dict.get("tanh_out", False)),
    ).to(device)
    return model


def load_checkpoint_model(ckpt_path: Path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "args" in ckpt:
        args_dict = dict(ckpt["args"])
        state_dict = ckpt.get("model_state_dict")
        if state_dict is None:
            raise ValueError(f"Checkpoint {ckpt_path} has args but no model_state_dict.")
    elif isinstance(ckpt, dict):
        raise ValueError(
            "Unsupported checkpoint format. Expected a dict with keys 'args' and 'model_state_dict'."
        )
    else:
        raise ValueError("Unsupported checkpoint object.")
    return args_dict, state_dict


def to_images(series_bct, embedder):
    series_btc = series_bct.permute(0, 2, 1).contiguous()
    images, pad_mask = embedder.ts_to_img(series_btc, pad=True, mask=0, return_pad_mask=True)
    return images, pad_mask


@torch.no_grad()
def run_eval(model, val_loader, embedder, device, max_samples):
    model.eval()
    all_orig = []
    all_recon = []
    all_feat = []
    all_mse = []

    collected = 0
    for x in val_loader:
        x = x.to(device, non_blocking=True)  # (B, C, T)
        images, _ = to_images(x, embedder)
        out = model(images)
        recon_images = out["recon"]
        recon_series = embedder.img_to_ts(recon_images)  # (B, T, C)
        orig_series = x.permute(0, 2, 1).contiguous()  # (B, T, C)

        mse = (recon_series - orig_series).pow(2).mean(dim=(1, 2))
        feat = model.get_embedding(images)

        all_orig.append(orig_series.detach().cpu())
        all_recon.append(recon_series.detach().cpu())
        all_feat.append(feat.detach().cpu())
        all_mse.append(mse.detach().cpu())

        collected += x.size(0)
        if max_samples > 0 and collected >= max_samples:
            break

    orig = torch.cat(all_orig, dim=0)
    recon = torch.cat(all_recon, dim=0)
    feat = torch.cat(all_feat, dim=0)
    mse = torch.cat(all_mse, dim=0)

    if max_samples > 0:
        orig = orig[:max_samples]
        recon = recon[:max_samples]
        feat = feat[:max_samples]
        mse = mse[:max_samples]
    return orig, recon, feat, mse


def plot_recon_examples(orig, recon, save_path, num_plot=12, channel=0):
    n = min(num_plot, orig.shape[0])
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3 * nrows), squeeze=False)
    t = np.arange(orig.shape[1])
    for i in range(nrows * ncols):
        ax = axes[i // ncols][i % ncols]
        if i >= n:
            ax.axis("off")
            continue
        ax.plot(t, orig[i, :, channel], label="orig", linewidth=1.5)
        ax.plot(t, recon[i, :, channel], label="recon", linewidth=1.2, alpha=0.9)
        ax.set_title(f"sample {i}")
        if i == 0:
            ax.legend()
    fig.suptitle("Validation reconstruction")
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def nearest_neighbor_analysis(orig, feat, k=1):
    feat = feat.float()
    n = feat.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 samples for neighbor analysis.")

    d_feat = torch.cdist(feat, feat, p=2)
    d_feat.fill_diagonal_(float("inf"))
    nn_dist, nn_idx = torch.topk(d_feat, k=k, dim=1, largest=False)

    ts_flat = orig.reshape(orig.shape[0], -1).float()
    d_ts = torch.cdist(ts_flat, ts_flat, p=2)
    row_idx = torch.arange(n).unsqueeze(1)
    nn_ts_dist = d_ts[row_idx, nn_idx]
    return nn_idx, nn_dist, nn_ts_dist


def plot_neighbor_examples(orig, nn_idx, nn_feat_dist, nn_ts_dist, save_path, num_plot=12, channel=0):
    n = min(num_plot, orig.shape[0])
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3 * nrows), squeeze=False)
    t = np.arange(orig.shape[1])
    for i in range(nrows * ncols):
        ax = axes[i // ncols][i % ncols]
        if i >= n:
            ax.axis("off")
            continue
        j = int(nn_idx[i, 0].item())
        ax.plot(t, orig[i, :, channel], label="query", linewidth=1.5)
        ax.plot(t, orig[j, :, channel], label="nn-feature", linewidth=1.2, alpha=0.9)
        ax.set_title(
            f"q={i}, nn={j}\n"
            f"feat_d={float(nn_feat_dist[i,0]):.4f}, ts_d={float(nn_ts_dist[i,0]):.4f}"
        )
        if i == 0:
            ax.legend()
    fig.suptitle("Nearest neighbor in feature space (shown in time-series domain)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def plot_distance_scatter(nn_feat_dist, nn_ts_dist, save_path):
    x = nn_feat_dist[:, 0].numpy()
    y = nn_ts_dist[:, 0].numpy()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, s=12, alpha=0.7)
    ax.set_xlabel("Nearest feature distance")
    ax.set_ylabel("Time-series L2 distance to same neighbor")
    ax.set_title("Neighbor consistency: feature vs time-series distance")
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate SoftVQ-VAE reconstruction and neighbor behavior on val set.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint file (best.pt).")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write visualizations and metrics.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_val_samples", type=int, default=2000)
    parser.add_argument("--num_plot", type=int, default=12)
    parser.add_argument("--neighbor_k", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--override_datasets_dir", type=str, default=None)
    parser.add_argument("--override_rel_path", type=str, default=None)
    return parser.parse_args()


def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.ckpt)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg, state_dict = load_checkpoint_model(ckpt_path, device)
    if args.override_datasets_dir is not None:
        cfg["datasets_dir"] = args.override_datasets_dir
    if args.override_rel_path is not None:
        cfg["rel_path"] = args.override_rel_path

    _, val_dataset = load_datasets(cfg)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    sample = val_dataset[0]  # (C, T)
    channels, seq_len = sample.shape
    embedder = DelayEmbedder(
        device=device,
        seq_len=seq_len,
        delay=int(cfg["delay"]),
        embedding=int(cfg["embedding"]),
    )
    sample_img, _ = to_images(sample.unsqueeze(0).to(device), embedder)
    resolution = int(sample_img.shape[-1])

    model = build_model(cfg, resolution, channels, device)
    model.load_state_dict(state_dict, strict=True)

    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Dataset: {cfg['dataset_name']} | val size: {len(val_dataset)}")
    print(f"Series shape: (C={channels}, T={seq_len}) | Image resolution: {resolution}")

    orig, recon, feat, mse = run_eval(model, val_loader, embedder, device, max_samples=args.max_val_samples)
    print(f"Evaluated samples: {orig.shape[0]}")
    print(f"Val recon MSE mean={float(mse.mean()):.6f}, std={float(mse.std()):.6f}")

    nn_idx, nn_feat_dist, nn_ts_dist = nearest_neighbor_analysis(orig, feat, k=args.neighbor_k)
    print(
        "Nearest-neighbor stats: "
        f"feature_dist_mean={float(nn_feat_dist[:,0].mean()):.6f}, "
        f"ts_dist_mean={float(nn_ts_dist[:,0].mean()):.6f}"
    )

    plot_recon_examples(orig.numpy(), recon.numpy(), out_dir / "reconstruction_examples.png", num_plot=args.num_plot)
    plot_neighbor_examples(
        orig.numpy(),
        nn_idx,
        nn_feat_dist,
        nn_ts_dist,
        out_dir / "feature_neighbors_timeseries_examples.png",
        num_plot=args.num_plot,
    )
    plot_distance_scatter(nn_feat_dist, nn_ts_dist, out_dir / "feature_vs_ts_distance_scatter.png")

    np.savez(
        out_dir / "metrics_and_neighbors.npz",
        recon_mse=mse.numpy(),
        feature_embeddings=feat.numpy(),
        nn_idx=nn_idx.numpy(),
        nn_feature_distance=nn_feat_dist.numpy(),
        nn_ts_distance=nn_ts_dist.numpy(),
    )
    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
