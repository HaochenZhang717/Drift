import os
import argparse
import math
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import wandb

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
import sys
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from models.vqvae import VQVAE
from data_provider.data_provider import get_train, get_test


def get_args():
    parser = argparse.ArgumentParser()

    # data (aligned with scripts/benchmark_drift.sh)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--datasets_dir", type=str, required=True)
    parser.add_argument("--rel_path", type=str, default=None)
    parser.add_argument("--rel_path_train", type=str, default=None)
    parser.add_argument("--rel_path_valid", type=str, default=None)
    parser.add_argument("--ts_seq_len", type=int, default=256)
    parser.add_argument("--window_stride", type=int, default=None)
    parser.add_argument("--ts_stride", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--train_split", type=str, default="train", choices=["train"])
    parser.add_argument("--val_split", type=str, default="test", choices=["test"])

    # training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)

    # model
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--code_dim", type=int, default=4)
    parser.add_argument("--num_codes", type=int, default=512)
    parser.add_argument("--latent_downsample", type=int, default=16)
    parser.add_argument("--decoder_upsample_rate", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--commitment_weight", type=float, default=0.25)
    parser.add_argument("--one_channel", action="store_true")

    # misc
    parser.add_argument("--save_dir", type=str, default="./fid_vae_ckpts/benchmark")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="drifting-model")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    args = parser.parse_args()
    if args.rel_path is None and not (args.rel_path_train and args.rel_path_valid):
        parser.error(
            "Provide --rel_path, or provide both --rel_path_train and --rel_path_valid."
        )
    if args.rel_path is None:
        args.rel_path = args.rel_path_train
    return args


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
        series = _extract_series(base_dataset[idx])  # (T, C)
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
    """Expose benchmark samples as VQVAE tensors, shape (C, T)."""

    def __init__(self, base_dataset, data_min, data_max, one_channel=False):
        self.base_dataset = base_dataset
        self.data_min = data_min
        self.data_max = data_max
        self.denom = torch.clamp(self.data_max - self.data_min, min=1e-6)
        self.one_channel = one_channel

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        series = _extract_series(self.base_dataset[idx])  # (T, C)
        if self.one_channel:
            series = series[:, :1]
        tensor = torch.from_numpy(series).to(torch.float32)
        tensor = torch.clamp((tensor - self.data_min) / self.denom, 0.0, 1.0)
        tensor = tensor * 2.0 - 1.0
        tensor = tensor.permute(1, 0).contiguous()  # (C, T)
        return (tensor,)


def _make_dataset_config(args, flag):
    stride = args.window_stride
    if stride is None:
        stride = args.ts_stride
    if stride is None:
        stride = args.stride

    config = {
        "name": args.dataset_name,
        "data": args.data,
        "datasets_dir": args.datasets_dir,
        "rel_path": args.rel_path,
        # Some dataset backends (e.g., Mujoco) expect `path` instead of `rel_path`.
        "path": args.rel_path,
        "seq_len": args.ts_seq_len,
        "flag": flag,
    }
    if args.rel_path_train is not None:
        config["rel_path_train"] = args.rel_path_train
    if args.rel_path_valid is not None:
        config["rel_path_valid"] = args.rel_path_valid
    if stride is not None:
        config["window_stride"] = stride
        config["ts_stride"] = stride
        config["stride"] = stride
    return config


def load_benchmark_datasets(args):
    train_base = get_train(_make_dataset_config(args, args.train_split))
    val_base = get_test(_make_dataset_config(args, args.val_split))
    data_min, data_max = _fit_minmax_stats(train_base, one_channel=args.one_channel)

    train_dataset = BenchmarkTensorDataset(
        train_base,
        data_min=data_min,
        data_max=data_max,
        one_channel=args.one_channel,
    )
    val_dataset = BenchmarkTensorDataset(
        val_base,
        data_min=data_min,
        data_max=data_max,
        one_channel=args.one_channel,
    )

    sample = train_dataset[0][0]
    print(
        f"Loaded benchmark dataset | data={args.data} | rel_path={args.rel_path} | "
        f"train={len(train_dataset)} | val={len(val_dataset)} | sample={tuple(sample.shape)}",
        flush=True,
    )
    return train_dataset, val_dataset


def validate_decoder_shape_args(seq_len, latent_downsample, decoder_upsample_rate):
    if latent_downsample < 1 or latent_downsample & (latent_downsample - 1) != 0:
        raise ValueError("--latent_downsample must be a power of two.")
    if decoder_upsample_rate < 1:
        raise ValueError("--decoder_upsample_rate must be positive.")
    if seq_len % latent_downsample != 0:
        raise ValueError(
            f"--ts_seq_len ({seq_len}) must be divisible by --latent_downsample ({latent_downsample})."
        )
    num_upsample_blocks = int(math.log(latent_downsample, decoder_upsample_rate))
    decoded_len = (seq_len // latent_downsample) * (decoder_upsample_rate ** num_upsample_blocks)
    if decoded_len != seq_len:
        raise ValueError(
            "Decoder output length will not match input length without final interpolation: "
            f"seq_len={seq_len}, latent_downsample={latent_downsample}, "
            f"decoder_upsample_rate={decoder_upsample_rate}, "
            f"num_upsample_blocks={num_upsample_blocks}, decoded_len={decoded_len}. "
            "Use compatible values where latent_downsample is a power of decoder_upsample_rate, "
            "e.g. latent_downsample=16 with decoder_upsample_rate=4."
        )


def _used_codes_count(indices: torch.Tensor) -> int:
    # indices: (B, T_latent)
    return int(torch.unique(indices).numel())


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_vq = 0.0
    total_codebook = 0.0
    total_commitment = 0.0
    total_perplexity = 0.0
    num_codes = int(model.quantizer.num_codes)
    epoch_used_mask = torch.zeros(num_codes, dtype=torch.bool)
    pbar = tqdm(dataloader, desc="Train", file=sys.stdout)

    for batch in pbar:
        x = batch[0].to(device)
        out = model(x)
        loss_dict = model.loss_function(x, out["recon"], out["vq_loss"])
        loss = loss_dict["loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_recon += loss_dict["recon_loss"].item()
        total_vq += loss_dict["vq_loss"].item()
        total_codebook += out["codebook_loss"].item()
        total_commitment += out["commitment_loss"].item()
        total_perplexity += out["perplexity"].item()
        batch_unique = torch.unique(out["indices"].detach().cpu())
        epoch_used_mask[batch_unique] = True

        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "recon": f"{loss_dict['recon_loss'].item():.4f}",
                "vq": f"{loss_dict['vq_loss'].item():.4f}",
                "ppx": f"{out['perplexity'].item():.2f}",
                "used": f"{_used_codes_count(out['indices'])}",
            }
        )

    n = len(dataloader)
    return (
        total_loss / n,
        total_recon / n,
        total_vq / n,
        total_codebook / n,
        total_commitment / n,
        total_perplexity / n,
        float(epoch_used_mask.sum().item()),
    )


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_vq = 0.0
    total_codebook = 0.0
    total_commitment = 0.0
    total_perplexity = 0.0
    num_codes = int(model.quantizer.num_codes)
    epoch_used_mask = torch.zeros(num_codes, dtype=torch.bool)

    for batch in dataloader:
        x = batch[0].to(device)
        out = model(x)
        loss_dict = model.loss_function(x, out["recon"], out["vq_loss"])
        total_loss += loss_dict["loss"].item()
        total_recon += loss_dict["recon_loss"].item()
        total_vq += loss_dict["vq_loss"].item()
        total_codebook += out["codebook_loss"].item()
        total_commitment += out["commitment_loss"].item()
        total_perplexity += out["perplexity"].item()
        batch_unique = torch.unique(out["indices"].detach().cpu())
        epoch_used_mask[batch_unique] = True

    n = len(dataloader)
    return (
        total_loss / n,
        total_recon / n,
        total_vq / n,
        total_codebook / n,
        total_commitment / n,
        total_perplexity / n,
        float(epoch_used_mask.sum().item()),
    )


def train(args):
    device = args.device if torch.cuda.is_available() else "cpu"
    train_dataset, val_dataset = load_benchmark_datasets(args)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    sample = train_dataset[0][0]
    print(f"data shape: {sample.shape}")
    c, t = sample.shape
    validate_decoder_shape_args(t, args.latent_downsample, args.decoder_upsample_rate)

    model = VQVAE(
        input_dim=c,
        output_dim=c,
        seq_len=t,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        code_dim=args.code_dim,
        num_codes=args.num_codes,
        latent_downsample=args.latent_downsample,
        decoder_upsample_rate=args.decoder_upsample_rate,
        dropout=args.dropout,
        commitment_weight=args.commitment_weight,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.one_channel:
        save_dir = os.path.join(args.save_dir, f"{args.dataset_name}_one_channel")
    else:
        save_dir = os.path.join(args.save_dir, args.dataset_name)

    os.makedirs(save_dir, exist_ok=True)

    wb = None
    if args.wandb:
        if wandb is None:
            raise ImportError("wandb is not installed. Please install wandb or run without --wandb.")
        wb = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                **vars(args),
                "train_size": len(train_dataset),
                "val_size": len(val_dataset),
                "input_channels": c,
                "input_seq_len": t,
                "save_dir": save_dir,
            },
            dir=save_dir,
        )

    np.savez(
        os.path.join(save_dir, "dataset_metadata.npz"),
        dataset_name=np.array(args.dataset_name),
        data=np.array(args.data),
        rel_path=np.array(args.rel_path),
        ts_seq_len=np.array(args.ts_seq_len, dtype=np.int64),
        code_dim=np.array(args.code_dim, dtype=np.int64),
        num_codes=np.array(args.num_codes, dtype=np.int64),
        window_stride=np.array(-1 if args.window_stride is None else args.window_stride, dtype=np.int64),
        ts_stride=np.array(-1 if args.ts_stride is None else args.ts_stride, dtype=np.int64),
        stride=np.array(-1 if args.stride is None else args.stride, dtype=np.int64),
        train_size=np.array(len(train_dataset), dtype=np.int64),
        val_size=np.array(len(val_dataset), dtype=np.int64),
        extra_scale=np.array(False, dtype=bool),
    )

    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        print(f"\n===== Epoch {epoch} =====", flush=True)
        (
            train_loss,
            train_recon,
            train_vq,
            train_codebook,
            train_commitment,
            train_ppx,
            train_used_codes,
        ) = train_one_epoch(model, train_loader, optimizer, device)
        (
            val_loss,
            val_recon,
            val_vq,
            val_codebook,
            val_commitment,
            val_ppx,
            val_used_codes,
        ) = validate(model, val_loader, device)
        train_used_ratio = train_used_codes / float(args.num_codes)
        val_used_ratio = val_used_codes / float(args.num_codes)

        print(
            f"\nTrain Loss: {train_loss:.6f} | Recon: {train_recon:.6f} | "
            f"VQ: {train_vq:.6f} | Codebook: {train_codebook:.6f} | "
            f"Commit: {train_commitment:.6f} | PPL: {train_ppx:.3f} | "
            f"Used: {train_used_codes:.2f}/{args.num_codes} ({train_used_ratio:.3f})",
            flush=True,
        )
        print(
            f"Val   Loss: {val_loss:.6f} | Recon: {val_recon:.6f} | "
            f"VQ: {val_vq:.6f} | Codebook: {val_codebook:.6f} | "
            f"Commit: {val_commitment:.6f} | PPL: {val_ppx:.3f} | "
            f"Used: {val_used_codes:.2f}/{args.num_codes} ({val_used_ratio:.3f})",
            flush=True,
        )

        if wb is not None:
            wandb.log(
                {
                    "train/loss": train_loss,
                    "train/recon_loss": train_recon,
                    "train/vq_loss": train_vq,
                    "train/codebook_loss": train_codebook,
                    "train/commitment_loss": train_commitment,
                    "train/perplexity": train_ppx,
                    "train/used_codes": train_used_codes,
                    "train/used_code_ratio": train_used_ratio,
                    "val/loss": val_loss,
                    "val/recon_loss": val_recon,
                    "val/vq_loss": val_vq,
                    "val/codebook_loss": val_codebook,
                    "val/commitment_loss": val_commitment,
                    "val/perplexity": val_ppx,
                    "val/used_codes": val_used_codes,
                    "val/used_code_ratio": val_used_ratio,
                    "val/best_loss": min(best_val_loss, val_loss),
                    "epoch": epoch + 1,
                    "lr": optimizer.param_groups[0]["lr"],
                },
                step=epoch + 1,
            )

        torch.save(model.state_dict(), os.path.join(save_dir, "last.pt"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best.pt"))
            print("Saved BEST model", flush=True)

    if wb is not None:
        wb.finish()


if __name__ == "__main__":
    train(get_args())
