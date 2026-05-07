import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
try:
    import wandb
except ImportError:
    wandb = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
import sys
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from fid_vae import FIDVAE
from data_provider.data_provider import get_train, get_test


def get_args():
    parser = argparse.ArgumentParser()

    # data (aligned with scripts/benchmark_drift.sh)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--datasets_dir", type=str, required=True)
    parser.add_argument("--rel_path", type=str, required=True)
    parser.add_argument("--ts_seq_len", type=int, default=256)
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
    parser.add_argument("--latent_dim", type=int, default=4)
    parser.add_argument("--latent_downsample", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--one_channel", action="store_true")

    # misc
    parser.add_argument("--save_dir", type=str, default="./fid_vae_ckpts/benchmark")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="drifting-model")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    return parser.parse_args()


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
    """Expose benchmark samples as FIDVAE tensors, shape (C, T)."""

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
    return {
        "name": args.dataset_name,
        "data": args.data,
        "datasets_dir": args.datasets_dir,
        "rel_path": args.rel_path,
        # Some dataset backends (e.g., Mujoco) expect `path` instead of `rel_path`.
        "path": args.rel_path,
        "seq_len": args.ts_seq_len,
        "flag": flag,
    }


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


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss, total_recon, total_kl = 0.0, 0.0, 0.0
    pbar = tqdm(dataloader, desc="Train", file=sys.stdout)

    for batch in pbar:
        x = batch[0].to(device)
        out = model(x)
        loss_dict = model.loss_function(x, out["recon"], out["mu"], out["logvar"])
        loss = loss_dict["loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_recon += loss_dict["recon_loss"].item()
        total_kl += loss_dict["kl_loss"].item()

        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "recon": f"{loss_dict['recon_loss'].item():.4f}",
                "kl": f"{loss_dict['kl_loss'].item():.4f}",
            }
        )

    n = len(dataloader)
    return total_loss / n, total_recon / n, total_kl / n


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    total_loss, total_recon, total_kl = 0.0, 0.0, 0.0

    for batch in dataloader:
        x = batch[0].to(device)
        out = model(x)
        loss_dict = model.loss_function(x, out["recon"], out["mu"], out["logvar"])
        total_loss += loss_dict["loss"].item()
        total_recon += loss_dict["recon_loss"].item()
        total_kl += loss_dict["kl_loss"].item()

    n = len(dataloader)
    return total_loss / n, total_recon / n, total_kl / n


def train(args):
    device = args.device if torch.cuda.is_available() else "cpu"
    train_dataset, val_dataset = load_benchmark_datasets(args)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    sample = train_dataset[0][0]
    print(f"data shape: {sample.shape}")
    c, t = sample.shape

    model = FIDVAE(
        input_dim=c,
        output_dim=c,
        seq_len=t,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        latent_dim=args.latent_dim,
        latent_downsample=args.latent_downsample,
        dropout=args.dropout,
        beta=args.beta,
    ).to(device)

    print(model)
    breakpoint()
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
        train_size=np.array(len(train_dataset), dtype=np.int64),
        val_size=np.array(len(val_dataset), dtype=np.int64),
        extra_scale=np.array(False, dtype=bool),
    )

    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        print(f"\n===== Epoch {epoch} =====", flush=True)
        train_loss, train_recon, train_kl = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_recon, val_kl = validate(model, val_loader, device)

        print(
            f"\nTrain Loss: {train_loss:.6f} | Recon: {train_recon:.6f} | KL: {train_kl:.6f}",
            flush=True,
        )
        print(
            f"Val   Loss: {val_loss:.6f} | Recon: {val_recon:.6f} | KL: {val_kl:.6f}",
            flush=True,
        )

        if wb is not None:
            wandb.log(
                {
                    "train/loss": train_loss,
                    "train/recon_loss": train_recon,
                    "train/kl_loss": train_kl,
                    "val/loss": val_loss,
                    "val/recon_loss": val_recon,
                    "val/kl_loss": val_kl,
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
