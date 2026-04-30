import os
import argparse
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
import sys
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from fid_vae import FIDVAE
from utils.utils_dataset import AIREADIModalityImputationDataset


# =========================
# Args
# =========================
def get_args():
    parser = argparse.ArgumentParser()

    # ===== data =====
    parser.add_argument("--data_root", type=str, default="./AI-READI")
    parser.add_argument("--participants_tsv_path", type=str, default=None)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="test")
    parser.add_argument("--ts_seq_len", type=int, default=288)
    parser.add_argument("--daily_min_events", type=int, default=288)
    parser.add_argument("--max_anchor_gap_minutes", type=float, default=10.0)
    parser.add_argument("--max_window_span_hours", type=float, default=24.0)
    parser.add_argument("--anchor_sampling_minutes", type=float, default=5.0)
    parser.add_argument("--anchor_sampling_tolerance_seconds", type=float, default=2.0)
    parser.add_argument(
        "--raw_glucose",
        action="store_true",
        help="Use raw mg/dL glucose values. By default the AI-READI dataset normalizes glucose to [-1, 1].",
    )

    # ===== training =====
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)

    # ===== model =====
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--beta", type=float, default=0.001)

    # ===== misc =====
    parser.add_argument("--save_dir", type=str, default="./fid_vae_ckpts/vae_glucose_daily")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--scale",
        action="store_true",
        help="Fit an additional MinMaxScaler on train data and scale all splits to [-1, 1]. Usually unnecessary unless --raw_glucose is set.",
    )

    return parser.parse_args()


# =========================
# Dataset
# =========================
def _load_numpy_series(npy_path):
    data = np.load(npy_path, allow_pickle=True)

    if data.dtype == object:
        data = np.stack(data)

    if data.ndim == 2:
        data = data[:, :, None]

    return data.astype(np.float32)


def fit_scaler(train_data):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(train_data.reshape(-1, train_data.shape[-1]))
    return scaler


def transform_data(data, scaler=None):
    if scaler is None:
        return data
    original_shape = data.shape
    data = scaler.transform(data.reshape(-1, original_shape[-1])).reshape(original_shape)
    return data.astype(np.float32)


def load_dataset(npy_path, scaler=None):
    data = _load_numpy_series(npy_path)
    data = transform_data(data, scaler=scaler)

    data = torch.tensor(data, dtype=torch.float32).permute(0, 2, 1)

    print(f"Loaded {npy_path}: {data.shape}")

    return TensorDataset(data)


class DailyGlucoseTensorDataset(Dataset):
    """Expose daily AI-READI glucose windows as FIDVAE tensors, shape (C, T)."""

    def __init__(self, base_dataset, scaler=None):
        self.base_dataset = base_dataset
        self.scaler = scaler

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        series = item["target"].detach().cpu().numpy().astype(np.float32)
        if self.scaler is not None:
            series = transform_data(series[None, :, :], scaler=self.scaler)[0]
        tensor = torch.from_numpy(series).permute(1, 0).contiguous()
        return (tensor,)


def make_daily_glucose_dataset(
    args,
    split,
    value_ranges=None,
):
    return AIREADIModalityImputationDataset(
        root=args.data_root,
        split=split,
        modalities=["glucose"],
        anchor_modality="glucose",
        target_modality="glucose",
        window_size=args.ts_seq_len,
        window_stride=args.ts_seq_len,
        window_mode="daily",
        daily_min_events=args.daily_min_events,
        max_events_per_modality={"glucose": args.ts_seq_len},
        min_events_per_modality={"glucose": args.daily_min_events},
        normalize=not args.raw_glucose,
        value_ranges=value_ranges,
        max_anchor_gap_minutes=args.max_anchor_gap_minutes,
        max_window_span_hours=args.max_window_span_hours,
        anchor_sampling_minutes=args.anchor_sampling_minutes,
        anchor_sampling_tolerance_seconds=args.anchor_sampling_tolerance_seconds,
        participants_tsv_path=args.participants_tsv_path,
        include_clinical_static=False,
        include_participant_metadata=args.participants_tsv_path is not None,
        include_study_group=args.participants_tsv_path is not None,
        include_clinical_site=False,
        pad=True,
        return_dict=True,
    )


def collect_dataset_array(dataset):
    """Collect target arrays as (N, T, C), used only for optional scaler fitting."""
    data = []
    for item in dataset:
        data.append(item["target"].detach().cpu().numpy().astype(np.float32))
    if not data:
        raise ValueError("No daily glucose windows were collected.")
    return np.stack(data, axis=0)


def load_daily_glucose_datasets(args):
    train_base = make_daily_glucose_dataset(args, args.train_split)
    val_base = make_daily_glucose_dataset(
        args,
        args.val_split,
        value_ranges=train_base.value_ranges,
    )

    scaler = None
    if args.scale:
        train_data = collect_dataset_array(train_base)
        scaler = fit_scaler(train_data)

    train_dataset = DailyGlucoseTensorDataset(train_base, scaler=scaler)
    val_dataset = DailyGlucoseTensorDataset(val_base, scaler=scaler)

    sample = train_dataset[0][0]
    print(
        f"Loaded daily AI-READI glucose | "
        f"train: {len(train_dataset)} | val: {len(val_dataset)} | "
        f"sample: {tuple(sample.shape)} | "
        f"normalized: {not args.raw_glucose} | extra_scale: {args.scale}"
    )
    return train_dataset, val_dataset, scaler, train_base


# =========================
# Train One Epoch
# =========================
def train_one_epoch(model, dataloader, optimizer, device):

    model.train()

    total_loss = 0
    total_recon = 0
    total_kl = 0

    pbar = tqdm(dataloader, desc="Train")

    for batch in pbar:
        x = batch[0].to(device)
        out = model(x)

        loss_dict = model.loss_function(
            x,
            out["recon"],
            out["mu"],
            out["logvar"]
        )

        loss = loss_dict["loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_recon += loss_dict["recon_loss"].item()
        total_kl += loss_dict["kl_loss"].item()

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "recon": f"{loss_dict['recon_loss'].item():.4f}",
            "kl": f"{loss_dict['kl_loss'].item():.4f}",
        })

    n = len(dataloader)
    return total_loss / n, total_recon / n, total_kl / n


# =========================
# Validation
# =========================
@torch.no_grad()
def validate(model, dataloader, device):

    model.eval()

    total_loss = 0
    total_recon = 0
    total_kl = 0

    for batch in dataloader:
        x = batch[0].to(device)

        out = model(x)

        loss_dict = model.loss_function(
            x,
            out["recon"],
            out["mu"],
            out["logvar"]
        )

        total_loss += loss_dict["loss"].item()
        total_recon += loss_dict["recon_loss"].item()
        total_kl += loss_dict["kl_loss"].item()

    n = len(dataloader)
    return total_loss / n, total_recon / n, total_kl / n


# =========================
# Train
# =========================
def train(args):

    device = args.device if torch.cuda.is_available() else "cpu"

    # ===== dataset =====
    train_dataset, val_dataset, scaler, train_base = load_daily_glucose_datasets(args)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # ===== infer shape =====
    sample = train_dataset[0][0]
    C, T = sample.shape

    # ===== model =====
    model = FIDVAE(
        input_dim=C,
        output_dim=C,
        seq_len=T,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        latent_dim=args.latent_dim,
        beta=args.beta,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.001)

    os.makedirs(args.save_dir, exist_ok=True)

    np.savez(
        os.path.join(args.save_dir, "dataset_metadata.npz"),
        ts_seq_len=np.array(args.ts_seq_len, dtype=np.int64),
        daily_min_events=np.array(args.daily_min_events, dtype=np.int64),
        raw_glucose=np.array(args.raw_glucose, dtype=bool),
        normalized=np.array(not args.raw_glucose, dtype=bool),
        train_size=np.array(len(train_dataset), dtype=np.int64),
        val_size=np.array(len(val_dataset), dtype=np.int64),
        glucose_value_range=np.asarray(
            train_base.value_ranges.get("glucose", (np.nan, np.nan)),
            dtype=np.float32,
        ),
    )

    if scaler is not None:
        np.savez(
            os.path.join(args.save_dir, "scaler_stats.npz"),
            data_min=scaler.data_min_.astype(np.float32),
            data_max=scaler.data_max_.astype(np.float32),
            data_range=scaler.data_range_.astype(np.float32),
            scale=scaler.scale_.astype(np.float32),
            min=scaler.min_.astype(np.float32),
            feature_range=np.array(scaler.feature_range, dtype=np.float32),
        )

    best_val_loss = float("inf")

    # =========================
    # Training loop
    # =========================
    for epoch in range(args.epochs):

        print(f"\n===== Epoch {epoch} =====")

        train_loss, train_recon, train_kl = train_one_epoch(
            model, train_loader, optimizer, device
        )

        val_loss, val_recon, val_kl = validate(
            model, val_loader, device
        )

        # ===== print =====
        print(f"\nTrain Loss: {train_loss:.6f} | Recon: {train_recon:.6f} | KL: {train_kl:.6f}")
        print(f"Val   Loss: {val_loss:.6f} | Recon: {val_recon:.6f} | KL: {val_kl:.6f}")

        # ===== save last =====
        torch.save(
            model.state_dict(),
            os.path.join(args.save_dir, "last.pt")
        )

        # ===== save best =====
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model.state_dict(),
                os.path.join(args.save_dir, "best.pt")
            )
            print("Saved BEST model")

# =========================
# Main
# =========================
if __name__ == "__main__":
    args = get_args()
    train(args)
