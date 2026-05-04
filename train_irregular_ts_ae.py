#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.modality_imputation.ts_ae import IrregularTimeSeriesAE
from utils.utils_dataset import AIREADIModalityImputationDataset, AIREADI_MODALITY_SPECS


ALIGNED_MODALITY_KEY = {
    "calorie": "calorie_aligned_to_glucose",
    "heart_rate": "heart_rate_aligned_to_glucose",
    "respiratory_rate": "respiratory_rate_aligned_to_glucose",
    "physical_activity": "physical_activity_aligned_to_glucose",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class CachedSample:
    x: torch.Tensor  # (T, C)
    observed_mask: torch.Tensor  # (T, C)


class IrregularTSDataContainer(Dataset):
    """Preload and filter samples from AIREADIModalityImputationDataset."""

    def __init__(
        self,
        base_dataset: AIREADIModalityImputationDataset,
        modality_key: str,
        max_missing_ratio: float,
    ):
        super().__init__()
        if not (0.0 <= max_missing_ratio <= 1.0):
            raise ValueError("max_missing_ratio must be in [0, 1]")

        self.samples: List[CachedSample] = []
        self.total_seen = 0
        self.total_kept = 0

        for idx in tqdm(range(len(base_dataset)), desc=f"Caching {modality_key}"):
            item = base_dataset[idx]
            self.total_seen += 1
            modality_pack = item["modalities"].get(modality_key)
            if modality_pack is None:
                continue

            x = modality_pack["values"].float()
            observed_mask = modality_pack["mask"].float()
            if x.ndim != 2 or observed_mask.ndim != 2:
                continue
            if x.shape != observed_mask.shape:
                continue

            valid_ratio = float(observed_mask.mean().item()) if observed_mask.numel() > 0 else 0.0
            missing_ratio = 1.0 - valid_ratio
            if missing_ratio > max_missing_ratio:
                continue

            self.samples.append(CachedSample(x=x, observed_mask=observed_mask))

        self.total_kept = len(self.samples)
        if self.total_kept == 0:
            raise ValueError(
                f"No samples kept for modality_key={modality_key}. "
                f"Try increasing --max_missing_ratio or changing data constraints."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.samples[idx]
        return s.x, s.observed_mask


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    # Sum over all selected positions then divide by number of selected positions.
    sq_err = (pred - target).pow(2) * loss_mask
    denom = loss_mask.sum().clamp_min(1.0)
    return sq_err.sum() / denom


def make_random_input_mask(
    observed_mask: torch.Tensor,
    random_drop_prob: float,
) -> torch.Tensor:
    """Create extra training mask only on observed positions.

    Returns input_mask where:
      - 1 indicates "additionally masked for model input"
      - 0 indicates "not additionally masked"
    """
    input_mask = torch.zeros_like(observed_mask)
    if random_drop_prob <= 0.0:
        return input_mask

    bsz = observed_mask.size(0)
    for b in range(bsz):
        observed_pos = torch.nonzero(observed_mask[b] > 0, as_tuple=False)  # (N_obs, 2) for (T, C)
        n_obs = int(observed_pos.size(0))
        if n_obs == 0:
            continue

        n_drop = int(round(float(random_drop_prob) * n_obs))
        n_drop = max(0, min(n_drop, n_obs))
        if n_drop == 0:
            continue

        perm = torch.randperm(n_obs, device=observed_mask.device)[:n_drop]
        chosen = observed_pos[perm]

        input_mask[b, chosen[:, 0], chosen[:, 1]] = 1.0

    return input_mask


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    input_random_drop_prob: float,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_valid = 0.0

    iterator = tqdm(loader, desc="train" if is_train else "val", leave=False)
    for x, observed_mask in iterator:
        x = x.to(device)
        observed_mask = observed_mask.to(device)
        input_mask = make_random_input_mask(observed_mask, input_random_drop_prob)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        pred = model(x, observed_mask=observed_mask, input_mask=input_mask)
        loss = masked_mse_loss(pred, x, input_mask)

        if is_train:
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            batch_valid = float(input_mask.sum().item())
            total_loss += float(loss.item()) * batch_valid
            total_valid += batch_valid

    mean_loss = total_loss / max(total_valid, 1.0)
    return {"loss": mean_loss}


@torch.no_grad()
def save_reconstruction_plots(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    out_dir: Path,
    num_samples: int = 4,
    input_random_drop_prob: float = 0.0,
) -> List[str]:
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[str] = []

    try:
        x, observed_mask = next(iter(loader))
    except StopIteration:
        return saved_paths

    x = x.to(device)
    observed_mask = observed_mask.to(device)
    # 1) Reconstruction with extra input masking (training-like setting)
    input_mask_with = make_random_input_mask(observed_mask, input_random_drop_prob)
    pred_with_mask = model(x, observed_mask=observed_mask, input_mask=input_mask_with)
    # For visualization as an imputed series:
    # keep original values only where data is observed and not additionally masked;
    # use model predictions for additionally masked positions and dataset-missing positions.
    keep_original_mask = observed_mask * (1.0 - input_mask_with)
    recon_with_imputed = pred_with_mask * (1.0 - keep_original_mask) + x * keep_original_mask
    # 2) Reconstruction without extra input masking (regularization view)
    input_mask_without = torch.zeros_like(observed_mask)
    pred_without_mask = model(x, observed_mask=observed_mask, input_mask=input_mask_without)

    bsz = min(num_samples, x.size(0))
    for i in range(bsz):
        real = x[i].detach().cpu().numpy()[:, 0]
        recon_with = recon_with_imputed[i].detach().cpu().numpy()[:, 0]
        recon_without = pred_without_mask[i].detach().cpu().numpy()[:, 0]
        obs = observed_mask[i].detach().cpu().numpy()[:, 0] > 0
        t = np.arange(real.shape[0])

        # with input_mask
        fig, ax = plt.subplots(figsize=(10, 4.2))
        ax.plot(t, real, label="real", linewidth=1.2)
        ax.plot(t, recon_with, label="recon_with_input_mask", linewidth=1.2, alpha=0.9)
        if np.any(obs):
            ax.scatter(t[~obs], real[~obs], s=12, marker="x", label="missing_in_dataset")
        ax.set_title(f"Reconstruction (with_input_mask) | epoch={epoch} | sample={i}")
        ax.set_xlabel("Time Index")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        out_path_with = out_dir / f"epoch_{epoch:04d}_sample_{i}_with_input_mask.png"
        fig.savefig(out_path_with, dpi=150)
        plt.close(fig)
        saved_paths.append(str(out_path_with))

        # without input_mask
        fig, ax = plt.subplots(figsize=(10, 4.2))
        ax.plot(t, real, label="real", linewidth=1.2)
        ax.plot(t, recon_without, label="recon_without_input_mask", linewidth=1.2, alpha=0.9)
        if np.any(obs):
            ax.scatter(t[~obs], real[~obs], s=12, marker="x", label="missing_in_dataset")
        ax.set_title(f"Reconstruction (without_input_mask) | epoch={epoch} | sample={i}")
        ax.set_xlabel("Time Index")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        out_path_without = out_dir / f"epoch_{epoch:04d}_sample_{i}_without_input_mask.png"
        fig.savefig(out_path_without, dpi=150)
        plt.close(fig)
        saved_paths.append(str(out_path_without))

    return saved_paths


def build_base_dataset(args: argparse.Namespace, split: str, modalities: List[str]) -> AIREADIModalityImputationDataset:
    max_events = {m: args.ts_seq_len for m in modalities}
    min_events = {args.anchor_modality: args.daily_min_events}

    return AIREADIModalityImputationDataset(
        root=args.data_root,
        split=split,
        modalities=modalities,
        anchor_modality=args.anchor_modality,
        target_modality=args.anchor_modality,
        window_size=args.ts_seq_len,
        window_stride=args.ts_seq_len,
        window_mode="daily",
        daily_min_events=args.daily_min_events,
        max_events_per_modality=max_events,
        min_events_per_modality=min_events,
        normalize=not args.raw_values,
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


def resolve_modality_key(train_modality: str, use_aligned: bool) -> Tuple[List[str], str]:
    if train_modality not in AIREADI_MODALITY_SPECS:
        raise ValueError(f"Unknown train_modality: {train_modality}")

    if use_aligned and train_modality in ALIGNED_MODALITY_KEY:
        return ["glucose", train_modality], ALIGNED_MODALITY_KEY[train_modality]
    return [train_modality], train_modality


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AE on one irregular AI-READI modality.")

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--participants_tsv_path", type=str, default=None)
    parser.add_argument("--train_split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--val_split", type=str, default="test", choices=["train", "valid", "test"])

    parser.add_argument("--train_modality", type=str, required=True, choices=list(AIREADI_MODALITY_SPECS.keys()))
    parser.add_argument("--use_aligned_modality", action="store_true")

    parser.add_argument("--anchor_modality", type=str, default="glucose")
    parser.add_argument("--ts_seq_len", type=int, default=288)
    parser.add_argument("--daily_min_events", type=int, default=288)
    parser.add_argument("--max_anchor_gap_minutes", type=float, default=10.0)
    parser.add_argument("--max_window_span_hours", type=float, default=24.0)
    parser.add_argument("--anchor_sampling_minutes", type=float, default=5.0)
    parser.add_argument("--anchor_sampling_tolerance_seconds", type=float, default=2.0)
    parser.add_argument("--raw_values", action="store_true")
    parser.add_argument("--max_missing_ratio", type=float, default=0.8)

    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=4)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--input_random_drop_prob", type=float, default=0.1)

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--save_dir", type=str, default="./outputs/irregular_ts_ae")
    parser.add_argument("--save_name", type=str, default="best.pt")
    parser.add_argument("--plot_every_epochs", type=int, default=100)
    parser.add_argument("--plot_num_samples", type=int, default=4)

    parser.add_argument("--wandb_project", type=str, default="irregular-ts-ae")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])

    return parser.parse_args()


def main() -> None:
    args = get_args()
    set_seed(args.seed)
    wandb = None
    if args.wandb_mode != "disabled":
        try:
            import wandb as _wandb
            wandb = _wandb
        except ImportError as exc:
            raise ImportError(
                "wandb is required unless --wandb_mode disabled. "
                "Install with: pip install wandb"
            ) from exc

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    source_modalities, modality_key = resolve_modality_key(args.train_modality, args.use_aligned_modality)

    train_base = build_base_dataset(args, args.train_split, source_modalities)
    val_base = build_base_dataset(args, args.val_split, source_modalities)

    train_data = IrregularTSDataContainer(
        base_dataset=train_base,
        modality_key=modality_key,
        max_missing_ratio=args.max_missing_ratio,
    )
    val_data = IrregularTSDataContainer(
        base_dataset=val_base,
        modality_key=modality_key,
        max_missing_ratio=args.max_missing_ratio,
    )

    print(f"training set size: {len(train_data)}")
    print(f"validation set size: {len(val_data)}")

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    sample_x, _ = train_data[0]
    input_dim = int(sample_x.shape[-1])
    model = IrregularTimeSeriesAE(
        input_dim=input_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        max_len=args.ts_seq_len,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run = None
    if wandb is not None:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            config=vars(args),
        )

    best_val = float("inf")
    best_path = str(Path(args.save_dir) / args.save_name)

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, device, optimizer,
            input_random_drop_prob=args.input_random_drop_prob
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model, val_loader, device, optimizer=None,
                input_random_drop_prob=args.input_random_drop_prob
            )

        log_data = {
            "epoch": epoch,
            "train/loss": train_metrics["loss"],
            "val/loss": val_metrics["loss"],
            "data/train_seen": train_data.total_seen,
            "data/train_kept": train_data.total_kept,
            "data/val_seen": val_data.total_seen,
            "data/val_kept": val_data.total_kept,
        }
        if wandb is not None:
            wandb.log(log_data)

        if args.plot_every_epochs > 0 and (epoch % args.plot_every_epochs == 0):
            plot_dir = Path(args.save_dir) / "recon_plots"
            saved = save_reconstruction_plots(
                model=model,
                loader=val_loader,
                device=device,
                epoch=epoch,
                out_dir=plot_dir,
                num_samples=args.plot_num_samples,
                input_random_drop_prob=args.input_random_drop_prob,
            )
            # if wandb is not None and saved:
            #     wandb.log({"recon/examples": [wandb.Image(p) for p in saved], "epoch": epoch})

        print(
            f"Epoch {epoch:03d} | train_loss={train_metrics['loss']:.6f} | "
            f"val_loss={val_metrics['loss']:.6f}",
            flush=True,
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_loss": best_val,
                    "train_modality": args.train_modality,
                    "modality_key": modality_key,
                    "args": vars(args),
                },
                best_path,
            )
            print(f"Saved best checkpoint: {best_path}", flush=True)

    print(f"Training done. Best val loss: {best_val:.6f}", flush=True)
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()



    class example_model(nn.Module):
        def __init__(self,):
            super().__init__()
            # (n, 120, 1) -> (n, 1, 64)
            # (n, 1, 120) -> (n, 64, 1)
            # 64 dimensional
            self.baseline_weight_proj = nn.Sequential(
                nn.Conv1d(1, 16, ),
                nn.

                nn.Mean(dim=-1)
            )

            self.q_proj()
            self.k_proj()



        def forward(self, x):
            '''
            :param past_week_history: (B, num_days, 96)
            :return: (B, 1, 96)
            '''
            baseline = 0.7 * one_day_ago + 0.3 * one_week_ago
            baseline = self.baseline_weight_proj(baseline) * baseline

            historry*(q@k.transpose()/sqrt(d))
