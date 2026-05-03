#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.ts_ae import IrregularTimeSeriesAE
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
    mask: torch.Tensor  # (T, C)


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
            mask = modality_pack["mask"].float()
            if x.ndim != 2 or mask.ndim != 2:
                continue
            if x.shape != mask.shape:
                continue

            valid_ratio = float(mask.mean().item()) if mask.numel() > 0 else 0.0
            missing_ratio = 1.0 - valid_ratio
            if missing_ratio > max_missing_ratio:
                continue

            self.samples.append(CachedSample(x=x, mask=mask))

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
        return s.x, s.mask


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # Sum over all valid positions then divide by number of valid positions.
    sq_err = (pred - target).pow(2) * mask
    denom = mask.sum().clamp_min(1.0)
    return sq_err.sum() / denom


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_valid = 0.0

    iterator = tqdm(loader, desc="train" if is_train else "val", leave=False)
    for x, mask in iterator:
        x = x.to(device)
        mask = mask.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        pred = model(x, mask)
        loss = masked_mse_loss(pred, x, mask)

        if is_train:
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            batch_valid = float(mask.sum().item())
            total_loss += float(loss.item()) * batch_valid
            total_valid += batch_valid

    mean_loss = total_loss / max(total_valid, 1.0)
    return {"loss": mean_loss}


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
    parser.add_argument("--val_split", type=str, default="test", choices=["train", "test"])

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

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--save_dir", type=str, default="./outputs/irregular_ts_ae")
    parser.add_argument("--save_name", type=str, default="best.pt")

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
        train_metrics = run_epoch(model, train_loader, device, optimizer)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, device, optimizer=None)

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
