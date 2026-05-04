#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

from models.ts_ae import IrregularTimeSeriesAE
from utils.utils_dataset import AIREADIModalityImputationDataset, AIREADI_MODALITY_SPECS


ALIGNED_MODALITY_KEY = {
    "calorie": "calorie_aligned_to_glucose",
    "heart_rate": "heart_rate_aligned_to_glucose",
    "respiratory_rate": "respiratory_rate_aligned_to_glucose",
    "physical_activity": "physical_activity_aligned_to_glucose",
}


@dataclass
class CachedSample:
    x: torch.Tensor
    observed_mask: torch.Tensor


class IrregularTSDataContainer(Dataset):
    def __init__(
        self,
        base_dataset: AIREADIModalityImputationDataset,
        modality_key: str,
        max_missing_ratio: float,
    ):
        super().__init__()
        self.samples: List[CachedSample] = []

        for idx in tqdm(range(len(base_dataset)), desc=f"Caching {modality_key}"):
            item = base_dataset[idx]
            modality_pack = item["modalities"].get(modality_key)
            if modality_pack is None:
                continue

            x = modality_pack["values"].float()
            observed_mask = modality_pack["mask"].float()
            if x.ndim != 2 or observed_mask.ndim != 2 or x.shape != observed_mask.shape:
                continue

            valid_ratio = float(observed_mask.mean().item()) if observed_mask.numel() > 0 else 0.0
            missing_ratio = 1.0 - valid_ratio
            if missing_ratio > max_missing_ratio:
                continue

            self.samples.append(CachedSample(x=x, observed_mask=observed_mask))

        if len(self.samples) == 0:
            raise ValueError(
                f"No samples kept for modality_key={modality_key}. "
                "Try increasing max_missing_ratio or changing data constraints."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.samples[idx]
        return s.x, s.observed_mask


def resolve_modality_key(train_modality: str, use_aligned: bool) -> Tuple[List[str], str]:
    if train_modality not in AIREADI_MODALITY_SPECS:
        raise ValueError(f"Unknown train_modality: {train_modality}")
    if use_aligned and train_modality in ALIGNED_MODALITY_KEY:
        return ["glucose", train_modality], ALIGNED_MODALITY_KEY[train_modality]
    return [train_modality], train_modality


def build_base_dataset(train_args: Dict, split: str, modalities: List[str]) -> AIREADIModalityImputationDataset:
    ts_seq_len = int(train_args["ts_seq_len"])
    anchor_modality = str(train_args["anchor_modality"])
    max_events = {m: ts_seq_len for m in modalities}
    min_events = {anchor_modality: int(train_args["daily_min_events"])}

    return AIREADIModalityImputationDataset(
        root=str(train_args["data_root"]),
        split=split,
        modalities=modalities,
        anchor_modality=anchor_modality,
        target_modality=anchor_modality,
        window_size=ts_seq_len,
        window_stride=ts_seq_len,
        window_mode="daily",
        daily_min_events=int(train_args["daily_min_events"]),
        max_events_per_modality=max_events,
        min_events_per_modality=min_events,
        normalize=not bool(train_args.get("raw_values", False)),
        max_anchor_gap_minutes=float(train_args["max_anchor_gap_minutes"]),
        max_window_span_hours=float(train_args["max_window_span_hours"]),
        anchor_sampling_minutes=float(train_args["anchor_sampling_minutes"]),
        anchor_sampling_tolerance_seconds=float(train_args["anchor_sampling_tolerance_seconds"]),
        participants_tsv_path=train_args.get("participants_tsv_path"),
        include_clinical_static=False,
        include_participant_metadata=train_args.get("participants_tsv_path") is not None,
        include_study_group=train_args.get("participants_tsv_path") is not None,
        include_clinical_site=False,
        pad=True,
        return_dict=True,
    )


@torch.no_grad()
def save_without_input_mask_imputed_plots(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    out_dir: Path,
    num_samples: int,
    channel_idx: int,
    tag: str,
) -> List[str]:
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []

    try:
        x, observed_mask = next(iter(loader))
    except StopIteration:
        return saved

    x = x.to(device)
    observed_mask = observed_mask.to(device)
    input_mask_without = torch.zeros_like(observed_mask)

    pred = model(x, observed_mask=observed_mask, input_mask=input_mask_without)
    # Without input_mask: copy observed values, and fill dataset-missing positions by prediction.
    recon_imputed = pred * (1.0 - observed_mask) + x * observed_mask

    bsz = min(num_samples, x.size(0))
    for i in range(bsz):
        real = x[i].detach().cpu().numpy()[:, channel_idx]
        recon = recon_imputed[i].detach().cpu().numpy()[:, channel_idx]
        obs = observed_mask[i].detach().cpu().numpy()[:, channel_idx] > 0
        t = np.arange(real.shape[0])

        fig, ax = plt.subplots(figsize=(10, 4.2))
        ax.plot(t, real, label="real", linewidth=1.2)
        ax.plot(t, recon, label="recon_without_input_mask_imputed", linewidth=1.2, alpha=0.9)
        if np.any(~obs):
            ax.scatter(t[~obs], real[~obs], s=12, marker="x", label="missing_in_dataset")
        ax.set_title(f"Reconstruction (without_input_mask_imputed) | {tag} | sample={i}")
        ax.set_xlabel("Time Index")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        out_path = out_dir / f"{tag}_sample_{i}_without_input_mask_imputed.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        saved.append(str(out_path))

    return saved


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize irregular_ts_ae best.pt with without_input_mask imputation view.")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--split", type=str, default=None, choices=["train", "valid", "test"], help="Dataset split override")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--channel_idx", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--out_dir", type=str, default=None, help="Output dir override")
    return parser.parse_args()


def main() -> None:
    args = get_args()
    ckpt_path = Path(args.ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "model_state_dict" not in ckpt or "args" not in ckpt:
        raise ValueError("Checkpoint must contain model_state_dict and args.")

    train_args: Dict = ckpt["args"]
    train_modality = str(ckpt.get("train_modality", train_args["train_modality"]))
    use_aligned = bool(train_args.get("use_aligned_modality", False))
    source_modalities, modality_key = resolve_modality_key(train_modality, use_aligned)

    split = args.split if args.split is not None else str(train_args.get("val_split", "test"))
    base = build_base_dataset(train_args, split=split, modalities=source_modalities)
    dataset = IrregularTSDataContainer(
        base_dataset=base,
        modality_key=modality_key,
        max_missing_ratio=float(train_args.get("max_missing_ratio", 0.8)),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    sample_x, _ = dataset[0]
    input_dim = int(sample_x.shape[-1])
    model = IrregularTimeSeriesAE(
        input_dim=input_dim,
        d_model=int(train_args["d_model"]),
        nhead=int(train_args["nhead"]),
        num_layers=int(train_args["num_layers"]),
        max_len=int(train_args["ts_seq_len"]),
    )
    model.load_state_dict(ckpt["model_state_dict"])

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model = model.to(device)

    default_out = ckpt_path.parent / "recon_plots_imputed"
    out_dir = Path(args.out_dir) if args.out_dir is not None else default_out
    tag = f"ckpt_epoch_{int(ckpt.get('epoch', -1)):04d}_{split}"
    saved = save_without_input_mask_imputed_plots(
        model=model,
        loader=loader,
        device=device,
        out_dir=out_dir,
        num_samples=args.num_samples,
        channel_idx=args.channel_idx,
        tag=tag,
    )

    print(f"Saved {len(saved)} plot(s) to: {out_dir}")
    for p in saved:
        print(p)


if __name__ == "__main__":
    main()

