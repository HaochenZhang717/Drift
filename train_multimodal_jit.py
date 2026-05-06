import argparse
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
try:
    import wandb
except ImportError:
    wandb = None

from img_transformations import DelayEmbedder
from utils.utils_dataset import AI_READI_STUDY_GROUPS, AIREADIModalityImputationDataset
from utils.utils_drift import count_parameters
from models.multimodal_jit.denoiser import Denoiser


MODALITY_KEYS = {
    "heart_rate": "heart_rate_aligned_to_glucose",
    "calorie": "calorie_aligned_to_glucose",
    "physical_activity": "physical_activity_aligned_to_glucose",
    "respiratory_rate": "respiratory_rate_aligned_to_glucose",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _select_modality_pack(batch: Dict[str, Any], modality: str) -> Dict[str, torch.Tensor]:
    packs = batch["modalities"]
    preferred_key = MODALITY_KEYS[modality]
    if preferred_key in packs:
        return packs[preferred_key]
    if modality in packs:
        return packs[modality]
    raise KeyError(f"Missing modality pack for {modality}; checked keys: {preferred_key}, {modality}")


def build_dataset(args: argparse.Namespace, split: str) -> AIREADIModalityImputationDataset:
    modalities = ["glucose", "heart_rate", "calorie", "physical_activity", "respiratory_rate"]
    max_events = {m: args.ts_seq_len for m in modalities}
    min_events = {"glucose": args.daily_min_events}

    return AIREADIModalityImputationDataset(
        root=args.data_root,
        split=split,
        modalities=modalities,
        anchor_modality="glucose",
        target_modality="glucose",
        window_size=args.ts_seq_len,
        window_stride=args.ts_seq_len,
        window_mode=args.window_mode,
        daily_min_events=args.daily_min_events,
        max_events_per_modality=max_events,
        min_events_per_modality=min_events,
        normalize=True,
        max_anchor_gap_minutes=args.max_anchor_gap_minutes,
        max_window_span_hours=args.max_window_span_hours,
        anchor_sampling_minutes=args.anchor_sampling_minutes,
        anchor_sampling_tolerance_seconds=args.anchor_sampling_tolerance_seconds,
        participants_tsv_path=args.participants_tsv_path,
        include_clinical_static=False,
        include_participant_metadata=True,
        include_study_group=True,
        include_clinical_site=False,
        pad=True,
        return_dict=True,
    )


def batch_to_model_inputs(
    batch: Dict[str, Any],
    delay_embedder: DelayEmbedder,
    num_classes: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    labels = batch["study_group_label"].long()
    valid = (labels >= 0) & (labels < num_classes)
    if not valid.any():
        return {}

    labels = labels[valid].to(device)

    target = batch["target"][valid].float().to(device)
    if target.ndim == 2:
        target = target.unsqueeze(-1)
    x, x_mask = delay_embedder.ts_to_img(target, return_pad_mask=True)

    hr = _select_modality_pack(batch, "heart_rate")
    cal = _select_modality_pack(batch, "calorie")
    pa = _select_modality_pack(batch, "physical_activity")
    rr = _select_modality_pack(batch, "respiratory_rate")

    return {
        "x": x,
        "x_mask": x_mask,
        "study_group": labels,
        "heart_rate": hr["values"][valid].float().to(device),
        "calorie": cal["values"][valid].float().to(device),
        "physical_activity": pa["values"][valid].float().to(device),
        "respiratory_rate": rr["values"][valid].float().to(device),
        "heart_rate_observed_mask": hr["mask"][valid].float().to(device),
        "calorie_observed_mask": cal["mask"][valid].float().to(device),
        "physical_activity_observed_mask": pa["mask"][valid].float().to(device),
        "respiratory_rate_observed_mask": rr["mask"][valid].float().to(device),
    }


def build_model_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        num_tokens_per_modality=args.num_tokens_per_modality,
        mm_dim_in=args.mm_dim_in,
        hidden_channels=args.hidden_channels,
        mm_n_heads=args.mm_n_heads,
        ae_input_dim=1,
        ae_d_model=args.ae_d_model,
        ae_max_len=args.ae_max_len,
        ae_nheads=args.ae_nheads,
        ae_num_layers=args.ae_num_layers,
        mm_missing_ratio_threshold=args.max_missing_ratio,
        ae_cpt_paths={
            "heart_rate": args.ckpt_heart_rate,
            "calorie": args.ckpt_calorie,
            "physical_activity": args.ckpt_physical_activity,
            "respiratory_rate": args.ckpt_respiratory_rate,
        },
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_channels=args.in_channels,
        depth=args.depth,
        num_heads=args.num_heads,
        attn_dropout=args.attn_dropout,
        proj_dropout=args.proj_dropout,
        bottleneck_dim=args.bottleneck_dim,
        in_context_start=args.in_context_start,
        label_drop_prob=args.label_drop_prob,
        P_mean=args.P_mean,
        P_std=args.P_std,
        t_eps=args.t_eps,
        noise_scale=args.noise_scale,
        ema_decay1=args.ema_decay1,
        ema_decay2=args.ema_decay2,
        sampling_method=args.sampling_method,
        num_sampling_steps=args.num_sampling_steps,
        cfg=args.cfg,
        interval_min=args.interval_min,
        interval_max=args.interval_max,
        num_classes=args.num_classes,
    )


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = build_dataset(args, args.train_split)
    val_dataset = build_dataset(args, args.val_split)
    print(f"Raw windows | train: {len(train_dataset)} | val: {len(val_dataset)}")
    print(f"Missing-token threshold (per modality): {args.max_missing_ratio:.2f}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    model_args = build_model_args(args)
    model = Denoiser(model_args).to(device)
    print(f"Trainable parameters: {count_parameters(model):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    delay_embedder = DelayEmbedder(device=device, seq_len=args.ts_seq_len, delay=args.ts_delay, embedding=args.ts_embedding)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wb = None
    if args.wandb:
        if wandb is None:
            raise ImportError("wandb is not installed. Please install wandb or run without --wandb.")
        wb = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
            dir=str(output_dir),
        )

    global_step = 0
    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_steps = 0

        for batch in train_loader:
            inputs = batch_to_model_inputs(batch, delay_embedder, args.num_classes, device)
            # if not inputs:
            #     continue
            # breakpoint()
            optimizer.zero_grad(set_to_none=True)
            loss = model(**inputs)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            loss_val = float(loss.item())
            epoch_loss += loss_val
            n_steps += 1
            global_step += 1

            if global_step % args.log_interval == 0:
                print(f"Epoch {epoch + 1}/{args.epochs} | Step {global_step} | Loss {loss_val:.6f}")
                if wb is not None:
                    wandb.log(
                        {
                            "train/loss_step": loss_val,
                            "train/lr": optimizer.param_groups[0]["lr"],
                            "epoch": epoch + 1,
                        },
                        step=global_step,
                    )

        avg_loss = epoch_loss / max(1, n_steps)
        model.eval()
        val_loss_sum = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch_to_model_inputs(batch, delay_embedder, args.num_classes, device)
                if not inputs:
                    continue
                val_loss = model(**inputs)
                val_loss_sum += float(val_loss.item())
                val_steps += 1
        avg_val_loss = val_loss_sum / max(1, val_steps)
        best_val_loss = min(best_val_loss, avg_val_loss)

        print(
            f"Epoch {epoch + 1} done | train_loss={avg_loss:.6f} | "
            f"val_loss={avg_val_loss:.6f} | best_val={best_val_loss:.6f} | steps={n_steps}"
        )
        if wb is not None:
            wandb.log(
                {
                    "train/loss_epoch": avg_loss,
                    "val/loss_epoch": avg_val_loss,
                    "val/best_loss": best_val_loss,
                    "epoch": epoch + 1,
                },
                step=global_step,
            )

        if (epoch + 1) % args.save_interval == 0:
            ckpt_path = output_dir / f"checkpoint_epoch{epoch + 1}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "step": global_step,
                    "best_val_loss": best_val_loss,
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"Saved: {ckpt_path}")

    final_path = output_dir / "checkpoint_final.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": args.epochs - 1,
            "step": global_step,
            "best_val_loss": best_val_loss,
            "args": vars(args),
        },
        final_path,
    )
    print(f"Training complete. Final checkpoint: {final_path}")
    if wb is not None:
        wb.finish()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train multimodal JiT denoiser on AI-READI modality-imputation dataset")

    parser.add_argument("--data_root", type=str, default="./AI-READI")
    parser.add_argument("--participants_tsv_path", type=str, default=None)
    parser.add_argument("--train_split", type=str, default="train", choices=["train", "valid", "test"])
    parser.add_argument("--val_split", type=str, default="valid", choices=["train", "valid", "test"])
    parser.add_argument("--output_dir", type=str, default="./outputs/multimodal_jit")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--save_interval", type=int, default=10)

    parser.add_argument("--img_size", type=int, default=32)
    parser.add_argument("--patch_size", type=int, default=2)
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--attn_dropout", type=float, default=0.0)
    parser.add_argument("--proj_dropout", type=float, default=0.0)
    parser.add_argument("--bottleneck_dim", type=int, default=64)
    parser.add_argument("--in_context_start", type=int, default=0)
    parser.add_argument("--num_classes", type=int, default=len(AI_READI_STUDY_GROUPS))

    parser.add_argument("--num_tokens_per_modality", type=int, default=4)
    parser.add_argument("--mm_dim_in", type=int, default=128)
    parser.add_argument("--mm_n_heads", type=int, default=4)
    parser.add_argument("--ae_d_model", type=int, default=128)
    parser.add_argument("--ae_nheads", type=int, default=4)
    parser.add_argument("--ae_num_layers", type=int, default=4)
    parser.add_argument("--ae_max_len", type=int, default=10000)

    parser.add_argument("--ckpt_heart_rate", type=str, required=True)
    parser.add_argument("--ckpt_calorie", type=str, required=True)
    parser.add_argument("--ckpt_physical_activity", type=str, required=True)
    parser.add_argument("--ckpt_respiratory_rate", type=str, required=True)

    parser.add_argument("--label_drop_prob", type=float, default=0.1)
    parser.add_argument("--P_mean", type=float, default=-1.2)
    parser.add_argument("--P_std", type=float, default=1.2)
    parser.add_argument("--t_eps", type=float, default=1e-5)
    parser.add_argument("--noise_scale", type=float, default=1.0)
    parser.add_argument("--ema_decay1", type=float, default=0.999)
    parser.add_argument("--ema_decay2", type=float, default=0.9999)

    parser.add_argument("--sampling_method", type=str, default="euler", choices=["euler", "heun"])
    parser.add_argument("--num_sampling_steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--interval_min", type=float, default=0.0)
    parser.add_argument("--interval_max", type=float, default=1.0)

    parser.add_argument("--ts_seq_len", type=int, default=288)
    parser.add_argument("--ts_delay", type=int, default=1)
    parser.add_argument("--ts_embedding", type=int, default=32)
    parser.add_argument("--window_mode", type=str, default="daily", choices=["daily", "sliding"])
    parser.add_argument("--daily_min_events", type=int, default=288)
    parser.add_argument("--max_missing_ratio", type=float, default=0.8)
    parser.add_argument("--max_anchor_gap_minutes", type=float, default=30.0)
    parser.add_argument("--max_window_span_hours", type=float, default=24.0)
    parser.add_argument("--anchor_sampling_minutes", type=float, default=5.0)
    parser.add_argument("--anchor_sampling_tolerance_seconds", type=float, default=120.0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="drifting-model")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    train(args)
