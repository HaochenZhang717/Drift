"""
Train MultiModalDrift on AI-READI daily glucose with study-group + 4-modality conditioning.
Style follows train_ts_uncond_daily.py / train_ts_cond_daily.py.
"""

import argparse
import random
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from img_transformations import DelayEmbedder
from drifting import compute_V
from ts_quality_eval import delay_images_to_series
from utils.utils_dataset import AI_READI_STUDY_GROUPS, AIREADIModalityImputationDataset
from utils.utils_drift import EMA, WarmupLRScheduler, count_parameters, save_checkpoint, set_seed

from models.modality_imputation.multi_modal_drift import MultiModalDrift
import wandb


MODALITY_KEYS = {
    "heart_rate": "heart_rate_aligned_to_glucose",
    "calorie": "calorie_aligned_to_glucose",
    "physical_activity": "physical_activity_aligned_to_glucose",
    "respiratory_rate": "respiratory_rate_aligned_to_glucose",
}


def parse_temperatures(value: str) -> list[float]:
    temps = [float(x) for x in value.split(",") if x.strip()]
    if not temps:
        raise argparse.ArgumentTypeError("temperatures must contain at least one value")
    return temps


def make_delay_embedder(config: dict, device: torch.device) -> DelayEmbedder:
    return DelayEmbedder(
        device=device,
        seq_len=config["ts_seq_len"],
        delay=config["ts_delay"],
        embedding=config["ts_embedding"],
    )


def target_series_to_images(series: torch.Tensor, config: dict, device: torch.device) -> torch.Tensor:
    if series.ndim == 2:
        series = series.unsqueeze(-1)
    return make_delay_embedder(config, device).ts_to_img(series.to(device))


def compute_drifting_loss(
    x_gen: torch.Tensor,
    labels_gen: torch.Tensor,
    x_pos: torch.Tensor,
    labels_pos: torch.Tensor,
    temperatures: list[float],
    ts_loss_config: Optional[dict] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    device = x_gen.device
    num_classes = int(labels_gen.max().item()) + 1

    rep_gen = delay_images_to_series(x_gen, ts_loss_config, device)
    rep_pos = delay_images_to_series(x_pos, ts_loss_config, device)

    feat_gen = rep_gen.flatten(start_dim=1)
    feat_pos = rep_pos.flatten(start_dim=1)

    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    total_drift_norm = 0.0
    num_losses = 0

    for c in range(num_classes):
        mask_gen = labels_gen == c
        mask_pos = labels_pos == c
        if not mask_gen.any() or not mask_pos.any():
            continue

        gen_c = feat_gen[mask_gen]
        pos_c = feat_pos[mask_pos]
        neg_c = gen_c

        v_total = torch.zeros_like(gen_c)
        for tau in temperatures:
            v_tau = compute_V(gen_c, pos_c, neg_c, tau, mask_self=True)
            v_norm = torch.sqrt(torch.mean(v_tau ** 2) + 1e-8)
            v_tau = v_tau / (v_norm + 1e-8)
            v_total = v_total + v_tau

        target = (gen_c + v_total).detach()
        loss_c = F.mse_loss(gen_c, target)

        total_loss = total_loss + loss_c
        total_drift_norm += (v_total ** 2).mean().item() ** 0.5
        num_losses += 1

    if num_losses == 0:
        return (
            torch.tensor(0.0, device=device, requires_grad=True),
            {"loss": 0.0, "drift_norm": 0.0},
        )

    loss = total_loss / num_losses
    return loss, {"loss": loss.item(), "drift_norm": total_drift_norm / num_losses}


class MultiModalConditionalQueue:
    """Per-class queue storing real glucose images + multimodal condition tensors."""

    def __init__(self, num_classes: int, queue_size: int):
        self.num_classes = num_classes
        self.buffers = [deque(maxlen=queue_size) for _ in range(num_classes)]

    def add(self, images: torch.Tensor, labels: torch.Tensor, cond: Dict[str, torch.Tensor]):
        n = images.shape[0]
        for i in range(n):
            cls = int(labels[i].item())
            if cls < 0 or cls >= self.num_classes:
                continue
            item = {
                "image": images[i].detach().cpu(),
                "heart_rate": cond["heart_rate"][i].detach().cpu(),
                "calorie": cond["calorie"][i].detach().cpu(),
                "physical_activity": cond["physical_activity"][i].detach().cpu(),
                "respiratory_rate": cond["respiratory_rate"][i].detach().cpu(),
                "heart_rate_observed_mask": cond["heart_rate_observed_mask"][i].detach().cpu(),
                "calorie_observed_mask": cond["calorie_observed_mask"][i].detach().cpu(),
                "physical_activity_observed_mask": cond["physical_activity_observed_mask"][i].detach().cpu(),
                "respiratory_rate_observed_mask": cond["respiratory_rate_observed_mask"][i].detach().cpu(),
            }
            self.buffers[cls].append(item)

    def is_ready(self, n_per_class: int) -> bool:
        return all(len(buf) >= n_per_class for buf in self.buffers)

    def sample_positive_images(self, n_per_class: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        xs = []
        ys = []
        for c in range(self.num_classes):
            picks = random.sample(list(self.buffers[c]), n_per_class)
            xs.append(torch.stack([p["image"] for p in picks], dim=0))
            ys.append(torch.full((n_per_class,), c, dtype=torch.long))
        x = torch.cat(xs, dim=0).to(device)
        y = torch.cat(ys, dim=0).to(device)
        return x, y

    def sample_conditions(self, n_per_class: int, device: torch.device) -> Dict[str, torch.Tensor]:
        out: Dict[str, list] = {
            "heart_rate": [],
            "calorie": [],
            "physical_activity": [],
            "respiratory_rate": [],
            "heart_rate_observed_mask": [],
            "calorie_observed_mask": [],
            "physical_activity_observed_mask": [],
            "respiratory_rate_observed_mask": [],
        }
        for c in range(self.num_classes):
            picks = random.sample(list(self.buffers[c]), n_per_class)
            for k in out.keys():
                out[k].extend([p[k] for p in picks])
        return {k: torch.stack(v, dim=0).to(device) for k, v in out.items()}


def _select_modality_pack(batch: Dict[str, Any], modality: str) -> Dict[str, torch.Tensor]:
    packs = batch["modalities"]
    preferred_key = MODALITY_KEYS[modality]
    if preferred_key in packs:
        return packs[preferred_key]
    if modality in packs:
        return packs[modality]
    raise KeyError(f"Missing modality pack for {modality}; checked keys: {preferred_key}, {modality}")


def batch_to_training_tensors(
    batch: Dict[str, Any],
    config: dict,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    labels = batch["study_group_label"].long()
    valid = (labels >= 0) & (labels < config["num_classes"])
    if not valid.any():
        return (
            torch.empty(0),
            torch.empty(0, dtype=torch.long),
            {},
        )

    labels = labels[valid]
    target = batch["target"][valid]
    images = target_series_to_images(target, config, device).detach().cpu()

    hr = _select_modality_pack(batch, "heart_rate")
    cal = _select_modality_pack(batch, "calorie")
    pa = _select_modality_pack(batch, "physical_activity")
    rr = _select_modality_pack(batch, "respiratory_rate")

    cond = {
        "heart_rate": hr["values"][valid].float(),
        "calorie": cal["values"][valid].float(),
        "physical_activity": pa["values"][valid].float(),
        "respiratory_rate": rr["values"][valid].float(),
        "heart_rate_observed_mask": hr["mask"][valid].float(),
        "calorie_observed_mask": cal["mask"][valid].float(),
        "physical_activity_observed_mask": pa["mask"][valid].float(),
        "respiratory_rate_observed_mask": rr["mask"][valid].float(),
    }

    return images, labels.detach().cpu(), cond


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    queue: MultiModalConditionalQueue,
    config: dict,
    device: torch.device,
) -> Dict[str, float]:
    model.train()

    n_neg = config["batch_n_neg"]
    n_pos = config["batch_n_pos"]
    num_classes = config["num_classes"]

    labels = torch.arange(num_classes, device=device).repeat_interleave(n_neg)
    cond = queue.sample_conditions(n_neg, device)

    noise = torch.randn(
        labels.shape[0],
        config["in_channels"],
        config["img_size"],
        config["img_size"],
        device=device,
    )

    x_gen = model(
        x=noise,
        study_group=labels,
        heart_rate=cond["heart_rate"],
        calorie=cond["calorie"],
        physical_activity=cond["physical_activity"],
        respiratory_rate=cond["respiratory_rate"],
        heart_rate_observed_mask=cond["heart_rate_observed_mask"],
        calorie_observed_mask=cond["calorie_observed_mask"],
        physical_activity_observed_mask=cond["physical_activity_observed_mask"],
        respiratory_rate_observed_mask=cond["respiratory_rate_observed_mask"],
    )

    x_pos, labels_pos = queue.sample_positive_images(n_pos, device)

    loss, info = compute_drifting_loss(
        x_gen=x_gen,
        labels_gen=labels,
        x_pos=x_pos,
        labels_pos=labels_pos,
        temperatures=config["temperatures"],
        ts_loss_config=config,
    )

    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
    optimizer.step()

    info["grad_norm"] = float(grad_norm.item())
    return info


def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "dataset": "aireadi_imputation",
        "img_size": args.img_size,
        "in_channels": args.in_channels,
        "num_classes": args.num_classes,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "batch_n_pos": args.batch_n_pos,
        "batch_n_neg": args.batch_n_neg,
        "queue_size": args.queue_size,
        "temperatures": args.temperatures,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "ema_decay": args.ema_decay,
        "warmup_steps": args.warmup_steps,
        "ts_seq_len": args.ts_seq_len,
        "ts_delay": args.ts_delay,
        "ts_embedding": args.ts_embedding,
        "ts_stride": args.ts_stride,
        "window_mode": args.window_mode,
        "daily_min_events": args.daily_min_events,
        "modalities": ["glucose", "heart_rate", "calorie", "physical_activity", "respiratory_rate"],
        "anchor_modality": "glucose",
        "target_modality": "glucose",
        "max_events_per_modality": {"glucose": args.ts_seq_len},
        "min_events_per_modality": {"glucose": args.daily_min_events or args.ts_seq_len},
        "max_anchor_gap_minutes": args.max_anchor_gap_minutes,
        "max_window_span_hours": args.max_window_span_hours,
        "anchor_sampling_minutes": args.anchor_sampling_minutes,
        "anchor_sampling_tolerance_seconds": args.anchor_sampling_tolerance_seconds,
        "clinical_root": args.clinical_root,
        "participants_tsv_path": args.participants_tsv_path,
        "include_clinical_static": False,
        "include_participant_metadata": True,
        "include_study_group": True,
        "include_clinical_site": False,
    }

def build_base_dataset(
    args: argparse.Namespace,
    split: str,
    modalities: list[str],
) -> AIREADIModalityImputationDataset:
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
        include_participant_metadata=args.participants_tsv_path is not None,
        include_study_group=True,
        include_clinical_site=False,
        pad=True,
        return_dict=True,
    )


def train(args: argparse.Namespace) -> None:
    config = build_config(args)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    modalities = config["modalities"]
    train_dataset = build_base_dataset(
        args=args,
        split=args.train_split,
        modalities=modalities,
    )
    val_dataset = build_base_dataset(
        args=args,
        split=args.val_split,
        modalities=modalities,
    )
    print(f"Dataset sizes | train: {len(train_dataset)} | val: {len(val_dataset)}")
    max_items = min(len(train_dataset), 1024)
    valid_labels = 0
    for idx in range(max_items):
        item = train_dataset[idx]
        label = int(item.get("study_group_label", torch.tensor(-1)).item())
        if 0 <= label < args.num_classes:
            valid_labels += 1
    if valid_labels == 0:
        raise ValueError(
            "No valid study_group_label found in train split. "
            "Please provide --participants_tsv_path with study-group metadata."
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    mm_encoder_kwargs = {
        "n_tokens": args.mm_n_tokens,
        "dim_in": args.mm_dim,
        "dim_out": args.hidden_size,
        "nhead_cross": args.mm_nhead_cross,
        "ae_input_dim": 1,
        "ae_d_model": args.mm_ae_d_model,
        "ae_nhead": args.mm_ae_nhead,
        "ae_num_layers": args.mm_ae_num_layers,
        "ae_max_len": args.ts_seq_len,
        "ckpt_paths": {
            "heart_rate": args.ckpt_heart_rate,
            "calorie": args.ckpt_calorie,
            "physical_activity": args.ckpt_physical_activity,
            "respiratory_rate": args.ckpt_respiratory_rate,
        },
        "strict_load": True,
    }

    model = MultiModalDrift(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_channels=args.in_channels,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        num_study_groups=args.num_classes,
        label_dropout=args.label_dropout,
        num_register_tokens=args.num_register_tokens,
        use_alpha_embed=False,
        multi_modal_encoder_kwargs=mm_encoder_kwargs,
    ).to(device)
    print(f"Model params: {count_parameters(model):,}")

    ema = EMA(model, decay=args.ema_decay)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    scheduler = WarmupLRScheduler(optimizer, warmup_steps=args.warmup_steps, base_lr=args.lr)

    queue = MultiModalConditionalQueue(num_classes=args.num_classes, queue_size=args.queue_size)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wb = None
    if args.wandb and wandb is not None:
        wb = wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args), dir=str(output_dir))

    global_step = 0
    for epoch in range(args.epochs):
        start = time.time()
        epoch_loss = 0.0
        n_steps = 0

        for batch in train_loader:
            images_cpu, labels_cpu, cond = batch_to_training_tensors(batch, config, device)
            if images_cpu.numel() == 0:
                continue
            queue.add(images_cpu, labels_cpu, cond)

            if not queue.is_ready(args.batch_n_pos):
                continue

            info = train_step(model, optimizer, queue, config, device)
            ema.update(model)
            scheduler.step()
            global_step += 1
            n_steps += 1
            epoch_loss += info["loss"]

            if global_step % args.log_interval == 0:
                lr = scheduler.get_lr()
                print(
                    f"Epoch {epoch+1}/{args.epochs} | Step {global_step} | "
                    f"Loss {info['loss']:.4f} | Drift {info['drift_norm']:.4f} | "
                    f"Grad {info['grad_norm']:.4f} | LR {lr:.6f}"
                )
                if wb is not None:
                    wandb.log({
                        "train/loss": info["loss"],
                        "train/drift_norm": info["drift_norm"],
                        "train/grad_norm": info["grad_norm"],
                        "train/lr": lr,
                        "train/epoch": epoch + 1,
                    }, step=global_step)

        avg_loss = epoch_loss / max(1, n_steps)
        print(f"Epoch {epoch+1} finished in {time.time()-start:.1f}s | avg_loss={avg_loss:.4f}")

        if (epoch + 1) % args.save_interval == 0:
            ckpt_path = output_dir / f"checkpoint_epoch{epoch+1}.pt"
            save_checkpoint(
                str(ckpt_path),
                model,
                ema,
                optimizer,
                scheduler,
                epoch,
                global_step,
                config,
            )
            print(f"Saved checkpoint: {ckpt_path}")

    final_path = output_dir / "checkpoint_final.pt"
    save_checkpoint(
        str(final_path),
        model,
        ema,
        optimizer,
        scheduler,
        args.epochs - 1,
        global_step,
        config,
    )
    print(f"Training complete. Final checkpoint: {final_path}")

    if wb is not None:
        wb.finish()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MultiModalDrift on AI-READI glucose daily windows")

    parser.add_argument("--data_root", type=str, default="./AI-READI")
    parser.add_argument("--train_split", type=str, default="train", choices=["train", "valid", "test"])
    parser.add_argument("--val_split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--output_dir", type=str, default="./outputs/glucose_multi_modal_cond")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--img_size", type=int, default=32)
    parser.add_argument("--patch_size", type=int, default=2)
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)
    parser.add_argument("--num_register_tokens", type=int, default=8)
    parser.add_argument("--num_classes", type=int, default=len(AI_READI_STUDY_GROUPS))
    parser.add_argument("--label_dropout", type=float, default=0.1)

    parser.add_argument("--mm_n_tokens", type=int, default=4)
    parser.add_argument("--mm_dim", type=int, default=128)
    parser.add_argument("--mm_nhead_cross", type=int, default=4)
    parser.add_argument("--mm_ae_d_model", type=int, default=128)
    parser.add_argument("--mm_ae_nhead", type=int, default=4)
    parser.add_argument("--mm_ae_num_layers", type=int, default=4)

    parser.add_argument("--ckpt_heart_rate", type=str, required=True)
    parser.add_argument("--ckpt_calorie", type=str, required=True)
    parser.add_argument("--ckpt_physical_activity", type=str, required=True)
    parser.add_argument("--ckpt_respiratory_rate", type=str, required=True)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--warmup_steps", type=int, default=1000)

    parser.add_argument("--batch_n_pos", type=int, default=8)
    parser.add_argument("--batch_n_neg", type=int, default=8)
    parser.add_argument("--queue_size", type=int, default=2048)
    parser.add_argument("--temperatures", type=parse_temperatures, default=[0.1, 0.2])

    parser.add_argument("--ts_seq_len", type=int, default=288)
    parser.add_argument("--ts_delay", type=int, default=1)
    parser.add_argument("--ts_embedding", type=int, default=32)
    parser.add_argument("--ts_stride", type=int, default=32)
    parser.add_argument("--window_mode", type=str, default="daily", choices=["daily", "sliding"])
    parser.add_argument("--daily_min_events", type=int, default=288)
    parser.add_argument("--max_anchor_gap_minutes", type=float, default=30.0)
    parser.add_argument("--max_window_span_hours", type=float, default=24.0)
    parser.add_argument("--anchor_sampling_minutes", type=float, default=5.0)
    parser.add_argument("--anchor_sampling_tolerance_seconds", type=float, default=120.0)

    parser.add_argument("--clinical_root", type=str, default=None)
    parser.add_argument("--participants_tsv_path", type=str, default=None)

    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=10)

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="drifting-model-ts")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    train(args)
