"""
Training script for Drifting Models with AI-READI study-group-conditional
glucose time-series generation.
"""
import argparse
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from img_transformations import DelayEmbedder
from utils.utils_dataset import AI_READI_STUDY_GROUPS, get_dataset
from models.cls_cond_model import DriftDiT_models
from drifting import compute_V
# from feature_extractors.ts2vec.models.encoder import TSEncoder
from utils.utils_drift import (
    EMA,
    WarmupLRScheduler,
    save_checkpoint,
    count_parameters,
    set_seed,
)

from utils.utils_drift import ClsCondSampleQueue as ConditionalSampleQueue

from ts_quality_eval import (
    delay_images_to_series,
    parse_metric_names,
)

try:
    import wandb
except ImportError:
    wandb = None


def parse_temperatures(value: str) -> list:
    """Parse a comma-separated temperature list."""
    temperatures = [float(item) for item in value.split(",") if item.strip()]
    if not temperatures:
        raise argparse.ArgumentTypeError("temperatures must contain at least one value")
    return temperatures


def parse_modalities(value: str) -> list[str]:
    """Parse comma-separated AI-READI modalities."""
    modalities = [item.strip() for item in value.split(",") if item.strip()]
    if not modalities:
        raise argparse.ArgumentTypeError("modalities must contain at least one value")
    return modalities


def delay_embedding_num_cols(seq_len: int, delay: int, embedding: int) -> int:
    """Return the delay-image column count for the configured time-series length."""
    col = 0
    while (col * delay + embedding) <= seq_len:
        col += 1
    if (
        col < embedding
        and col * delay != seq_len
        and col * delay + embedding > seq_len
    ):
        col += 1
    return max(1, col)


def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Build the training config from parsed argparse values."""
    num_delay_cols = delay_embedding_num_cols(
        args.ts_seq_len,
        args.ts_delay,
        args.ts_embedding,
    )
    if num_delay_cols > args.ts_embedding:
        raise ValueError(
            "Delay embedding configuration does not fit in a square image: "
            f"ts_seq_len={args.ts_seq_len}, ts_delay={args.ts_delay}, "
            f"ts_embedding={args.ts_embedding} requires {num_delay_cols} columns. "
            "Increase --ts_embedding/--img_size or increase --ts_delay."
        )

    config_keys = [
        "model",
        "img_size",
        "in_channels",
        "num_classes",
        "batch_n_pos",
        "batch_n_neg",
        "temperatures",
        "lr",
        "weight_decay",
        "grad_clip",
        "ema_decay",
        "warmup_steps",
        "epochs",
        "use_feature_encoder",
        "loss_domain",
        "queue_size",
        "label_dropout",
        "alpha_min",
        "alpha_max",
        "cfg_sample_alpha",
        "ts_seq_len",
        "ts_delay",
        "ts_embedding",
        "ts_stride",
        "window_mode",
        "daily_min_events",
        "modalities",
        "anchor_modality",
        "target_modality",
        "max_anchor_gap_minutes",
        "max_window_span_hours",
        "anchor_sampling_minutes",
        "anchor_sampling_tolerance_seconds",
        "clinical_root",
        "participants_tsv_path",
        "include_clinical_static",
        "include_participant_metadata",
        "include_study_group",
        "include_clinical_site",
        "eval_per_class_samples",
    ]
    config = {key: getattr(args, key) for key in config_keys}
    config["dataset"] = "aireadi_imputation"
    config["study_group_names"] = list(AI_READI_STUDY_GROUPS)
    config["max_events_per_modality"] = {"glucose": args.ts_seq_len}
    min_glucose_events = args.daily_min_events or args.ts_seq_len
    config["min_events_per_modality"] = {"glucose": min_glucose_events}
    if args.window_mode == "daily" and config["max_window_span_hours"] is not None:
        config["max_window_span_hours"] = max(float(config["max_window_span_hours"]), 24.0)
    return config


def _strip_module_prefix(state_dict: dict) -> dict:
    if not state_dict:
        return state_dict
    if not all(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k[len("module."):]: v for k, v in state_dict.items()}


# def load_ts_feature_encoder_from_ckpt(
#     ckpt_path: str,
#     device: torch.device,
# ) -> nn.Module:
#     """Load a frozen TS encoder checkpoint saved by full-series TS2Vec training."""
#     ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
#     metadata = ckpt.get("metadata", {})
#     required = ["input_dims", "output_dims", "hidden_dims", "depth", "mask_prob"]
#     missing = [key for key in required if key not in metadata]
#     if missing:
#         raise ValueError(
#             f"Missing fields {missing} in TS encoder checkpoint metadata at {ckpt_path}"
#         )
#
#     encoder = TSEncoder(
#         input_dims=metadata["input_dims"],
#         output_dims=metadata["output_dims"],
#         hidden_dims=metadata["hidden_dims"],
#         depth=metadata["depth"],
#         mask_prob=metadata["mask_prob"],
#     ).to(device)
#
#     state_dict = ckpt.get("model_state_dict")
#     if state_dict is None:
#         raise ValueError(f"Checkpoint at {ckpt_path} does not contain model_state_dict")
#     encoder.load_state_dict(_strip_module_prefix(state_dict), strict=True)
#     encoder.eval()
#     for param in encoder.parameters():
#         param.requires_grad = False
#     return encoder


def make_delay_embedder(config: dict, device: torch.device) -> DelayEmbedder:
    return DelayEmbedder(
        device=device,
        seq_len=config["ts_seq_len"],
        delay=config["ts_delay"],
        embedding=config["ts_embedding"],
    )


def target_series_to_images(
    series: torch.Tensor,
    config: dict,
    device: torch.device,
) -> torch.Tensor:
    """Convert glucose target sequences from AI-READI batches to delay images."""
    if series.ndim == 2:
        series = series.unsqueeze(-1)
    embedder = make_delay_embedder(config, device)
    return embedder.ts_to_img(series.to(device))


def sample_batch(
    queue: ConditionalSampleQueue,
    num_classes: int,
    n_pos: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample positive delay-image examples from every study group."""
    x_pos_list = []
    labels_list = []

    for c in range(num_classes):
        x_c = queue.sample(c, n_pos, device)
        x_pos_list.append(x_c)
        labels_list.append(torch.full((n_pos,), c, device=device, dtype=torch.long))

    return torch.cat(x_pos_list, dim=0), torch.cat(labels_list, dim=0)


def compute_drifting_loss(
    x_gen: torch.Tensor,
    labels_gen: torch.Tensor,
    x_pos: torch.Tensor,
    labels_pos: torch.Tensor,
    feature_encoder: Optional[nn.Module],
    temperatures: list,
    ts_loss_config: Optional[dict] = None,
) -> tuple[torch.Tensor, dict]:
    """Compute study-group-conditional drifting loss."""
    device = x_gen.device
    num_classes = int(labels_gen.max().item()) + 1

    rep_gen = delay_images_to_series(x_gen, ts_loss_config, device)
    rep_pos = delay_images_to_series(x_pos, ts_loss_config, device)

    if feature_encoder is None:
        feat_gen_list = [rep_gen.flatten(start_dim=1)]
        feat_pos_list = [rep_pos.flatten(start_dim=1)]
    else:
        raise NotImplementedError
        # feat_gen_seq = feature_encoder(rep_gen, mask="all_true")
        # with torch.no_grad():
        #     feat_pos_seq = feature_encoder(rep_pos, mask="all_true")
        #
        # if not isinstance(feature_encoder, TSEncoder):
        #     raise ValueError("feature_encoder must be an instance of [TSEncoder,]")
        #
        # feat_gen = F.max_pool1d(
        #     feat_gen_seq.transpose(1, 2),
        #     kernel_size=feat_gen_seq.size(1),
        # ).squeeze(-1)
        # feat_pos = F.max_pool1d(
        #     feat_pos_seq.transpose(1, 2),
        #     kernel_size=feat_pos_seq.size(1),
        # ).squeeze(-1)
        # feat_gen_list = [F.normalize(feat_gen, p=2, dim=1)]
        # feat_pos_list = [F.normalize(feat_pos, p=2, dim=1)]

    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    total_drift_norm = 0.0
    num_losses = 0

    for c in range(num_classes):
        mask_gen = labels_gen == c
        mask_pos = labels_pos == c
        if not mask_gen.any() or not mask_pos.any():
            continue

        # Compute loss at each scale
        for scale_idx, (feat_gen, feat_pos) in enumerate(zip(feat_gen_list, feat_pos_list)):
            feat_gen_c = feat_gen[mask_gen]
            feat_pos_c = feat_pos[mask_pos]

            # Negatives: generated samples from current class (following Algorithm 1: y_neg = x)
            feat_neg_c = feat_gen_c

            # Simple L2 normalization (projects to unit sphere)
            feat_gen_c_norm = feat_gen_c
            feat_pos_c_norm = feat_pos_c
            feat_neg_c_norm = feat_neg_c

            # Compute V with multiple temperatures
            V_total = torch.zeros_like(feat_gen_c_norm)
            for tau in temperatures:
                V_tau = compute_V(
                    feat_gen_c_norm,
                    feat_pos_c_norm,
                    feat_neg_c_norm,
                    tau,
                    mask_self=True,  # y_neg = x, so mask self
                )
                # Normalize each V before summing
                v_norm = torch.sqrt(torch.mean(V_tau ** 2) + 1e-8)
                V_tau = V_tau / (v_norm + 1e-8)
                V_total = V_total + V_tau

            # Loss: MSE(phi(x), stopgrad(phi(x) + V))
            target = (feat_gen_c_norm + V_total).detach()
            loss_scale = F.mse_loss(feat_gen_c_norm, target)

            total_loss = total_loss + loss_scale
            total_drift_norm += (V_total ** 2).mean().item() ** 0.5
            num_losses += 1

    if num_losses == 0:
        return (
            torch.tensor(0.0, device=device, requires_grad=True),
            {"loss": 0.0, "drift_norm": 0.0},
        )

    loss = total_loss / num_losses
    return loss, {
        "loss": loss.item(),
        "drift_norm": total_drift_norm / num_losses,
    }


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    queue: ConditionalSampleQueue,
    config: dict,
    device: torch.device,
    feature_encoder: Optional[nn.Module] = None,
) -> dict:
    """Single class-conditional drifting training step with CFG alpha."""
    model.train()
    num_classes = config["num_classes"]
    n_pos = config["batch_n_pos"]
    n_neg = config["batch_n_neg"]
    temperatures = config["temperatures"]
    ts_loss_config = config if config.get("loss_domain") == "time_series" else None

    batch_size = num_classes * n_neg
    labels = torch.arange(num_classes, device=device).repeat_interleave(n_neg)
    alpha = torch.empty(batch_size, device=device).uniform_(
        config["alpha_min"],
        config["alpha_max"],
    )
    noise = torch.randn(
        batch_size,
        config["in_channels"],
        config["img_size"],
        config["img_size"],
        device=device,
    )

    x_gen = model(noise, labels, alpha)
    x_pos, labels_pos = sample_batch(queue, num_classes, n_pos, device)
    loss, info = compute_drifting_loss(
        x_gen,
        labels,
        x_pos,
        labels_pos,
        feature_encoder,
        temperatures,
        ts_loss_config=ts_loss_config,
    )

    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        config["grad_clip"],
    )
    info["grad_norm"] = grad_norm.item()
    optimizer.step()
    return info


def _batch_to_images_and_labels(
    batch: dict,
    config: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    series = batch["target"]
    labels = batch["study_group_label"].long()
    valid = (labels >= 0) & (labels < config["num_classes"])
    if not valid.any():
        return torch.empty(0), torch.empty(0, dtype=torch.long)
    images = target_series_to_images(series[valid], config, device)
    return images.detach().cpu(), labels[valid].detach().cpu()


def fill_queue(
    queue: ConditionalSampleQueue,
    dataloader: DataLoader,
    config: dict,
    device: torch.device,
    min_samples: int = 64,
):
    """Fill the sample queue with real AI-READI glucose windows."""
    for batch in dataloader:
        x, labels = _batch_to_images_and_labels(batch, config, device)
        if x.numel() > 0:
            queue.add(x, labels)
        if queue.is_ready(min_samples):
            break


def prefix_metric_results(results: Dict[str, Any], split: str) -> Dict[str, Any]:
    """Prefix metric keys with the evaluated data split."""
    return {
        key.replace("metric/", f"metric/{split}/", 1): value
        for key, value in results.items()
    }


def validate_study_group_labels(dataset: Dataset, split: str, config: dict):
    """Fail early when participant metadata did not provide study-group labels."""
    labels = []
    max_items = min(len(dataset), 1024)
    for idx in range(max_items):
        label = int(dataset[idx]["study_group_label"].item())
        labels.append(label)

    valid_labels = [label for label in labels if 0 <= label < config["num_classes"]]
    if not valid_labels:
        raise ValueError(
            f"No valid study_group_label values found in the {split} split. "
            "They are all -1, which usually means participants.tsv was not loaded. "
            "Pass --participants_tsv_path /path/to/participants.tsv; for example "
            "--participants_tsv_path /Users/zhc/Downloads/AI-READI/participants.tsv"
        )

    counts = {
        name: valid_labels.count(idx)
        for idx, name in enumerate(config["study_group_names"])
    }
    print(f"{split} study-group label counts in first {max_items} windows: {counts}")


def train(
    config: Dict[str, Any],
    output_dir: str = "./outputs",
    data_root: str = "./AI-READI",
    seed: int = 42,
    num_workers: int = 1,
    log_interval: int = 1,
    save_interval: int = 10,
    sample_interval: int = 10,
    eval_step_interval: int = 500,
    eval_metrics: str = "disc",
    eval_num_samples: Optional[int] = None,
    metric_iteration: int = 1,
    wandb_enabled: bool = False,
    wandb_project: str = "drifting-model-ts",
    wandb_run_name: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_mode: Optional[str] = None,
    metrics_base_path: Optional[str] = None,
    vae_ckpt_root: Optional[str] = None,
    ts_feature_encoder_ckpt: Optional[str] = None,
    batch_size: int = 256,
    argparse_config: Optional[Dict[str, Any]] = None,
):
    """Main training function."""
    set_seed(seed)

    dataset = "aireadi_study_group_glucose"
    metric_names = parse_metric_names(eval_metrics)
    metric_iteration = max(1, metric_iteration)
    config["eval_metrics"] = metric_names
    config["eval_step_interval"] = eval_step_interval
    config["eval_num_samples"] = eval_num_samples
    config["metric_iteration"] = metric_iteration
    config["ts_feature_encoder_ckpt"] = ts_feature_encoder_ckpt
    config["train_batch_size"] = batch_size
    if ts_feature_encoder_ckpt is not None:
        config["use_feature_encoder"] = True
    wandb_config = dict(argparse_config) if argparse_config is not None else dict(config)
    wandb_config["resolved_training_config"] = dict(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(output_dir) / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = None
    if wandb_enabled:
        if wandb is None:
            print("wandb is not installed; continuing without wandb logging.")
        else:
            wandb_run = wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=wandb_run_name,
                mode=wandb_mode,
                config=wandb_config,
                dir=str(output_dir),
            )

    train_dataset, test_dataset = get_dataset(
        "aireadi_imputation",
        config=config,
        root=data_root,
        seed=seed,
    )
    print(f"Dataset sizes | train: {len(train_dataset)} | test: {len(test_dataset)}")
    validate_study_group_labels(train_dataset, "train", config)
    validate_study_group_labels(test_dataset, "test", config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    model_fn = DriftDiT_models[config["model"]]
    model = model_fn(
        img_size=config["img_size"],
        in_channels=config["in_channels"],
        num_classes=config["num_classes"],
        label_dropout=config["label_dropout"],
    ).to(device)
    print(f"Model: {config['model']}, Parameters: {count_parameters(model):,}")

    ema = EMA(model, decay=config["ema_decay"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        betas=(0.9, 0.95),
        weight_decay=config["weight_decay"],
    )
    scheduler = WarmupLRScheduler(
        optimizer,
        warmup_steps=config["warmup_steps"],
        base_lr=config["lr"],
    )
    queue = ConditionalSampleQueue(
        num_classes=config["num_classes"],
        queue_size=config["queue_size"],
        sample_shape=(config["in_channels"], config["img_size"], config["img_size"]),
    )

    feature_encoder = None
    if ts_feature_encoder_ckpt is not None:
        raise NotImplementedError("ts_feature_encoder_ckpt is not implemented.")
        # print(f"Loading time-series feature encoder from {ts_feature_encoder_ckpt}")
        # feature_encoder = load_ts_feature_encoder_from_ckpt(
        #     ts_feature_encoder_ckpt,
        #     device,
        # )
        # print("Using pretrained TS feature encoder for drifting loss.")

    global_step = 0
    print(f"\nStarting training for {config['epochs']} epochs...")
    for epoch in range(config["epochs"]):
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_drift_norm = 0.0
        num_batches = 0

        fill_queue(queue, train_loader, config, device, min_samples=config["batch_n_pos"])

        for batch_idx, batch in enumerate(train_loader):
            x_real, labels_real = _batch_to_images_and_labels(batch, config, device)
            if x_real.numel() > 0:
                queue.add(x_real, labels_real)
            if not queue.is_ready(config["batch_n_pos"]):
                continue

            info = train_step(
                model,
                optimizer,
                queue,
                config,
                device,
                feature_encoder,
            )
            ema.update(model)
            scheduler.step()

            epoch_loss += info["loss"]
            epoch_drift_norm += info["drift_norm"]
            num_batches += 1
            global_step += 1

            if global_step % log_interval == 0:
                lr = scheduler.get_lr()
                print(
                    f"Epoch {epoch+1}/{config['epochs']} | "
                    f"Step {global_step} | "
                    f"Loss: {info['loss']:.4f} | "
                    f"Drift: {info['drift_norm']:.4f} | "
                    f"Grad: {info['grad_norm']:.4f} | "
                    f"LR: {lr:.6f}"
                )
                if wandb_run is not None:
                    wandb.log(
                        {
                            "train/loss": info["loss"],
                            "train/drift_norm": info["drift_norm"],
                            "train/grad_norm": info["grad_norm"],
                            "train/lr": lr,
                            "train/epoch": epoch + 1,
                        },
                        step=global_step,
                    )

            if global_step % 500 == 0:
                sample_path = output_dir / f"samples_step{global_step}.png"
                real_sample_path = output_dir / f"real_samples_step{global_step}.png"
                generate_samples(
                    ema.shadow,
                    config,
                    device,
                    str(sample_path),
                    num_per_group=20,
                    alpha=config["cfg_sample_alpha"],
                )
                save_real_time_series_samples(
                    train_dataset,
                    config,
                    device,
                    str(real_sample_path),
                    num_per_group=20,
                    num_workers=num_workers,
                )
                print(f"Saved samples to {sample_path}")
                print(f"Saved real samples to {real_sample_path}")
                if wandb_run is not None:
                    wandb.log(
                        {
                            "samples/step": wandb.Image(str(sample_path)),
                            "samples/real_step": wandb.Image(str(real_sample_path)),
                        },
                        step=global_step,
                    )

            if (
                metric_names
                and eval_step_interval > 0
                and global_step % eval_step_interval == 0
            ):
                try:
                    metric_results = {}
                    for split, eval_dataset in (
                        ("train", train_dataset),
                        ("test", test_dataset),
                    ):
                        split_output_dir = output_dir / "metric_samples" / split
                        split_results = evaluate_conditional_time_series_metrics(
                            ema.shadow,
                            eval_dataset,
                            config,
                            device,
                            eval_metrics=metric_names,
                            num_samples=eval_num_samples,
                            metric_iteration=metric_iteration,
                            num_workers=num_workers,
                            base_path=metrics_base_path,
                            vae_ckpt_root=vae_ckpt_root,
                            output_dir=split_output_dir,
                            step=global_step,
                        )
                        metric_results.update(prefix_metric_results(split_results, split))
                    metric_str = " | ".join(
                        f"{key}: {value:.4f}" for key, value in metric_results.items()
                    )
                    print(f"Metrics step {global_step} | {metric_str}")
                    if wandb_run is not None:
                        wandb.log(metric_results, step=global_step)
                except Exception as exc:
                    print(f"Metric evaluation failed at step {global_step}: {exc}")
                    if wandb_run is not None:
                        wandb.log(
                            {"metric/eval_failed": 1, "metric/eval_error": str(exc)},
                            step=global_step,
                        )

        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / max(num_batches, 1)
        avg_drift = epoch_drift_norm / max(num_batches, 1)
        print(
            f"\nEpoch {epoch+1} completed in {epoch_time:.1f}s | "
            f"Avg Loss: {avg_loss:.4f} | "
            f"Avg Drift Norm: {avg_drift:.4f}\n"
        )
        if wandb_run is not None:
            wandb.log(
                {
                    "epoch/loss": avg_loss,
                    "epoch/drift_norm": avg_drift,
                    "epoch/time_sec": epoch_time,
                },
                step=global_step,
            )

        if (epoch + 1) % save_interval == 0:
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
            print(f"Saved checkpoint to {ckpt_path}")

        if (epoch + 1) % sample_interval == 0:
            sample_path = output_dir / f"samples_epoch{epoch+1}.png"
            real_sample_path = output_dir / f"real_samples_epoch{epoch+1}.png"
            generate_samples(
                ema.shadow,
                config,
                device,
                str(sample_path),
                num_per_group=20,
                alpha=config["cfg_sample_alpha"],
            )
            save_real_time_series_samples(
                train_dataset,
                config,
                device,
                str(real_sample_path),
                num_per_group=20,
                num_workers=num_workers,
            )
            print(f"Saved samples to {sample_path}")
            print(f"Saved real samples to {real_sample_path}")
            if wandb_run is not None:
                wandb.log(
                    {
                        "samples/epoch": wandb.Image(str(sample_path)),
                        "samples/real_epoch": wandb.Image(str(real_sample_path)),
                    },
                    step=global_step,
                )

    final_path = output_dir / "checkpoint_final.pt"
    save_checkpoint(
        str(final_path),
        model,
        ema,
        optimizer,
        scheduler,
        config["epochs"] - 1,
        global_step,
        config,
    )
    print(f"Training complete! Final checkpoint saved to {final_path}")
    if wandb_run is not None:
        wandb.finish()


def save_time_series_grid(
    series: torch.Tensor,
    save_path: str,
    ncol: int = 8,
    row_labels: Optional[list[str]] = None,
):
    """Save time-series samples as a grid of line plots."""
    if series.ndim == 2:
        series = series.unsqueeze(-1)

    num_samples, seq_len, _ = series.shape
    ncol = max(1, ncol)
    nrow = math.ceil(num_samples / ncol)

    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(ncol * 2.0, nrow * 1.5),
        sharex=True,
        sharey=True,
    )
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    t = torch.arange(seq_len).cpu().numpy()
    series_np = series.detach().cpu().numpy()

    for i, ax in enumerate(axes):
        if i < num_samples:
            ax.plot(t, series_np[i, :, 0], linewidth=1.0)
            if row_labels is not None and i % ncol == 0 and i // ncol < len(row_labels):
                ax.set_ylabel(row_labels[i // ncol], fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis("off")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.3)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def generate_conditioned_time_series_samples(
    model: nn.Module,
    config: dict,
    device: torch.device,
    labels: torch.Tensor,
    batch_size: int = 256,
    alpha: Optional[float] = None,
) -> torch.Tensor:
    """Generate delay-image samples conditioned on study-group labels."""
    model.eval()
    all_series = []
    alpha = config["cfg_sample_alpha"] if alpha is None else alpha
    labels = labels.long().cpu()

    for start in range(0, labels.numel(), batch_size):
        labels_batch = labels[start:start + batch_size].to(device)
        current_batch = labels_batch.numel()
        noise = torch.randn(
            current_batch,
            config["in_channels"],
            config["img_size"],
            config["img_size"],
            device=device,
        )
        samples = model.forward_with_cfg(noise, labels_batch, alpha=alpha)
        all_series.append(delay_images_to_series(samples, config, device).detach().cpu())

    return torch.cat(all_series, dim=0)


@torch.no_grad()
def collect_real_time_series_and_labels(
    dataset: Dataset,
    config: dict,
    num_samples: Optional[int] = None,
    batch_size: int = 256,
    num_workers: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect real AI-READI glucose target sequences and study-group labels."""
    if num_samples is None:
        num_samples = len(dataset)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    all_series = []
    all_labels = []
    num_collected = 0

    for batch in loader:
        labels = batch["study_group_label"].long()
        valid = (labels >= 0) & (labels < config["num_classes"])
        if not valid.any():
            continue
        series = batch["target"][valid]
        labels = labels[valid]
        needed = num_samples - num_collected
        all_series.append(series[:needed].detach().cpu())
        all_labels.append(labels[:needed].detach().cpu())
        num_collected += min(series.shape[0], needed)
        if num_collected >= num_samples:
            break

    if not all_series:
        raise ValueError("No real time-series samples were collected.")

    return (
        torch.cat(all_series, dim=0)[:num_samples],
        torch.cat(all_labels, dim=0)[:num_samples],
    )


def select_fixed_count(series: torch.Tensor, n: int) -> torch.Tensor:
    """Select exactly n samples deterministically, repeating if needed."""
    count = series.shape[0]
    if count <= 0:
        raise ValueError("Cannot select samples from an empty tensor.")
    if count >= n:
        indices = torch.linspace(0, count - 1, steps=n).long()
        return series[indices]
    repeats = math.ceil(n / count)
    return series.repeat((repeats, 1, 1))[:n]


def save_real_time_series_samples(
    dataset: Dataset,
    config: dict,
    device: torch.device,
    save_path: str,
    num_per_group: int = 20,
    num_workers: int = 0,
):
    """Save real glucose samples grouped by study group."""
    series, labels = collect_real_time_series_and_labels(
        dataset,
        config,
        num_samples=len(dataset),
        num_workers=num_workers,
    )
    grouped = []
    row_labels = []
    for c, name in enumerate(config["study_group_names"]):
        group_series = series[labels == c][:num_per_group]
        if group_series.numel() == 0:
            continue
        grouped.append(group_series)
        row_labels.append(name)
    if not grouped:
        raise ValueError("No grouped real samples were collected.")
    save_time_series_grid(
        torch.cat(grouped, dim=0),
        save_path,
        ncol=num_per_group,
        row_labels=row_labels,
    )


@torch.no_grad()
def generate_samples(
    model: nn.Module,
    config: dict,
    device: torch.device,
    save_path: str,
    num_per_group: int = 20,
    alpha: float = 1.5,
):
    """Generate study-group-conditioned samples and save visualization."""
    labels = torch.arange(config["num_classes"]).repeat_interleave(num_per_group)
    series = generate_conditioned_time_series_samples(
        model,
        config,
        device,
        labels,
        alpha=alpha,
    )
    save_time_series_grid(
        series,
        save_path,
        ncol=num_per_group,
        row_labels=config["study_group_names"],
    )
    return series


def evaluate_conditional_time_series_metrics(
    model: nn.Module,
    test_dataset: Dataset,
    config: dict,
    device: torch.device,
    eval_metrics: list,
    metric_iteration: int,
    num_workers: int,
    num_samples: Optional[int] = None,
    base_path: Optional[str] = None,
    vae_ckpt_root: Optional[str] = None,
    output_dir: Optional[Path] = None,
    step: Optional[int] = None,
) -> Dict[str, Any]:
    """Run time-series metrics with generated labels matched to real labels."""
    eval_size = len(test_dataset) if num_samples is None else min(num_samples, len(test_dataset))
    real_series, labels = collect_real_time_series_and_labels(
        test_dataset,
        config,
        num_samples=eval_size,
        num_workers=num_workers,
    )
    gen_series = generate_conditioned_time_series_samples(
        model,
        config,
        device,
        labels,
        alpha=config["cfg_sample_alpha"],
    )

    real_sig = real_series.numpy().astype(np.float32)
    gen_sig = gen_series.numpy().astype(np.float32)

    if output_dir is not None and step is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / f"real_test_step{step}.npy", real_sig)
        np.save(output_dir / f"generated_step{step}.npy", gen_sig)
        np.save(output_dir / f"labels_step{step}.npy", labels.numpy())

    results: Dict[str, Any] = {"metric/eval_num_samples": int(real_sig.shape[0])}

    if "disc" in eval_metrics:
        from metrics.discriminative_torch import discriminative_score_metrics

        disc_scores = [
            discriminative_score_metrics(real_sig, gen_sig, device)
            for _ in range(metric_iteration)
        ]
        results["metric/disc_mean"] = float(np.round(np.mean(disc_scores), 4))
        results["metric/disc_std"] = float(np.round(np.std(disc_scores), 4))

    if "pred" in eval_metrics:
        from metrics.predictive_metrics_pytorch import predictive_score_metrics

        pred_scores = [
            predictive_score_metrics(real_sig, gen_sig, device=device)
            for _ in range(metric_iteration)
        ]
        results["metric/pred_mean"] = float(np.round(np.nanmean(pred_scores), 4))
        results["metric/pred_std"] = float(np.round(np.nanstd(pred_scores), 4))

    if "contextFID" in eval_metrics:
        from metrics.context_fid import Context_FID

        results["metric/context_fid"] = float(
            Context_FID(real_sig, gen_sig, "glucose", device, base_path)
        )

    if "vaeFID" in eval_metrics:
        from metrics.vae_fid import VAE_FID

        vae_dataset = "glucose_daily" if config.get("window_mode") == "daily" else "glucose"
        results["metric/vae_fid"] = float(
            VAE_FID(real_sig, gen_sig, vae_dataset, device, vae_ckpt_root=vae_ckpt_root)
        )

        per_class_samples = int(config.get("eval_per_class_samples", 1000))
        real_all, labels_all = collect_real_time_series_and_labels(
            test_dataset,
            config,
            num_samples=len(test_dataset),
            num_workers=num_workers,
        )
        results["metric/class_eval_num_samples"] = per_class_samples

        class_dir = None
        if output_dir is not None and step is not None:
            class_dir = output_dir / "class_vae_fid"
            class_dir.mkdir(parents=True, exist_ok=True)

        for class_id, class_name in enumerate(config["study_group_names"]):
            class_real_all = real_all[labels_all == class_id]
            available = int(class_real_all.shape[0])
            results[f"metric/vae_fid_class_{class_id}_real_available"] = available
            if available == 0:
                continue

            class_real = select_fixed_count(class_real_all, per_class_samples)
            class_labels = torch.full(
                (per_class_samples,),
                class_id,
                dtype=torch.long,
            )
            class_gen = generate_conditioned_time_series_samples(
                model,
                config,
                device,
                class_labels,
                alpha=config["cfg_sample_alpha"],
            )

            class_real_np = class_real.numpy().astype(np.float32)
            class_gen_np = class_gen.numpy().astype(np.float32)
            safe_name = class_name.replace("/", "_")
            results[f"metric/vae_fid_class_{class_id}_{safe_name}"] = float(
                VAE_FID(
                    class_real_np,
                    class_gen_np,
                    vae_dataset,
                    device,
                    vae_ckpt_root=vae_ckpt_root,
                )
            )

            if class_dir is not None:
                np.save(class_dir / f"real_class_{class_id}_step{step}.npy", class_real_np)
                np.save(class_dir / f"generated_class_{class_id}_step{step}.npy", class_gen_np)

    return results


def main():
    parser = argparse.ArgumentParser(description="Train study-group-conditional glucose Drifting Models")
    parser.add_argument("--output_dir", type=str, default="./outputs/aireadi_study_group_glucose")
    parser.add_argument("--data_root", type=str, default="./AI-READI")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--sample_interval", type=int, default=10)
    parser.add_argument("--eval_step_interval", "--eval_interval", dest="eval_step_interval", type=int, default=500)
    parser.add_argument("--eval_metrics", type=str, default="disc")
    parser.add_argument("--eval_num_samples", type=int, default=None)
    parser.add_argument(
        "--eval_per_class_samples",
        type=int,
        default=1000,
        help="Number of real/generated samples per class for class-wise vaeFID.",
    )
    parser.add_argument("--metric_iteration", type=int, default=10)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="drifting-model-ts")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default=None, choices=[None, "online", "offline", "disabled"])
    parser.add_argument("--metrics_base_path", type=str, default=None)
    parser.add_argument("--vae_ckpt_root", type=str, default=None)
    parser.add_argument("--ts_feature_encoder_ckpt", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=256)

    parser.add_argument("--model", type=str, default="DriftDiT-Tiny", choices=sorted(DriftDiT_models.keys()))
    parser.add_argument("--img_size", type=int, default=18)
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--num_classes", type=int, default=len(AI_READI_STUDY_GROUPS))
    parser.add_argument("--batch_n_pos", type=int, default=80)
    parser.add_argument("--batch_n_neg", type=int, default=80)
    parser.add_argument("--temperatures", type=parse_temperatures, default=[0.02, 0.05, 0.2])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--use_feature_encoder", action="store_true")
    parser.add_argument("--loss_domain", type=str, default="time_series", choices=["time_series"])
    parser.add_argument("--queue_size", type=int, default=1280)
    parser.add_argument("--label_dropout", type=float, default=0.0)
    parser.add_argument("--alpha_min", type=float, default=1.0)
    parser.add_argument("--alpha_max", type=float, default=1.0)
    parser.add_argument("--cfg_sample_alpha", type=float, default=1.0)

    parser.add_argument("--ts_seq_len", type=int, default=288)
    parser.add_argument("--ts_delay", type=int, default=18)
    parser.add_argument("--ts_embedding", type=int, default=18)
    parser.add_argument("--ts_stride", "--glucose_stride", dest="ts_stride", type=int, default=32)
    parser.add_argument("--window_mode", type=str, default="daily", choices=["sliding", "daily"])
    parser.add_argument(
        "--daily_min_events",
        type=int,
        default=None,
        help="Minimum glucose events required for a local calendar-day window. Defaults to ts_seq_len.",
    )

    parser.add_argument("--modalities", type=parse_modalities, default=["glucose"])
    parser.add_argument("--anchor_modality", type=str, default="glucose")
    parser.add_argument("--target_modality", type=str, default="glucose")
    parser.add_argument("--max_anchor_gap_minutes", type=float, default=10.0)
    parser.add_argument("--max_window_span_hours", type=float, default=14.0)
    parser.add_argument("--anchor_sampling_minutes", type=float, default=5.0)
    parser.add_argument("--anchor_sampling_tolerance_seconds", type=float, default=2.0)
    parser.add_argument("--clinical_root", type=str, default=None)
    parser.add_argument("--participants_tsv_path", type=str, default='/Users/zhc/Downloads/AI-READI/participants.tsv')
    parser.add_argument("--include_clinical_static", action="store_true")
    parser.add_argument("--include_participant_metadata", action="store_true", default=True)
    parser.add_argument("--include_study_group", action="store_true", default=True)
    parser.add_argument("--include_clinical_site", action="store_true")

    args = parser.parse_args()
    config = build_config(args)

    train(
        config=config,
        output_dir=args.output_dir,
        data_root=args.data_root,
        seed=args.seed,
        num_workers=args.num_workers,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        sample_interval=args.sample_interval,
        eval_step_interval=args.eval_step_interval,
        eval_metrics=args.eval_metrics,
        eval_num_samples=args.eval_num_samples,
        metric_iteration=args.metric_iteration,
        wandb_enabled=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode,
        metrics_base_path=args.metrics_base_path,
        vae_ckpt_root=args.vae_ckpt_root,
        ts_feature_encoder_ckpt=args.ts_feature_encoder_ckpt,
        batch_size=args.batch_size,
        argparse_config=vars(args),
    )


if __name__ == "__main__":
    main()
