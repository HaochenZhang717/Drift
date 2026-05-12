"""Train the two-loop trend/full Drift model for unconditional time-series generation."""

import argparse
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import wandb
except ImportError:
    wandb = None

from benchmarking_drift import (
    DelayEmbeddingImageDataset,
    MinMaxNormalizedTimeSeriesDataset,
    _fit_minmax_stats,
    fill_queue,
    parse_eval_splits,
    parse_metric_names,
    parse_temperatures,
    prefix_metric_results,
    save_real_time_series_samples,
    save_time_series_grid,
)
from data_provider.data_provider import get_test, get_train
from losses.trend_full_drifting_loss import compute_trend_full_drifting_loss
from models.trend_full_looped_unconditional_model import TrendFullLoopedDriftDiT_models
from ts_quality_eval import delay_images_to_series, evaluate_time_series_metrics
from utils.utils_drift import (
    EMA,
    SampleQueue,
    WarmupLRScheduler,
    count_parameters,
    save_checkpoint,
    set_seed,
)


class FullOutputWrapper(nn.Module):
    """Expose only the final full output to existing evaluation utilities."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x, output_component="full")


def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    keys = [
        "model",
        "img_size",
        "in_channels",
        "one_channel",
        "batch_n_pos",
        "batch_n_neg",
        "temperatures",
        "lr",
        "weight_decay",
        "grad_clip",
        "ema_decay",
        "warmup_steps",
        "epochs",
        "queue_size",
        "ts_seq_len",
        "ts_delay",
        "ts_embedding",
        "window_stride",
        "ts_stride",
        "stride",
        "dataset_name",
        "data",
        "datasets_dir",
        "rel_path",
        "rel_path_train",
        "rel_path_valid",
        "tf_num_loops",
        "tf_loop_depth",
        "tf_trend_kernel_size",
        "tf_trend_sigma",
        "tf_weight_schedule",
        "tf_trend_start",
        "tf_trend_end",
        "tf_full_start",
        "tf_full_end",
        "tf_consistency_weight",
    ]
    config = {key: getattr(args, key) for key in keys}
    if config["one_channel"]:
        config["in_channels"] = 1
    return config


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    queue: SampleQueue,
    config: dict,
    device: torch.device,
    *,
    step: int,
    total_steps: int,
) -> dict:
    model.train()
    batch_size = config["batch_n_neg"]
    noise = torch.randn(
        batch_size,
        config["in_channels"],
        config["img_size"],
        config["img_size"],
        device=device,
    )
    x_outputs = model(noise)
    x_pos = queue.sample(config["batch_n_pos"], device)

    loss, info = compute_trend_full_drifting_loss(
        x_outputs,
        x_pos,
        temperatures=config["temperatures"],
        config=config,
        step=step,
        total_steps=total_steps,
    )

    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        config["grad_clip"],
    )
    optimizer.step()
    info["grad_norm"] = grad_norm.item()
    return info


@torch.no_grad()
def generate_samples(
    model: nn.Module,
    config: dict,
    device: torch.device,
    save_path: str,
    num_samples: int = 80,
    component: str = "full",
):
    model.eval()
    noise = torch.randn(
        num_samples,
        config["in_channels"],
        config["img_size"],
        config["img_size"],
        device=device,
    )
    samples = model(noise, output_component=component)
    series = delay_images_to_series(samples, config, device)
    save_time_series_grid(series, save_path, ncol=8)
    return samples


@torch.no_grad()
def generate_component_samples(
    model: nn.Module,
    config: dict,
    device: torch.device,
    output_dir: Path,
    prefix: str,
    num_samples: int = 80,
):
    model.eval()
    noise = torch.randn(
        num_samples,
        config["in_channels"],
        config["img_size"],
        config["img_size"],
        device=device,
    )
    outputs = model(noise)
    for component, samples in outputs.items():
        series = delay_images_to_series(samples, config, device)
        save_time_series_grid(
            series,
            str(output_dir / f"{prefix}_{component}.png"),
            ncol=8,
        )


def _make_dataset_config(config: dict) -> dict:
    effective_stride = config.get("window_stride")
    if effective_stride is None:
        effective_stride = config.get("ts_stride")
    if effective_stride is None:
        effective_stride = config.get("stride")

    dataset_config = {
        "name": config["dataset_name"],
        "data": config["data"],
        "datasets_dir": config["datasets_dir"],
        "rel_path": config["rel_path"],
        "seq_len": config["ts_seq_len"],
        "flag": "train",
    }
    if config.get("rel_path_train") is not None:
        dataset_config["rel_path_train"] = config["rel_path_train"]
    if config.get("rel_path_valid") is not None:
        dataset_config["rel_path_valid"] = config["rel_path_valid"]
    if effective_stride is not None:
        dataset_config["window_stride"] = int(effective_stride)
        dataset_config["ts_stride"] = int(effective_stride)
        dataset_config["stride"] = int(effective_stride)
    return dataset_config


def train(
    config: Dict[str, Any],
    output_dir: str,
    *,
    seed: int = 42,
    num_workers: int = 1,
    log_interval: int = 100,
    save_interval: int = 100,
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
    vae_ckpt_name: str = "best.pt",
    batch_size: int = 256,
    argparse_config: Optional[Dict[str, Any]] = None,
    eval_splits: Optional[list] = None,
):
    set_seed(seed)
    metric_names = parse_metric_names(eval_metrics)
    eval_splits = eval_splits or ["train", "test"]
    config["eval_metrics"] = metric_names
    config["eval_step_interval"] = eval_step_interval
    config["eval_num_samples"] = eval_num_samples
    config["metric_iteration"] = max(1, metric_iteration)
    config["train_batch_size"] = batch_size
    config["dataset"] = (
        f"{config['dataset_name']}_one_channel"
        if config.get("one_channel")
        else config["dataset_name"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(output_dir) / config["dataset_name"]
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
                config=argparse_config or config,
                dir=str(output_dir),
            )

    dataset_config = _make_dataset_config(config)
    train_dataset = get_train(dataset_config.copy())
    test_dataset = get_test(dataset_config.copy())
    data_min, data_max = _fit_minmax_stats(
        train_dataset,
        one_channel=bool(config.get("one_channel", False)),
    )
    train_dataset = MinMaxNormalizedTimeSeriesDataset(
        train_dataset,
        data_min=data_min,
        data_max=data_max,
        one_channel=bool(config.get("one_channel", False)),
    )
    test_dataset = MinMaxNormalizedTimeSeriesDataset(
        test_dataset,
        data_min=data_min,
        data_max=data_max,
        one_channel=bool(config.get("one_channel", False)),
    )
    train_dataset = DelayEmbeddingImageDataset(train_dataset, config)
    test_dataset = DelayEmbeddingImageDataset(test_dataset, config)
    print(f"Dataset sizes | train: {len(train_dataset)} | test: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    model_fn = TrendFullLoopedDriftDiT_models[config["model"]]
    model = model_fn(
        img_size=config["img_size"],
        in_channels=config["in_channels"],
        loop_depth=config["tf_loop_depth"],
        num_loops=config["tf_num_loops"],
    ).to(device)
    print(f"Model: {config['model']}, Parameters: {count_parameters(model):,}")

    ema = EMA(model, decay=config["ema_decay"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    scheduler = WarmupLRScheduler(
        optimizer,
        warmup_steps=config["warmup_steps"],
        base_lr=config["lr"],
    )
    queue = SampleQueue(
        queue_size=config["queue_size"],
        sample_shape=(config["in_channels"], config["img_size"], config["img_size"]),
    )

    total_steps = max(1, config["epochs"] * len(train_loader))
    global_step = 0
    best_monitored_fid = float("inf")

    print(f"\nStarting training for {config['epochs']} epochs...")
    for epoch in range(config["epochs"]):
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_drift_norm = 0.0
        num_batches = 0

        fill_queue(queue, train_loader, device, min_samples=64)

        for batch in train_loader:
            x_real = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            queue.add(x_real.cpu())
            if not queue.is_ready(config["batch_n_pos"]):
                continue

            info = train_step(
                model,
                optimizer,
                queue,
                config,
                device,
                step=global_step,
                total_steps=total_steps,
            )
            ema.update(model)
            scheduler.step()

            epoch_loss += info["loss"]
            epoch_drift_norm += info["drift_norm"]
            num_batches += 1
            global_step += 1
            lr = scheduler.get_lr()

            if global_step % log_interval == 0:
                print(
                    f"Epoch {epoch+1}/{config['epochs']} | "
                    f"Step {global_step} | "
                    f"Loss: {info['loss']:.4f} | "
                    f"Trend: {info['loss_trend']:.4f} "
                    f"(w={info['weight_trend']:.3f}) | "
                    f"Full: {info['loss_full']:.4f} "
                    f"(w={info['weight_full']:.3f}) | "
                    f"Grad: {info['grad_norm']:.4f} | "
                    f"LR: {lr:.6f}"
                )
                if wandb_run is not None:
                    wandb.log(
                        {
                            "train/loss": info["loss"],
                            "train/loss_trend": info["loss_trend"],
                            "train/loss_full": info["loss_full"],
                            "train/loss_consistency": info["loss_consistency"],
                            "train/weight_trend": info["weight_trend"],
                            "train/weight_full": info["weight_full"],
                            "train/drift_norm": info["drift_norm"],
                            "train/drift_norm_trend": info["drift_norm_trend"],
                            "train/drift_norm_full": info["drift_norm_full"],
                            "train/v_norm": info["v_norm"],
                            "train/true_v_norm": info["true_v_norm"],
                            "train/grad_norm": info["grad_norm"],
                            "train/lr": lr,
                            "train/epoch": epoch + 1,
                        },
                        step=global_step,
                    )

            if global_step % 500 == 0:
                generate_component_samples(
                    ema.shadow,
                    config,
                    device,
                    output_dir,
                    prefix=f"samples_step{global_step}",
                    num_samples=80,
                )
                real_sample_path = output_dir / f"real_samples_step{global_step}.png"
                save_real_time_series_samples(
                    train_dataset,
                    config,
                    device,
                    str(real_sample_path),
                    num_samples=80,
                    num_workers=num_workers,
                )
                print(f"Saved component samples for step {global_step}")

            if (
                metric_names
                and eval_step_interval > 0
                and global_step % eval_step_interval == 0
            ):
                try:
                    metric_results = {}
                    split_to_dataset = {"train": train_dataset, "test": test_dataset}
                    eval_model = FullOutputWrapper(ema.shadow)
                    for split in eval_splits:
                        split_results = evaluate_time_series_metrics(
                            eval_model,
                            split_to_dataset[split],
                            config,
                            device,
                            eval_metrics=metric_names,
                            num_samples=eval_num_samples,
                            metric_iteration=metric_iteration,
                            num_workers=num_workers,
                            base_path=metrics_base_path,
                            vae_ckpt_root=vae_ckpt_root,
                            vae_ckpt_name=vae_ckpt_name,
                            output_dir=output_dir / "metric_samples" / split,
                            step=global_step,
                        )
                        metric_results.update(prefix_metric_results(split_results, split))
                    metric_str = " | ".join(
                        f"{key}: {value:.4f}" for key, value in metric_results.items()
                    )
                    print(f"Metrics step {global_step} | {metric_str}")
                    if wandb_run is not None:
                        wandb.log(metric_results, step=global_step)

                    monitored_fid_key = "metric/test/vae_fid"
                    if monitored_fid_key in metric_results:
                        current_fid = float(metric_results[monitored_fid_key])
                        if current_fid < best_monitored_fid:
                            best_monitored_fid = current_fid
                            ckpt_path = output_dir / "checkpoint_best_fid.pt"
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
                            print(f"Saved best-FID checkpoint -> {ckpt_path}")
                except Exception as exc:
                    print(f"Metric evaluation failed at step {global_step}: {exc}")

        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / max(num_batches, 1)
        avg_drift = epoch_drift_norm / max(num_batches, 1)
        print(
            f"\nEpoch {epoch+1} completed in {epoch_time:.1f}s | "
            f"Avg Loss: {avg_loss:.4f} | Avg Drift Norm: {avg_drift:.4f}\n"
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
            generate_component_samples(
                ema.shadow,
                config,
                device,
                output_dir,
                prefix=f"samples_epoch{epoch+1}",
                num_samples=80,
            )
            print(f"Saved epoch component samples to {output_dir}")

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


def main():
    parser = argparse.ArgumentParser(description="Train Trend/Full Looped Drifting Models")
    parser.add_argument("--output_dir", type=str, default="./outputs/trend_full_looped_drift")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--sample_interval", type=int, default=10)
    parser.add_argument("--eval_step_interval", "--eval_interval", dest="eval_step_interval", type=int, default=500)
    parser.add_argument("--eval_metrics", type=str, default="disc")
    parser.add_argument("--eval_splits", type=parse_eval_splits, default=["train", "test"])
    parser.add_argument("--eval_num_samples", type=int, default=None)
    parser.add_argument("--metric_iteration", type=int, default=10)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="drifting-model-ts")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default=None)
    parser.add_argument("--metrics_base_path", type=str, default=None)
    parser.add_argument("--vae_ckpt_root", type=str, default=None)
    parser.add_argument("--vae_ckpt_name", type=str, default="best.pt")
    parser.add_argument("--batch_size", type=int, default=256)

    parser.add_argument("--model", type=str, default="TrendFullLoopedDriftDiT-Tiny", choices=sorted(TrendFullLoopedDriftDiT_models.keys()))
    parser.add_argument("--img_size", type=int, default=32)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--one_channel", action="store_true")
    parser.add_argument("--batch_n_pos", type=int, default=256)
    parser.add_argument("--batch_n_neg", type=int, default=256)
    parser.add_argument("--temperatures", type=parse_temperatures, default=[0.02, 0.05, 0.2])
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--queue_size", type=int, default=1280)

    parser.add_argument("--ts_seq_len", type=int, default=128)
    parser.add_argument("--ts_delay", type=int, default=12)
    parser.add_argument("--ts_embedding", type=int, default=12)
    parser.add_argument("--window_stride", type=int, default=None)
    parser.add_argument("--ts_stride", "--glucose_stride", dest="ts_stride", type=int, default=128)
    parser.add_argument("--stride", type=int, default=None)

    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--datasets_dir", type=str, required=True)
    parser.add_argument("--rel_path", type=str, default=None)
    parser.add_argument("--rel_path_train", type=str, default=None)
    parser.add_argument("--rel_path_valid", type=str, default=None)

    parser.add_argument("--tf_num_loops", type=int, default=2)
    parser.add_argument("--tf_loop_depth", type=int, default=3)
    parser.add_argument("--tf_trend_kernel_size", type=int, default=15)
    parser.add_argument("--tf_trend_sigma", type=float, default=3.0)
    parser.add_argument("--tf_weight_schedule", type=str, default="cosine", choices=["cosine", "linear", "constant"])
    parser.add_argument("--tf_trend_start", type=float, default=1.0)
    parser.add_argument("--tf_trend_end", type=float, default=0.2)
    parser.add_argument("--tf_full_start", type=float, default=0.2)
    parser.add_argument("--tf_full_end", type=float, default=1.0)
    parser.add_argument("--tf_consistency_weight", type=float, default=0.0)

    args = parser.parse_args()
    if args.rel_path is None and not (args.rel_path_train and args.rel_path_valid):
        parser.error("Provide --rel_path, or provide both --rel_path_train and --rel_path_valid.")
    if args.rel_path is None:
        args.rel_path = args.rel_path_train

    config = build_config(args)
    train(
        config=config,
        output_dir=args.output_dir,
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
        vae_ckpt_name=args.vae_ckpt_name,
        batch_size=args.batch_size,
        argparse_config=vars(args),
        eval_splits=args.eval_splits,
    )


if __name__ == "__main__":
    main()
