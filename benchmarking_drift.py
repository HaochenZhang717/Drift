"""
Training script for Drifting Models with unconditional glucose time-series generation.
"""
import argparse
import math
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from data_provider.data_provider import get_train, get_test
from img_transformations import DelayEmbedder



from models.unconditional_model import DriftDiT_Tiny, DriftDiT_Small, DriftDiT_models
from models.vqvae import VQVAE
from drifting import (
    compute_V,
)
from utils.utils_drift import (
    EMA,
    WarmupLRScheduler,
    SampleQueue,
    save_checkpoint,
    save_image_grid,
    count_parameters,
    set_seed,
)
from ts_quality_eval import (
    collect_real_time_series,
    delay_images_to_series,
    evaluate_time_series_metrics,
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


def parse_eval_splits(value: str) -> list:
    """Parse comma-separated eval split names."""
    splits = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not splits:
        raise argparse.ArgumentTypeError("eval_splits must contain at least one split")
    valid = {"train", "test"}
    invalid = [s for s in splits if s not in valid]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Unsupported eval split(s): {invalid}. Valid splits: {sorted(valid)}"
        )
    # Keep order while removing duplicates.
    return list(dict.fromkeys(splits))


def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Build the training config from parsed argparse values."""
    config_keys = [
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
        "drift_loss_mode",
        "vqvae_ckpt_path",
        "vqvae_hidden_size",
        "vqvae_num_layers",
        "vqvae_code_dim",
        "vqvae_num_codes",
        "vqvae_latent_downsample",
        "vqvae_decoder_upsample_rate",
        "vqvae_dropout",
        "vqvae_commitment_weight",
    ]

    config = {key: getattr(args, key) for key in config_keys}
    if config["one_channel"]:
        config["in_channels"] = 1
    return config

def load_vqvae_feature_encoder_from_ckpt(
    ckpt_path: str,
    device: torch.device,
    *,
    input_dim: int,
    seq_len: int,
    hidden_size: int,
    num_layers: int,
    code_dim: int,
    num_codes: int,
    latent_downsample: int,
    decoder_upsample_rate: int,
    dropout: float,
    commitment_weight: float,
) -> VQVAE:
    model = VQVAE(
        input_dim=input_dim,
        output_dim=input_dim,
        seq_len=seq_len,
        hidden_size=hidden_size,
        num_layers=num_layers,
        code_dim=code_dim,
        num_codes=num_codes,
        latent_downsample=latent_downsample,
        decoder_upsample_rate=decoder_upsample_rate,
        dropout=dropout,
        commitment_weight=commitment_weight,
    ).to(device)
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def compute_drifting_loss(
    x_gen: torch.Tensor,
    x_pos: torch.Tensor,
    feature_encoders: Optional[list],
    temperatures: list,
    ts_loss_config: Optional[dict] = None,
) -> tuple:

    device = x_gen.device

    rep_gen = delay_images_to_series(x_gen, ts_loss_config, device)
    rep_pos = delay_images_to_series(x_pos, ts_loss_config, device)

    if not feature_encoders:
        feature_encoders = ["raw_ts"]

    feat_gen_list = []
    feat_pos_list = []
    for encoder in feature_encoders:
        if encoder == "raw_ts":
            feat_gen = rep_gen.flatten(start_dim=1)
            feat_pos = rep_pos.flatten(start_dim=1)
        elif isinstance(encoder, VQVAE):
            rep_gen_vq = rep_gen.transpose(1, 2).contiguous()  # (B, C, T)
            rep_pos_vq = rep_pos.transpose(1, 2).contiguous()  # (B, C, T)
            feat_gen = encoder.get_embedding(rep_gen_vq)
            with torch.no_grad():
                feat_pos = encoder.get_embedding(rep_pos_vq)
            breakpoint()
        else:
            raise ValueError(f"Unsupported feature encoder type: {type(encoder)}")

        feat_gen_list.append(feat_gen)
        feat_pos_list.append(feat_pos)

    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    total_drift_norm = 0.0
    total_v_norm = 0.0
    total_true_v_norm = 0.0   # ✅ 新增

    for scale_idx, (feat_gen, feat_pos) in enumerate(zip(feat_gen_list, feat_pos_list)):

        feat_neg = feat_gen

        V_total = torch.zeros_like(feat_gen)        # normalized（训练用）
        V_total_raw = torch.zeros_like(feat_gen)    # ✅ raw V（log 用）

        for tau in temperatures:
            V_tau = compute_V(
                feat_gen,
                feat_pos,
                feat_neg,
                tau,
                mask_self=True,
            )

            # ✅ 累积 raw V（关键！！！）
            V_total_raw = V_total_raw + V_tau

            # ===== 原逻辑（不要改）=====
            v_norm = torch.sqrt(torch.mean(V_tau ** 2) + 1e-8)
            V_tau = V_tau / (v_norm + 1e-8)

            V_total = V_total + V_tau
            total_v_norm += v_norm.item()

        # ===== 计算真实 drift norm =====
        true_v_norm = torch.sqrt(torch.mean(V_total_raw ** 2) + 1e-8)
        total_true_v_norm += true_v_norm.item()

        # ===== 原 loss =====
        target = (feat_gen + V_total).detach()
        loss_scale = F.mse_loss(feat_gen, target)

        total_loss = total_loss + loss_scale
        total_drift_norm += (V_total ** 2).mean().item() ** 0.5

    info = {
        "loss": total_loss.item(),
        "drift_norm": total_drift_norm,
        "v_norm": total_v_norm,                   # 原来的
        "true_v_norm": total_true_v_norm,         # ✅ 新的（最重要）
    }

    return total_loss, info


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    queue: SampleQueue,
    config: dict,
    device: torch.device,
    feature_encoders: Optional[list] = None,
) -> dict:
    """
    Single training step (Algorithm 1).

    1. Generate samples from noise
    2. Sample positive samples from queue
    3. Compute drifting field and loss
    4. Update model
    """
    model.train()
    n_pos = config["batch_n_pos"]
    n_neg = config["batch_n_neg"]
    temperatures = config["temperatures"]
    ts_loss_config = config

    # Total batch size
    batch_size = n_neg

    # Sample noise
    noise = torch.randn(
        batch_size,
        config["in_channels"],
        config["img_size"],
        config["img_size"],
        device=device,
    )

    # Generate samples
    x_gen = model(noise)

    # Sample positive samples from queue
    x_pos = queue.sample(n_pos, device)

    # Compute drifting loss
    loss, info = compute_drifting_loss(
        x_gen,
        x_pos,
        feature_encoders,
        temperatures,
        ts_loss_config=ts_loss_config,
    )

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Gradient clipping
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), config["grad_clip"]
    )
    info["grad_norm"] = grad_norm.item()

    # Optimizer step
    optimizer.step()

    return info


def fill_queue(
    queue: SampleQueue,
    dataloader: DataLoader,
    device: torch.device,
    min_samples: int = 64,
):
    """Fill the sample queue with real data."""
    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch

        queue.add(x)

        if queue.is_ready(min_samples):
            break


class DelayEmbeddingImageDataset(Dataset):
    """Wrap a time-series dataset and emit delay-embedding images (C, H, W)."""

    def __init__(self, base_dataset: Dataset, config: Dict[str, Any]):
        self.base_dataset = base_dataset
        self.one_channel = bool(config.get("one_channel", False))
        self.embedder = DelayEmbedder(
            device=torch.device("cpu"),
            seq_len=config["ts_seq_len"],
            delay=config["ts_delay"],
            embedding=config["ts_embedding"],
        )

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        sample = self.base_dataset[idx]
        if isinstance(sample, (list, tuple)):
            sample = sample[0]
        if not torch.is_tensor(sample):
            sample = torch.as_tensor(sample, dtype=torch.float32)
        sample = sample.to(torch.float32)
        if sample.ndim == 1:
            sample = sample.unsqueeze(-1)
        if sample.ndim != 2:
            raise ValueError(f"Expected time-series sample shape (T, C), got {tuple(sample.shape)}")
        if self.one_channel:
            sample = sample[:, :1]

        img = self.embedder.ts_to_img(sample.unsqueeze(0), pad=True)
        return img.squeeze(0)


def _extract_ts_sample(sample: Any) -> torch.Tensor:
    if isinstance(sample, (list, tuple)):
        sample = sample[0]
    if not torch.is_tensor(sample):
        sample = torch.as_tensor(sample, dtype=torch.float32)
    sample = sample.to(torch.float32)
    if sample.ndim == 1:
        sample = sample.unsqueeze(-1)
    if sample.ndim != 2:
        raise ValueError(f"Expected time-series sample shape (T, C), got {tuple(sample.shape)}")
    return sample


def _fit_minmax_stats(base_dataset: Dataset, one_channel: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    data_min = None
    data_max = None
    for idx in range(len(base_dataset)):
        sample = _extract_ts_sample(base_dataset[idx])
        if one_channel:
            sample = sample[:, :1]
        sample_min = sample.amin(dim=0)
        sample_max = sample.amax(dim=0)
        data_min = sample_min if data_min is None else torch.minimum(data_min, sample_min)
        data_max = sample_max if data_max is None else torch.maximum(data_max, sample_max)
    if data_min is None or data_max is None:
        raise ValueError("Cannot fit min-max statistics on an empty dataset.")
    return data_min, data_max


class MinMaxNormalizedTimeSeriesDataset(Dataset):
    """Normalize each channel to [-1, 1] using train-split min/max statistics."""

    def __init__(
        self,
        base_dataset: Dataset,
        data_min: torch.Tensor,
        data_max: torch.Tensor,
        one_channel: bool = False,
    ):
        self.base_dataset = base_dataset
        self.data_min = data_min.to(torch.float32)
        self.data_max = data_max.to(torch.float32)
        self.one_channel = one_channel
        self.denom = torch.clamp(self.data_max - self.data_min, min=1e-6)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        sample = _extract_ts_sample(self.base_dataset[idx])
        if self.one_channel:
            sample = sample[:, :1]
        sample = torch.clamp((sample - self.data_min) / self.denom, 0.0, 1.0)
        sample = sample * 2.0 - 1.0
        return sample


def prefix_metric_results(results: Dict[str, Any], split: str) -> Dict[str, Any]:
    """Prefix metric keys with the evaluated data split."""
    return {
        key.replace("metric/", f"metric/{split}/", 1): value
        for key, value in results.items()
    }


def train(
    config: Dict[str, Any],
    output_dir: str = "./outputs",
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
    vae_ckpt_name: str = "best.pt",
    batch_size: int = 256,
    argparse_config: Optional[Dict[str, Any]] = None,
    eval_splits: Optional[list] = None,
):
    """Main training function."""
    set_seed(seed)

    metric_names = parse_metric_names(eval_metrics)
    metric_iteration = max(1, metric_iteration)
    config["eval_metrics"] = metric_names
    config["eval_step_interval"] = eval_step_interval
    config["eval_num_samples"] = eval_num_samples
    config["metric_iteration"] = metric_iteration
    config["train_batch_size"] = batch_size
    config["dataset"] = (
        f"{config['dataset_name']}_one_channel"
        if config.get("one_channel")
        else config["dataset_name"]
    )
    # config["dataset"] = config["data"]

    eval_splits = eval_splits or ["train", "test"]
    wandb_config = dict(argparse_config) if argparse_config is not None else dict(config)
    wandb_config["resolved_training_config"] = dict(config)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
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
                config=wandb_config,
                dir=str(output_dir),
            )

    effective_stride = config.get("window_stride")
    if effective_stride is None:
        effective_stride = config.get("ts_stride")
    if effective_stride is None:
        effective_stride = config.get("stride")

    dataset_config = {
        "name": config["dataset_name"],
        "data": config["data"],  # 对应 data_dict 的 key
        "datasets_dir": config["datasets_dir"],
        "rel_path": config["rel_path"],
        "seq_len": config["ts_seq_len"],  # get_train 会写入，但底层 verbal_ts 实际不使用这个字段
        "flag": "train",  # get_train/get_test 会覆盖这个
    }
    if config.get("rel_path_train") is not None:
        dataset_config["rel_path_train"] = config["rel_path_train"]
    if config.get("rel_path_valid") is not None:
        dataset_config["rel_path_valid"] = config["rel_path_valid"]
    if effective_stride is not None:
        dataset_config["window_stride"] = int(effective_stride)
        dataset_config["ts_stride"] = int(effective_stride)
        dataset_config["stride"] = int(effective_stride)

    train_dataset = get_train(dataset_config.copy())  # torch.utils.data.Dataset
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
    # Load dataset
    # train_dataset, test_dataset = get_dataset(
    #     dataset,
    #     config=config,
    #     root=data_root,
    #     seed=seed,
    # )
    print(f"Dataset sizes | train: {len(train_dataset)} | test: {len(test_dataset)}")
    test_eval_size = len(test_dataset)
    real_sig_test = collect_real_time_series(
        test_dataset,
        config,
        device,
        num_samples=test_eval_size,
        batch_size=batch_size,
        num_workers=num_workers,
    ).numpy().astype(np.float32)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Create model
    model_fn = DriftDiT_models[config["model"]]
    model = model_fn(
        img_size=config["img_size"],
        in_channels=config["in_channels"],
    ).to(device)

    print(f"Model: {config['model']}, Parameters: {count_parameters(model):,}")

    # Create EMA
    ema = EMA(model, decay=config["ema_decay"])

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    # Create scheduler
    scheduler = WarmupLRScheduler(
        optimizer,
        warmup_steps=config["warmup_steps"],
        base_lr=config["lr"],
    )

    # Create sample queue
    queue = SampleQueue(
        queue_size=config["queue_size"],
        sample_shape=(config["in_channels"], config["img_size"], config["img_size"]),
    )

    # Feature encoders used by drifting loss.
    feature_encoders = []
    if config.get("drift_loss_mode") in {"time_series", "both"}:
        feature_encoders.append("raw_ts")

    if config.get("drift_loss_mode") in {"vqvae", "both"}:
        vqvae_ckpt_path = config.get("vqvae_ckpt_path")
        if not vqvae_ckpt_path:
            raise ValueError(
                "drift_loss_mode is 'vqvae' or 'both', but --vqvae_ckpt_path is missing."
            )
        print(f"Loading VQVAE feature encoder from {vqvae_ckpt_path}")
        vq_feature_encoder = load_vqvae_feature_encoder_from_ckpt(
            vqvae_ckpt_path,
            device,
            input_dim=config["in_channels"],
            seq_len=config["ts_seq_len"],
            hidden_size=config["vqvae_hidden_size"],
            num_layers=config["vqvae_num_layers"],
            code_dim=config["vqvae_code_dim"],
            num_codes=config["vqvae_num_codes"],
            latent_downsample=config["vqvae_latent_downsample"],
            decoder_upsample_rate=config["vqvae_decoder_upsample_rate"],
            dropout=config["vqvae_dropout"],
            commitment_weight=config["vqvae_commitment_weight"],
        )
        feature_encoders.append(vq_feature_encoder)
        print("Using pretrained VQVAE feature encoder for drifting loss.")

    if not feature_encoders:
        raise ValueError("No loss feature source selected. Check drift_loss_mode and encoder args.")

    start_epoch = 0
    global_step = 0
    best_monitored_fid = float("inf")
    # v_norm_ema = None
    # v_norm_ema_decay = 0.98

    # Training loop
    print(f"\nStarting training for {config['epochs']} epochs...")
    for epoch in range(start_epoch, config["epochs"]):
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_drift_norm = 0.0
        num_batches = 0

        # Fill queue at start of each epoch
        fill_queue(queue, train_loader, device, min_samples=64)

        for batch_idx, batch in enumerate(train_loader):
            if isinstance(batch, (list, tuple)):
                x_real = batch[0].to(device)
            else:
                x_real = batch.to(device)

            # Add to queue
            queue.add(x_real.cpu())

            # Skip if queue not ready
            if not queue.is_ready(config["batch_n_pos"]):
                continue

            # Training step
            info = train_step(
                model,
                optimizer,
                queue,
                config,
                device,
                feature_encoders,
            )

            # Update EMA and scheduler
            ema.update(model)
            scheduler.step()

            # Accumulate metrics
            epoch_loss += info["loss"]
            epoch_drift_norm += info["drift_norm"]
            num_batches += 1
            global_step += 1
            # if v_norm_ema is None:
            #     v_norm_ema = info["v_norm"]
            # else:
            #     v_norm_ema = (
            #         v_norm_ema_decay * v_norm_ema
            #         + (1.0 - v_norm_ema_decay) * info["v_norm"]
            #     )

            lr = scheduler.get_lr()
            if wandb_run is not None:
                # Log |V| every step so the trend can be visualized like toy examples.
                # wandb.log(
                #     {
                #         "train/v_norm_step": info["v_norm"],
                #         # "train/v_norm_step_ema": v_norm_ema,
                #     },
                #     step=global_step,
                # )

                wandb.log(
                    {
                        "train/v_norm_step": info["v_norm"],
                        # "train/v_norm_step_ema": v_norm_ema,
                        "train/true_v_norm_step": info["true_v_norm"],  # ✅ 新增
                    },
                    step=global_step,
                )


            # Logging
            if global_step % log_interval == 0:
                print(
                    f"Epoch {epoch+1}/{config['epochs']} | "
                    f"Step {global_step} | "
                    f"Loss: {info['loss']:.4f} | "
                    f"Drift: {info['drift_norm']:.4f} | "
                    f"|V|: {info['v_norm']:.4f} | "
                    f"Grad: {info['grad_norm']:.4f} | "
                    f"LR: {lr:.6f}"
                )
                if wandb_run is not None:
                    wandb.log(
                        {
                            "train/loss": info["loss"],
                            "train/drift_norm": info["drift_norm"],
                            "train/v_norm": info["v_norm"],
                            "train/grad_norm": info["grad_norm"],
                            "train/lr": lr,
                            "train/epoch": epoch + 1,
                        },
                        step=global_step,
                    )

            # Generate samples every 500 steps for quick visualization
            if global_step % 500 == 0:
                sample_path = output_dir / f"samples_step{global_step}.png"
                real_sample_path = output_dir / f"real_samples_step{global_step}.png"
                generate_samples(
                    ema.shadow,
                    config,
                    device,
                    str(sample_path),
                    num_samples=80,
                )
                save_real_time_series_samples(
                    train_dataset,
                    config,
                    device,
                    str(real_sample_path),
                    num_samples=80,
                    num_workers=num_workers,
                )
                print(f"Saved samples to {sample_path}")
                print(f"Saved real samples to {real_sample_path}")
                # if wandb_run is not None:
                #     wandb.log(
                #         {
                #             "samples/step": wandb.Image(str(sample_path)),
                #             "samples/real_step": wandb.Image(str(real_sample_path)),
                #         },
                #         step=global_step,
                #     )

            if (
                metric_names
                and eval_step_interval > 0
                and global_step % eval_step_interval == 0
            ):
                try:
                    metric_results = {}
                    split_to_dataset = {
                        "train": train_dataset,
                        "test": test_dataset,
                    }
                    for split in eval_splits:
                        eval_dataset = split_to_dataset[split]
                        split_output_dir = output_dir / "metric_samples" / split
                        split_results = evaluate_time_series_metrics(
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
                            vae_ckpt_name=vae_ckpt_name,
                            output_dir=split_output_dir,
                            step=global_step,
                        )
                        metric_results.update(
                            prefix_metric_results(split_results, split)
                        )
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
                            best_fid_ckpt_path = output_dir / "checkpoint_best_fid.pt"
                            save_checkpoint(
                                str(best_fid_ckpt_path),
                                model,
                                ema,
                                optimizer,
                                scheduler,
                                epoch,
                                global_step,
                                config,
                            )
                            print(f"Saved best-FID checkpoint (test) {current_fid:.6f} -> {best_fid_ckpt_path}")
                            if wandb_run is not None:
                                wandb.log(
                                    {"metric/test/best_vae_fid": best_monitored_fid},
                                    step=global_step,
                                )
                except Exception as exc:
                    print(f"Metric evaluation failed at step {global_step}: {exc}")
                    if wandb_run is not None:
                        wandb.log(
                            {"metric/eval_failed": 1, "metric/eval_error": str(exc)},
                            step=global_step,
                        )

        # Epoch summary
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

        # Save checkpoint
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

        # Generate samples
        if (epoch + 1) % sample_interval == 0:
            sample_path = output_dir / f"samples_epoch{epoch+1}.png"
            real_sample_path = output_dir / f"real_samples_epoch{epoch+1}.png"
            generate_samples(
                ema.shadow,
                config,
                device,
                str(sample_path),
                num_samples=80,
            )
            save_real_time_series_samples(
                train_dataset,
                config,
                device,
                str(real_sample_path),
                num_samples=80,
                num_workers=num_workers,
            )
            print(f"Saved samples to {sample_path}")
            print(f"Saved real samples to {real_sample_path}")
            # if wandb_run is not None:
            #     wandb.log(
            #         {
            #             "samples/epoch": wandb.Image(str(sample_path)),
            #             "samples/real_epoch": wandb.Image(str(real_sample_path)),
            #         },
            #         step=global_step,
            #     )

        # Every 100 epochs: generate test-sized samples and evaluate discriminative score.
        if (epoch + 1) % 100 == 0:
            try:
                from metrics.discriminative_torch import discriminative_score_metrics

                gen_sig_test = generate_time_series_samples(
                    ema.shadow,
                    config,
                    device,
                    num_samples=test_eval_size,
                    batch_size=batch_size,
                ).numpy().astype(np.float32)

                disc_score = float(
                    discriminative_score_metrics(real_sig_test, gen_sig_test, device)
                )
                print(
                    f"Epoch {epoch+1} | test discriminative score (n={test_eval_size}): "
                    f"{disc_score:.6f}"
                )
                if wandb_run is not None:
                    wandb.log(
                        {
                            "metric/test/disc_epoch100": disc_score,
                            "metric/test/disc_epoch100_num_samples": test_eval_size,
                            "metric/test/disc_epoch100_epoch": epoch + 1,
                        },
                        step=global_step,
                    )
            except Exception as exc:
                print(f"Epoch {epoch+1} discriminative evaluation failed: {exc}")
                if wandb_run is not None:
                    wandb.log(
                        {
                            "metric/test/disc_epoch100_failed": 1,
                            "metric/test/disc_epoch100_error": str(exc),
                        },
                        step=global_step,
                    )

    # Final checkpoint
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
):
    """Save time-series samples as a grid of line plots."""
    if series.ndim == 2:
        series = series.unsqueeze(-1)

    num_samples, seq_len, channels = series.shape
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
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis("off")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.3)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def save_real_time_series_samples(
    dataset: Dataset,
    config: dict,
    device: torch.device,
    save_path: str,
    num_samples: int = 80,
    num_workers: int = 0,
):
    """Collect randomly sampled real delay-embedded samples and plot them."""
    n = min(num_samples, len(dataset))
    if n <= 0:
        raise ValueError("Dataset is empty; cannot save real samples.")

    # For visualization, random sampling gives a much more representative grid
    # than always taking the first contiguous windows.
    indices = random.sample(range(len(dataset)), k=n)
    sampled_dataset = torch.utils.data.Subset(dataset, indices)

    series = collect_real_time_series(
        sampled_dataset,
        config,
        device,
        num_samples=n,
        num_workers=num_workers,
    )
    save_time_series_grid(series, save_path, ncol=8)


@torch.no_grad()
def generate_samples(
    model: nn.Module,
    config: dict,
    device: torch.device,
    save_path: str,
    num_samples: int = 80,
):
    """Generate samples and save visualization."""
    model.eval()

    in_channels = config["in_channels"]
    img_size = config["img_size"]

    noise = torch.randn(num_samples, in_channels, img_size, img_size, device=device)
    samples = model(noise)

    series = delay_images_to_series(samples, config, device)
    save_time_series_grid(series, save_path, ncol=8)  # 10 x 8 for 80 samples

    return samples


def main():
    parser = argparse.ArgumentParser(description="Train Drifting Models")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/glucose_unconditional_debug",
        help="Output directory",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="Logging interval (steps)",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=100,
        help="Checkpoint save interval (epochs)",
    )
    parser.add_argument(
        "--sample_interval",
        type=int,
        default=10,
        help="Sample generation interval (epochs)",
    )
    parser.add_argument(
        "--eval_step_interval",
        "--eval_interval",
        dest="eval_step_interval",
        type=int,
        default=500,
        help="Metric evaluation interval in training steps. Set <= 0 to disable.",
    )
    parser.add_argument(
        "--eval_metrics",
        type=str,
        default="disc",
        help="Comma-separated metrics to run: disc,pred,contextFID,vaeFID",
    )
    parser.add_argument(
        "--eval_splits",
        type=parse_eval_splits,
        default=["train", "test"],
        help="Comma-separated splits for metric evaluation (default: train,test).",
    )
    parser.add_argument(
        "--eval_num_samples",
        type=int,
        default=None,
        help="Number of real/generated samples for metrics. Defaults to the full test set.",
    )
    parser.add_argument(
        "--metric_iteration",
        type=int,
        default=10,
        help="Number of repeated metric runs for mean/std metrics",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="drifting-model-ts",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Weights & Biases run name",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Weights & Biases entity",
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default=None,
        choices=[None, "online", "offline", "disabled"],
        help="Weights & Biases mode",
    )
    parser.add_argument(
        "--metrics_base_path",
        type=str,
        default=None,
        help="Base path for cached metric checkpoints/representations",
    )
    parser.add_argument(
        "--vae_ckpt_root",
        type=str,
        default=None,
        help="Checkpoint root for vaeFID",
    )
    parser.add_argument(
        "--vae_ckpt_name",
        type=str,
        default="best.pt",
        help="FID-VAE checkpoint filename to use for vaeFID (e.g., best.pt or last.pt).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Training DataLoader batch size.",
    )

    parser.add_argument("--model", type=str, default="DriftDiT-Tiny", choices=sorted(DriftDiT_models.keys()))
    parser.add_argument("--img_size", type=int, default=12)
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--one_channel", action="store_true")
    parser.add_argument("--batch_n_pos", type=int, default=320)
    parser.add_argument("--batch_n_neg", type=int, default=320)
    parser.add_argument("--temperatures", type=parse_temperatures, default=[0.02, 0.05, 0.2])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--drift_loss_mode",
        type=str,
        default="time_series",
        choices=["time_series", "vqvae", "both"],
        help="Loss feature space: raw time-series only, VQVAE only, or both.",
    )
    parser.add_argument("--queue_size", type=int, default=1280)

    parser.add_argument("--ts_seq_len", type=int, default=128)
    parser.add_argument("--ts_delay", type=int, default=12)
    parser.add_argument("--ts_embedding", type=int, default=12)
    parser.add_argument("--window_stride", type=int, default=None)
    parser.add_argument("--ts_stride", "--glucose_stride", dest="ts_stride", type=int, default=128)
    parser.add_argument("--stride", type=int, default=None)

    parser.add_argument("--vqvae_ckpt_path", type=str, default=None, help="Path to trained VQVAE checkpoint (e.g., best.pt).")
    parser.add_argument("--vqvae_hidden_size", type=int, default=32)
    parser.add_argument("--vqvae_num_layers", type=int, default=1)
    parser.add_argument("--vqvae_code_dim", type=int, default=8)
    parser.add_argument("--vqvae_num_codes", type=int, default=150)
    parser.add_argument("--vqvae_latent_downsample", type=int, default=16)
    parser.add_argument("--vqvae_decoder_upsample_rate", type=int, default=4)
    parser.add_argument("--vqvae_dropout", type=float, default=0.1)
    parser.add_argument("--vqvae_commitment_weight", type=float, default=0.25)


    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--datasets_dir", type=str, required=True)
    parser.add_argument("--rel_path", type=str, default=None)
    parser.add_argument("--rel_path_train", type=str, default=None)
    parser.add_argument("--rel_path_valid", type=str, default=None)

    args = parser.parse_args()
    if args.rel_path is None and not (args.rel_path_train and args.rel_path_valid):
        parser.error(
            "Provide --rel_path, or provide both --rel_path_train and --rel_path_valid."
        )
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
