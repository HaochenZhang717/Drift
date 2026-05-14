import argparse
import copy
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    import wandb
except ImportError:
    wandb = None


_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_THIS_FILE.parents[1]) not in sys.path:
    sys.path.insert(0, str(_THIS_FILE.parents[1]))

from baselines.JITT1.denoiser import Denoiser
from data_provider.data_provider import data_provider


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_ema_decays(value: str) -> List[float]:
    decays = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not decays:
        raise ValueError("--ema_decays must contain at least one decay.")
    for d in decays:
        if d < 0.0 or d >= 1.0:
            raise ValueError(f"EMA decay must be in [0, 1), got {d}")
    return decays


def to_tensor_data(batch):
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch


def update_ema_model(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    with torch.no_grad():
        msd = model.state_dict()
        esd = ema_model.state_dict()
        for k in esd.keys():
            if not torch.is_floating_point(esd[k]):
                esd[k].copy_(msd[k])
            else:
                esd[k].mul_(decay).add_(msd[k], alpha=(1.0 - decay))


def run_validation(
    model: torch.nn.Module,
    dataset_loader,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    dataset_losses: Dict[str, float] = {}
    dataset_counts: Dict[str, int] = {}

    with torch.no_grad():
        for dataset_name, testset in dataset_loader.testsets.items():
            test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            sum_loss = 0.0
            n_batches = 0
            for batch in test_loader:
                x = to_tensor_data(batch).to(device=device, dtype=torch.float32)
                loss = model(x)
                sum_loss += float(loss.item())
                n_batches += 1
            if n_batches == 0:
                continue
            dataset_losses[dataset_name] = sum_loss / n_batches
            dataset_counts[dataset_name] = n_batches

    if not dataset_losses:
        return float("nan"), {}

    total_batches = sum(dataset_counts.values())
    weighted = sum(dataset_losses[k] * dataset_counts[k] for k in dataset_losses) / max(total_batches, 1)
    return weighted, dataset_losses


def build_data_args(args) -> SimpleNamespace:
    dataset_configs = []
    if "GlucoseSliding" in args.train_on_datasets:
        dataset_configs.append({
            "name": "GlucoseSliding",
            "data": "GlucoseSliding",
            "rel_path": args.glucose_rel_path,
            "window_stride": args.glucose_stride,
        })
    if "ErcotData" in args.train_on_datasets:
        dataset_configs.append({
            "name": "ErcotData",
            "data": "ErcotData",
            "rel_path": args.ercot_rel_path,
            "window_stride": args.ercot_stride,
        })
    if "HouseholdData" in args.train_on_datasets:
        dataset_configs.append({
            "name": "HouseholdData",
            "data": "HouseholdData",
            "rel_path": args.household_rel_path,
            "window_stride": args.household_stride,
        })

    if not dataset_configs:
        raise ValueError("No valid datasets selected in --train_on_datasets.")

    return SimpleNamespace(
        datasets=dataset_configs,
        train_on_datasets=args.train_on_datasets,
        seq_len=args.seq_len,
        datasets_dir=args.datasets_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        ddp=False,
        finetune=False,
        input_channels=args.input_channels,
        handler="",
        load_long_clip_context=False,
        verbalts_context_suffix=None,
        verbalts_text_npy_name="my_text_caps",
        caption_embeddings_path=None,
    )


def build_model_args(args, enc_in: int) -> SimpleNamespace:
    return SimpleNamespace(
        seq_len=args.seq_len,
        enc_in=enc_in,
        n_heads=args.n_heads,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        n_blocks=tuple(args.n_blocks),
        kernel_size_large=tuple(args.kernel_size_large),
        kernel_size_small=args.kernel_size_small,
        ffn_ratio=args.ffn_ratio,
        downsample_ratio=args.downsample_ratio,
        qkv_bias=not args.no_qkv_bias,
        drop_attn=args.drop_attn,
        drop_ffn=args.drop_ffn,
        drop_proj=args.drop_proj,
        drop_head=args.drop_head,
        positional_encoding=not args.no_positional_encoding,
        P_mean=args.P_mean,
        P_std=args.P_std,
        t_eps=args.t_eps,
        noise_scale=args.noise_scale,
        sampling_method=args.sampling_method,
        num_sampling_steps=args.num_sampling_steps,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train JITT1 diffusion denoiser")

    parser.add_argument("--datasets_dir", type=str, required=True)
    parser.add_argument("--train_on_datasets", type=parse_csv_list, default=parse_csv_list("GlucoseSliding,ErcotData,HouseholdData"))
    parser.add_argument("--glucose_rel_path", type=str, default="glucose_{split}.parquet")
    parser.add_argument("--ercot_rel_path", type=str, default="ERCOT_merged.csv")
    parser.add_argument("--household_rel_path", type=str, default="HouseHold_6.csv")
    parser.add_argument("--glucose_stride", type=int, default=1)
    parser.add_argument("--ercot_stride", type=int, default=1)
    parser.add_argument("--household_stride", type=int, default=1)

    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--input_channels", type=int, default=None)

    parser.add_argument("--n_heads", type=int, default=128)
    parser.add_argument("--patch_size", type=int, default=2)
    parser.add_argument("--patch_stride", type=int, default=1)
    parser.add_argument("--n_blocks", type=int, nargs="+", default=[2, 2])
    parser.add_argument("--kernel_size_large", type=int, nargs="+", default=[71, 31])
    parser.add_argument("--kernel_size_small", type=int, default=5)
    parser.add_argument("--ffn_ratio", type=float, default=1.0)
    parser.add_argument("--downsample_ratio", type=int, default=2)
    parser.add_argument("--no_qkv_bias", action="store_true")
    parser.add_argument("--drop_attn", type=float, default=0.0)
    parser.add_argument("--drop_ffn", type=float, default=0.0)
    parser.add_argument("--drop_proj", type=float, default=0.0)
    parser.add_argument("--drop_head", type=float, default=0.0)
    parser.add_argument("--no_positional_encoding", action="store_true")

    parser.add_argument("--P_mean", type=float, default=0.0)
    parser.add_argument("--P_std", type=float, default=1.0)
    parser.add_argument("--t_eps", type=float, default=1e-5)
    parser.add_argument("--noise_scale", type=float, default=1.0)
    parser.add_argument("--sampling_method", type=str, default="euler", choices=["euler", "heun"])
    parser.add_argument("--num_sampling_steps", type=int, default=50)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=0)
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=100)

    parser.add_argument("--ema_decays", type=parse_ema_decays, default=parse_ema_decays("0.999,0.9999"))

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--output_dir", type=str, default="baselines/JITT1/checkpoints")
    parser.add_argument("--run_name", type=str, default=None)

    parser.add_argument("--wandb_project", type=str, default="JITT1")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])

    args = parser.parse_args()

    if len(args.kernel_size_large) != len(args.n_blocks):
        if len(args.kernel_size_large) == 1:
            args.kernel_size_large = args.kernel_size_large * len(args.n_blocks)
        else:
            raise ValueError("--kernel_size_large length must be 1 or equal to --n_blocks length")

    set_seed(args.seed)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    run_name = args.run_name or f"jitt1_{'_'.join(args.train_on_datasets)}_L{args.seq_len}_{int(time.time())}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    data_args = build_data_args(args)
    dataset_loader, _, _, metadatas = data_provider(data_args)

    enc_in = max(int(meta["channels"]) for meta in metadatas.values())
    model_args = build_model_args(args, enc_in)
    model = Denoiser(model_args).to(device)

    ema_models = {}
    for decay in args.ema_decays:
        key = f"ema_{str(decay).replace('.', 'p')}"
        ema_model = copy.deepcopy(model).to(device)
        ema_model.eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)
        ema_models[key] = {"decay": decay, "model": ema_model}

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    use_wandb = args.wandb_mode != "disabled" and wandb is not None
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            mode=args.wandb_mode,
            config={
                **vars(args),
                "enc_in": enc_in,
                "resolved_metadatas": metadatas,
            },
        )

    global_step = 0
    best_val = float("inf")

    def build_models_state() -> Dict[str, Dict[str, torch.Tensor]]:
        states = {"online": model.state_dict()}
        for name, st in ema_models.items():
            states[name] = st["model"].state_dict()
        return states

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        epoch_start = time.time()

        for batch, _dataset_label in iter(dataset_loader):
            x = to_tensor_data(batch).to(device=device, dtype=torch.float32)

            optimizer.zero_grad(set_to_none=True)
            loss = model(x)
            loss.backward()

            if args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            for ema_state in ema_models.values():
                update_ema_model(ema_state["model"], model, ema_state["decay"])

            loss_value = float(loss.item())
            train_losses.append(loss_value)

            global_step += 1
            if args.log_interval and args.log_interval > 0 and global_step % args.log_interval == 0:
                log_payload = {
                    "train/loss_step": loss_value,
                    "train/epoch": epoch,
                    "train/global_step": global_step,
                }
                if use_wandb:
                    wandb.log(log_payload, step=global_step)

        epoch_train_loss = float(np.mean(train_losses)) if train_losses else float("nan")

        log_dict = {
            "train/loss_epoch": epoch_train_loss,
            "train/epoch": epoch,
            "train/global_step": global_step,
            "time/epoch_seconds": time.time() - epoch_start,
        }

        do_val = (epoch % args.val_every == 0)
        if do_val:
            val_loss, val_per_dataset = run_validation(
                model=model,
                dataset_loader=dataset_loader,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
            )
            log_dict["val/loss"] = val_loss
            for ds_name, ds_loss in val_per_dataset.items():
                log_dict[f"val/{ds_name}_loss"] = ds_loss

            for ema_name, ema_state in ema_models.items():
                ema_val_loss, ema_val_per_dataset = run_validation(
                    model=ema_state["model"],
                    dataset_loader=dataset_loader,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    device=device,
                )
                log_dict[f"val/{ema_name}_loss"] = ema_val_loss
                for ds_name, ds_loss in ema_val_per_dataset.items():
                    log_dict[f"val/{ema_name}_{ds_name}_loss"] = ds_loss

            if not math.isnan(val_loss) and val_loss < best_val:
                best_val = val_loss
                best_ckpt = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model": model.state_dict(),
                    "models": build_models_state(),
                    "optimizer": optimizer.state_dict(),
                    "best_val": best_val,
                    "args": vars(args),
                    "model_args": vars(model_args),
                    "ema_models": {name: st["model"].state_dict() for name, st in ema_models.items()},
                    "ema_decays": {name: st["decay"] for name, st in ema_models.items()},
                }
                torch.save(best_ckpt, output_dir / "best.pt")

        ckpt = {
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "models": build_models_state(),
            "optimizer": optimizer.state_dict(),
            "best_val": best_val,
            "args": vars(args),
            "model_args": vars(model_args),
            "ema_models": {name: st["model"].state_dict() for name, st in ema_models.items()},
            "ema_decays": {name: st["decay"] for name, st in ema_models.items()},
        }
        torch.save(ckpt, output_dir / "last.pt")
        if args.save_every and args.save_every > 0 and epoch % args.save_every == 0:
            torch.save(ckpt, output_dir / f"epoch_{epoch:04d}.pt")

        if use_wandb:
            wandb.log(log_dict, step=global_step)

        print(json.dumps({k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in log_dict.items()}))

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
