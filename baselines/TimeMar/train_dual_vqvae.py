import os
import time
import math
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lrs

from dataset.get_datasets import build_dataloader
from Utils.base_utils import load_model_path_by_config, ConfigLoader
from Utils.context_fid import Context_FID
from Utils.metric_utils import display_scores
from models import DualVQVAE
from omegaconf import OmegaConf




# -------------------------
# Helper
# -------------------------
def get_loss_fn(loss_name: str):
    loss = loss_name.lower()
    if loss == "mse":
        return F.mse_loss
    if loss == "l1":
        return F.l1_loss
    if loss == "bce":
        return F.binary_cross_entropy
    raise ValueError(f"Invalid loss type: {loss_name}")


def fft_loss(batch: torch.Tensor, x_hat: torch.Tensor):
    fft1 = torch.fft.fft(batch.transpose(1, 2), norm="forward")
    fft2 = torch.fft.fft(x_hat.transpose(1, 2), norm="forward")
    fft1, fft2 = fft1.transpose(1, 2), fft2.transpose(1, 2)

    fourier = F.l1_loss(torch.real(fft1), torch.real(fft2), reduction="none") + \
              F.l1_loss(torch.imag(fft1), torch.imag(fft2), reduction="none")

    ff_weight = math.sqrt(24) / 5
    return ff_weight * fourier.mean(dim=1).mean(dim=1).mean(dim=0)


def build_optimizer_and_scheduler(model, config):
    weight_decay = getattr(config, "weight_decay", 0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=weight_decay)

    if getattr(config, "lr_scheduler", None) is None:
        return optimizer, None

    sch = config.lr_scheduler
    if sch == "step":
        scheduler = lrs.StepLR(
            optimizer,
            step_size=config.lr_decay_steps,
            gamma=config.lr_decay_rate,
        )
    elif sch == "cosine":
        scheduler = lrs.CosineAnnealingLR(
            optimizer,
            T_max=config.lr_decay_steps,
            eta_min=config.lr_decay_min_lr,
        )
    elif sch == "Reduce":
        scheduler = lrs.ReduceLROnPlateau(optimizer, mode="min", factor=0.8, patience=5)
    else:
        raise ValueError(f"Invalid lr_scheduler type: {sch}")
    return optimizer, scheduler


def model_size_stats(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return {
        "total_params": total,
        "trainable_params": trainable,
        "param_mb": param_bytes / 1024 ** 2,
        "buffer_mb": buffer_bytes / 1024 ** 2,
    }


def print_model_size(name, model):
    stats = model_size_stats(model)
    print(
        f"[MODEL SIZE] {name}: "
        f"total_params={stats['total_params']:,} "
        f"trainable_params={stats['trainable_params']:,} "
        f"param_memory={stats['param_mb']:.2f} MiB "
        f"buffer_memory={stats['buffer_mb']:.2f} MiB"
    )
    return stats


def save_ckpt(path, model, optimizer, scheduler, epoch, best_fid, config_dict=None):
    ckpt = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_fid": best_fid,
    }
    if scheduler is not None and not isinstance(scheduler, lrs.ReduceLROnPlateau):
        ckpt["scheduler"] = scheduler.state_dict()
    if config_dict is not None:
        ckpt["config"] = config_dict
    torch.save(ckpt, path)


def load_ckpt(path, model, optimizer=None, scheduler=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["state_dict"], strict=True)

    start_epoch = ckpt.get("epoch", -1) + 1
    best_fid = ckpt.get("best_fid", float("inf"))

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    if scheduler is not None and "scheduler" in ckpt:
        try:
            scheduler.load_state_dict(ckpt["scheduler"])
        except Exception:
            pass

    return start_epoch, best_fid


# @torch.no_grad()
def run_validation_and_fid(model, val_loader, loss_fn, device):
    model.eval()

    total_loss = 0.0
    total_recon = 0.0
    total_vq = 0.0
    n_batches = 0

    outs = []
    gts = []

    for batch in val_loader:
        batch = batch.to(device)

        with torch.no_grad():
            (trend, seasonal, coarse_seasonal,
             recon_trend, recon_seasonal, recon_coarse_seasonal,
             x_hat, usages, vq_loss) = model(batch, ret_usages=True)

        l_reconstruct = loss_fn(batch, x_hat)
        l_trend = loss_fn(trend, recon_trend)
        l_season = loss_fn(seasonal, recon_seasonal)
        l_coarse = loss_fn(coarse_seasonal, recon_coarse_seasonal)
        l_fft = fft_loss(seasonal, recon_seasonal)

        loss = l_reconstruct * 50 + vq_loss * 0.25 + l_trend * 1 + l_season * 2.5 + l_fft * 1 + l_coarse * 0.1

        total_loss += float(loss.item())
        total_recon += float(l_reconstruct.item())
        total_vq += float(vq_loss.item())
        n_batches += 1

        outs.append(x_hat.detach().cpu())
        gts.append(batch.detach().cpu())

    outputs_np = torch.cat(outs, dim=0).numpy()
    ground_truth_np = torch.cat(gts, dim=0).numpy()

    if len(outputs_np) < 1000:
        fid = Context_FID(ground_truth_np[:outputs_np.shape[0]], outputs_np)
    else:
        fid = Context_FID(ground_truth_np, outputs_np)

    return {
        "eval/valid_loss": total_loss / max(1, n_batches),
        "eval/valid_l_reconstruct": total_recon / max(1, n_batches),
        "eval/valid_vq_loss": total_vq / max(1, n_batches),
        "eval/fid": float(fid),
        "eval/generate_len": int(len(outputs_np)),
    }


# -------------------------
# Argparse
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser("DualVQVAE Training (Pure PyTorch)")

    parser.add_argument("--data", type=str, default="stock",
                        help="dataset name, e.g., stock/energy/traffic")

    parser.add_argument("--config", type=str, default=None,
                        help="path to yaml config; if None, use configs/train_vq_{data}.yaml")

    parser.add_argument("--device", type=str, default=None,
                        help="cuda / cuda:0 / cpu; default auto")

    parser.add_argument("--max_epochs", type=int, default=None,
                        help="override config max_epochs")
    parser.add_argument("--val_every", type=int, default=None,
                        help="override config val_every")

    parser.add_argument("--save_dir", type=str, default="log_torch",
                        help="root directory to save checkpoints/logs")
    parser.add_argument("--dry_run_model_size", action="store_true",
                        help="build the model, print parameter counts, then exit before training")
    parser.add_argument("--max_model_params", type=int, default=None,
                        help="abort before training if total parameter count is above this value")






    return parser.parse_args()


def main():
    args = parse_args()

    # ---- load config ----
    if args.config is None:
        config_path = f"configs/train_vq_{args.data}.yaml"
    else:
        config_path = args.config

    config = ConfigLoader.load_vq_config(config=config_path)

    # ---- seed ----
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # ---- device ----
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("Using device:", device)
    print("Using config:", config_path)
    max_epochs = args.max_epochs if args.max_epochs is not None else getattr(config, "max_epochs", 5000)
    val_every = args.val_every if args.val_every is not None else getattr(config, "val_every", 100)
    print(f"Training schedule: max_epochs={max_epochs}, val_every={val_every}")

    # ---- init model ----
    import inspect
    init_args = inspect.getfullargspec(DualVQVAE.__init__).args[1:]
    cfg_keys = set(config.keys()) if hasattr(config, "keys") else set(vars(config).keys())

    kwargs = {}
    for a in init_args:
        if a in cfg_keys:
            kwargs[a] = getattr(config, a)

    model = DualVQVAE(**kwargs).to(device)
    size_stats = print_model_size("DualVQVAE", model)
    if args.max_model_params is not None and size_stats["total_params"] > args.max_model_params:
        raise SystemExit(
            f"Model has {size_stats['total_params']:,} parameters, "
            f"which exceeds --max_model_params={args.max_model_params:,}."
        )
    if args.dry_run_model_size:
        print("Dry run requested; exiting before optimizer/dataloader training loop.")
        return

    import wandb

    wandb.init(
        project="DualVQVAE",
        name=f"vq_{args.data}",
        config=OmegaConf.to_container(config, resolve=True),
    )

    # ---- dataloader ----
    train_loader, val_loader = build_dataloader(config)

    loss_fn = get_loss_fn(config.loss)
    optimizer, scheduler = build_optimizer_and_scheduler(model, config)

    start_epoch = 0
    best_fid = float("inf")

    # ---- dirs ----
    run_dir = os.path.join(args.save_dir, f"vq_{args.data}")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---- train loop ----
    global_step = 0
    for epoch in range(start_epoch, max_epochs):
        model.train()
        t0 = time.time()

        running_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            batch = batch.to(device)

            (trend, seasonal, coarse_seasonal,
             recon_trend, recon_seasonal, recon_coarse_seasonal,
             x_hat, usages, vq_loss) = model(batch, ret_usages=True)

            l_reconstruct = loss_fn(batch, x_hat)
            l_trend = loss_fn(trend, recon_trend)
            l_season = loss_fn(seasonal, recon_seasonal)
            l_coarse = loss_fn(coarse_seasonal, recon_coarse_seasonal)
            l_fft = fft_loss(seasonal, recon_seasonal)

            loss = l_reconstruct * 50 + vq_loss * 0.25 + l_trend * 1 + l_season * 2.5 + l_fft * 1 + l_coarse * 0.1
            wandb.log({
                "train/loss_total": loss.item(),

                "train/loss_reconstruct": l_reconstruct.item(),
                "train/loss_trend": l_trend.item(),
                "train/loss_season": l_season.item(),
                "train/loss_coarse": l_coarse.item(),
                "train/loss_fft": l_fft.item(),

                "train/epoch": epoch,
                "train/step": global_step,
                "train/usage": usages,
            })

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            running_loss += float(loss.item())
            n_batches += 1

        # scheduler step
        if scheduler is not None and not isinstance(scheduler, lrs.ReduceLROnPlateau):
            scheduler.step()

        dt = time.time() - t0
        print(f"Epoch {epoch} done | train_loss={running_loss/max(1,n_batches):.4f} time={dt:.1f}s")

        # ---- validation & fid ----
        if epoch % val_every == 0:
            stats = run_validation_and_fid(model, val_loader, loss_fn, device)
            wandb.log(stats)
            print(
                f"[VAL E{epoch:04d}] "
                f"valid_loss={stats['eval/valid_loss']:.4f} "
                f"fid={stats['eval/fid']:.6f} "
                f"generate_len={stats['eval/generate_len']}"
            )
            display_scores([stats["eval/fid"]])

            # ReduceLROnPlateau step needs metric
            if scheduler is not None and isinstance(scheduler, lrs.ReduceLROnPlateau):
                scheduler.step(stats["eval/valid_loss"])

            # save latest
            latest_path = os.path.join(ckpt_dir, "latest.pt")
            save_ckpt(latest_path, model, optimizer, scheduler, epoch, best_fid, config_dict=vars(config))

            # save best by fid
            if stats["eval/fid"] < best_fid:
                best_fid = stats["eval/fid"]
                # best_path = os.path.join(ckpt_dir, f"best_epoch{epoch:04d}_fid{best_fid:.6f}.pt")
                best_path = os.path.join(ckpt_dir, f"best.pt")
                save_ckpt(best_path, model, optimizer, scheduler, epoch, best_fid, config_dict=vars(config))
                print("Saved best:", best_path)


if __name__ == "__main__":
    main()
