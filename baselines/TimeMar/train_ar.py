import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau

from models import DualVQVAE
from models import VAR
from Utils.context_fid import Context_FID

from dataset.get_datasets import build_dataloader_var
from Utils.base_utils import load_model_path_by_config, ConfigLoader

import argparse


# -------------------------
# Utils
# -------------------------
def set_seed(seed=0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_scheduler(optimizer, config):
    if config.lr_scheduler is None:
        return None

    if config.lr_scheduler == "step":
        return StepLR(optimizer, step_size=config.lr_decay_steps, gamma=config.lr_decay_rate)

    if config.lr_scheduler == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=config.lr_decay_steps,
            eta_min=config.lr_decay_min_lr
        )

    if config.lr_scheduler == "Reduce":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.8,
            patience=5,
            verbose=True
        )

    raise ValueError(f"Unknown lr_scheduler: {config.lr_scheduler}")


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


@torch.no_grad()
def encode_ts_to_tokens(vqvae: DualVQVAE, batch):
    """
    batch: [B, T, C] or [B, T]
    return: gt_BL [B, L]
    """
    gt_idx_Bl = vqvae.ts_to_idxBl(batch)
    gt_BL = torch.cat(gt_idx_Bl, dim=1)
    return gt_BL


def shift_right(gt_BL, start_token_id):
    """
    teacher forcing input: shift gt right by 1
    """
    inp = gt_BL.clone()
    inp[:, 1:] = inp[:, :-1]
    inp[:, 0] = start_token_id
    return inp


def compute_loss_acc(model, gt_BL, vocab_size, loss_weight):
    """
    model: AR model, output logits [B, L, V]
    gt_BL: [B, L]
    """
    B, L = gt_BL.shape
    start_token_id = vocab_size

    inp = shift_right(gt_BL, start_token_id=start_token_id)
    logits_BLV = model(inp)  # [B, L, V]

    loss_fn = nn.CrossEntropyLoss(reduction="none")
    loss = loss_fn(
        logits_BLV.reshape(-1, vocab_size),
        gt_BL.reshape(-1)
    ).reshape(B, L)

    lw = loss_weight.to(gt_BL.device)
    loss = (loss * lw).sum(dim=-1).mean()

    acc = (logits_BLV.argmax(dim=-1) == gt_BL).float().mean()
    return loss, acc


# -------------------------
# Train / Eval
# -------------------------
def train_one_epoch(model, vqvae, loader, optimizer, device, loss_weight):
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0

    vocab_size = vqvae.vocab_size

    for batch in loader:
        batch = batch.to(device)

        with torch.no_grad():
            gt_BL = encode_ts_to_tokens(vqvae, batch)

        loss, acc = compute_loss_acc(model, gt_BL, vocab_size, loss_weight)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_acc += acc.item()
        n += 1

    return total_loss / n, total_acc / n


@torch.no_grad()
def eval_one_epoch(model, vqvae, loader, device, loss_weight):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0

    vocab_size = vqvae.vocab_size

    for batch in loader:
        batch = batch.to(device)
        gt_BL = encode_ts_to_tokens(vqvae, batch)

        loss, acc = compute_loss_acc(model, gt_BL, vocab_size, loss_weight)

        total_loss += loss.item()
        total_acc += acc.item()
        n += 1

    return total_loss / n, total_acc / n


# -------------------------
# Sampling + FID
# -------------------------
@torch.no_grad()
def sample_tokens(model, vqvae, num_samples, batch_size, L, device):
    model.eval()
    vocab_size = vqvae.vocab_size

    n_iters = num_samples // batch_size + 1
    out_data_list = []

    for _ in range(n_iters):
        inp = vocab_size * torch.ones(batch_size, L, dtype=torch.int32, device=device)
        pre_idx = model.sample(input_idx=inp, if_categorial=True)
        out_data_list.append(pre_idx)

    # keep your original reorder logic
    out_reorder = []
    for i in range(len(out_data_list[0])):
        for j in range(len(out_data_list)):
            out_reorder.append(out_data_list[j][i])

    generated = torch.stack(out_reorder)
    return generated.detach().cpu().numpy()


def compute_context_fid(dataset, window_size, generated_np):
    file_path = f"./output/samples/{dataset}_norm_truth_{window_size}_train.npy"
    if dataset == "Sines":
        file_path = f"./output/samples/{dataset}_ground_truth_{window_size}_train.npy"

    gt = np.load(file_path)
    generated_np = generated_np[:gt.shape[0]]

    fid = Context_FID(gt, generated_np)
    return float(fid)


# -------------------------
# Main train loop
# -------------------------
def train_ar(
    config,
    dataset_name,
    train_loader,
    val_loader,
    vqvae_ckpt_path,
    save_dir,
    dry_run_model_size=False,
    max_model_params=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)

    set_seed(config.seed)

    # ===== build vqvae =====
    vqvae = DualVQVAE(**config.vqvae_args)
    if vqvae_ckpt_path is not None:
        ckpt = torch.load(vqvae_ckpt_path, map_location="cpu", weights_only=False)
        assert 'state_dict' in ckpt
        state_dict = ckpt['state_dict']
        vqvae.load_state_dict(state_dict, strict=True)

    vqvae.to(device)
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad = False
    vqvae_stats = print_model_size("DualVQVAE frozen tokenizer", vqvae)

    # ===== build AR model =====
    model = VAR(vqvae, **config.var_args)   # 你可能要换成 VAR(**config.var_args)
    model.to(device)
    var_stats = print_model_size("VAR trainable prior", model)
    combined_params = vqvae_stats["total_params"] + var_stats["total_params"]
    print(f"[MODEL SIZE] Combined VQ-VAE + VAR total_params={combined_params:,}")

    if max_model_params is not None and combined_params > max_model_params:
        raise SystemExit(
            f"Combined model has {combined_params:,} parameters, "
            f"which exceeds --max_model_params={max_model_params:,}."
        )
    if dry_run_model_size:
        print("Dry run requested; exiting before optimizer/dataloader training loop.")
        return model


    # ===== loss weight =====
    patch_nums = config.vqvae_args.v_patch_nums
    L = sum(patch_nums)
    loss_weight = torch.ones(1, L, device=device) / L

    # ===== optimizer + scheduler =====
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=getattr(config, "weight_decay", 0.0))
    scheduler = build_scheduler(optimizer, config)

    # best_fid = 1e9

    for epoch in range(config.max_epochs):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, vqvae, train_loader, optimizer, device, loss_weight
        )
        val_loss, val_acc = eval_one_epoch(
            model, vqvae, val_loader, device, loss_weight
        )

        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        print(f"[Epoch {epoch}] "
              f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, "
              f"time={time.time()-t0:.2f}s")

        if epoch % 100 == 0 and epoch > 0:
            # ---- save latest ----
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, os.path.join(save_dir, f"epoch_{epoch}.pt"))

        # ---- FID ----
        # if epoch % 100 == 0 and epoch > 0:
        #     if dataset_name in ["Sines", "Mujoco", "fMRI"]:
        #         num = 10000
        #     elif dataset_name == "stock":
        #         num = 4000
        #     else:
        #         num = 20000
        #
        #     generated_np = sample_tokens(
        #         model=model,
        #         vqvae=vqvae,
        #         num_samples=num,
        #         batch_size=config.batch_size,
        #         L=L,
        #         device=device,
        #     )
        #
        #     fid = compute_context_fid(
        #         dataset_name,
        #         window_size=config.dataloader.params.window,
        #         generated_np=generated_np
        #     )
        #     print(f"[Epoch {epoch}] Context-FID = {fid:.4f}")
        #
        #     if fid < best_fid:
        #         best_fid = fid
        #         ckpt_path = os.path.join(save_dir, f"best_epoch{epoch}_fid{fid:.4f}.pt")
        #         torch.save({
        #             "epoch": epoch,
        #             "fid": fid,
        #             "var_model": model.state_dict(),
        #             "dual_vqvae": vqvae.state_dict(),
        #             "optimizer": optimizer.state_dict(),
        #         }, ckpt_path)
        #         print(f"Saved BEST checkpoint to {ckpt_path}")

    return model




def parse_args():
    parser = argparse.ArgumentParser("TimeMAR Training (Pure PyTorch)")
    parser.add_argument("--data", type=str, default="stock", help="dataset name, e.g., stock/energy/traffic")
    parser.add_argument("--config", type=str, default=None, help="path to yaml config; if None, use configs/train_vq_{data}.yaml")
    parser.add_argument("--vqvae_path", type=str, default=None, help="trained vqvae ckpt path")
    parser.add_argument("--save_dir", type=str, default="log_torch", help="root directory to save checkpoints/logs")
    parser.add_argument("--dry_run_model_size", action="store_true",
                        help="build the models, print parameter counts, then exit before training")
    parser.add_argument("--max_model_params", type=int, default=None,
                        help="abort before training if combined VQ-VAE + VAR parameter count is above this value")

    return parser.parse_args()


def main():
    args = parse_args()

    # config = ConfigLoader.load_var_config(config=f"configs/train_var_{args.data}.yaml")
    config = ConfigLoader.load_var_config(config=args.config)

    if args.dry_run_model_size:
        train_loader, val_loader = None, None
    else:
        train_loader, val_loader = build_dataloader_var(config, args.data)
    # vqvae_path = load_model_path_by_config(config)

    # save_dir = f"output/{args.data}/ar_checkpoints"

    train_ar(
        config=config,
        dataset_name=args.data,
        train_loader=train_loader,
        val_loader=val_loader,
        vqvae_ckpt_path=args.vqvae_path,
        save_dir=args.save_dir,
        dry_run_model_size=args.dry_run_model_size,
        max_model_params=args.max_model_params,
    )

if __name__ == "__main__":
    main()
