"""
SDFlow
"""

import math
import os
import sys
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    class SummaryWriter:
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def close(self):
            pass
from tqdm import tqdm
import argparse
import hashlib
from sklearn.mixture import GaussianMixture

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import options.option_transformer as option_trans
import models.vqvae as vqvae
import utils.utils_model as utils_model
from dataset import dataset_VQ
try:
    from metrics.discriminative_metrics import discriminative_score_metrics
    from metrics.context_fid import Context_FID
    from metrics.predictive_metrics import predictive_score_metrics2 as predictive_score_metrics
except ModuleNotFoundError:
    discriminative_score_metrics = None
    Context_FID = None
    predictive_score_metrics = None
from models.sdflow import SDFlowModel, KDEPrior


def infer_input_emb_width_from_checkpoint(checkpoint):
    weight = checkpoint['net'].get('vqvae.encoder.model.0.weight')
    return int(weight.shape[1]) if weight is not None else None


def apply_vqvae_checkpoint_args(vqvae_args, checkpoint):
    saved_args = checkpoint.get('args') or {}
    for key in [
        'down_t',
        'stride_t',
        'width',
        'depth',
        'dilation_growth_rate',
        'vq_act',
        'vq_norm',
        'quantizer',
        'input_emb_width',
    ]:
        if key in saved_args:
            setattr(vqvae_args, key, saved_args[key])
    if getattr(vqvae_args, 'input_emb_width', None) is None:
        vqvae_args.input_emb_width = infer_input_emb_width_from_checkpoint(checkpoint)
    return vqvae_args


def auto_encode_and_cache(vqvae_ckpt, dataname, cache_dir, device='cuda',
                          window_size=None, down_t=None, quantizer=None,
                          datasets_dir=None, rel_path=None,
                          rel_path_train=None, rel_path_valid=None,
                          rel_path_test=None, stride=1,
                          window_stride=None, ts_stride=None,
                          input_emb_width=None, column='glucose'):
    print("\n" + "="*60)
    print("Preprocessing")
    print("="*60)

    vqvae_checkpoint = torch.load(vqvae_ckpt, map_location='cpu', weights_only=False)
    codebook_raw = vqvae_checkpoint['net']['vqvae.quantizer.codebook']
    num_codes, code_dim = codebook_raw.shape

    sys_argv_backup = sys.argv.copy()
    sys.argv = [sys.argv[0], '--dataname', dataname]
    vqvae_args = option_trans.get_args_parser()
    sys.argv = sys_argv_backup
    vqvae_args = apply_vqvae_checkpoint_args(vqvae_args, vqvae_checkpoint)

    if quantizer:
        vqvae_args.quantizer = quantizer
    if window_size:
        vqvae_args.window_size = window_size
    if down_t:
        vqvae_args.down_t = down_t
    if input_emb_width is not None:
        vqvae_args.input_emb_width = input_emb_width
    elif getattr(vqvae_args, 'input_emb_width', None) is None:
        vqvae_args.input_emb_width = infer_input_emb_width_from_checkpoint(vqvae_checkpoint)

    window_size = vqvae_args.window_size
    down_t = vqvae_args.down_t

    quantizer_str = quantizer if quantizer else "default"
    data_str = f"{datasets_dir}_{rel_path}_{rel_path_train}_{rel_path_valid}_{rel_path_test}_{stride}_{window_stride}_{ts_stride}_{column}"
    config_str = f"{dataname}_{window_size}_{down_t}_{num_codes}_{code_dim}_{quantizer_str}_{data_str}"
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

    cache_subdir = os.path.join(cache_dir, f'auto_cache_{dataname}_{config_hash}')
    codebook_path = os.path.join(cache_subdir, 'codebook.pth')
    indices_path = os.path.join(cache_subdir, 'train_indices.pth')
    config_path = os.path.join(cache_subdir, 'config.json')

    cache_valid = False
    if os.path.exists(config_path) and os.path.exists(codebook_path) and os.path.exists(indices_path):
        try:
            with open(config_path, 'r') as f:
                saved_config = json.load(f)
            if (saved_config['dataname'] == dataname and
                saved_config['window_size'] == window_size and
                saved_config['down_t'] == down_t and
                saved_config.get('data_str') == data_str):
                cache_valid = True
        except:
            pass

    if cache_valid:
        codebook = torch.load(codebook_path, weights_only=False)
        train_indices = torch.load(indices_path, weights_only=False)
        print(f"  Codebook: {codebook.shape}")
        print(f"  Indices: {train_indices.shape}")
        print("="*60 + "\n")
        return codebook, train_indices, train_indices.shape[1]

    print("\nEncoding data...")
    os.makedirs(cache_subdir, exist_ok=True)

    vq_model = vqvae.VQVAE(
        vqvae_args, num_codes, code_dim,
        vqvae_args.down_t, vqvae_args.stride_t,
        vqvae_args.width, vqvae_args.depth,
        vqvae_args.dilation_growth_rate
    ).to(device)
    vq_model.load_state_dict(vqvae_checkpoint['net'])
    vq_model.eval()

    train_loader = dataset_VQ.DATALoader(
        dataname, batch_size=128, num_workers=0,
        window_size=window_size, unit_length=2**down_t,
        dataset_type='train',
        datasets_dir=datasets_dir,
        rel_path=rel_path,
        rel_path_train=rel_path_train,
        rel_path_valid=rel_path_valid,
        rel_path_test=rel_path_test,
        stride=stride,
        window_stride=window_stride,
        ts_stride=ts_stride,
        column=column
    )

    train_indices_list = []
    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Encoding"):
            batch_data = batch.to(device).float()
            indices = vq_model.encode(batch_data)
            train_indices_list.append(indices.cpu())

    train_indices = torch.cat(train_indices_list, dim=0)
    codebook = codebook_raw.cpu()

    torch.save(codebook, codebook_path)
    torch.save(train_indices, indices_path)
    config = {
        'dataname': dataname, 'window_size': window_size,
        'down_t': down_t, 'num_codes': num_codes,
        'code_dim': code_dim, 'vqvae_ckpt': vqvae_ckpt,
        'data_str': data_str,
        'seq_len': train_indices.shape[1],
        'n_samples': train_indices.shape[0]
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print("="*60 + "\n")
    return codebook, train_indices, train_indices.shape[1]


def decode_in_batches(vq_model, indices, batch_size=32):
    decoded_list = []
    num_samples = len(indices)
    device = indices.device

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i : i + batch_size]
            batch_recon = vq_model.forward_decoder(batch_indices)
            decoded_list.append(batch_recon.detach().cpu())

    return torch.cat(decoded_list, dim=0).numpy()


def log_model_size(model, logger, name):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    msg = (
        f"{name} parameters: total={total_params:,} "
        f"({total_params / 1e6:.3f}M), trainable={trainable_params:,} "
        f"({trainable_params / 1e6:.3f}M)"
    )
    print(msg)
    logger.info(msg)

    

def train():
    parser = argparse.ArgumentParser(description="SDFlow")
    parser.add_argument('--vqvae_ckpt', type=str, required=True)
    parser.add_argument('--dataname', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./checkpoints_SDFlow')
    
    parser.add_argument('--quantizer', type=str, default='ema_reset_sim')
    parser.add_argument('--window_size', type=int, default=24)
    parser.add_argument('--down_t', type=int, default=2)
    parser.add_argument('--datasets_dir', '--datasets-dir', dest='datasets_dir', type=str, default=None)
    parser.add_argument('--rel_path', '--rel-path', dest='rel_path', type=str, default=None)
    parser.add_argument('--rel_path_train', '--rel-path-train', dest='rel_path_train', type=str, default=None)
    parser.add_argument('--rel_path_valid', '--rel-path-valid', dest='rel_path_valid', type=str, default=None)
    parser.add_argument('--rel_path_test', '--rel-path-test', dest='rel_path_test', type=str, default=None)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--window_stride', '--window-stride', dest='window_stride', type=int, default=None)
    parser.add_argument('--ts_stride', '--ts-stride', dest='ts_stride', type=int, default=None)
    parser.add_argument('--input_emb_width', '--input-emb-width', dest='input_emb_width', type=int, default=None)
    parser.add_argument('--column', type=str, default='glucose')
    
    parser.add_argument('--rank', type=int, default=128)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    parser.add_argument('--noise_std', type=float, default=0.01)
    parser.add_argument('--lambda_mean', type=float, default=0.1)
    parser.add_argument('--lambda_std', type=float, default=0.1)
    
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_uv', type=float, default=5e-4)
    parser.add_argument('--max_iters', type=int, default=100000)
    parser.add_argument('--print_interval', type=int, default=200)
    parser.add_argument('--eval_interval', type=int, default=5000)
    parser.add_argument('--eval_steps', type=int, default=50)
    parser.add_argument('--num_samples', type=int, default=50000)
    parser.add_argument('--sample_batch_size', type=int, default=128)
    
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(args.output_dir, exist_ok=True)
    logger = utils_model.get_logger(args.output_dir)
    writer = SummaryWriter(args.output_dir)
    
    # Load data
    codebook, train_indices, seq_len = auto_encode_and_cache(
        vqvae_ckpt=args.vqvae_ckpt,
        dataname=args.dataname,
        cache_dir=args.output_dir,
        device=device,
        window_size=args.window_size,
        down_t=args.down_t,
        quantizer=args.quantizer,
        datasets_dir=args.datasets_dir,
        rel_path=args.rel_path,
        rel_path_train=args.rel_path_train,
        rel_path_valid=args.rel_path_valid,
        rel_path_test=args.rel_path_test,
        stride=args.stride,
        window_stride=args.window_stride,
        ts_stride=args.ts_stride,
        input_emb_width=args.input_emb_width,
        column=args.column
    )
    
    num_codes, code_dim = codebook.shape
    n_samples = len(train_indices)
    
    print(f"\nData Statistics:")
    print(f"  samples: {n_samples}")
    print(f"  Codebook: {num_codes} codes × {code_dim} dim")
    
    codebook = codebook.to(device)
    train_indices = train_indices.to(device)
    
    # Load VQ-VAE
    print(f"\nLoading VQ-VAE...")
    sys_argv_backup = sys.argv.copy()
    sys.argv = [sys.argv[0], '--dataname', args.dataname]
    vqvae_args = option_trans.get_args_parser()
    sys.argv = sys_argv_backup
    vqvae_checkpoint = torch.load(args.vqvae_ckpt, map_location='cpu', weights_only=False)
    vqvae_args = apply_vqvae_checkpoint_args(vqvae_args, vqvae_checkpoint)
    
    if args.quantizer:
        vqvae_args.quantizer = args.quantizer
    if args.window_size:
        vqvae_args.window_size = args.window_size
    if args.down_t:
        vqvae_args.down_t = args.down_t
    
    if args.input_emb_width is not None:
        vqvae_args.input_emb_width = args.input_emb_width
    elif getattr(vqvae_args, 'input_emb_width', None) is None:
        vqvae_args.input_emb_width = infer_input_emb_width_from_checkpoint(vqvae_checkpoint)
    vq_model = vqvae.VQVAE(vqvae_args, num_codes, code_dim,
                           vqvae_args.down_t, vqvae_args.stride_t,
                           vqvae_args.width, vqvae_args.depth,
                           vqvae_args.dilation_growth_rate).to(device)
    vq_model.load_state_dict(vqvae_checkpoint['net'])
    vq_model.eval()
    log_model_size(vq_model, logger, 'VQ-VAE')
    
    # Create model
    model = SDFlowModel(
        num_samples=n_samples,
        num_codes=num_codes,
        code_dim=code_dim,
        seq_len=seq_len,
        rank=args.rank,
        d_model=args.d_model,
        n_layers=args.n_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        noise_std=args.noise_std,
        lambda_mean=args.lambda_mean,
        lambda_std=args.lambda_std,
        t_scheduler='cosine'
    ).to(device)
    log_model_size(model, logger, 'SDFlow')

    # Optimizer
    optimizer = optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if n not in ['U', 'V']],
         'lr': args.lr},
        {'params': [model.U, model.V], 'lr': args.lr_uv}
    ], weight_decay=0.01)
    
    warmup_steps = 500
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.0, (args.max_iters - step) / (args.max_iters - warmup_steps))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    
    model.train()
    sample_dir = os.path.join(args.output_dir, 'samples')
    os.makedirs(sample_dir, exist_ok=True)
    
    for iter_idx in range(args.max_iters):
        # Sample batch
        batch_sample_ids = torch.randint(0, n_samples, (args.batch_size,), device=device)
        batch_indices = train_indices[batch_sample_ids]
        
        # Compute loss
        loss, metrics = model.compute_loss(codebook, batch_indices, batch_sample_ids)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Logging
        if (iter_idx + 1) % args.print_interval == 0:
            lr = optimizer.param_groups[0]['lr']
            lr_uv = optimizer.param_groups[1]['lr']
            msg = f"Iter {iter_idx+1}: Loss={metrics['loss']:.6f}, CE={metrics['ce_loss']:.6f}, Mean={metrics['mean_loss']:.6f}, Std={metrics['std_loss']:.6f}, structure={metrics['structure_loss']:.6f}, Acc={metrics['accuracy']:.6f}"
            print(msg)
            logger.info(msg)
            writer.add_scalar('Train/Loss', metrics['loss'], iter_idx + 1)
            writer.add_scalar('Train/CE_Loss', metrics['ce_loss'], iter_idx + 1)
            writer.add_scalar('Train/Mean_Loss', metrics['mean_loss'], iter_idx + 1)
            writer.add_scalar('Train/Std_Loss', metrics['std_loss'], iter_idx + 1)
            writer.add_scalar('Train/Accuracy', metrics['accuracy'], iter_idx + 1)
        
        # Sampling. Metric evaluation is intentionally disabled for baseline training.
        if (iter_idx + 1) % args.eval_interval == 0 or (iter_idx + 1) == args.max_iters:
            print(f"\n{'='*60}")
            print(f"Sampling - Iter {iter_idx+1}")
            print(f"{'='*60}")
            
            model.eval()
            
            kde = KDEPrior(model.U, device=device, bandwidth_factor=0.001)
            
            # Check U statistics
            with torch.no_grad():
                u_mean = model.U.mean(dim=0).abs().mean().item()
                u_std = model.U.std(dim=0).mean().item()
                print(f"\nU Statistics:")
                print(f"  Mean: {u_mean:.6f} (target: 0)")
                print(f"  Std: {u_std:.6f} (target: 1)")

            num_gen = min(args.num_samples, n_samples)
            gen_batch_size = args.sample_batch_size
            gen_indices_list = []

            with torch.no_grad():
                for i in tqdm(range(0, num_gen, gen_batch_size), desc="Generating"):
                    bs = min(gen_batch_size, num_gen - i)
                    _, gen_idx = model.sample_kde(
                        codebook=codebook,
                        batch_size=bs,
                        seq_len=seq_len,
                        steps=args.eval_steps,
                        temperature=0.9,
                        device=device,
                        kde_solver=kde 
                    )
                    gen_indices_list.append(gen_idx.cpu())

            gen_indices = torch.cat(gen_indices_list, dim=0).to(device)
            gen_timeseries = decode_in_batches(vq_model, gen_indices, batch_size=64)
            sample_prefix = os.path.join(sample_dir, f'iter_{iter_idx + 1:08d}')
            np.save(f'{sample_prefix}_samples.npy', gen_timeseries)
            np.save(f'{sample_prefix}_indices.npy', gen_indices.detach().cpu().numpy())
            torch.save({
                'model_state_dict': model.state_dict(),
                'iter': iter_idx + 1,
                'args': vars(args)
            }, os.path.join(args.output_dir, 'latest.pth'))
            print(f"Saved {len(gen_timeseries)} samples to {sample_prefix}_samples.npy")
            logger.info(f"Saved {len(gen_timeseries)} samples to {sample_prefix}_samples.npy")
            
            print(f"{'='*60}\n")
            model.train()
    
    print(f"\nTraining finished. Samples saved under: {sample_dir}")
    writer.close()


if __name__ == "__main__":
    train()
