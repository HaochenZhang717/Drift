# 2D CNN SoftVQ-VAE for Delay-Image Time Series

## Goal

Adapt the SoftVQ-VAE paper's continuous soft-codebook tokenizer to time series by first converting each series into a delay-embedding image with `DelayEmbedder`. The VQ-VAE itself is a latent-diffusion-style 2D CNN autoencoder.

## Architecture

1. Load benchmark time series as `(B, C, T)`.
2. Convert to delay images with `DelayEmbedder`, producing `(B, C, H, W)`.
3. Use an LDM-style 2D CNN `Encoder` with ResNet blocks, optional attention, and strided downsampling.
4. Replace hard nearest-neighbor VQ with a differentiable soft codebook over every spatial latent location:

   ```text
   q = softmax(-||z_e - C||^2 / tau)
   z_q = q C
   ```

5. Use the matching LDM-style 2D CNN `Decoder` to reconstruct the delay image.

## Loss

The base training objective is:

```text
loss = recon_weight * masked_MSE(x_hat_img, x_img) + kl_weight * (mean_token_entropy - batch_code_entropy)
```

The reconstruction loss can ignore padded image regions. The entropy term is the SoftVQ-style prior regularizer. Minimizing it encourages confident token-to-codeword assignments while keeping aggregate codebook usage broad. Unlike hard VQ-VAE, there is no straight-through estimator, codebook loss, or commitment loss.

## Metrics

Track:

- reconstruction loss
- entropy/KL regularizer
- soft codebook perplexity from the aggregate code distribution
- hard argmax code usage as a diagnostic only
- assignment entropy

## First Experiments

Start with conservative settings:

```bash
python train_soft_vqvae_benchmark.py \
  --dataset_name GlucoseSliding \
  --data GlucoseSliding \
  --datasets_dir ./AI-READI \
  --rel_path 'glucose_{split}.parquet' \
  --ts_seq_len 128 \
  --delay 12 \
  --embedding 12 \
  --hidden_size 128 \
  --code_dim 32 \
  --num_codes 512 \
  --ch_mult 1,2,4 \
  --temperature 0.07 \
  --kl_weight 0.01
```

If recon is poor, increase `code_dim`, reduce `ch_mult` depth, increase `hidden_size`, or lower `kl_weight`. If code usage collapses, raise `kl_weight` slightly or use `--learnable_temperature`.
