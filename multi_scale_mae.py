"""
Multi-scale masked autoencoder for time series.

The model follows a three-scale patching design:
    x -> per-scale patch embedding/encoder -> z1, z2, z3
      -> concat all scales -> self-attention bridge
      -> per-scale linear decoders -> reconstruction losses

Input tensors use shape [batch, time, channels].
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn


def _default_strides(patch_sizes: Sequence[int]) -> Tuple[int, ...]:
    return tuple(max(1, patch_size // 2) for patch_size in patch_sizes)


def _sinusoidal_position_encoding(length: int, dim: int, device: torch.device) -> torch.Tensor:
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / dim)
    )
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe.unsqueeze(0)


def _make_random_mask(
    batch_size: int,
    num_tokens: int,
    mask_ratio: float,
    device: torch.device,
) -> torch.Tensor:
    if mask_ratio <= 0:
        return torch.zeros(batch_size, num_tokens, dtype=torch.bool, device=device)
    if mask_ratio >= 1:
        return torch.ones(batch_size, num_tokens, dtype=torch.bool, device=device)

    num_mask = max(1, int(round(num_tokens * mask_ratio)))
    noise = torch.rand(batch_size, num_tokens, device=device)
    ids_shuffle = torch.argsort(noise, dim=1)
    mask = torch.zeros(batch_size, num_tokens, dtype=torch.bool, device=device)
    mask.scatter_(1, ids_shuffle[:, :num_mask], True)
    return mask


def _extract_patches(x: torch.Tensor, patch_size: int, stride: int) -> torch.Tensor:
    """Extract overlapping patches from [B, T, C] as [B, N, patch_size * C]."""
    patches = x.unfold(dimension=1, size=patch_size, step=stride)
    # unfold returns [B, N, C, P]; make each patch time-major.
    patches = patches.permute(0, 1, 3, 2).contiguous()
    return patches.flatten(start_dim=2)


def _patches_to_series(
    patches: torch.Tensor,
    seq_len: int,
    channels: int,
    patch_size: int,
    stride: int,
) -> torch.Tensor:
    """Overlap-add patch predictions [B, N, patch_size * C] to [B, T, C]."""
    batch_size, num_patches, _ = patches.shape
    patches = patches.view(batch_size, num_patches, patch_size, channels)
    series = patches.new_zeros(batch_size, seq_len, channels)
    counts = patches.new_zeros(batch_size, seq_len, channels)

    for idx in range(num_patches):
        start = idx * stride
        end = min(start + patch_size, seq_len)
        width = end - start
        series[:, start:end] += patches[:, idx, :width]
        counts[:, start:end] += 1

    return series / counts.clamp_min(1)


class PatchEmbed1D(nn.Module):
    """Strided 1D patch embedding for time series."""

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int, stride: int):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.proj = nn.Conv1d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.proj(x)
        return x.transpose(1, 2)


class TransformerStack(nn.Module):
    """Small batch-first Transformer encoder stack."""

    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.blocks(x))


@dataclass
class MultiScaleMAEOutput:
    loss: torch.Tensor
    losses: Dict[str, torch.Tensor]
    reconstructions: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    masks: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    z1: torch.Tensor
    z2: torch.Tensor
    z3: torch.Tensor
    z1_fused: torch.Tensor
    z2_fused: torch.Tensor
    z3_fused: torch.Tensor


class MultiScaleTimeSeriesMAE(nn.Module):
    """
    Three-scale time-series MAE with a concatenated self-attention bridge.

    Args:
        input_dims: Number of input channels C.
        seq_len: Input window length T.
        patch_sizes: Patch sizes for z1, z2, z3.
        strides: Optional patch strides. Defaults to patch_size // 2 per scale.
        embed_dim: Latent token dimension d.
        encoder_depth: Per-scale encoder depth.
        bridge_depth: Self-attention bridge depth after concatenating all scales.
        decoder_depth: Optional per-scale decoder Transformer depth before linear heads.
        num_heads: Attention heads.
        mask_ratio: Token mask ratio for MAE pretraining.
    """

    def __init__(
        self,
        input_dims: int = 1,
        seq_len: int = 128,
        patch_sizes: Sequence[int] = (4, 16, 64),
        strides: Optional[Sequence[int]] = None,
        embed_dim: int = 128,
        encoder_depth: int = 2,
        bridge_depth: int = 2,
        decoder_depth: int = 1,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        mask_ratio: float = 0.5,
        dropout: float = 0.0,
        loss_weights: Sequence[float] = (1.0, 1.0, 1.0),
    ):
        super().__init__()
        if len(patch_sizes) != 3:
            raise ValueError("patch_sizes must contain exactly three scales")
        if strides is None:
            strides = _default_strides(patch_sizes)
        if len(strides) != 3:
            raise ValueError("strides must contain exactly three scales")
        if len(loss_weights) != 3:
            raise ValueError("loss_weights must contain exactly three values")

        self.input_dims = input_dims
        self.seq_len = seq_len
        self.patch_sizes = tuple(int(p) for p in patch_sizes)
        self.strides = tuple(int(s) for s in strides)
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        self.loss_weights = tuple(float(w) for w in loss_weights)

        self.patch_embeds = nn.ModuleList(
            [
                PatchEmbed1D(input_dims, embed_dim, patch_size, stride)
                for patch_size, stride in zip(self.patch_sizes, self.strides)
            ]
        )
        self.encoders = nn.ModuleList(
            [
                TransformerStack(embed_dim, encoder_depth, num_heads, mlp_ratio, dropout)
                for _ in range(3)
            ]
        )

        self.mask_tokens = nn.Parameter(torch.zeros(3, 1, 1, embed_dim))
        self.scale_embeddings = nn.Parameter(torch.zeros(3, 1, 1, embed_dim))

        self.bridge = TransformerStack(embed_dim, bridge_depth, num_heads, mlp_ratio, dropout)
        self.decoders = nn.ModuleList(
            [
                TransformerStack(embed_dim, decoder_depth, num_heads, mlp_ratio, dropout)
                if decoder_depth > 0
                else nn.Identity()
                for _ in range(3)
            ]
        )
        self.heads = nn.ModuleList(
            [
                nn.Linear(embed_dim, patch_size * input_dims)
                for patch_size in self.patch_sizes
            ]
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.mask_tokens, std=0.02)
        nn.init.normal_(self.scale_embeddings, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _encode_scale(
        self,
        x: torch.Tensor,
        scale_idx: int,
        do_mask: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.patch_embeds[scale_idx](x)
        tokens = tokens + _sinusoidal_position_encoding(
            tokens.size(1),
            tokens.size(2),
            tokens.device,
        )

        mask = None
        if do_mask:
            mask = _make_random_mask(
                tokens.size(0),
                tokens.size(1),
                self.mask_ratio,
                tokens.device,
            )
            mask_token = self.mask_tokens[scale_idx].expand(tokens.size(0), tokens.size(1), -1)
            tokens = torch.where(mask.unsqueeze(-1), mask_token, tokens)

        return self.encoders[scale_idx](tokens), mask

    def encode(
        self,
        x: torch.Tensor,
        masks: Optional[Sequence[torch.Tensor]] = None,
        fuse: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return z1, z2, z3 features. Set fuse=False for pre-bridge latents."""
        masks = masks or (None, None, None)
        z_list = [
            self._encode_scale(x, idx, masks[idx])[0]
            for idx in range(3)
        ]
        if not fuse:
            return z_list[0], z_list[1], z_list[2]
        return self._bridge(z_list)

    def _bridge(
        self,
        z_list: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lengths = [z.size(1) for z in z_list]
        tokens = [
            z + self.scale_embeddings[idx].expand(z.size(0), z.size(1), -1)
            for idx, z in enumerate(z_list)
        ]

        fused = self.bridge(torch.cat(tokens, dim=1))
        return tuple(torch.split(fused, lengths, dim=1))  # type: ignore[return-value]

    def _decode_scale(
        self,
        z: torch.Tensor,
        scale_idx: int,
    ) -> torch.Tensor:
        z = self.decoders[scale_idx](z)
        return self.heads[scale_idx](z)

    def _scale_loss(
        self,
        x: torch.Tensor,
        pred_patches: torch.Tensor,
        mask: torch.Tensor,
        scale_idx: int,
    ) -> torch.Tensor:
        target = _extract_patches(
            x,
            self.patch_sizes[scale_idx],
            self.strides[scale_idx],
        )
        per_patch_loss = (pred_patches - target).pow(2).mean(dim=-1)

        if mask.any():
            return (per_patch_loss * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
        return per_patch_loss.mean()

    def forward(
        self,
        x: torch.Tensor,
        do_mask: bool,
    ) -> MultiScaleMAEOutput:
        if x.ndim != 3:
            raise ValueError("Expected input shape [batch, time, channels]")
        if x.size(1) != self.seq_len:
            raise ValueError(f"Expected sequence length {self.seq_len}, got {x.size(1)}")
        if x.size(2) != self.input_dims:
            raise ValueError(f"Expected {self.input_dims} channels, got {x.size(2)}")

        z1, mask1 = self._encode_scale(x, 0, do_mask)
        z2, mask2 = self._encode_scale(x, 1, do_mask)
        z3, mask3 = self._encode_scale(x, 2, do_mask)


        z1_fused, z2_fused, z3_fused = self._bridge((z1, z2, z3))

        pred1 = self._decode_scale(z1_fused, 0)
        pred2 = self._decode_scale(z2_fused, 1)
        pred3 = self._decode_scale(z3_fused, 2)


        loss1 = self._scale_loss(x, pred1, mask1, 0)
        loss2 = self._scale_loss(x, pred2, mask2, 1)
        loss3 = self._scale_loss(x, pred3, mask3, 2)

        loss = (
            self.loss_weights[0] * loss1
            + self.loss_weights[1] * loss2
            + self.loss_weights[2] * loss3
        )

        return loss


if __name__ == "__main__":
    model = MultiScaleTimeSeriesMAE(input_dims=1, seq_len=128, patch_sizes=[8, 16, 32])
    x = torch.randn(2, 128, 1)
    out = model(x, do_mask=True)
    print(out.loss.item(), out.z1.shape, out.z2.shape, out.z3.shape)
