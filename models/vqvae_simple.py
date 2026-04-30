import os
import json
import math
import random
import numpy as np
from dataclasses import dataclass
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import MinMaxScaler
try:
    import wandb
except ImportError:
    wandb = None

import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Subset



# --------------------------
# Utils: pad mask
# --------------------------

def make_valid_mask(lengths: torch.Tensor, T: int) -> torch.Tensor:
    """
    lengths: [B]
    returns: [B, T] float mask, 1 for valid, 0 for pad
    """
    device = lengths.device
    return (torch.arange(T, device=device)[None, :] < lengths[:, None]).float()


# --------------------------
# Vector Quantizer (VQ-VAE)
# --------------------------

class VectorQuantizer(nn.Module):
    """
    Standard VQ-VAE (nearest neighbor) with straight-through gradient.

    z_e: [B, T', D]  (encoder outputs)
    z_q: [B, T', D]  (quantized)
    ids: [B, T']     (code indices)
    loss: vq loss (codebook + commitment)
    """
    def __init__(self, num_codes: int, code_dim: int, beta: float = 0.25):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.beta = beta

        self.codebook = nn.Embedding(num_codes, code_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / num_codes, 1.0 / num_codes)

    def forward(self, z_e: torch.Tensor):
        B, T, D = z_e.shape
        assert D == self.code_dim

        flat = z_e.reshape(B * T, D)  # [BT, D]
        e = self.codebook.weight      # [K, D]

        # squared euclidean distances: ||x||^2 - 2 x·e + ||e||^2
        dist = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * flat @ e.t()
            + e.pow(2).sum(dim=1, keepdim=True).t()
        )  # [BT, K]

        ids = torch.argmin(dist, dim=1)             # [BT]
        z_q = self.codebook(ids).view(B, T, D)      # [B, T, D]

        # VQ losses
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commit_loss   = F.mse_loss(z_e, z_q.detach())
        vq_loss = codebook_loss + self.beta * commit_loss

        # Straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()

        return z_q_st, ids.view(B, T), vq_loss


# --------------------------
# 1D Conv VQ-VAE
# --------------------------

class ResBlock1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        downsample=False,
    ):
        super().__init__()
        padding = kernel_size // 2
        # self.downsample = downsample or (stride != 1) or (in_channels != out_channels)

        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )

        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)

        # if self.downsample:
        #     self.skip = nn.Conv1d(
        #         in_channels, out_channels,
        #         kernel_size=1,
        #         stride=stride,
        #         bias=False,
        #     )
        # else:
        #     self.skip = nn.Identity()
        self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out = out + identity
        out = self.act(out)
        return out


class ResNetEncoder1D(nn.Module):
    """
    Input:  x [B, T, C]
    Output: z_e [B, T', D]
    """
    def __init__(
        self,
        seq_len,
        in_channels,
        channels=(64, 128, 256),
        blocks_per_stage=2,
        code_dim=128,
        kernel_size=3,
        down_ratio=2,
        code_len=4
    ):
        super().__init__()

        self.stem = nn.Conv1d(
            in_channels, channels[0],
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
        )

        stages = []
        in_ch = channels[0]

        for stage_idx, out_ch in enumerate(channels):
            for block_idx in range(blocks_per_stage):
                # downsample only at first block of each stage (except stage 0)
                stride = down_ratio if (block_idx == 0 and stage_idx > 0) else 1
                stages.append(
                    ResBlock1D(
                        in_ch, out_ch,
                        kernel_size=kernel_size,
                        stride=stride,
                    )
                )
                in_ch = out_ch

        self.stages = nn.Sequential(*stages)

        self.global_pooling = nn.Linear(seq_len, code_len)
        self.proj = nn.Conv1d(
            in_ch, code_dim,
            kernel_size=1,
            bias=False,
        )

    def forward(self, x, loss_mask):
        # x: [B, T, C] → [B, C, T]
        x = x.transpose(1, 2)
        h = self.stem(x)
        h = self.stages(h)
        h = h * loss_mask.permute(0, 2, 1)
        z = self.global_pooling(h)
        z = torch.relu(z)
        z = self.proj(z)          # [B, D, T']
        z = z.transpose(1, 2)     # [B, T', D]
        return z


class UpResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, up_ratio):
        super().__init__()
        padding = kernel_size // 2

        self.upsample = nn.Upsample(scale_factor=up_ratio, mode="nearest")

        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)

        self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.upsample(x)
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out = out + identity
        out = self.act(out)
        return out


class ResNetDecoder1D(nn.Module):
    """
    Input:  z_q [B, T', D]
    Output: x_hat [B, T, C]
    """
    def __init__(
        self,
        out_channels,
        channels=(256, 128, 64),
        blocks_per_stage=1,
        code_dim=128,
        kernel_size=3,
        up_ratio=2,
        code_len=123,
        seq_len=123
    ):
        super().__init__()

        stages = []
        in_ch = channels[0]

        for out_ch in channels[1:]:
            for _ in range(blocks_per_stage):
                stages.append(
                    UpResBlock1D(
                        in_ch, out_ch,
                        kernel_size=kernel_size,
                        up_ratio=up_ratio,
                    )
                )
                in_ch = out_ch

        self.stages = nn.Sequential(*stages)

        self.head = nn.Conv1d(
            in_ch, out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )

        n_upsample = len(channels) - 1
        T0 = math.ceil(seq_len / (up_ratio ** n_upsample))

        self.input_proj_1 = nn.Linear(code_dim, channels[0])   # channel
        self.input_proj_2 = nn.Linear(code_len, T0)            # time


    def forward(self, z, target_len):
        '''
        z: [B, T, C]
        '''
        z = self.input_proj_1(z)

        z = z.transpose(1, 2)           # [B, C0, 4]
        z = self.input_proj_2(z)
        h = self.stages(z)
        x_hat = self.head(h)       # [B, C, T_recon]
        x_hat = x_hat.transpose(1, 2)

        # crop / pad to target_len
        if x_hat.size(1) > target_len:
            x_hat = x_hat[:, :target_len]
        elif x_hat.size(1) < target_len:
            x_hat = F.pad(x_hat, (0, 0, 0, target_len - x_hat.size(1)))

        return x_hat


class VQVAE1D(nn.Module):
    def __init__(
        self,
        in_channels, encoder_channels, decoder_channels, code_dim, num_codes,
        down_ratio, up_ratio, code_len, seq_len
    ):
        super().__init__()
        # self.encoder = ResNetEncoder1D(
        #     in_channels=in_channels,
        #     channels=(16, 16, 32, 32, 64, 64),
        #     blocks_per_stage=1,
        #     code_dim=code_dim,
        # )
        self.encoder = ResNetEncoder1D(
            seq_len=seq_len,
            in_channels=in_channels,
            channels=encoder_channels,
            blocks_per_stage=1,
            code_dim=code_dim,
            down_ratio=down_ratio,
            code_len=code_len
        )
        self.quantizer = VectorQuantizer(num_codes, code_dim)
        # self.decoder = ResNetDecoder1D(
        #     out_channels=in_channels,
        #     channels=(64, 64, 32, 32, 16, 16),
        #     blocks_per_stage=1,
        #     code_dim=code_dim,
        # )
        self.decoder = ResNetDecoder1D(
            out_channels=in_channels,
            channels=decoder_channels,
            blocks_per_stage=1,
            code_dim=code_dim,
            up_ratio=up_ratio,
            code_len=code_len,
            seq_len=seq_len
        )

    def forward(self, x, loss_mask):
        z_e = self.encoder(x, loss_mask)
        z_q, ids, vq_loss = self.quantizer(z_e)
        x_hat = self.decoder(z_q, x.size(1))
        return x_hat, ids, vq_loss
