import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def compute_out_len(L: int, k: int, s: int) -> int:
    return ceil_div(L, s)


def pick(val, idx: int) -> int:
    return int(val[idx]) if isinstance(val, (list, tuple)) else int(val)


class FrontPadding(nn.Module):
    def __init__(self, patch_size: int, stride: int):
        super().__init__()
        self.k = patch_size
        self.s = stride

    def forward(self, x):
        T = x.size(-1)
        out_len = ceil_div(T, self.s)
        total_len_needed = (out_len - 1) * self.s + self.k
        pad = max(0, total_len_needed - T)
        if pad == 0:
            return x
        return torch.cat([x[..., -1:].repeat(*(1,) * (x.dim() - 1), pad), x], dim=-1)


class DepthwiseMix(nn.Module):
    def __init__(self, C: int, kL: int, kS: int, bias: bool):
        super().__init__()
        self.large = nn.Conv1d(C, C, kL, padding=kL // 2, groups=C, bias=bias)
        self.small = nn.Conv1d(C, C, kS, padding=kS // 2, groups=C, bias=False)

    def forward(self, x):
        return self.large(x) + self.small(x)


class FeedForward(nn.Module):
    def __init__(self, hidden: int, ratio: float, drop: float):
        super().__init__()
        inner = int(hidden * ratio)
        self.conv1 = nn.Conv1d(hidden, inner, 1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)
        self.conv2 = nn.Conv1d(inner, hidden, 1)

    def forward(self, x):
        B, M, H, T = x.shape
        y = x.reshape(B * M, H, T)
        y = self.conv2(self.drop(self.act(self.conv1(y))))
        return y.reshape(B, M, H, T)


class CHeadAttention(nn.Module):
    def __init__(self, hidden: int, kL: int, kS: int, qkv_bias: bool, drop_attn: float, drop_proj: float):
        super().__init__()
        self.q_proj = DepthwiseMix(hidden, kL, kS, qkv_bias)
        self.k_proj = DepthwiseMix(hidden, kL, kS, qkv_bias)
        self.v_proj = DepthwiseMix(hidden, kL, kS, qkv_bias)
        self.out_proj = nn.Conv1d(hidden, hidden, 1)
        self.drop_attn = drop_attn
        self.drop_proj = nn.Dropout(drop_proj)

    def forward(self, x):
        B, M, H, T = x.shape
        xr = x.reshape(B * M, H, T)
        q = self.q_proj(xr).reshape(B, M, H, T).permute(0, 2, 1, 3)
        k = self.k_proj(xr).reshape(B, M, H, T).permute(0, 2, 1, 3)
        v = self.v_proj(xr).reshape(B, M, H, T).permute(0, 2, 1, 3)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.drop_attn if self.training else 0.0,
            scale=1.0 / math.sqrt(T),
            is_causal=False,
        )
        out = self.out_proj(out.transpose(1, 2).reshape(B * M, H, T))
        return self.drop_proj(out.reshape(B, M, H, T))


class T1BlockAdaLN(nn.Module):
    def __init__(self, hidden: int, T: int, kL: int, kS: int, ffn_ratio: float, drop_ffn: float, qkv_bias: bool, drop_attn: float, drop_proj: float):
        super().__init__()
        self.hidden = hidden
        self.T = T
        self.attn = CHeadAttention(hidden, kL, kS, qkv_bias, drop_attn, drop_proj)
        self.ffn = FeedForward(hidden, ffn_ratio, drop_ffn)
        self.norm1 = nn.LayerNorm((hidden, T), eps=1e-5)
        self.norm2 = nn.LayerNorm((hidden, T), eps=1e-5)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 6 * hidden, bias=True))

    def _modulate_channel(self, x, shift, scale):
        B, M, H, T = x.shape
        y = x.permute(0, 1, 3, 2).reshape(B, M * T, H)
        y = modulate(y, shift, scale)
        return y.reshape(B, M, T, H).permute(0, 1, 3, 2)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(c).chunk(6, dim=-1)

        y1 = self._modulate_channel(self.norm1(self.attn(x)), shift_msa, scale_msa)
        x = x + gate_msa[:, None, :, None] * y1

        y2 = self._modulate_channel(self.norm2(self.ffn(x)), shift_mlp, scale_mlp)
        x = x + gate_mlp[:, None, :, None] * y2
        return x


class DownSample(nn.Module):
    def __init__(self, k: int, s: int, C: int):
        super().__init__()
        self.k = k
        self.s = s
        self.dw = nn.Conv1d(C, C, k, s, groups=C)

    def forward(self, x):
        B, M, C, T = x.shape
        out_len = ceil_div(T, self.s)
        total_len_needed = (out_len - 1) * self.s + self.k
        pad = max(0, total_len_needed - T)
        if pad:
            x = torch.cat([x[..., -1:].repeat(1, 1, 1, pad), x], -1)
        y = self.dw(x.reshape(B * M, C, -1))
        return y.reshape(B, M, C, y.size(-1))


class PixelShuffle1D(nn.Module):
    def __init__(self, r: int):
        super().__init__()
        self.r = r

    def forward(self, x):
        B, C, L = x.shape
        assert C % self.r == 0
        y = x.reshape(B, C // self.r, self.r, L).permute(0, 1, 3, 2)
        return y.reshape(B, C // self.r, L * self.r)


class ReconHead(nn.Module):
    def __init__(self, hidden: int, T_out: int, pred_len: int, drop_head: float):
        super().__init__()
        self.pred_len = pred_len
        self.up = ceil_div(pred_len, T_out)
        self.adjusted = ((hidden + self.up - 1) // self.up) * self.up
        self.channel_adjust = nn.Conv1d(hidden, self.adjusted, 1)
        self.ps = PixelShuffle1D(self.up)
        self.ps_len = T_out * self.up
        outC = self.adjusted // self.up
        self.proj = nn.Linear(outC, 1)
        self.drop = nn.Dropout(drop_head)

    def forward(self, x):
        B, M, H, T = x.shape
        y = self.channel_adjust(x.reshape(B * M, H, T))
        y = self.ps(y.reshape(B, M * self.adjusted, T)).reshape(B * M, self.adjusted // self.up, self.ps_len)
        if self.ps_len > self.pred_len:
            st = (self.ps_len - self.pred_len) // 2
            y = y[..., st:st + self.pred_len]
        y = y.transpose(1, 2)
        return self.drop(self.proj(y)).reshape(B, M, self.pred_len)


@dataclass
class JITT1Config:
    seq_len: int
    enc_in: int
    n_heads: int = 128
    patch_size: int = 2
    patch_stride: int = 1
    n_blocks: tuple = (2, 2)
    kernel_size_large: tuple = (71, 31)
    kernel_size_small: int = 5
    ffn_ratio: float = 1.0
    downsample_ratio: int = 2
    qkv_bias: bool = True
    drop_attn: float = 0.0
    drop_ffn: float = 0.0
    drop_proj: float = 0.0
    drop_head: float = 0.0
    positional_encoding: bool = True
    out_len: int = None


class JITT1(nn.Module):
    """
    Unconditional T1 backbone adapted for JiT-style diffusion denoising.
    Input/Output: [B, T, M]
    """
    def __init__(self, cfg: JITT1Config):
        super().__init__()
        self.cfg = cfg
        self.seq_len = cfg.seq_len
        self.enc_in = cfg.enc_in
        self.hidden = cfg.n_heads
        self.out_len = cfg.out_len if cfg.out_len is not None else cfg.seq_len

        self.t_embedder = TimestepEmbedder(self.hidden)
        self.stem_pad = FrontPadding(cfg.patch_size, cfg.patch_stride)
        self.stem = nn.Conv1d(1, self.hidden, cfg.patch_size, cfg.patch_stride)

        T_after_stem = compute_out_len(cfg.seq_len, cfg.patch_size, cfg.patch_stride)
        if cfg.positional_encoding:
            self.pos = nn.Parameter(torch.randn(1, cfg.enc_in, cfg.n_heads, T_after_stem) * 0.02)
        else:
            self.pos = None

        self.stages = nn.ModuleList()
        self.downs = nn.ModuleList()
        curT = T_after_stem
        for i, n in enumerate(cfg.n_blocks):
            kL = pick(cfg.kernel_size_large, i)
            kS = pick(cfg.kernel_size_small, i)
            blocks = nn.ModuleList(
                [
                    T1BlockAdaLN(
                        hidden=self.hidden,
                        T=curT,
                        kL=kL,
                        kS=kS,
                        ffn_ratio=cfg.ffn_ratio,
                        drop_ffn=cfg.drop_ffn,
                        qkv_bias=cfg.qkv_bias,
                        drop_attn=cfg.drop_attn,
                        drop_proj=cfg.drop_proj,
                    )
                    for _ in range(n)
                ]
            )
            self.stages.append(blocks)
            is_last = i == len(cfg.n_blocks) - 1
            if is_last:
                self.downs.append(nn.Identity())
            else:
                self.downs.append(DownSample(cfg.downsample_ratio, cfg.downsample_ratio, cfg.n_heads))
                curT = compute_out_len(curT, cfg.downsample_ratio, cfg.downsample_ratio)

        self.head = ReconHead(self.hidden, curT, self.out_len, cfg.drop_head)

    def _normalize(self, x):
        mean = x.mean(1, keepdim=True).detach()
        centered = x - mean
        std = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        return centered / std, mean, std

    def _embed(self, x):
        B, T, M = x.shape
        x_pad = self.stem_pad(x.permute(0, 2, 1))
        y = self.stem(x_pad.reshape(B * M, 1, -1)).reshape(B, M, self.hidden, -1)
        if self.pos is not None:
            y = y + self.pos
        return y

    def forward(self, x, t):
        """
        x: [B, T, M]
        t: [B]
        """
        # x_norm, mean, std = self._normalize(x)
        # x_norm, mean, std = self._normalize(x)

        c = self.t_embedder(t)
        y = self._embed(x)
        for blocks, down in zip(self.stages, self.downs):
            for blk in blocks:
                y = blk(y, c)
            y = down(y)
        out = self.head(y).permute(0, 2, 1)
        return out

