import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modality_imputation.multi_modal_encoder import MultiModalEncoder


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 1024, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SwiGLU(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=False)
        self.w2 = nn.Linear(hidden_features, out_features, bias=False)
        self.w3 = nn.Linear(in_features, hidden_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, use_qk_norm: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        qkv = self.qkv(x).reshape(bsz, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if rope_cos is not None and rope_sin is not None:
            q, k = apply_rope(q, k, rope_cos, rope_sin)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(bsz, seq_len, dim)
        return self.proj(out)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, use_qk_norm: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        bsz, n_main, dim = x.shape
        n_cond = c.shape[1]

        q = self.q_proj(x).reshape(bsz, n_main, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(c).reshape(bsz, n_cond, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(c).reshape(bsz, n_cond, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(bsz, n_main, dim)
        return self.proj(out)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlockCrossAttention(nn.Module):
    """
    DiT block with:
    1) self-attention on main tokens
    2) cross-attention where main tokens attend condition tokens
    3) MLP
    Each sub-layer is modulated by study-group conditioning via adaLN-style params.
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, use_qk_norm: bool = True):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.norm3 = RMSNorm(dim)
        self.cond_norm = RMSNorm(dim)

        self.self_attn = Attention(dim, num_heads=num_heads, use_qk_norm=use_qk_norm)
        self.cross_attn = CrossAttention(dim, num_heads=num_heads, use_qk_norm=use_qk_norm)

        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = SwiGLU(dim, mlp_hidden, dim)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 9 * dim, bias=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        condition_tokens: torch.Tensor,
        c_vec: torch.Tensor,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        (
            shift_sa,
            scale_sa,
            gate_sa,
            shift_ca,
            scale_ca,
            gate_ca,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c_vec).chunk(9, dim=1)

        x = x + gate_sa.unsqueeze(1) * self.self_attn(
            modulate(self.norm1(x), shift_sa, scale_sa), rope_cos, rope_sin
        )

        x = x + gate_ca.unsqueeze(1) * self.cross_attn(
            modulate(self.norm2(x), shift_ca, scale_ca),
            self.cond_norm(condition_tokens),
        )

        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm3(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer(nn.Module):
    def __init__(self, dim: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.linear = nn.Linear(dim, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim, bias=True),
        )

    def forward(self, x: torch.Tensor, c_vec: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c_vec).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 32, patch_size: int = 4, in_channels: int = 3, embed_dim: int = 256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class LabelEmbedder(nn.Module):
    """Same conditioning path as cls_cond_model: class embedding + optional dropout-to-null for CFG."""

    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float = 0.1):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels: torch.Tensor, force_drop_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids.bool()
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(
        self,
        labels: torch.Tensor,
        train: bool = True,
        force_drop_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.dropout_prob > 0 and train:
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding_table(labels)


class AlphaEmbedder(nn.Module):
    """Optional alpha embedding, same style as cls_cond_model."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def fourier_features(alpha: torch.Tensor, dim: int, max_period: float = 10.0) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=alpha.device) / half)
        args = alpha[:, None] * freqs[None, :]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, alpha: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.fourier_features(alpha, self.frequency_embedding_size))


class MultiModalDrift(nn.Module):
    """
    Drift model conditioned by:
    1) study group embedding (same mechanism as cls_cond_model LabelEmbedder path)
    2) multi-modal time-series tokens from MultiModalEncoder injected via cross-attention.
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 1,
        hidden_size: int = 256,
        depth: int = 6,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        num_study_groups: int = 8,
        label_dropout: float = 0.1,
        num_register_tokens: int = 8,
        use_alpha_embed: bool = False,
        multi_modal_encoder_kwargs: Optional[Dict] = None,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.num_patches = (img_size // patch_size) ** 2
        self.use_alpha_embed = use_alpha_embed

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=hidden_size,
        )

        self.num_register_tokens = num_register_tokens
        self.register_tokens = nn.Parameter(torch.randn(1, num_register_tokens, hidden_size) * 0.02)

        head_dim = hidden_size // num_heads
        self.rope = RotaryPositionEmbedding(dim=head_dim, max_seq_len=self.num_patches + num_register_tokens + 64)

        # Study-group conditioning path: same as cls_cond_model label embedding route
        self.study_group_embed = LabelEmbedder(num_study_groups, hidden_size, label_dropout)
        self.alpha_embed = AlphaEmbedder(hidden_size) if use_alpha_embed else None

        # Multi-modal encoder that returns condition tokens
        mm_kwargs = dict(multi_modal_encoder_kwargs or {})
        mm_kwargs.setdefault("dim_out", hidden_size)
        self.multi_modal_encoder = MultiModalEncoder(**mm_kwargs)
        self.cond_proj = nn.Identity() if self.multi_modal_encoder.dim_out == hidden_size else nn.Linear(
            self.multi_modal_encoder.dim_out, hidden_size
        )

        self.group_token_proj = nn.Linear(hidden_size, hidden_size)

        self.blocks = nn.ModuleList(
            [
                DiTBlockCrossAttention(
                    dim=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_qk_norm=True,
                )
                for _ in range(depth)
            ]
        )

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self._init_weights()

    def _init_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

        self.apply(_basic_init)

        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)

        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.normal_(self.final_layer.linear.weight, std=0.02)
        nn.init.zeros_(self.final_layer.linear.bias)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        c = self.out_channels
        p = self.patch_size
        h = w = self.img_size // p

        x = x.reshape(-1, h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        x = x.reshape(-1, c, h * p, w * p)
        return x

    def forward(
        self,
        x: torch.Tensor,
        study_group: torch.Tensor,
        heart_rate: torch.Tensor,
        calorie: torch.Tensor,
        physical_activity: torch.Tensor,
        respiratory_rate: torch.Tensor,
        heart_rate_observed_mask: torch.Tensor,
        calorie_observed_mask: torch.Tensor,
        physical_activity_observed_mask: torch.Tensor,
        respiratory_rate_observed_mask: torch.Tensor,
        alpha: Optional[torch.Tensor] = None,
        force_drop_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz = x.shape[0]

        # Patch/main tokens
        x_tokens = self.patch_embed(x)
        register = self.register_tokens.expand(bsz, -1, -1)
        x_tokens = torch.cat([register, x_tokens], dim=1)

        seq_len = x_tokens.shape[1]
        rope_cos, rope_sin = self.rope(seq_len)

        # Study-group condition vector (same mechanism as cls_cond_model label path)
        c_vec = self.study_group_embed(study_group, self.training, force_drop_ids)
        if self.use_alpha_embed:
            if alpha is None:
                alpha = torch.zeros(bsz, device=x.device, dtype=x.dtype)
            c_vec = c_vec + self.alpha_embed(alpha)

        # Multi-modal condition tokens
        cond_tokens = self.multi_modal_encoder(
            heart_rate=heart_rate,
            calorie=calorie,
            physical_activity=physical_activity,
            respiratory_rate=respiratory_rate,
            heart_rate_observed_mask=heart_rate_observed_mask,
            calorie_observed_mask=calorie_observed_mask,
            physical_activity_observed_mask=physical_activity_observed_mask,
            respiratory_rate_observed_mask=respiratory_rate_observed_mask,
        )
        cond_tokens = self.cond_proj(cond_tokens)

        # Inject study-group as an additional condition token for cross-attn
        group_token = self.group_token_proj(c_vec).unsqueeze(1)
        cond_tokens = torch.cat([group_token, cond_tokens], dim=1)

        # Transformer with cross-attention injection
        for block in self.blocks:
            x_tokens = block(x_tokens, cond_tokens, c_vec, rope_cos, rope_sin)

        # Remove register tokens then decode patches
        x_tokens = x_tokens[:, self.num_register_tokens :, :]
        x_tokens = self.final_layer(x_tokens, c_vec)
        x = self.unpatchify(x_tokens)

        return torch.tanh(x)
