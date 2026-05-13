import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)


def nonlinearity(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def _num_groups(channels: int, requested: int = 32) -> int:
    for groups in (requested, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def Normalize(in_channels: int, num_groups: int = 32) -> nn.GroupNorm:
    return nn.GroupNorm(
        num_groups=_num_groups(in_channels, num_groups),
        num_channels=in_channels,
        eps=1e-6,
        affine=True,
    )


class Upsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.with_conv:
            x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int | None = None,
        conv_shortcut: bool = False,
        dropout: float,
        temb_channels: int = 0,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.temb_proj = nn.Linear(temb_channels, out_channels) if temb_channels > 0 else None
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor, temb: torch.Tensor | None = None) -> torch.Tensor:
        h = self.conv1(nonlinearity(self.norm1(x)))
        if temb is not None and self.temb_proj is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]
        h = self.conv2(self.dropout(nonlinearity(self.norm2(h))))

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        bsz, channels, height, width = q.shape
        q = q.reshape(bsz, channels, height * width).permute(0, 2, 1)
        k = k.reshape(bsz, channels, height * width)
        weights = torch.bmm(q, k) * (channels ** -0.5)
        weights = F.softmax(weights, dim=2)

        v = v.reshape(bsz, channels, height * width)
        h_ = torch.bmm(v, weights.permute(0, 2, 1)).reshape(bsz, channels, height, width)
        return x + self.proj_out(h_)


def make_attn(in_channels: int, attn_type: str = "vanilla") -> nn.Module:
    if attn_type not in {"vanilla", "linear", "none"}:
        raise ValueError(f"Unknown attn_type: {attn_type}")
    if attn_type == "none":
        return nn.Identity()
    return AttnBlock(in_channels)


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        ch: int,
        ch_mult: tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int,
        attn_resolutions: tuple[int, ...],
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
        in_channels: int,
        resolution: int,
        z_channels: int,
        double_z: bool = False,
        use_linear_attn: bool = False,
        attn_type: str = "vanilla",
        **ignore_kwargs,
    ):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        temb = None
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        h = self.conv_out(nonlinearity(self.norm_out(h)))
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        ch: int,
        out_ch: int,
        ch_mult: tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int,
        attn_resolutions: tuple[int, ...],
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
        in_channels: int | None,
        resolution: int,
        z_channels: int,
        give_pre_end: bool = False,
        tanh_out: bool = False,
        use_linear_attn: bool = False,
        attn_type: str = "vanilla",
        **ignore_kwargs,
    ):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)

        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        temb = None
        h = self.conv_in(z)
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        if self.give_pre_end:
            return h
        h = self.conv_out(nonlinearity(self.norm_out(h)))
        if self.tanh_out:
            h = torch.tanh(h)
        return h


class SoftVectorQuantizer2D(nn.Module):
    """SoftVQ quantizer: each spatial latent is a soft weighted sum of codewords."""

    def __init__(
        self,
        num_codes: int,
        code_dim: int,
        temperature: float = 0.07,
        learnable_temperature: bool = False,
        eps: float = 1e-8,
    ):
        super().__init__()
        if num_codes <= 1:
            raise ValueError("num_codes must be > 1.")
        if code_dim <= 0:
            raise ValueError("code_dim must be positive.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")

        self.num_codes = num_codes
        self.code_dim = code_dim
        self.eps = eps
        self.codebook = nn.Embedding(num_codes, code_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)

        log_temperature = torch.tensor(math.log(temperature), dtype=torch.float32)
        if learnable_temperature:
            self.log_temperature = nn.Parameter(log_temperature)
        else:
            self.register_buffer("log_temperature", log_temperature, persistent=True)

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp_min(self.eps)

    def forward(self, z_e: torch.Tensor) -> dict[str, torch.Tensor]:
        if z_e.ndim != 4:
            raise ValueError(f"Expected z_e with shape (B, D, H, W), got {tuple(z_e.shape)}")
        bsz, dim, height, width = z_e.shape
        if dim != self.code_dim:
            raise ValueError(f"Expected latent dim {self.code_dim}, got {dim}")

        flat = z_e.permute(0, 2, 3, 1).contiguous().view(-1, dim)
        codebook = self.codebook.weight
        distances = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * flat @ codebook.t()
            + codebook.pow(2).sum(dim=1, keepdim=True).t()
        )
        probs = F.softmax(-distances / self.temperature, dim=-1)
        quantized_flat = probs @ codebook
        z_q = quantized_flat.view(bsz, height, width, dim).permute(0, 3, 1, 2).contiguous()

        token_entropy = -(probs * torch.log(probs.clamp_min(self.eps))).sum(dim=-1)
        avg_token_entropy = token_entropy.mean()
        avg_probs = probs.mean(dim=0)
        batch_entropy = -(avg_probs * torch.log(avg_probs.clamp_min(self.eps))).sum()
        kl_loss = avg_token_entropy - batch_entropy

        with torch.no_grad():
            hard_indices = torch.argmax(probs, dim=-1).view(bsz, height, width)
            perplexity = torch.exp(batch_entropy)

        return {
            "z_q": z_q,
            "probs": probs.view(bsz, height, width, self.num_codes),
            "indices": hard_indices,
            "kl_loss": kl_loss,
            "avg_token_entropy": avg_token_entropy,
            "batch_entropy": batch_entropy,
            "perplexity": perplexity,
            "assignment_entropy": avg_token_entropy.detach(),
        }


class SoftVQVAE(nn.Module):
    """Latent-diffusion-style 2D CNN VQ-VAE with a differentiable SoftVQ codebook."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        resolution: int,
        hidden_size: int = 128,
        num_res_blocks: int = 2,
        code_dim: int = 32,
        num_codes: int = 512,
        ch_mult: tuple[int, ...] = (1, 2, 4),
        attn_resolutions: tuple[int, ...] = (),
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
        temperature: float = 0.07,
        learnable_temperature: bool = False,
        attn_type: str = "vanilla",
        tanh_out: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.resolution = resolution
        self.code_dim = code_dim
        self.num_codes = num_codes
        self.ch_mult = tuple(ch_mult)
        self.latent_downsample = 2 ** (len(self.ch_mult) - 1)
        self.latent_resolution = resolution // self.latent_downsample

        self.encoder = Encoder(
            ch=hidden_size,
            ch_mult=self.ch_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=tuple(attn_resolutions),
            dropout=dropout,
            resamp_with_conv=resamp_with_conv,
            in_channels=input_dim,
            resolution=resolution,
            z_channels=code_dim,
            double_z=False,
            attn_type=attn_type,
        )
        self.quantizer = SoftVectorQuantizer2D(
            num_codes=num_codes,
            code_dim=code_dim,
            temperature=temperature,
            learnable_temperature=learnable_temperature,
        )
        self.decoder = Decoder(
            ch=hidden_size,
            out_ch=output_dim,
            ch_mult=self.ch_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=tuple(attn_resolutions),
            dropout=dropout,
            resamp_with_conv=resamp_with_conv,
            in_channels=None,
            resolution=resolution,
            z_channels=code_dim,
            attn_type=attn_type,
            tanh_out=tanh_out,
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        z_e = self.encoder(x)
        q = self.quantizer(z_e)
        return z_e, q

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        return self.decoder(z_q)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        _, q = self.encode(x)
        return q["z_q"].flatten(1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z_e, q = self.encode(x)
        recon = self.decode(q["z_q"])
        return {
            "recon": recon,
            "z_e": z_e,
            "z_q": q["z_q"],
            "probs": q["probs"],
            "indices": q["indices"],
            "kl_loss": q["kl_loss"],
            "avg_token_entropy": q["avg_token_entropy"],
            "batch_entropy": q["batch_entropy"],
            "perplexity": q["perplexity"],
            "assignment_entropy": q["assignment_entropy"],
            "temperature": self.quantizer.temperature.detach(),
        }

    def loss_function(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        kl_loss: torch.Tensor,
        recon_weight: float = 1.0,
        kl_weight: float = 0.01,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if mask is None:
            recon_loss = F.mse_loss(recon, x)
        else:
            # Per-sample masked MSE so each sample contributes equally even
            # when valid (non-padded) pixel counts differ across the batch.
            sq_err = (recon - x).square()
            weighted = sq_err * mask
            per_sample_num = weighted.flatten(1).sum(dim=1)
            per_sample_den = mask.flatten(1).sum(dim=1).clamp_min(1.0)
            recon_loss = (per_sample_num / per_sample_den).mean()
        loss = recon_weight * recon_loss + kl_weight * kl_loss
        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }


if __name__ == "__main__":
    model = SoftVQVAE(
        input_dim=1,
        output_dim=1,
        resolution=32,
        hidden_size=32,
        num_res_blocks=1,
        code_dim=8,
        num_codes=64,
        ch_mult=(1, 2, 4),
        attn_resolutions=(8,),
        attn_type="none",
    )
    x = torch.randn(4, 1, 32, 32)
    out = model(x)
    print("recon:", out["recon"].shape)
    print("z_e:", out["z_e"].shape)
    print("z_q:", out["z_q"].shape)
    print("probs:", out["probs"].shape)
    print("indices:", out["indices"].shape)
    print("perplexity:", float(out["perplexity"]))
    model.loss_function(x, out["recon"], out["kl_loss"])["loss"].backward()
    print("backward ok")
