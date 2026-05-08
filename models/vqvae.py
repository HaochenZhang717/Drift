import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _num_groups(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ConvResBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(_num_groups(channels), channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.Dropout(dropout),
            nn.GroupNorm(_num_groups(channels), channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class DownsampleBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.res = ConvResBlock(channels, dropout=dropout)
        self.down = nn.Conv1d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.down(self.res(x))


class UpsampleBlock(nn.Module):
    def __init__(self, channels: int, scale_factor: int = 4, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose1d(
                channels,
                channels,
                kernel_size=scale_factor,
                stride=scale_factor,
            ),
            nn.GroupNorm(_num_groups(channels), channels),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class VQEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 128,
        num_layers: int = 4,
        code_dim: int = 64,
        latent_downsample: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        if latent_downsample < 1 or latent_downsample & (latent_downsample - 1) != 0:
            raise ValueError("latent_downsample must be a power of two.")

        self.stem = nn.Conv1d(input_dim, hidden_size, kernel_size=7, padding=3)
        self.down_blocks = nn.ModuleList(
            [DownsampleBlock(hidden_size, dropout=dropout) for _ in range(int(math.log2(latent_downsample)))]
        )
        self.layers = nn.ModuleList([ConvResBlock(hidden_size, dropout=dropout) for _ in range(num_layers)])
        self.final = nn.Sequential(
            nn.GroupNorm(_num_groups(hidden_size), hidden_size),
            nn.SiLU(),
            nn.Conv1d(hidden_size, code_dim, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.stem(x)
        for block in self.down_blocks:
            x = block(x)
        for layer in self.layers:
            x = layer(x)
        return self.final(x)


class VQDecoder(nn.Module):
    def __init__(
        self,
        code_dim: int,
        output_dim: int,
        hidden_size: int = 128,
        seq_len: int = 128,
        num_layers: int = 4,
        latent_downsample: int = 8,
        decoder_upsample_rate: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        if latent_downsample < 1 or latent_downsample & (latent_downsample - 1) != 0:
            raise ValueError("latent_downsample must be a power of two.")
        if decoder_upsample_rate < 1:
            raise ValueError("decoder_upsample_rate must be positive.")
        if latent_downsample % decoder_upsample_rate != 0 and latent_downsample != 1:
            raise ValueError("latent_downsample must be divisible by decoder_upsample_rate.")

        self.seq_len = seq_len
        self.code_dim = code_dim
        self.latent_downsample = latent_downsample
        self.decoder_upsample_rate = decoder_upsample_rate
        self.latent_seq_len = max(1, math.ceil(seq_len / latent_downsample))

        self.register_buffer("seq_len_buffer", torch.tensor(seq_len, dtype=torch.long), persistent=True)
        self.register_buffer(
            "latent_downsample_buffer", torch.tensor(latent_downsample, dtype=torch.long), persistent=True
        )

        self.in_proj = nn.Sequential(
            nn.Conv1d(code_dim, hidden_size, kernel_size=3, padding=1),
            nn.GroupNorm(_num_groups(hidden_size), hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        num_upsample_blocks = int(math.log(latent_downsample, decoder_upsample_rate)) if latent_downsample > 1 else 0
        self.up_blocks = nn.ModuleList(
            [UpsampleBlock(hidden_size, scale_factor=decoder_upsample_rate, dropout=dropout) for _ in range(num_upsample_blocks)]
        )

        self.refine = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
                    nn.GroupNorm(_num_groups(hidden_size), hidden_size),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                )
                for _ in range(max(0, num_layers - num_upsample_blocks))
            ]
        )

        self.out = nn.Sequential(
            nn.GroupNorm(_num_groups(hidden_size), hidden_size),
            nn.SiLU(),
            nn.Conv1d(hidden_size, output_dim, kernel_size=3, padding=1),
        )

    def forward(self, z_q):
        if z_q.ndim == 2:
            z_q = z_q.view(z_q.shape[0], self.code_dim, self.latent_seq_len)

        x = self.in_proj(z_q)
        for block in self.up_blocks:
            x = block(x)
        x = self.refine(x)
        x = self.out(x)

        if x.shape[-1] != self.seq_len:
            raise ValueError(
                f"Decoder output length {x.shape[-1]} does not match seq_len {self.seq_len}. "
                "Adjust latent_downsample and decoder_upsample_rate."
            )
        return x


class VectorQuantizer(nn.Module):
    """
    Standard VQ layer with straight-through estimator.
    Returns quantized latents and VQ losses for codebook + commitment.
    """

    def __init__(self, num_codes: int, code_dim: int, commitment_weight: float = 0.25):
        super().__init__()
        if num_codes <= 1:
            raise ValueError("num_codes must be > 1.")
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.commitment_weight = commitment_weight

        self.codebook = nn.Embedding(num_codes, code_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)

    def forward(self, z_e):
        # z_e: (B, D, T_latent)
        bsz, dim, t_latent = z_e.shape
        flat = z_e.permute(0, 2, 1).contiguous().view(-1, dim)  # (B*T, D)

        codebook = self.codebook.weight  # (K, D)
        distances = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * flat @ codebook.t()
            + codebook.pow(2).sum(dim=1, keepdim=True).t()
        )
        encoding_indices = torch.argmin(distances, dim=1)  # (B*T)

        quantized_flat = self.codebook(encoding_indices)  # (B*T, D)
        quantized = quantized_flat.view(bsz, t_latent, dim).permute(0, 2, 1).contiguous()  # (B, D, T)

        # Losses: codebook and commitment.
        codebook_loss = F.mse_loss(quantized, z_e.detach())
        commitment_loss = F.mse_loss(z_e, quantized.detach())
        vq_loss = codebook_loss + self.commitment_weight * commitment_loss

        # Straight-through estimator.
        quantized_st = z_e + (quantized - z_e).detach()

        # Optional stats.
        with torch.no_grad():
            one_hot = F.one_hot(encoding_indices, num_classes=self.num_codes).to(z_e.dtype)
            usage = one_hot.mean(dim=0)
            perplexity = torch.exp(-(usage * torch.log(usage + 1e-10)).sum())

        return {
            "z_q": quantized_st,
            "indices": encoding_indices.view(bsz, t_latent),
            "vq_loss": vq_loss,
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
            "perplexity": perplexity,
        }


class VQVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        seq_len: int,
        hidden_size: int = 128,
        num_layers: int = 4,
        code_dim: int = 64,
        num_codes: int = 512,
        latent_downsample: int = 8,
        decoder_upsample_rate: int = 4,
        dropout: float = 0.0,
        commitment_weight: float = 0.25,
    ):
        super().__init__()
        self.code_dim = code_dim
        self.latent_downsample = latent_downsample
        self.latent_seq_len = max(1, math.ceil(seq_len / latent_downsample))

        self.encoder = VQEncoder(
            input_dim=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            code_dim=code_dim,
            latent_downsample=latent_downsample,
            dropout=dropout,
        )
        self.quantizer = VectorQuantizer(
            num_codes=num_codes,
            code_dim=code_dim,
            commitment_weight=commitment_weight,
        )
        self.decoder = VQDecoder(
            code_dim=code_dim,
            output_dim=output_dim,
            hidden_size=hidden_size,
            seq_len=seq_len,
            num_layers=num_layers,
            latent_downsample=latent_downsample,
            decoder_upsample_rate=decoder_upsample_rate,
            dropout=dropout,
        )

    def encode(self, x):
        z_e = self.encoder(x)
        q = self.quantizer(z_e)
        return z_e, q

    def decode(self, z_q):
        return self.decoder(z_q)

    def get_embedding(self, x):
        _, q = self.encode(x)
        return q["z_q"].flatten(1)

    def forward(self, x):
        z_e, q = self.encode(x)
        recon = self.decode(q["z_q"])
        return {
            "recon": recon,
            "z_e": z_e,
            "z_q": q["z_q"],
            "indices": q["indices"],
            "vq_loss": q["vq_loss"],
            "codebook_loss": q["codebook_loss"],
            "commitment_loss": q["commitment_loss"],
            "perplexity": q["perplexity"],
        }

    def loss_function(self, x, recon, vq_loss):
        recon_loss = F.mse_loss(recon, x)
        loss = recon_loss + vq_loss
        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "vq_loss": vq_loss,
        }


if __name__ == "__main__":
    model = VQVAE(
        input_dim=1,
        output_dim=1,
        seq_len=128,
        hidden_size=128,
        num_layers=2,
        code_dim=32,
        num_codes=256,
        latent_downsample=16,
        decoder_upsample_rate=4,
        commitment_weight=0.25,
    )

    x = torch.randn(8, 1, 128)
    out = model(x)

    print("recon:", out["recon"].shape)      # (8, 1, 128)
    print("z_e:", out["z_e"].shape)          # (8, code_dim, latent_seq_len)
    print("z_q:", out["z_q"].shape)          # (8, code_dim, latent_seq_len)
    print("indices:", out["indices"].shape)  # (8, latent_seq_len)
    print("perplexity:", float(out["perplexity"]))

    loss_dict = model.loss_function(x, out["recon"], out["vq_loss"])
    loss = loss_dict["loss"]
    loss.backward()
    print("backward ok")
