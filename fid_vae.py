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
        self.scale_factor = scale_factor
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.res = ConvResBlock(channels, dropout=dropout)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="linear", align_corners=False)
        return self.res(self.conv(x))


class FIDEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_size=128,
        num_layers=4,
        latent_dim=4,
        latent_downsample=8,
        dropout=0.0,
    ):
        super().__init__()
        if latent_downsample < 1 or latent_downsample & (latent_downsample - 1) != 0:
            raise ValueError("latent_downsample must be a power of two.")

        self.latent_downsample = latent_downsample
        self.stem = nn.Conv1d(input_dim, hidden_size, kernel_size=7, padding=3)
        self.down_blocks = nn.ModuleList([
            DownsampleBlock(hidden_size, dropout=dropout)
            for _ in range(int(math.log2(latent_downsample)))
        ])
        self.layers = nn.ModuleList([
            ConvResBlock(hidden_size, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.GroupNorm(_num_groups(hidden_size), hidden_size)
        self.final_act = nn.SiLU()
        self.to_mu = nn.Conv1d(hidden_size, latent_dim, kernel_size=3, padding=1)
        self.to_logvar = nn.Conv1d(hidden_size, latent_dim, kernel_size=3, padding=1)

    def forward(self, x):
        """
        x: (B, C, T)
        return:
            mu:     (B, latent_dim, T // latent_downsample)
            logvar: (B, latent_dim, T // latent_downsample)
        """
        x = self.stem(x)
        for block in self.down_blocks:
            x = block(x)
        for layer in self.layers:
            x = layer(x)

        x = self.final_act(self.final_norm(x))

        mu = self.to_mu(x)
        logvar = self.to_logvar(x)
        logvar = torch.clamp(logvar, min=-6.0, max=6.0)

        return mu, logvar


class FIDDecoder(nn.Module):
    def __init__(
        self,
        latent_dim,
        output_dim,
        hidden_size=128,
        seq_len=128,
        num_layers=4,
        latent_downsample=8,
        decoder_upsample_rate=4,
        dropout=0.0,
    ):
        super().__init__()
        if latent_downsample < 1 or latent_downsample & (latent_downsample - 1) != 0:
            raise ValueError("latent_downsample must be a power of two.")
        if decoder_upsample_rate < 1:
            raise ValueError("decoder_upsample_rate must be positive.")
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.latent_downsample = latent_downsample
        self.decoder_upsample_rate = decoder_upsample_rate
        self.latent_seq_len = max(1, math.ceil(seq_len / latent_downsample))
        self.register_buffer("seq_len_buffer", torch.tensor(seq_len, dtype=torch.long), persistent=True)
        self.register_buffer("latent_downsample_buffer", torch.tensor(latent_downsample, dtype=torch.long), persistent=True)
        self.register_buffer("decoder_upsample_rate_buffer", torch.tensor(decoder_upsample_rate, dtype=torch.long), persistent=True)

        self.in_proj = nn.Conv1d(latent_dim, hidden_size, kernel_size=3, padding=1)
        self.layers = nn.ModuleList([
            ConvResBlock(hidden_size, dropout=dropout)
            for _ in range(num_layers)
        ])
        num_upsample_blocks = int(math.log(latent_downsample, decoder_upsample_rate))
        self.up_blocks = nn.ModuleList([
            UpsampleBlock(hidden_size, scale_factor=decoder_upsample_rate, dropout=dropout)
            for _ in range(num_upsample_blocks)
        ])
        self.out = nn.Sequential(
            nn.GroupNorm(_num_groups(hidden_size), hidden_size),
            nn.SiLU(),
            nn.Conv1d(hidden_size, output_dim, kernel_size=3, padding=1),
        )

    def forward(self, z):
        """
        z: (B, latent_dim, T // latent_downsample)
        return: (B, C, T)
        """
        if z.ndim == 2:
            z = z.view(z.shape[0], self.latent_dim, self.latent_seq_len)

        x = self.in_proj(z)
        for layer in self.layers:
            x = layer(x)
        for block in self.up_blocks:
            x = block(x)
        x = self.out(x)
        if x.shape[-1] != self.seq_len:
            x = F.interpolate(x, size=self.seq_len, mode="linear", align_corners=False)
        return x


class FIDVAE(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        seq_len,
        hidden_size=128,
        num_layers=4,
        latent_dim=4,
        latent_downsample=8,
        decoder_upsample_rate=4,
        dropout=0.0,
        beta=0.001,
    ):
        super().__init__()

        self.beta = beta
        self.latent_dim = latent_dim
        self.latent_downsample = latent_downsample
        self.latent_seq_len = max(1, math.ceil(seq_len / latent_downsample))

        self.encoder = FIDEncoder(
            input_dim=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            latent_dim=latent_dim,
            latent_downsample=latent_downsample,
            dropout=dropout,
        )

        self.decoder = FIDDecoder(
            latent_dim=latent_dim,
            output_dim=output_dim,
            hidden_size=hidden_size,
            seq_len=seq_len,
            num_layers=num_layers,
            latent_downsample=latent_downsample,
            decoder_upsample_rate=decoder_upsample_rate,
            dropout=dropout,
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        """
        x: (B, C, T)
        returns:
            mu, logvar, z with shape (B, latent_dim, T // latent_downsample)
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return mu, logvar, z

    def decode(self, z):
        return self.decoder(z)

    def get_embedding(self, x, use_mu=True):
        """
        x: (B, C, T)
        return: (B, latent_dim * latent_seq_len)
        """
        mu, logvar = self.encoder(x)
        if use_mu:
            return mu.flatten(1)
        return self.reparameterize(mu, logvar).flatten(1)

    def forward(self, x):
        """
        x: (B, C, T)
        """
        mu, logvar, z = self.encode(x)
        recon = self.decode(z)

        return {
            "recon": recon,
            "mu": mu.flatten(1),
            "logvar": logvar.flatten(1),
            "z": z,
        }

    def loss_function(self, x, recon, mu, logvar):
        recon_loss = F.mse_loss(recon, x)

        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl.mean()

        loss = recon_loss + self.beta * kl_loss

        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }


if __name__ == "__main__":
    model = FIDVAE(
        input_dim=1,
        output_dim=1,
        seq_len=128,
        hidden_size=128,
        num_layers=2,
        latent_dim=4,
        latent_downsample=8,
        beta=0.001,
    )

    x = torch.randn(8, 1, 128)

    out = model(x)

    print("recon:", out["recon"].shape)   # (8, 1, 128)
    print("mu:", out["mu"].shape)         # (8, latent_dim * latent_seq_len)
    print("logvar:", out["logvar"].shape) # (8, latent_dim * latent_seq_len)
    print("z:", out["z"].shape)           # (8, latent_dim, latent_seq_len)

    loss_dict = model.loss_function(
        x,
        out["recon"],
        out["mu"],
        out["logvar"]
    )

    loss = loss_dict["loss"]
    loss.backward()
    print("backward ok")
