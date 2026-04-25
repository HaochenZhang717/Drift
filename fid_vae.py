import torch
import torch.nn as nn
import torch.nn.functional as F


class NormCausalAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q, k = self.q_norm(q), self.k_norm(k)
        q = q.to(v.dtype)
        k = k.to(v.dtype)

        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        mlp_ratio = 4.0

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.attn = NormCausalAttention(hidden_size, num_heads=num_heads)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(
            hidden_size,
            int(2 / 3 * mlp_hidden_dim),
            hidden_size
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class FIDEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_size=128,
        num_layers=4,
        num_heads=4,
        latent_dim=64,
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_size // 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_size // 2, hidden_size, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.layers = nn.ModuleList([
            EncoderLayer(hidden_size, num_heads)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_size)
        self.to_mu = nn.Linear(hidden_size, latent_dim)
        self.to_logvar = nn.Linear(hidden_size, latent_dim)

    def forward(self, x):
        """
        x: (B, C, T)
        return:
            mu:     (B, latent_dim)
            logvar: (B, latent_dim)
        """
        x = self.conv(x)              # (B, hidden, T')
        x = x.permute(0, 2, 1)        # (B, T', hidden)

        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)

        # global pooling over time
        x = x.mean(dim=1)             # (B, hidden)

        mu = self.to_mu(x)            # (B, latent_dim)
        logvar = self.to_logvar(x)    # (B, latent_dim)

        # 可选：数值稳定
        logvar = torch.clamp(logvar, min=-6.0, max=6.0)

        return mu, logvar


class FIDDecoder(nn.Module):
    def __init__(
        self,
        latent_dim,
        output_dim,
        hidden_size=128,
        seq_len=128,
    ):
        super().__init__()

        assert seq_len % 4 == 0, "seq_len must be divisible by 4"
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.base_len = seq_len // 4

        self.fc = nn.Linear(latent_dim, hidden_size * self.base_len)

        self.net = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),

            nn.Conv1d(hidden_size, hidden_size // 2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),

            nn.Conv1d(hidden_size // 2, output_dim, kernel_size=5, padding=2),
        )

    def forward(self, z):
        """
        z: (B, latent_dim)
        return: (B, C, T)
        """
        B = z.shape[0]

        x = self.fc(z)                                # (B, hidden * T/4)
        x = x.view(B, self.hidden_size, self.base_len)
        x = self.net(x)                               # (B, C, T)

        return x


class FIDVAE(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        seq_len,
        hidden_size=128,
        num_layers=4,
        num_heads=4,
        latent_dim=64,
        beta=0.001,
    ):
        super().__init__()

        self.beta = beta

        self.encoder = FIDEncoder(
            input_dim=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            latent_dim=latent_dim,
        )

        self.decoder = FIDDecoder(
            latent_dim=latent_dim,
            output_dim=output_dim,
            hidden_size=hidden_size,
            seq_len=seq_len,
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        """
        x: (B, C, T)
        returns:
            mu, logvar, z with shape (B, latent_dim)
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return mu, logvar, z

    def decode(self, z):
        return self.decoder(z)

    def get_embedding(self, x, use_mu=True):
        """
        x: (B, C, T)
        return: (B, latent_dim)
        """
        mu, logvar = self.encoder(x)
        if use_mu:
            return mu
        return self.reparameterize(mu, logvar)

    def forward(self, x):
        """
        x: (B, C, T)
        """
        mu, logvar, z = self.encode(x)
        recon = self.decode(z)

        return {
            "recon": recon,
            "mu": mu,
            "logvar": logvar,
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
        num_heads=8,
        latent_dim=128,
        beta=0.001,
    )

    x = torch.randn(8, 1, 128)

    out = model(x)

    print("recon:", out["recon"].shape)   # (8, 1, 128)
    print("mu:", out["mu"].shape)         # (8, 128)
    print("logvar:", out["logvar"].shape) # (8, 128)
    print("z:", out["z"].shape)           # (8, 128)

    loss_dict = model.loss_function(
        x,
        out["recon"],
        out["mu"],
        out["logvar"]
    )

    loss = loss_dict["loss"]
    loss.backward()
    print("backward ok")