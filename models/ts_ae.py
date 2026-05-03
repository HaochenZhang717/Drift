import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Positional Encoding（可选但推荐）
# =========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, T, D)

        self.register_buffer("pe", pe)

    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:, :T]


# =========================
# Model
# =========================
class IrregularTimeSeriesAE(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=4, max_len=10000):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        self.mask_embed = nn.Embedding(2, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.decoder = nn.Linear(d_model, input_dim)

    def forward(self, x, mask):
        """
        x: (B, T, C)
        mask: (B, T, C), 1=valid, 0=padding/missing
        """
        if x.dim() != 3:
            raise ValueError(f"x must be (B, T, C), got {tuple(x.shape)}")
        if mask.dim() != 3:
            raise ValueError(f"mask must be (B, T, C), got {tuple(mask.shape)}")
        if x.shape != mask.shape:
            raise ValueError(f"x and mask must share shape, got {tuple(x.shape)} vs {tuple(mask.shape)}")

        x = x.float()
        mask = mask.float()
        x_masked = x * mask

        # Valid timestamp if any channel is observed at that time.
        time_valid = (mask > 0).any(dim=-1)      # (B, T), bool
        time_valid_ids = time_valid.long()       # (B, T), in {0, 1}
        key_padding_mask = ~time_valid           # (B, T), True => ignore

        h = self.input_proj(x_masked)
        h = self.pos_enc(h) + self.mask_embed(time_valid_ids)
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        out = self.decoder(h)
        return out
