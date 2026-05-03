import torch
import torch.nn as nn


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
        # Two binary embeddings (0/1), one for observation mask and one for input mask.
        self.observation_mask_embed = nn.Embedding(2, d_model)
        self.input_mask_embed = nn.Embedding(2, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.decoder = nn.Linear(d_model, input_dim)

    def forward(self, x, observed_mask, input_mask=None):
        """
        x: (B, T, C)
        observed_mask: (B, T, C), 1=observed(real), 0=missing/padding in dataset
        input_mask: (B, T, C), optional extra training mask marker:
            1 means this observed value is additionally masked for model input.
            0 means not additionally masked.
        """
        if x.dim() != 3:
            raise ValueError(f"x must be (B, T, C), got {tuple(x.shape)}")
        if observed_mask.dim() != 3:
            raise ValueError(f"observed_mask must be (B, T, C), got {tuple(observed_mask.shape)}")
        if x.shape != observed_mask.shape:
            raise ValueError(
                f"x and observed_mask must share shape, got {tuple(x.shape)} vs {tuple(observed_mask.shape)}"
            )
        if input_mask is not None and input_mask.shape != x.shape:
            raise ValueError(f"input_mask must match x shape, got {tuple(input_mask.shape)} vs {tuple(x.shape)}")

        x = x.float()
        observed_mask = observed_mask.float()
        if input_mask is None:
            input_mask = torch.zeros_like(observed_mask)
        else:
            input_mask = input_mask.float()
        visible_input_mask = observed_mask * (1.0 - input_mask)
        x_masked = x * visible_input_mask
        observation_time_valid = (observed_mask > 0).any(dim=-1)  # (B, T), bool
        observation_time_valid_ids = observation_time_valid.long()
        input_time_mask_ids = (input_mask > 0).any(dim=-1).long()  # 1 means additionally masked
        # Padding should still be decided by dataset observation availability.
        key_padding_mask = ~observation_time_valid  # (B, T), True => ignore

        h = self.input_proj(x_masked)
        h = (
            self.pos_enc(h)
            + self.observation_mask_embed(observation_time_valid_ids)
            + self.input_mask_embed(input_time_mask_ids)
        )
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        out = self.decoder(h)
        return out
