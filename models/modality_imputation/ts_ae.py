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
        # A shared learnable token used for both:
        # 1) dataset-missing/padding positions from observed_mask
        # 2) additionally masked positions from input_mask
        self.special_token = nn.Parameter(torch.zeros(1, 1, d_model))

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

        # Time-step level mask:
        # - observed_mask==0 at a time step (dataset missing/padding)
        # - input_mask==1 at a time step (additional masking)
        observation_time_missing = ~(observed_mask > 0).any(dim=-1)  # (B, T), bool
        input_time_masked = (input_mask > 0).any(dim=-1)  # (B, T), bool
        special_token_mask = observation_time_missing | input_time_masked

        h = self.input_proj(x)  # (B, T, D)
        special = self.special_token.expand(h.size(0), h.size(1), -1)
        h = torch.where(special_token_mask.unsqueeze(-1), special, h)
        h = self.pos_enc(h)
        # No attention mask: all positions attend, including special-token positions.
        h = self.encoder(h)
        out = self.decoder(h)
        return out
