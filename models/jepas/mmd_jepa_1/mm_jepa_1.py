import copy
import math
from typing import Dict, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _num_groups(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


def _downsampled_length(seq_len: int, latent_downsample: int) -> int:
    length = seq_len
    for _ in range(int(math.log2(latent_downsample))):
        length = length // 2
    if length < 1:
        raise ValueError(
            f"seq_len={seq_len} is too short for latent_downsample={latent_downsample}."
        )
    return length


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class DownsampleBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.res = ConvResBlock(channels, dropout=dropout)
        self.down = nn.Conv1d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.res(x))


class TSJEPAEncoder(nn.Module):
    """
    Conv1d time-series encoder following the VQEncoder pattern in models/vqvae.py.

    Input shape: (B, C, T)
    Output shape: (B, D, T_latent)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 128,
        num_layers: int = 4,
        embed_dim: int = 64,
        latent_downsample: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        if latent_downsample < 1 or latent_downsample & (latent_downsample - 1) != 0:
            raise ValueError("latent_downsample must be a power of two.")

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.latent_downsample = latent_downsample

        self.stem = nn.Conv1d(input_dim, hidden_size, kernel_size=7, padding=3)
        self.down_blocks = nn.ModuleList(
            [
                DownsampleBlock(hidden_size, dropout=dropout)
                for _ in range(int(math.log2(latent_downsample)))
            ]
        )
        self.layers = nn.ModuleList(
            [ConvResBlock(hidden_size, dropout=dropout) for _ in range(num_layers)]
        )
        self.final = nn.Sequential(
            nn.GroupNorm(_num_groups(hidden_size), hidden_size),
            nn.SiLU(),
            nn.Conv1d(hidden_size, embed_dim, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must have shape (B, C, T), got {tuple(x.shape)}")

        x = self.stem(x)
        for block in self.down_blocks:
            x = block(x)
        for layer in self.layers:
            x = layer(x)
        return self.final(x)


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if positions.max().item() >= self.pe.shape[1]:
            raise ValueError(
                f"Position {int(positions.max())} exceeds max_len {self.pe.shape[1]}."
            )
        return x + self.pe[:, positions]


class MultimodalJEPAPredictor(nn.Module):
    """
    Transformer-encoder predictor for embedding-level modality imputation.

    The encoder input is the full multimodal latent sequence with missing modality
    positions replaced by learned special tokens.
    Predictor outputs are selected at missing positions by MultimodalTSJEPA.
    """

    def __init__(
        self,
        num_modalities: int,
        embed_dim: int,
        latent_seq_len: int,
        predictor_dim: Optional[int] = None,
        num_layers: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_len: int = 10000,
    ):
        super().__init__()
        predictor_dim = predictor_dim or embed_dim
        if predictor_dim % num_heads != 0:
            raise ValueError("predictor_dim must be divisible by num_heads.")

        self.num_modalities = num_modalities
        self.embed_dim = embed_dim
        self.predictor_dim = predictor_dim
        self.latent_seq_len = latent_seq_len

        self.in_proj = nn.Identity() if embed_dim == predictor_dim else nn.Linear(embed_dim, predictor_dim)
        self.out_proj = nn.Identity() if predictor_dim == embed_dim else nn.Linear(predictor_dim, embed_dim)
        self.pos_enc = SinusoidalPositionEncoding(predictor_dim, max_len=max_len)
        self.modality_embed = nn.Embedding(num_modalities, predictor_dim)
        self.missing_tokens = nn.Parameter(torch.randn(num_modalities, latent_seq_len, predictor_dim) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=predictor_dim,
            nhead=num_heads,
            dim_feedforward=int(predictor_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(predictor_dim)

    def _add_position_and_modality(
        self,
        tokens: torch.Tensor,
        modality_ids: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        tokens = self.pos_enc(tokens, position_ids)
        tokens = tokens + self.modality_embed(modality_ids).unsqueeze(0)
        return tokens

    def forward(self, embeddings: torch.Tensor, missing_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: online encoder embeddings, shape (B, M, T_latent, D).
            missing_mask: bool tensor, shape (B, M), True means predict this modality.

        Returns:
            Predicted embeddings for all positions, shape (B, M, T_latent, D).
            Callers should select only missing positions for loss.
        """
        if embeddings.ndim != 4:
            raise ValueError(
                f"embeddings must have shape (B, M, T_latent, D), got {tuple(embeddings.shape)}"
            )

        bsz, num_modalities, latent_seq_len, embed_dim = embeddings.shape
        if num_modalities != self.num_modalities:
            raise ValueError(f"Expected {self.num_modalities} modalities, got {num_modalities}.")
        if latent_seq_len != self.latent_seq_len:
            raise ValueError(f"Expected latent_seq_len={self.latent_seq_len}, got {latent_seq_len}.")
        if embed_dim != self.embed_dim:
            raise ValueError(f"Expected embed_dim={self.embed_dim}, got {embed_dim}.")
        if missing_mask.shape != (bsz, num_modalities):
            raise ValueError(
                f"missing_mask must have shape {(bsz, num_modalities)}, got {tuple(missing_mask.shape)}"
            )

        missing_mask = missing_mask.bool()
        if (~missing_mask).sum(dim=1).min().item() == 0:
            raise ValueError("Each sample must have at least one observed modality.")

        projected = self.in_proj(embeddings)
        special = self.missing_tokens.unsqueeze(0).expand(bsz, -1, -1, -1)
        tokens = torch.where(missing_mask[:, :, None, None], special, projected)

        modality_ids = torch.arange(num_modalities, device=embeddings.device).repeat_interleave(latent_seq_len)
        position_ids = torch.arange(latent_seq_len, device=embeddings.device).repeat(num_modalities)

        tokens = tokens.reshape(bsz, num_modalities * latent_seq_len, self.predictor_dim)
        tokens = self._add_position_and_modality(tokens, modality_ids, position_ids)

        encoded = self.encoder(tokens)
        encoded = self.norm(encoded)
        encoded = self.out_proj(encoded)
        return encoded.reshape(bsz, num_modalities, latent_seq_len, self.embed_dim)


class MultimodalTSJEPA(nn.Module):
    """
    Multimodal time-series JEPA with modality-specific online encoders, EMA target
    encoders, and a Transformer-encoder predictor.

    Args:
        input_dims: one channel count per modality.
        seq_len: input time-series length used to infer latent_seq_len.

    Forward:
        modalities: list/tuple of tensors, each (B, C_i, T).
        During training, exactly one modality is randomly masked per sample.
    """

    def __init__(
        self,
        input_dims: Sequence[int],
        seq_len: int,
        hidden_size: int = 128,
        encoder_layers: int = 4,
        embed_dim: int = 64,
        latent_downsample: int = 8,
        encoder_dropout: float = 0.0,
        predictor_dim: Optional[int] = None,
        predictor_layers: int = 4,
        predictor_heads: int = 4,
        predictor_mlp_ratio: float = 4.0,
        predictor_dropout: float = 0.0,
        ema_momentum: float = 0.996,
        max_len: int = 10000,
    ):
        super().__init__()
        if len(input_dims) < 2:
            raise ValueError("input_dims must contain at least two modalities.")
        if not (0.0 <= ema_momentum < 1.0):
            raise ValueError("ema_momentum must be in [0, 1).")

        self.input_dims = list(input_dims)
        self.num_modalities = len(input_dims)
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.latent_downsample = latent_downsample
        self.latent_seq_len = _downsampled_length(seq_len, latent_downsample)
        self.ema_momentum = ema_momentum

        self.encoders = nn.ModuleList(
            [
                TSJEPAEncoder(
                    input_dim=input_dim,
                    hidden_size=hidden_size,
                    num_layers=encoder_layers,
                    embed_dim=embed_dim,
                    latent_downsample=latent_downsample,
                    dropout=encoder_dropout,
                )
                for input_dim in input_dims
            ]
        )
        self.target_encoders = copy.deepcopy(self.encoders)
        self._freeze_target_encoders()

        self.predictor = MultimodalJEPAPredictor(
            num_modalities=self.num_modalities,
            embed_dim=embed_dim,
            latent_seq_len=self.latent_seq_len,
            predictor_dim=predictor_dim,
            num_layers=predictor_layers,
            num_heads=predictor_heads,
            mlp_ratio=predictor_mlp_ratio,
            dropout=predictor_dropout,
            max_len=max_len,
        )

    def _freeze_target_encoders(self) -> None:
        for encoder in self.target_encoders:
            encoder.eval()
            for param in encoder.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        for encoder in self.target_encoders:
            encoder.eval()
        return self

    @torch.no_grad()
    def update_target_encoders(self, momentum: Optional[float] = None) -> None:
        """EMA update. Call this after each optimizer step."""
        momentum = self.ema_momentum if momentum is None else momentum
        if not (0.0 <= momentum < 1.0):
            raise ValueError("momentum must be in [0, 1).")

        for online_encoder, target_encoder in zip(self.encoders, self.target_encoders):
            for online_param, target_param in zip(online_encoder.parameters(), target_encoder.parameters()):
                target_param.data.mul_(momentum).add_(online_param.data, alpha=1.0 - momentum)
            for online_buffer, target_buffer in zip(online_encoder.buffers(), target_encoder.buffers()):
                target_buffer.copy_(online_buffer)

    def _validate_modalities(self, modalities: Sequence[torch.Tensor]) -> None:
        if len(modalities) != self.num_modalities:
            raise ValueError(f"Expected {self.num_modalities} modality tensors, got {len(modalities)}.")
        batch_size = modalities[0].shape[0]
        for idx, (x, input_dim) in enumerate(zip(modalities, self.input_dims)):
            if x.ndim != 3:
                raise ValueError(f"modality {idx} must have shape (B, C, T), got {tuple(x.shape)}.")
            if x.shape[0] != batch_size:
                raise ValueError("All modalities must share the same batch size.")
            if x.shape[1] != input_dim:
                raise ValueError(f"modality {idx} expected {input_dim} channels, got {x.shape[1]}.")
            if x.shape[2] != self.seq_len:
                raise ValueError(f"modality {idx} expected seq_len={self.seq_len}, got {x.shape[2]}.")

    def _sample_missing_mask(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        missing_modality_ids = torch.randint(self.num_modalities, (batch_size,), device=device)
        missing_mask = F.one_hot(missing_modality_ids, num_classes=self.num_modalities).bool()
        return {
            "missing_mask": missing_mask,
            "missing_modality_ids": missing_modality_ids,
        }

    def _missing_mask_from_ids(
        self,
        missing_modalities: Union[int, Sequence[int], torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        if isinstance(missing_modalities, int):
            missing_modality_ids = torch.full(
                (batch_size,),
                missing_modalities,
                dtype=torch.long,
                device=device,
            )
        elif isinstance(missing_modalities, torch.Tensor):
            missing_modality_ids = missing_modalities.to(device=device, dtype=torch.long)
            if missing_modality_ids.ndim == 0:
                missing_modality_ids = missing_modality_ids.expand(batch_size)
        else:
            missing_modality_ids = torch.tensor(missing_modalities, dtype=torch.long, device=device)

        if missing_modality_ids.shape != (batch_size,):
            raise ValueError(
                f"missing_modalities must be scalar or shape {(batch_size,)}, "
                f"got {tuple(missing_modality_ids.shape)}."
            )
        if (missing_modality_ids < 0).any() or (missing_modality_ids >= self.num_modalities).any():
            raise ValueError(f"missing_modalities must be in [0, {self.num_modalities - 1}].")

        missing_mask = F.one_hot(missing_modality_ids, num_classes=self.num_modalities).bool()
        return {
            "missing_mask": missing_mask,
            "missing_modality_ids": missing_modality_ids,
        }

    def encode(self, modalities: Sequence[torch.Tensor], target: bool = False) -> torch.Tensor:
        self._validate_modalities(modalities)
        encoders = self.target_encoders if target else self.encoders
        outputs = [encoder(x).transpose(1, 2) for encoder, x in zip(encoders, modalities)]
        return torch.stack(outputs, dim=1)

    def predict(self, online_embeddings: torch.Tensor, missing_mask: torch.Tensor) -> torch.Tensor:
        return self.predictor(online_embeddings, missing_mask)

    def forward(
        self,
        modalities: Sequence[torch.Tensor],
        missing_modalities: Optional[Union[int, Sequence[int], torch.Tensor]] = None,
        return_all: bool = True,
    ) -> Dict[str, torch.Tensor]:
        self._validate_modalities(modalities)
        batch_size = modalities[0].shape[0]
        device = modalities[0].device
        if missing_modalities is None:
            missing = self._sample_missing_mask(batch_size, device)
        else:
            missing = self._missing_mask_from_ids(missing_modalities, batch_size, device)
        missing_mask = missing["missing_mask"]
        missing_modality_ids = missing["missing_modality_ids"]

        online_embeddings = self.encode(modalities, target=False)
        with torch.no_grad():
            target_embeddings = self.encode(modalities, target=True)

        predictions = self.predict(online_embeddings, missing_mask)
        token_mask = missing_mask[:, :, None, None].expand_as(predictions)
        loss = F.mse_loss(predictions[token_mask], target_embeddings[token_mask])
        batch_idx = torch.arange(batch_size, device=device)
        predicted_missing_tokens = predictions[batch_idx, missing_modality_ids]
        target_missing_tokens = target_embeddings[batch_idx, missing_modality_ids]

        output = {
            "loss": loss,
            "predictions": predictions,
            "targets": target_embeddings,
            "missing_mask": missing_mask,
            "missing_modality_ids": missing_modality_ids,
            "predicted_missing_tokens": predicted_missing_tokens,
            "target_missing_tokens": target_missing_tokens,
        }
        if return_all:
            output["online_embeddings"] = online_embeddings
            output["predicted_missing"] = predictions[token_mask].view(-1, self.embed_dim)
            output["target_missing"] = target_embeddings[token_mask].view(-1, self.embed_dim)
        return output


if __name__ == "__main__":
    model = MultimodalTSJEPA(
        input_dims=[1, 1, 1, 1],
        seq_len=128,
        hidden_size=64,
        encoder_layers=2,
        embed_dim=32,
        latent_downsample=8,
        predictor_layers=2,
        predictor_heads=4,
    )
    xs = [torch.randn(4, 1, 128) for _ in range(4)]
    out = model(xs)
    print("loss:", float(out["loss"].detach()))
    print("predictions:", out["predictions"].shape)
    print("predicted_missing_tokens:", out["predicted_missing_tokens"].shape)
    print("missing_modality_ids:", out["missing_modality_ids"])
    out["loss"].backward()
    model.update_target_encoders()
    print("backward and ema update ok")
