from pathlib import Path
from typing import Dict, Optional
import torch
import torch.nn as nn
from models.modality_imputation.ts_ae import IrregularTimeSeriesAE


class MultiModalEncoder(nn.Module):
    """
    Multi-modal encoder with 4 frozen modality-specific autoencoders.

    Workflow:
    1) Encode each modality sequence using its frozen AE encoder stack.
    2) Use modality-specific learnable special tokens as cross-attention queries.
    3) Concatenate attended special-token groups from all 4 modalities.

    Returned shape: (B, 4 * n_tokens, dim_out)
    """

    MODALITIES = (
        "heart_rate",
        "calorie",
        "physical_activity",
        "respiratory_rate",
    )

    def __init__(
        self,
        n_tokens: int,
        dim_in: int = 128,
        dim_out: int = 128,
        nhead_cross: int = 4,
        ae_input_dim: int = 1,
        ae_d_model: int = 128,
        ae_nhead: int = 4,
        ae_num_layers: int = 4,
        ae_max_len: int = 10000,
        ckpt_paths: Optional[Dict[str, str]] = None,
        strict_load: bool = True,
    ):
        super().__init__()

        if n_tokens <= 0:
            raise ValueError(f"n_tokens must be > 0, got {n_tokens}")

        self.n_tokens = n_tokens
        self.dim_in = dim_in
        self.dim_out = dim_out

        # 4 modality-specific AEs
        self.heart_rate_ae = IrregularTimeSeriesAE(
            input_dim=ae_input_dim,
            d_model=ae_d_model,
            nhead=ae_nhead,
            num_layers=ae_num_layers,
            max_len=ae_max_len,
        )
        self.calorie_ae = IrregularTimeSeriesAE(
            input_dim=ae_input_dim,
            d_model=ae_d_model,
            nhead=ae_nhead,
            num_layers=ae_num_layers,
            max_len=ae_max_len,
        )
        self.physical_activity_ae = IrregularTimeSeriesAE(
            input_dim=ae_input_dim,
            d_model=ae_d_model,
            nhead=ae_nhead,
            num_layers=ae_num_layers,
            max_len=ae_max_len,
        )
        self.respiratory_rate_ae = IrregularTimeSeriesAE(
            input_dim=ae_input_dim,
            d_model=ae_d_model,
            nhead=ae_nhead,
            num_layers=ae_num_layers,
            max_len=ae_max_len,
        )

        self._ae_map = {
            "heart_rate": self.heart_rate_ae,
            "calorie": self.calorie_ae,
            "physical_activity": self.physical_activity_ae,
            "respiratory_rate": self.respiratory_rate_ae,
        }

        # Load and freeze AEs
        if ckpt_paths is not None:
            self._load_ae_checkpoints(ckpt_paths=ckpt_paths, strict_load=strict_load)
        self._freeze_aes()

        # Project AE token dim -> cross-attn dim if needed
        self.token_proj = nn.Identity() if ae_d_model == dim_in else nn.Linear(ae_d_model, dim_in)
        self.out_proj = nn.Identity() if dim_in == dim_out else nn.Linear(dim_in, dim_out)

        # 4 groups of learnable special tokens
        self.modality_special_tokens = nn.ParameterDict(
            {
                m: nn.Parameter(torch.randn(1, n_tokens, dim_in) * 0.02)
                for m in self.MODALITIES
            }
        )

        # Modality-specific cross-attention blocks
        self.cross_attentions = nn.ModuleDict(
            {
                m: nn.MultiheadAttention(
                    embed_dim=dim_in,
                    num_heads=nhead_cross,
                    batch_first=True,
                )
                for m in self.MODALITIES
            }
        )

    def _extract_state_dict(self, ckpt: Dict) -> Dict[str, torch.Tensor]:
        if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            return ckpt["model_state_dict"]
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
        raise ValueError("Checkpoint must contain a model state dict (model_state_dict/state_dict).")

    def _load_ae_checkpoints(self, ckpt_paths: Dict[str, str], strict_load: bool = True) -> None:
        for modality in self.MODALITIES:
            if modality not in ckpt_paths:
                raise ValueError(f"Missing checkpoint path for modality: {modality}")

            ckpt_path = Path(ckpt_paths[modality])
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found for {modality}: {ckpt_path}")

            ckpt = torch.load(str(ckpt_path), map_location="cpu")
            state_dict = self._extract_state_dict(ckpt)
            self._ae_map[modality].load_state_dict(state_dict, strict=strict_load)

    def _freeze_aes(self) -> None:
        for ae in self._ae_map.values():
            ae.eval()
            for p in ae.parameters():
                p.requires_grad = False

    def _encode_with_ae(
        self,
        ae: IrregularTimeSeriesAE,
        x: torch.Tensor,
        observed_mask: torch.Tensor
    ) -> torch.Tensor:
        """Run only encoder path of IrregularTimeSeriesAE, return latent tokens (B, T, D_ae)."""
        x = x.float()
        observed_mask = observed_mask.float()
        input_mask = torch.zeros_like(observed_mask)

        observation_time_missing = ~(observed_mask > 0).any(dim=-1)
        input_time_masked = (input_mask > 0).any(dim=-1)
        special_token_mask = observation_time_missing | input_time_masked

        h = ae.input_proj(x)
        special = ae.special_token.expand(h.size(0), h.size(1), -1)
        h = torch.where(special_token_mask.unsqueeze(-1), special, h)
        h = ae.pos_enc(h)
        h = ae.encoder(h)
        return h

    def forward(
        self,
        heart_rate: torch.Tensor,
        calorie: torch.Tensor,
        physical_activity: torch.Tensor,
        respiratory_rate: torch.Tensor,
        heart_rate_observed_mask: torch.Tensor,
        calorie_observed_mask: torch.Tensor,
        physical_activity_observed_mask: torch.Tensor,
        respiratory_rate_observed_mask: torch.Tensor) -> torch.Tensor:
        """
        Each modality tensor/mask shape: (B, T, C).
        Returns: concatenated special tokens of shape (B, 4*n_tokens, dim_out).
        """
        with torch.no_grad():
            hr_tokens = self._encode_with_ae(self.heart_rate_ae, heart_rate, heart_rate_observed_mask)
            cal_tokens = self._encode_with_ae(self.calorie_ae, calorie, calorie_observed_mask)
            pa_tokens = self._encode_with_ae(self.physical_activity_ae, physical_activity, physical_activity_observed_mask)
            rr_tokens = self._encode_with_ae(self.respiratory_rate_ae, respiratory_rate, respiratory_rate_observed_mask)

        modality_tokens = {
            "heart_rate": hr_tokens,
            "calorie": cal_tokens,
            "physical_activity": pa_tokens,
            "respiratory_rate": rr_tokens,
        }

        attended_groups = []
        for modality in self.MODALITIES:
            encoded = self.token_proj(modality_tokens[modality])
            q = self.modality_special_tokens[modality].expand(encoded.size(0), -1, -1)
            attended, _ = self.cross_attentions[modality](
                query=q, key=encoded, value=encoded, need_weights=False
            )
            attended = self.out_proj(attended)
            attended_groups.append(attended)

        return torch.cat(attended_groups, dim=1)
