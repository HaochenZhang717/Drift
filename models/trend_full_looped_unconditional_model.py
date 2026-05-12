"""Two-output looped DiT generator for trend/full Drift training."""

import torch
import torch.nn as nn

from models.unconditional_model import (
    DiTBlock,
    FinalLayer,
    PatchEmbed,
    RotaryPositionEmbedding,
)


class TrendFullLoopedDriftDiT(nn.Module):
    """
    Looped DiT-style generator that emits a trend output after loop 1 and a
    full-resolution output after loop 2.

    The outputs are delay-image tensors with shape (B, C, H, W), matching the
    original unconditional Drift pipeline.
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        hidden_size: int = 256,
        loop_depth: int = 3,
        num_loops: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        num_register_tokens: int = 8,
        separate_heads: bool = False,
    ):
        super().__init__()
        if num_loops != 2:
            raise ValueError("The first trend/full prototype expects num_loops=2")
        if loop_depth <= 0:
            raise ValueError("loop_depth must be positive")

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.loop_depth = loop_depth
        self.num_loops = num_loops
        self.num_heads = num_heads
        self.num_patches = (img_size // patch_size) ** 2
        self.num_register_tokens = num_register_tokens
        self.separate_heads = separate_heads

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=hidden_size,
        )
        self.register_tokens = nn.Parameter(
            torch.randn(1, num_register_tokens, hidden_size) * 0.02
        )

        head_dim = hidden_size // num_heads
        self.rope = RotaryPositionEmbedding(
            dim=head_dim,
            max_seq_len=self.num_patches + num_register_tokens + 64,
        )

        self.loop_blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_qk_norm=True,
                )
                for _ in range(loop_depth)
            ]
        )

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        if separate_heads:
            self.trend_final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
            self.full_final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        else:
            self.trend_final_layer = self.final_layer
            self.full_final_layer = self.final_layer

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert patch tokens back to image."""
        c = self.out_channels
        p = self.patch_size
        h = w = self.img_size // p

        x = x.reshape(-1, h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(-1, c, h * p, w * p)

    def decode_tokens(self, x: torch.Tensor, component: str) -> torch.Tensor:
        """Decode patch tokens into a tanh-bounded delay image."""
        if component == "trend":
            final_layer = self.trend_final_layer
        elif component == "full":
            final_layer = self.full_final_layer
        else:
            raise ValueError(f"Unsupported component {component!r}")
        x = final_layer(x)
        x = self.unpatchify(x)
        return torch.tanh(x)

    def forward(
        self,
        x: torch.Tensor,
        output_component: str | None = None,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        """
        Args:
            x: Input noise, shape (B, C, H, W).
            output_component: If "trend" or "full", return only that tensor.
                If None, return {"trend": x_trend, "full": x_full}.
        """
        if output_component not in {None, "trend", "full"}:
            raise ValueError("output_component must be None, 'trend', or 'full'")

        batch_size = x.shape[0]
        x = self.patch_embed(x)
        register = self.register_tokens.expand(batch_size, -1, -1)
        x = torch.cat([register, x], dim=1)

        seq_len = x.shape[1]
        rope_cos, rope_sin = self.rope(x, seq_len)

        outputs = {}
        for loop_idx in range(self.num_loops):
            for block in self.loop_blocks:
                x = block(x, rope_cos, rope_sin)

            patch_tokens = x[:, self.num_register_tokens :, :]
            if loop_idx == 0:
                outputs["trend"] = self.decode_tokens(patch_tokens, "trend")
            elif loop_idx == 1:
                outputs["full"] = self.decode_tokens(patch_tokens, "full")

        if output_component is not None:
            return outputs[output_component]
        return outputs


def TrendFullLoopedDriftDiT_Tiny(
    img_size=32,
    in_channels=3,
    loop_depth=3,
    num_loops=2,
    **kwargs,
):
    return TrendFullLoopedDriftDiT(
        img_size=img_size,
        patch_size=2,
        in_channels=in_channels,
        hidden_size=256,
        loop_depth=loop_depth,
        num_loops=num_loops,
        num_heads=4,
        mlp_ratio=4.0,
        **kwargs,
    )


def TrendFullLoopedDriftDiT_Small(
    img_size=32,
    in_channels=3,
    loop_depth=4,
    num_loops=2,
    **kwargs,
):
    return TrendFullLoopedDriftDiT(
        img_size=img_size,
        patch_size=4,
        in_channels=in_channels,
        hidden_size=384,
        loop_depth=loop_depth,
        num_loops=num_loops,
        num_heads=6,
        mlp_ratio=4.0,
        **kwargs,
    )


TrendFullLoopedDriftDiT_models = {
    "TrendFullLoopedDriftDiT-Tiny": TrendFullLoopedDriftDiT_Tiny,
    "TrendFullLoopedDriftDiT-Small": TrendFullLoopedDriftDiT_Small,
}
