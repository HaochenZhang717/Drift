import torch
import torch.nn as nn
from typing import Sequence, Union

from img_transformations import DelayEmbedder
from models.jepas.mmd_jepa_1.mm_jepa_1 import MultimodalTSJEPA
from models.jepas.mmd_jepa_1.model_jit import JiT

class Denoiser(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()

        self._args = args
        self.num_modalities = self._cfg("num_modalities")
        self.modality_input_dims = list(self._cfg("jepa_input_dims"))
        self.ts_seq_len = self._cfg("ts_seq_len")
        self.delay_embedder = DelayEmbedder(
            device=self._cfg_optional("device", "cpu"),
            seq_len=self.ts_seq_len,
            delay=self._cfg("ts_delay"),
            embedding=self._cfg("ts_embedding"),
        )

        self.jepa = MultimodalTSJEPA(
            input_dims=self.modality_input_dims,
            seq_len=self.ts_seq_len,
            hidden_size=self._cfg("jepa_hidden_size"),
            encoder_layers=self._cfg("jepa_encoder_layers"),
            embed_dim=self._cfg("jepa_embed_dim"),
            latent_downsample=self._cfg("jepa_latent_downsample"),
            encoder_dropout=self._cfg("jepa_encoder_dropout"),
            predictor_dim=self._cfg("jepa_predictor_dim"),
            predictor_layers=self._cfg("jepa_predictor_layers"),
            predictor_heads=self._cfg("jepa_predictor_heads"),
            predictor_mlp_ratio=self._cfg("jepa_predictor_mlp_ratio"),
            predictor_dropout=self._cfg("jepa_predictor_dropout"),
            ema_momentum=self._cfg("jepa_ema_momentum"),
            max_len=self._cfg("jepa_max_len"),
        )

        self.jepa_cond_proj = (
            nn.Identity()
            if self.jepa.embed_dim == self._cfg("hidden_channels")
            else nn.Linear(self.jepa.embed_dim, self._cfg("hidden_channels"))
        )

        self.net = JiT(
            input_size=self._cfg("img_size"),
            patch_size=self._cfg("patch_size"),
            in_channels=self._cfg("in_channels"),
            num_classes=self.num_modalities,
            hidden_size=self._cfg("hidden_channels"),
            depth=self._cfg("depth"),
            num_heads=self._cfg("num_heads"),
            mlp_ratio=4.0,
            attn_drop=self._cfg("attn_dropout"),
            proj_drop=self._cfg("proj_dropout"),
            bottleneck_dim=self._cfg("bottleneck_dim"),
            in_context_len=self.jepa.latent_seq_len,
            in_context_start=self._cfg("in_context_start"),
            num_modalities=self.num_modalities,
        )
        self.img_size = self._cfg("img_size")
        self.P_mean = self._cfg("P_mean")
        self.P_std = self._cfg("P_std")
        self.t_eps = self._cfg("t_eps")
        self.noise_scale = self._cfg("noise_scale")

        # ema
        self.ema_decay1 = self._cfg("ema_decay1")
        self.ema_decay2 = self._cfg("ema_decay2")
        self.ema_params1 = None
        self.ema_params2 = None

        # generation hyper params
        self.method = self._cfg("sampling_method")
        self.steps = self._cfg("num_sampling_steps")
        self.cfg_scale = self._cfg("cfg")
        self.cfg_interval = (self._cfg("interval_min"), self._cfg("interval_max"))

    def _cfg(self, key: str):
        if isinstance(self._args, dict):
            return self._args[key]
        return getattr(self._args, key)

    def _cfg_optional(self, key: str, default):
        if isinstance(self._args, dict):
            return self._args.get(key, default)
        return getattr(self._args, key, default)

    def _to_jepa_input(self, x: torch.Tensor, modality_idx: int) -> torch.Tensor:
        expected_dim = self.modality_input_dims[modality_idx]
        if x.ndim != 3:
            raise ValueError(f"JEPA modality input must be 3D, got {tuple(x.shape)}.")
        if x.shape[1] == expected_dim:
            out = x
        elif x.shape[-1] == expected_dim:
            out = x.transpose(1, 2)
        else:
            raise ValueError(
                f"Modality {modality_idx} expected channel dim {expected_dim}; got {tuple(x.shape)}."
            )
        if out.shape[-1] != self.ts_seq_len:
            raise ValueError(
                f"Modality {modality_idx} expected seq_len={self.ts_seq_len}; got {out.shape[-1]}."
            )
        return out

    def _jepa_modalities(self, modalities: Sequence[torch.Tensor]):
        if len(modalities) != self.num_modalities:
            raise ValueError(f"Expected {self.num_modalities} modalities, got {len(modalities)}.")
        return [self._to_jepa_input(x.float(), idx) for idx, x in enumerate(modalities)]

    def _select_diffusion_target(
        self,
        normalized_modalities: Sequence[torch.Tensor],
        missing_modality: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = normalized_modalities[0].shape[0]
        pieces = [
            normalized_modalities[int(modality_id)][sample_idx]
            for sample_idx, modality_id in enumerate(missing_modality.tolist())
        ]
        target = torch.stack(pieces, dim=0)

        if target.shape[1] != self.net.in_channels:
            raise ValueError(
                f"Selected modality channel dim {target.shape[1]} does not match JiT in_channels "
                f"{self.net.in_channels}."
            )

        self.delay_embedder.device = target.device
        target_img, img_mask = self.delay_embedder.ts_to_img(target.transpose(1, 2), return_pad_mask=True)
        if target_img.shape[-2:] != (self.img_size, self.img_size):
            raise ValueError(
                f"DelayEmbedder produced image shape {tuple(target_img.shape[-2:])}, "
                f"but JiT expects {(self.img_size, self.img_size)}. "
                "Adjust img_size, ts_delay, or ts_embedding."
            )
        return target_img, img_mask

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(
        self,
        modalities: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        jepa_modalities = self._jepa_modalities(modalities)
        jepa_out = self.jepa.forward(jepa_modalities)
        cond_tokens = self.jepa_cond_proj(jepa_out["predicted_missing_tokens"])
        missing_modality = jepa_out["missing_modality_ids"]
        x, x_mask = self._select_diffusion_target(jepa_modalities, missing_modality)

        bsz = x.size(0)
        if cond_tokens.shape[0] != bsz:
            raise ValueError(
                f"Image batch size {bsz} must match modality batch size {cond_tokens.shape[0]}."
            )

        t = self.sample_t(bsz, device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale

        z = x_mask * (t * x + (1 - t) * e)
        v = x_mask * (x - z) / (1 - t).clamp_min(self.t_eps)

        labels = missing_modality
        x_pred = x_mask * self.net.forward(
            z,
            t.flatten(),
            labels,
            cond_tokens,
            missing_modality,
        )
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)

        diffusion_loss = (v - v_pred) ** 2
        diffusion_loss = (diffusion_loss.sum(dim=(1,2,3)) / x_mask.sum(dim=(1,2,3))).mean()
        return diffusion_loss + jepa_out["loss"], diffusion_loss.item(), jepa_out["loss"].item()

    @torch.no_grad()
    def _encode_conditions(
        self,
        modalities: Sequence[torch.Tensor],
        missing_modalities: Union[int, Sequence[int], torch.Tensor],
    ):
        jepa_out = self.jepa.forward(
            self._jepa_modalities(modalities),
            missing_modalities=missing_modalities,
        )
        cond_tokens = self.jepa_cond_proj(jepa_out["predicted_missing_tokens"])
        return cond_tokens, jepa_out["missing_modality_ids"]

    @torch.no_grad()
    def generate(
        self,
        modalities: Sequence[torch.Tensor],
        missing_modalities: Union[int, Sequence[int], torch.Tensor],
    ):
        """Generate samples with the same multimodal-conditioning schema as forward()."""
        cond_tokens, missing_modality = self._encode_conditions(
            modalities=modalities,
            missing_modalities=missing_modalities,
        )
        device = cond_tokens.device
        bsz = cond_tokens.size(0)
        labels = missing_modality
        cond_tokens = cond_tokens.to(device)
        missing_modality = missing_modality.to(device)

        z = self.noise_scale * torch.randn(
            bsz,
            self.net.in_channels,
            self.img_size,
            self.img_size,
            device=device,
        )
        timesteps = torch.linspace(0.0, 1.0, self.steps + 1, device=device)

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        # ode
        for i in range(self.steps - 1):
            t = timesteps[i].view(1, 1, 1, 1).expand(bsz, 1, 1, 1)
            t_next = timesteps[i + 1].view(1, 1, 1, 1).expand(bsz, 1, 1, 1)
            z = stepper(z, t, t_next, labels, cond_tokens, missing_modality)

        # last step euler
        t_last = timesteps[-2].view(1, 1, 1, 1).expand(bsz, 1, 1, 1)
        t_end = timesteps[-1].view(1, 1, 1, 1).expand(bsz, 1, 1, 1)
        z = self._euler_step(z, t_last, t_end, labels, cond_tokens, missing_modality)
        return z

    @torch.no_grad()
    def _forward_sample(self, z, t, labels, cond_tokens, missing_modality):
        x_cond = self.net(z, t.flatten(), labels, cond_tokens, missing_modality)
        return (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, labels, cond_tokens, missing_modality):
        v_pred = self._forward_sample(z, t, labels, cond_tokens, missing_modality)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, labels, cond_tokens, missing_modality):
        v_pred_t = self._forward_sample(z, t, labels, cond_tokens, missing_modality)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, labels, cond_tokens, missing_modality)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def update_ema(self):
        self.jepa.update_target_encoders()
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)
