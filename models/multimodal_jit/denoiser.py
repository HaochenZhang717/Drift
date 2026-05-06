import torch
import torch.nn as nn
from models.multimodal_jit.model_jit import JiT
from models.multi_modal_encoder import MultiModalEncoder

class Denoiser(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()

        self.mm_encoder = MultiModalEncoder(
            n_tokens=args.num_tokens_per_modality,
            dim_in=args.mm_dim_in,
            dim_out=args.hidden_channels,
            nhead_cross=args.mm_n_heads,
            ae_input_dim=args.ae_input_dim,
            ae_d_model=args.ae_d_model,
            ae_nhead=args.ae_nheads,
            ae_num_layers= args.ae_num_layers,
            ckpt_paths=args.ae_cpt_paths,
            strict_load=True
        )

        self.net = JiT(
            input_size=args.img_size,
            patch_size=args.patch_size,
            in_channels=args.in_channels,
            hidden_size=args.hidden_channels,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=4.0,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
            bottleneck_dim=args.bottleneck_dim,
            in_context_len=args.num_tokens_per_modality * 4,
            in_context_start=args.in_context_start,
        )
        self.img_size = args.img_size

        self.label_drop_prob = args.label_drop_prob
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale

        # ema
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None

        # generation hyper params
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.cfg_scale = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)

    def drop_labels(self, labels):
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        out = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
        return out

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(
            self,
            x: torch.Tensor,
            study_group: torch.Tensor,
            heart_rate: torch.Tensor,
            calorie: torch.Tensor,
            physical_activity: torch.Tensor,
            respiratory_rate: torch.Tensor,
            heart_rate_observed_mask: torch.Tensor,
            calorie_observed_mask: torch.Tensor,
            physical_activity_observed_mask: torch.Tensor,
            respiratory_rate_observed_mask: torch.Tensor,

        ):

        cond_tokens = self.mm_encoder.forward(
            heart_rate=heart_rate,
            calorie=calorie,
            physical_activity=physical_activity,
            respiratory_rate=respiratory_rate,
            heart_rate_observed_mask=heart_rate_observed_mask,
            calorie_observed_mask=calorie_observed_mask,
            physical_activity_observed_mask=physical_activity_observed_mask,
            respiratory_rate_observed_mask=respiratory_rate_observed_mask,
        )

        # labels_dropped = self.drop_labels(labels) if self.training else labels

        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale

        z = t * x + (1 - t) * e
        v = (x - z) / (1 - t).clamp_min(self.t_eps)

        x_pred = self.net.forward(z, t.flatten(), study_group, cond_tokens)
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)

        # l2 loss
        loss = (v - v_pred) ** 2
        loss = loss.mean(dim=(1, 2, 3)).mean()

        return loss

    @torch.no_grad()
    def generate(self, labels):
        device = labels.device
        bsz = labels.size(0)
        z = self.noise_scale * torch.randn(bsz, 3, self.img_size, self.img_size, device=device)
        timesteps = torch.linspace(0.0, 1.0, self.steps+1, device=device).view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        # ode
        for i in range(self.steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = stepper(z, t, t_next, labels)
        # last step euler
        z = self._euler_step(z, timesteps[-2], timesteps[-1], labels)
        return z

    @torch.no_grad()
    def _forward_sample(self, z, t, labels):
        # conditional
        x_cond = self.net(z, t.flatten(), labels)
        v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)

        # unconditional
        x_uncond = self.net(z, t.flatten(), torch.full_like(labels, self.num_classes))
        v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(self.t_eps)

        # cfg interval
        low, high = self.cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)

        return v_uncond + cfg_scale_interval * (v_cond - v_uncond)

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, labels):
        v_pred = self._forward_sample(z, t, labels)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, labels):
        v_pred_t = self._forward_sample(z, t, labels)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, labels)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)
