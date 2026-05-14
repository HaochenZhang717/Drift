import torch
import torch.nn as nn

from .model_jit_t1 import JITT1, JITT1Config


class Denoiser(nn.Module):
    """
    Unconditional JiT formulation with T1 backbone.
    """
    def __init__(self, args):
        super().__init__()
        cfg = JITT1Config(
            seq_len=args.seq_len,
            enc_in=args.enc_in,
            n_heads=args.n_heads,
            patch_size=args.patch_size,
            patch_stride=args.patch_stride,
            n_blocks=tuple(args.n_blocks),
            kernel_size_large=tuple(args.kernel_size_large),
            kernel_size_small=args.kernel_size_small,
            ffn_ratio=args.ffn_ratio,
            downsample_ratio=args.downsample_ratio,
            qkv_bias=args.qkv_bias,
            drop_attn=args.drop_attn,
            drop_ffn=args.drop_ffn,
            drop_proj=args.drop_proj,
            drop_head=args.drop_head,
            positional_encoding=args.positional_encoding,
            out_len=args.seq_len,
        )
        self.net = JITT1(cfg)

        self.seq_len = args.seq_len
        self.enc_in = args.enc_in

        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale

        self.method = args.sampling_method
        self.steps = args.num_sampling_steps

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, x: torch.Tensor):
        """
        x: [B, T, M]
        """
        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale

        z = t * x + (1 - t) * e
        v = (x - z) / (1 - t).clamp_min(self.t_eps)

        x_pred = self.net(z, t.flatten())
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)

        loss = (v - v_pred) ** 2
        loss = loss.mean()
        return loss

    @torch.no_grad()
    def generate(self, batch_size: int, device=None):
        if device is None:
            device = next(self.parameters()).device
        z = self.noise_scale * torch.randn(batch_size, self.seq_len, self.enc_in, device=device)
        timesteps = torch.linspace(0.0, 1.0, self.steps + 1, device=device)

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        for i in range(self.steps - 1):
            t = timesteps[i].view(1, 1, 1).expand(batch_size, 1, 1)
            t_next = timesteps[i + 1].view(1, 1, 1).expand(batch_size, 1, 1)
            z = stepper(z, t, t_next)

        t_last = timesteps[-2].view(1, 1, 1).expand(batch_size, 1, 1)
        t_end = timesteps[-1].view(1, 1, 1).expand(batch_size, 1, 1)
        z = self._euler_step(z, t_last, t_end)
        return z

    @torch.no_grad()
    def _forward_sample(self, z, t):
        x_pred = self.net(z, t.flatten())
        v_pred = (x_pred - z) / (1.0 - t).clamp_min(self.t_eps)
        return v_pred

    @torch.no_grad()
    def _euler_step(self, z, t, t_next):
        v_pred = self._forward_sample(z, t)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next):
        v_pred_t = self._forward_sample(z, t)
        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next)
        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next

