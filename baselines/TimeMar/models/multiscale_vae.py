import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .decomp_vae import Decoder, Encoder,GuidedDecoder, GuidedEncoder
from .quant import VectorQuantizer2


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1 - math.floor((self.kernel_size - 1) // 2), 1)
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp_multi(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]
        self.layer = torch.nn.Linear(1, len(kernel_size))

    def forward(self, x):
        moving_mean = []
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg.unsqueeze(-1))
        moving_mean = torch.cat(moving_mean, dim=-1)
        moving_mean = torch.sum(moving_mean * nn.Softmax(-1)(self.layer(x.unsqueeze(-1))), dim=-1)
        res = x - moving_mean
        return res, moving_mean


class SymmetricFusion(nn.Module):
    def __init__(self, z_channels):
        super().__init__()
        self.fc_fusion = nn.Linear(z_channels * 3, z_channels)

    def forward(self, z1, z2, z3):
        #  [B, C, L] -> [B, 3C, L] -> [B, C, L]
        fused = torch.cat([z1, z2, z3], dim=1)
        return self.fc_fusion(fused.permute(0, 2, 1)).permute(0, 2, 1)


class SymmetricDecomp(nn.Module):
    def __init__(self, z_channels):
        super().__init__()
        self.fc_decomp = nn.Linear(z_channels, z_channels * 3)

    def forward(self, f):
        #  [B, C, L] -> [B, 3C, L]
        decomp = self.fc_decomp(f.permute(0, 2, 1)).permute(0, 2, 1)
        return torch.chunk(decomp, 3, dim=1)



class MultiScaleVAE(nn.Module):
    def __init__(
            self,
            in_channels,
            vocab_size=512,
            z_channels=32,
            ch=128,
            dropout=0.0,
            beta=0.25,
            using_znorm=False,
            quant_conv_ks=3,
            quant_resi=0.5,
            share_quant_resi=0,
            default_qresi_counts=0,
            v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
            test_mode=True,
            ch_mult=(1, 1, 2),
    ):
        super().__init__()
        self.test_mode = test_mode
        self.V, self.Cvae = vocab_size, z_channels
        self.vocab_size = vocab_size
        self.decomp_ts = series_decomp_multi([5, 7])

        # /
        coarse_config = dict(
            dropout=dropout,
            ch=ch,
            z_channels=z_channels,
            in_channels=in_channels,
            ch_mult=ch_mult,
            num_res_blocks=1,
            using_sa=False,
            using_mid_sa=False,
        )
        fine_config = dict(
            dropout=dropout,
            ch=ch,
            z_channels=z_channels,
            in_channels=in_channels,
            ch_mult=ch_mult,
            num_res_blocks=1,
            using_sa=False,
            using_mid_sa=False,
        )

        self.encoder_trend = Encoder(**coarse_config)
        self.decoder_trend = Decoder(**coarse_config)
        self.quant_conv_trend = nn.Conv1d(z_channels, z_channels, quant_conv_ks, padding=quant_conv_ks // 2)
        self.post_quant_conv_trend = nn.Conv1d(z_channels, z_channels, quant_conv_ks, padding=quant_conv_ks // 2)


        self.encoder_seasonal = GuidedEncoder(**fine_config)
        self.decoder_seasonal = GuidedDecoder(**fine_config)

        self.quant_conv_seasonal = nn.Conv1d(z_channels, z_channels, quant_conv_ks, padding=quant_conv_ks // 2)
        self.post_quant_conv_seasonal = nn.Conv1d(z_channels, z_channels, quant_conv_ks, padding=quant_conv_ks // 2)


        self.downsample = nn.AvgPool1d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)

        if test_mode:
            self.eval()
            [p.requires_grad_(False) for p in self.parameters()]

    def forward(self, inp, ret_usages=True):
        seasonal, trend = self.decomp_ts(inp)

        coarse_seasonal_down = self.downsample(seasonal.transpose(1, 2))
        coarse_seasonal = self.upsample(coarse_seasonal_down).transpose(1, 2)

        h_trend = self.encoder_trend(trend)
        ze_trend = self.quant_conv_trend(h_trend)

        h_coarse_seasonal = self.encoder_trend(coarse_seasonal)
        ze_coarse_seasonal = self.quant_conv_trend(h_coarse_seasonal)

        h_seasonal = self.encoder_seasonal(seasonal)
        ze_seasonal = self.quant_conv_seasonal(h_seasonal)



        # zq_trend, zq_seasonal, zq_coarse_seasonal = self.decomp_latent(f_hat)

        recon_trend = self.decoder_trend(self.post_quant_conv_trend(zq_trend))
        recon_coarse_seasonal = self.decoder_trend(self.post_quant_conv_trend(zq_coarse_seasonal))
        recon_seasonal = self.decoder_seasonal(self.post_quant_conv_seasonal(zq_seasonal), recon_coarse_seasonal)

        total_recon = recon_trend + recon_seasonal

        # stop_gradient
        def stop_gradient(x):
            return x.detach() + (x - x.detach())  #

        consistency_loss = 0
        consistency_loss += F.mse_loss(zq_trend, stop_gradient(ze_trend))
        consistency_loss += F.mse_loss(zq_seasonal, stop_gradient(ze_seasonal))
        consistency_loss += F.mse_loss(zq_coarse_seasonal, stop_gradient(ze_coarse_seasonal))

        vq_loss = vq_loss + 0.1 * consistency_loss  #

        return trend, seasonal, coarse_seasonal, recon_trend, recon_seasonal, recon_coarse_seasonal, total_recon, usages, vq_loss

    def decomp_fhat(self, fhat):
        zq_trend, zq_seasonal, zq_coarse_seasonal = self.decomp_latent(fhat)
        return zq_trend, zq_seasonal, zq_coarse_seasonal

    def fhat_to_ts(self, f_hat: torch.Tensor):
        """Convert quantized latent to TS."""
        zq_trend, zq_seasonal, zq_coarse_seasonal = self.decomp_latent(f_hat)
        # zq_trend, zq_seasonal, zq_coarse_seasonal = f_hat

        recon_trend = self.decoder_trend(self.post_quant_conv_trend(zq_trend))
        recon_coarse_seasonal = self.decoder_trend(self.post_quant_conv_trend(zq_coarse_seasonal))
        recon_seasonal = self.decoder_seasonal(self.post_quant_conv_seasonal(zq_seasonal), recon_coarse_seasonal)

        return (recon_trend + recon_seasonal)

    def fhat_to_ts_decomp(self, f_hat: torch.Tensor):
        """Convert quantized latent to TS."""
        zq_trend, zq_seasonal, zq_coarse_seasonal = self.decomp_latent(f_hat)
        recon_trend = self.decoder_trend(self.post_quant_conv_trend(zq_trend))
        recon_coarse_seasonal = self.decoder_trend(self.post_quant_conv_trend(zq_coarse_seasonal))
        recon_seasonal = self.decoder_seasonal(self.post_quant_conv_seasonal(zq_seasonal), recon_coarse_seasonal)

        return recon_trend, recon_coarse_seasonal, recon_seasonal

    # def load_state_dict(self, state_dict, strict=True, assign=False):
    #     self.quantize_trend.load_state_dict(state_dict['quantize_trend'], strict)
    #     self.quantize_seasonal.load_state_dict(state_dict['quantize_seasonal'], strict)
    #     super().load_state_dict(state_dict, strict, assign)


if __name__ == '__main__':
    model = MultiScaleVAE(in_channels=6)
    input_ts = torch.randn(2, 128, 6)
    model(input_ts)