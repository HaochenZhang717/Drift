from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import distributed as tdist, nn as nn
from torch.nn import functional as F

from . import dist

# this file only provides the VectorQuantizer2 used in VQVAE
__all__ = ['VectorQuantizer2', ]


class VectorQuantizer2(nn.Module):
    # VQGAN originally use beta=1.0, never tried 0.25; SD seems using 0.25
    def __init__(
            self, vocab_size, Cvae, using_znorm, beta: float = 0.25,
            default_qresi_counts=0, v_patch_nums=None, quant_resi=0.5, share_quant_resi=4,  # share_quant_resi: args.qsr
    ):
        super().__init__()
        self.vocab_size: int = vocab_size
        self.Cvae: int = Cvae
        self.using_znorm: bool = using_znorm
        self.v_patch_nums: Tuple[int] = v_patch_nums

        self.quant_resi_ratio = quant_resi
        if share_quant_resi == 0:  # non-shared: \phi_{1 to K} for K scales
            self.quant_resi = PhiNonShared(
                [(Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()) for _ in
                 range(default_qresi_counts or len(self.v_patch_nums))])
        elif share_quant_resi == 1:  # fully shared: only a single \phi for K scales
            self.quant_resi = PhiShared(Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity())
        else:  # partially shared: \phi_{1 to share_quant_resi} for K scales
            self.quant_resi = PhiPartiallyShared(nn.ModuleList(
                [(Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()) for _ in
                 range(share_quant_resi)]))

        self.register_buffer('ema_vocab_hit_SV', torch.full((len(self.v_patch_nums), self.vocab_size), fill_value=0.0))
        self.record_hit = 0

        self.beta: float = beta
        self.embedding = nn.Embedding(self.vocab_size, self.Cvae)

        # only used for progressive training of VAR (not supported yet, will be tested and supported in the future)
        self.prog_si = -1  # progressive training: not supported yet, prog_si always -1

    def eini(self, eini):
        if eini > 0:
            nn.init.trunc_normal_(self.embedding.weight.data, std=eini)
        elif eini < 0:
            self.embedding.weight.data.uniform_(-abs(eini) / self.vocab_size, abs(eini) / self.vocab_size)

    def extra_repr(self) -> str:
        return f'{self.v_patch_nums}, znorm={self.using_znorm}, beta={self.beta}  |  S={len(self.v_patch_nums)}, quant_resi={self.quant_resi_ratio}'

    # ===================== `forward` is only used in VAE training =====================
    def forward(self, f_BCL: torch.Tensor, ret_usages=True) -> Tuple[torch.Tensor, List[float], torch.Tensor]:
        dtype = f_BCL.dtype
        if dtype != torch.float32: f_BCL = f_BCL.float()
        # f_BCL=f_BCL.permute(0,2,1)
        B, C, L = f_BCL.shape
        # print("f_BCL", f_BCL.shape)
        f_no_grad = f_BCL.detach()
        # print("self.training:",self.training)
        # print("dist.initialized():", dist.initialized())

        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)

        # with torch.cuda.amp.autocast(enabled=False):
        with torch.amp.autocast(enabled=False, device_type='cuda'):
            mean_vq_loss: torch.Tensor = 0.0
            vocab_hit_V = torch.zeros(self.vocab_size, dtype=torch.float, device=f_BCL.device)
            SN = len(self.v_patch_nums)
            # print("v_patch_nums", self.v_patch_nums)
            for si, pl in enumerate(self.v_patch_nums):  # from small to large
                if self.using_znorm:
                    rest_NC = F.interpolate(f_rest, size=(pl,), mode='linear').permute(0, 2, 1).reshape(-1, C) if (
                            si != SN - 1) else f_rest.permute(0, 2, 1).reshape(-1, C)
                    rest_NC = F.normalize(rest_NC, dim=-1)
                    # print("rest_NC", rest_NC.shape)
                    idx_N = torch.argmax(rest_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
                    # print("idx_N", idx_N.shape)

                else:
                    rest_NC = F.interpolate(f_rest, size=pl, mode='linear').permute(0, 2, 1).reshape(-1, C) if (
                            si != SN - 1) else f_rest.permute(0, 2, 1).reshape(-1, C)
                    d_no_grad = torch.sum(rest_NC.square(), dim=1, keepdim=True) + torch.sum(
                        self.embedding.weight.data.square(), dim=1, keepdim=False)
                    d_no_grad.addmm_(rest_NC, self.embedding.weight.data.T, alpha=-2, beta=1)  # (B*L, vocab_size)
                    idx_N = torch.argmin(d_no_grad, dim=1)

                hit_V = idx_N.bincount(minlength=self.vocab_size).float()
                if self.training:
                    if dist.initialized():
                        handler = tdist.all_reduce(hit_V, async_op=True)

                # calc loss
                idx_Bl = idx_N.view(B, pl)
                h_BCl = F.interpolate(self.embedding(idx_Bl).permute(0, 2, 1), size=L,
                                      mode='linear').contiguous() if (si != SN - 1) else self.embedding(idx_Bl).permute(
                    0, 2, 1).contiguous()

                if SN > 1:
                    ratio = si / (SN - 1)
                else:
                    ratio = 0
                h_BCl = self.quant_resi[ratio](h_BCl)
                f_hat = f_hat + h_BCl
                f_rest -= h_BCl

                if self.training:
                    if dist.initialized():
                        handler.wait()
                    if self.record_hit == 0:
                        self.ema_vocab_hit_SV[si].copy_(hit_V)
                    elif self.record_hit < 100:
                        self.ema_vocab_hit_SV[si].mul_(0.9).add_(hit_V.mul(0.1))
                    else:
                        self.ema_vocab_hit_SV[si].mul_(0.99).add_(hit_V.mul(0.01))
                    # print(f"Scale {si}, Hit V sum: {hit_V.sum().item()}, EMA sum: {self.ema_vocab_hit_SV[si].sum().item()}")
                    self.record_hit += 1
                    # =====================  =====================
                    # # scaleEMA
                    # ema_si = self.ema_vocab_hit_SV[si]
                    # # token
                    # unused_ratio = (ema_si == 0).sum().item() / self.vocab_size * 100
                    # print(f"[Scale {si}] Unused Tokens: {unused_ratio:.1f}%")
                    # =====================  =====================

                vocab_hit_V.add_(hit_V)
                mean_vq_loss += F.mse_loss(f_hat.data, f_BCL).mul_(self.beta) + F.mse_loss(f_hat, f_no_grad)

            mean_vq_loss *= 1. / SN
            f_hat = (f_hat.data - f_no_grad).add_(f_BCL)

        # margin = tdist.get_world_size() * (f_BCL.numel() / f_BCL.shape[1]) / self.vocab_size * 0.08
        margin = (f_BCL.numel() / f_BCL.shape[1]) / self.vocab_size * 0.08
        # margin = pn*pn / 100
        if ret_usages: usages = [(self.ema_vocab_hit_SV[si] >= margin).float().mean().item() * 100 for si, pn in enumerate(self.v_patch_nums)]
        # if ret_usages: usages = [(self.ema_vocab_hit_SV[si] == 0).sum().item() / self.vocab_size * 100 for si, pn in enumerate(self.v_patch_nums)]
        else: usages = None
        # print(usages)
        return f_hat, usages, mean_vq_loss

    # ===================== `forward` is only used in VAE training =====================

    def embed_to_fhat(self, ms_h_BChw: List[torch.Tensor], all_to_max_scale=True, last_one=False) -> Union[
        List[torch.Tensor], torch.Tensor]:
        ls_f_hat_BChw = []
        B = ms_h_BChw[0].shape[0]
        L= self.v_patch_nums[-1]
        SN = len(self.v_patch_nums)
        if all_to_max_scale:
            f_hat = ms_h_BChw[0].new_zeros(B, self.Cvae, L, dtype=torch.float32)
            for si, pn in enumerate(self.v_patch_nums):  # from small to large
                h_BChw = ms_h_BChw[si]
                if si < len(self.v_patch_nums) - 1:
                    h_BChw = F.interpolate(h_BChw, size=(L), mode='linear')
                h_BChw = self.quant_resi[si / (SN - 1)](h_BChw)
                f_hat.add_(h_BChw)
                if last_one:
                    ls_f_hat_BChw = f_hat
                else:
                    ls_f_hat_BChw.append(f_hat.clone())
        else:
            # WARNING: this is not the case in VQ-VAE training or inference (we'll interpolate every token map to the max H W, like above)
            # WARNING: this should only be used for experimental purpose
            f_hat = ms_h_BChw[0].new_zeros(B, self.Cvae, self.v_patch_nums[0], self.v_patch_nums[0],
                                           dtype=torch.float32)
            for si, pn in enumerate(self.v_patch_nums):  # from small to large
                f_hat = F.interpolate(f_hat, size=(pn, pn), mode='bicubic')
                h_BChw = self.quant_resi[si / (SN - 1)](ms_h_BChw[si])
                f_hat.add_(h_BChw)
                if last_one:
                    ls_f_hat_BChw = f_hat
                else:
                    ls_f_hat_BChw.append(f_hat)

        # print("ls_f_hat_BChw",ls_f_hat_BChw.shape)
        return ls_f_hat_BChw

    def f_to_idxBl_or_fhat(self, f_BCl: torch.Tensor, to_fhat: bool,
                           v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None) -> List[
        Union[torch.Tensor, torch.LongTensor]]:  # z_BChw is the feature from inp_img_no_grad
        B, C, L = f_BCl.shape
        f_no_grad = f_BCl.detach()
        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)

        f_hat_or_idx_Bl: List[torch.Tensor] = []

        patch_lens = [pn if isinstance(pn, int) else pn[0] for pn in
                      (v_patch_nums or self.v_patch_nums)]  # from small to large
        assert patch_lens[-1] == L, f'{patch_lens[-1]=} != ({L=})'

        SN = len(patch_lens)
        for si, pl in enumerate(patch_lens):  # from small to large
            if 0 <= self.prog_si < si: break  # progressive training: not supported yet, prog_si always -1
            # find the nearest embedding
            z_NC = F.interpolate(f_rest, size=(pl,), mode='linear').permute(0, 2, 1).reshape(-1, C) if (
                    si != SN - 1) else f_rest.permute(0, 2, 1).reshape(-1, C)
            if self.using_znorm:
                z_NC = F.normalize(z_NC, dim=-1)
                idx_N = torch.argmax(z_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
            else:
                d_no_grad = torch.sum(z_NC.square(), dim=1, keepdim=True) + torch.sum(
                    self.embedding.weight.data.square(), dim=1, keepdim=False)
                d_no_grad.addmm_(z_NC, self.embedding.weight.data.T, alpha=-2, beta=1)  # (B*L, vocab_size)
                idx_N = torch.argmin(d_no_grad, dim=1)

            # Ensure idx_N has correct size before reshaping
            if idx_N.size(0) != B * pl:
                raise ValueError(f"Unexpected idx_N size: {idx_N.size()}, expected {(B * pl,)}")

            idx_Bl = idx_N.view(B, pl)
            h_BCl = F.interpolate(self.embedding(idx_Bl).permute(0, 2, 1), size=(L,), mode='linear').contiguous() if (
                        si != SN - 1) else self.embedding(idx_Bl).permute(0, 2, 1).contiguous()

            if SN > 1:
                ratio = si / (SN - 1)
            else:
                ratio = 0
            h_BCl = self.quant_resi[ratio](h_BCl)
            f_hat.add_(h_BCl)
            f_rest.sub_(h_BCl)
            f_hat_or_idx_Bl.append(f_hat.clone() if to_fhat else idx_N.reshape(B, pl))

        return f_hat_or_idx_Bl

    # ===================== idxBl_to_var_input: only used in VAR training, for getting teacher-forcing input =====================
    def idxBl_to_var_input(self, gt_ms_idx_Bl: List[torch.Tensor]) -> torch.Tensor:
        next_scales = []
        B = gt_ms_idx_Bl[0].shape[0]
        C = self.Cvae
        L= self.v_patch_nums[-1]
        SN = len(self.v_patch_nums)

        f_hat = gt_ms_idx_Bl[0].new_zeros(B, C,L, dtype=torch.float32)
        pn_next: int = self.v_patch_nums[0]
        for si in range(SN - 1):
            if self.prog_si == 0 or (
                    0 <= self.prog_si - 1 < si): break  # progressive training: not supported yet, prog_si always -1
            h_BCL = F.interpolate(self.embedding(gt_ms_idx_Bl[si]).transpose_(1, 2).view(B, C, pn_next),
                                   size=L, mode='linear')
            f_hat.add_(self.quant_resi[si / (SN - 1)](h_BCL))
            pn_next = self.v_patch_nums[si + 1]
            next_scales.append(
                F.interpolate(f_hat, size=pn_next,  mode='linear').view(B, C, -1).transpose(1, 2))
        return torch.cat(next_scales, dim=1) if len(next_scales) else None  # cat BlCs to BLC, this should be float32

    # ===================== get_next_autoregressive_input: only used in VAR inference, for getting next step's input =====================
    def get_next_autoregressive_input(self, si: int, SN: int, f_hat: torch.Tensor, h_BCL: torch.Tensor) -> Tuple[
        Optional[torch.Tensor], torch.Tensor]:  # only used in VAR inference
        L = self.v_patch_nums[-1]
        if si != SN - 1:
            h = self.quant_resi[si / (SN - 1)](
                F.interpolate(h_BCL, size=L, mode='linear'))  # conv after upsample
            # print(f"f_hat shape: {f_hat.shape}, h shape: {h.shape}")
            f_hat.add_(h)
            return f_hat, F.interpolate(f_hat, size=(self.v_patch_nums[si + 1]), mode='linear')
        else:
            # print("h_BCL",h_BCL.shape)

            if SN > 1:
                ratio = si / (SN - 1)
            else:
                ratio = 0
            h = self.quant_resi[ratio](h_BCL)
            # print("h",h.shape)
            # print("f_hat",f_hat.shape)

            f_hat.add_(h)
            return f_hat, f_hat


class Phi(nn.Conv1d):
    def __init__(self, embed_dim, quant_resi):
        ks = 3
        super().__init__(in_channels=embed_dim, out_channels=embed_dim, kernel_size=ks, stride=1, padding=ks // 2)
        self.resi_ratio = abs(quant_resi)

    def forward(self, h_BCl):
        return h_BCl.mul(1 - self.resi_ratio) + super().forward(h_BCl).mul_(self.resi_ratio)


class PhiShared(nn.Module):
    def __init__(self, qresi: Phi):
        super().__init__()
        self.qresi: Phi = qresi

    def __getitem__(self, _) -> Phi:
        return self.qresi


class PhiPartiallyShared(nn.Module):
    def __init__(self, qresi_ls: nn.ModuleList):
        super().__init__()
        self.qresi_ls = qresi_ls
        K = len(qresi_ls)
        self.ticks = np.linspace(1 / 3 / K, 1 - 1 / 3 / K, K) if K == 4 else np.linspace(1 / 2 / K, 1 - 1 / 2 / K, K)

    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return self.qresi_ls[np.argmin(np.abs(self.ticks - at_from_0_to_1)).item()]

    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'


class PhiNonShared(nn.ModuleList):
    def __init__(self, qresi: List):
        super().__init__(qresi)
        # self.qresi = qresi
        K = len(qresi)
        self.ticks = np.linspace(1 / 3 / K, 1 - 1 / 3 / K, K) if K == 4 else np.linspace(1 / 2 / K, 1 - 1 / 2 / K, K)

    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return super().__getitem__(np.argmin(np.abs(self.ticks - at_from_0_to_1)).item())

    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'