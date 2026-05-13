import torch
from types import SimpleNamespace

from JITT1.model_jit_t1 import JITT1, JITT1Config
from JITT1.denoiser import Denoiser


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_backbone_shapes():
    # Official T1 main setting used by most scripts (ETT/Weather/ECL/Exchange/ILI)
    B, T, M = 4, 96, 7
    cfg = JITT1Config(
        seq_len=T,
        enc_in=M,
        n_heads=128,
        patch_size=2,
        patch_stride=1,
        n_blocks=(2, 2),
        kernel_size_large=(71, 31),
        kernel_size_small=5,
        ffn_ratio=1.0,
        downsample_ratio=2,
        positional_encoding=True,
    )
    model = JITT1(cfg)
    n_params = count_trainable_params(model)
    x = torch.randn(B, T, M)
    t = torch.rand(B)
    y = model(x, t)
    print("[JITT1 Backbone]")
    print(f"trainable params: {n_params:,}")
    print(f"input x: {tuple(x.shape)}")
    print(f"input t: {tuple(t.shape)}")
    print(f"output y: {tuple(y.shape)}")


def test_denoiser_shapes():
    B, T, M = 4, 96, 7
    args = SimpleNamespace(
        seq_len=T,
        enc_in=M,
        n_heads=128,
        patch_size=2,
        patch_stride=1,
        n_blocks=[2, 2, 2, 2],
        kernel_size_large=[71, 71, 31, 31],
        kernel_size_small=5,
        ffn_ratio=1.0,
        downsample_ratio=1,
        qkv_bias=True,
        drop_attn=0.1,
        drop_ffn=0.0,
        drop_proj=0.0,
        drop_head=0.0,
        positional_encoding=True,
        P_mean=0.0,
        P_std=1.0,
        t_eps=1e-5,
        noise_scale=1.0,
        sampling_method="euler",
        num_sampling_steps=8,
    )
    model = Denoiser(args)
    n_params = count_trainable_params(model)

    x = torch.randn(B, T, M)
    loss = model(x)
    samples = model.generate(batch_size=B, device=x.device)

    print("\n[JITT1 Denoiser]")
    print(f"trainable params: {n_params:,}")
    print(f"forward input x: {tuple(x.shape)}")
    print(f"forward output loss: {tuple(loss.shape)} (scalar tensor)")
    print(f"generate output samples: {tuple(samples.shape)}")


if __name__ == "__main__":
    test_backbone_shapes()
    test_denoiser_shapes()
