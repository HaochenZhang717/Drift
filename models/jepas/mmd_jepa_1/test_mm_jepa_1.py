import sys
import os
from pathlib import Path
from types import SimpleNamespace

import torch

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.jepas.mmd_jepa_1.denoiser import Denoiser
from models.jepas.mmd_jepa_1.mm_jepa_1 import MultimodalTSJEPA


def make_model(input_dims=(1, 1, 1, 1), seq_len=128):
    return MultimodalTSJEPA(
        input_dims=input_dims,
        seq_len=seq_len,
        hidden_size=32,
        encoder_layers=1,
        embed_dim=16,
        latent_downsample=8,
        predictor_layers=1,
        predictor_heads=4,
        predictor_dim=32,
        predictor_dropout=0.0,
        encoder_dropout=0.0,
        ema_momentum=0.99,
    )


def make_inputs(batch_size=4, input_dims=(1, 1, 1, 1), seq_len=128):
    return [torch.randn(batch_size, dim, seq_len) for dim in input_dims]


def make_denoiser_args():
    return SimpleNamespace(
        num_modalities=4,
        jepa_input_dims=[1, 1, 1, 1],
        ts_seq_len=64,
        jepa_hidden_size=16,
        jepa_encoder_layers=1,
        jepa_embed_dim=16,
        jepa_latent_downsample=8,
        jepa_encoder_dropout=0.0,
        jepa_predictor_dim=16,
        jepa_predictor_layers=1,
        jepa_predictor_heads=4,
        jepa_predictor_mlp_ratio=4.0,
        jepa_predictor_dropout=0.0,
        jepa_ema_momentum=0.99,
        jepa_max_len=1024,
        hidden_channels=16,
        mm_dim_in=16,
        mm_n_heads=4,
        ae_input_dim=1,
        ae_d_model=16,
        ae_num_layers=1,
        img_size=8,
        patch_size=2,
        in_channels=1,
        depth=1,
        num_heads=4,
        attn_dropout=0.0,
        proj_dropout=0.0,
        bottleneck_dim=8,
        in_context_start=0,
        label_drop_prob=0.0,
        P_mean=-1.2,
        P_std=1.2,
        t_eps=1e-5,
        noise_scale=1.0,
        ema_decay1=0.999,
        ema_decay2=0.9999,
        sampling_method="euler",
        num_sampling_steps=2,
        cfg=1.0,
        interval_min=0.0,
        interval_max=1.0,
        ts_delay=8,
        ts_embedding=8,
    )


def make_denoiser_batch(batch_size=2, seq_len=64):
    modalities = [torch.randn(batch_size, seq_len, 1) for _ in range(4)]
    return {
        "modalities": modalities,
    }


def test_forward_with_one_missing_modality():
    torch.manual_seed(0)
    model = make_model()
    xs = make_inputs()

    out = model(xs)

    assert out["loss"].ndim == 0
    assert out["predictions"].shape == (4, 4, 16, 16)
    assert out["targets"].shape == (4, 4, 16, 16)
    assert out["online_embeddings"].shape == (4, 4, 16, 16)
    assert out["missing_mask"].shape == (4, 4)
    assert out["missing_mask"].sum(dim=1).eq(1).all()
    assert out["missing_modality_ids"].shape == (4,)
    assert out["predicted_missing_tokens"].shape == (4, 16, 16)


def test_backward_and_ema_update():
    torch.manual_seed(1)
    model = make_model()
    xs = make_inputs()

    target_before = next(model.target_encoders[0].parameters()).detach().clone()
    out = model(xs)
    out["loss"].backward()

    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.encoders.parameters()
    )
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.predictor.parameters()
    )
    assert all(p.grad is None for p in model.target_encoders.parameters())

    with torch.no_grad():
        for p in model.encoders[0].parameters():
            p.add_(0.01)

    model.update_target_encoders()
    target_after = next(model.target_encoders[0].parameters()).detach()
    assert not torch.allclose(target_before, target_after)


def test_randomly_masks_one_modality_per_sample():
    torch.manual_seed(2)
    model = make_model()
    xs = make_inputs()

    out = model(xs)

    assert out["missing_mask"].sum(dim=1).eq(1).all()
    assert torch.equal(
        out["missing_mask"].long().argmax(dim=1),
        out["missing_modality_ids"],
    )
    assert out["predicted_missing"].shape == (4 * model.latent_seq_len, model.embed_dim)
    assert out["target_missing"].shape == (4 * model.latent_seq_len, model.embed_dim)
    assert torch.isfinite(out["loss"])


def test_heterogeneous_modality_channels():
    torch.manual_seed(3)
    input_dims = (1, 2, 3)
    model = make_model(input_dims=input_dims, seq_len=96)
    xs = make_inputs(batch_size=2, input_dims=input_dims, seq_len=96)

    out = model(xs)

    assert out["predictions"].shape == (2, 3, model.latent_seq_len, model.embed_dim)
    assert out["missing_mask"].shape == (2, 3)
    assert out["missing_mask"].sum(dim=1).eq(1).all()
    assert torch.isfinite(out["loss"])


def test_denoiser_overall_forward_backward():
    torch.manual_seed(4)
    model = Denoiser(make_denoiser_args())
    batch = make_denoiser_batch()

    loss, diffusion_loss, jepa_loss = model(**batch)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert isinstance(diffusion_loss, float)
    assert isinstance(jepa_loss, float)

    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.jepa.encoders.parameters()
    )
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.net.parameters()
    )


def test_denoiser_generate_with_explicit_missing_modalities():
    torch.manual_seed(5)
    model = Denoiser(make_denoiser_args())
    batch = make_denoiser_batch()
    missing_modalities = torch.tensor([0, 2])

    generated = model.generate(
        modalities=batch["modalities"],
        missing_modalities=missing_modalities,
    )

    assert generated.shape == (2, 1, 8, 8)
    assert torch.isfinite(generated).all()


def demo():
    torch.manual_seed(42)
    model = make_model()
    xs = make_inputs()

    print("=== MultimodalTSJEPA demo ===")
    print(f"num_modalities: {model.num_modalities}")
    print(f"latent_seq_len: {model.latent_seq_len}")
    print(f"embed_dim: {model.embed_dim}")

    out = model(xs)
    print("\nCase 1: randomly sampled missing modality per sample")
    print(f"loss: {float(out['loss'].detach()):.6f}")
    print(f"predictions: {tuple(out['predictions'].shape)}")
    print(f"targets: {tuple(out['targets'].shape)}")
    print(f"missing_mask:\n{out['missing_mask']}")

    out = model(xs)
    print("\nCase 2: another random missing-modality sample")
    print(f"loss: {float(out['loss'].detach()):.6f}")
    print(f"predicted_missing: {tuple(out['predicted_missing'].shape)}")
    print(f"target_missing: {tuple(out['target_missing'].shape)}")
    print(f"missing_mask:\n{out['missing_mask']}")

    out["loss"].backward()
    model.update_target_encoders()
    print("\nBackward pass and EMA target update finished.")


if __name__ == "__main__":
    test_forward_with_one_missing_modality()
    test_backward_and_ema_update()
    test_randomly_masks_one_modality_per_sample()
    test_heterogeneous_modality_channels()
    test_denoiser_overall_forward_backward()
    test_denoiser_generate_with_explicit_missing_modalities()
    demo()
