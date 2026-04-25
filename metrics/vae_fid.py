import importlib.util
import os
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import sqrtm


def _load_fid_vae_class():
    # module_path = Path(__file__).resolve().parents[1] / "VerbalTS" / "fid_vae.py"
    # spec = importlib.util.spec_from_file_location("verbalts_fid_vae", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load FIDVAE from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FIDVAE


# FIDVAE = _load_fid_vae_class()
from fid_vae import FIDVAE

_DEFAULT_CKPT_DIR_CANDIDATES = (
    os.getenv("FID_VAE_CKPT_ROOT"),
    "./fid_vae_ckpts",
    "../fid_vae_ckpts",
    "/playpen-shared/haochenz/ImagenFew/fid_vae_ckpts",
)

_DATASET_TO_CKPT_DIR = {
    "synthetic_u": "vae_synth_u",
    "synthetic_m": "vae_synth_m",
    "ETTm1": "vae_ettm1",
    "ettm1": "vae_ettm1",
    "istanbul_traffic": "vae_istanbul_traffic",
    "Weather": "vae_weather",
    "weather": "vae_weather",
}


def _compute_fid(real_embeddings, fake_embeddings):
    mu_r = np.mean(real_embeddings, axis=0)
    mu_f = np.mean(fake_embeddings, axis=0)

    sigma_r = np.cov(real_embeddings, rowvar=False)
    sigma_f = np.cov(fake_embeddings, rowvar=False)

    covmean = sqrtm(sigma_r @ sigma_f)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = np.sum((mu_r - mu_f) ** 2) + np.trace(sigma_r + sigma_f - 2 * covmean)
    return float(fid)


def _resolve_ckpt_root(ckpt_root=None):
    if ckpt_root:
        return ckpt_root
    for candidate in _DEFAULT_CKPT_DIR_CANDIDATES:
        if candidate and os.path.isdir(candidate):
            return candidate
    return None


def _resolve_ckpt_path(dataset, ckpt_root=None):
    ckpt_root = _resolve_ckpt_root(ckpt_root)
    if ckpt_root is None:
        raise FileNotFoundError(
            "Could not find fid_vae_ckpts root. Set FID_VAE_CKPT_ROOT or pass vae_ckpt_root explicitly."
        )

    ckpt_dir_name = _DATASET_TO_CKPT_DIR.get(dataset)
    if ckpt_dir_name is None:
        raise FileNotFoundError(
            f"No FID-VAE checkpoint mapping is defined for dataset '{dataset}'."
        )

    ckpt_path = os.path.join(ckpt_root, ckpt_dir_name, "best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Expected FID-VAE checkpoint at {ckpt_path} for dataset '{dataset}'."
        )
    return ckpt_path


def _to_bct(data):
    if torch.is_tensor(data):
        tensor = data.detach().cpu()
    else:
        tensor = torch.as_tensor(data)
    tensor = tensor.to(torch.float32)
    if tensor.ndim != 3:
        raise ValueError(f"Expected 3D time-series tensor/array, got shape {tuple(tensor.shape)}")
    if tensor.shape[1] > tensor.shape[2]:
        tensor = tensor.permute(0, 2, 1)
    return tensor.contiguous()


def _extract_embeddings(model, data, device, batch_size=128):
    embeddings = []
    model.eval()
    with torch.no_grad():
        for start in range(0, data.shape[0], batch_size):
            batch = data[start:start + batch_size].to(device)
            out = model(batch)
            embeddings.append(out["mu"].detach().cpu())
    return torch.cat(embeddings, dim=0).numpy()


def _load_model_state(model, ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=True)
    return model


def VAE_FID(ori_data, generated_data, dataset, device, vae_ckpt_root=None, batch_size=128):
    real_tensor = _to_bct(ori_data)
    fake_tensor = _to_bct(generated_data)

    ckpt_path = _resolve_ckpt_path(dataset, vae_ckpt_root)
    channels, seq_len = real_tensor.shape[1], real_tensor.shape[2]

    model = FIDVAE(
        input_dim=channels,
        output_dim=channels,
        seq_len=seq_len,
        hidden_size=128,
        num_layers=2,
        num_heads=8,
        latent_dim=64,
    ).to(device).eval()
    model = _load_model_state(model, ckpt_path, device)

    real_embeddings = _extract_embeddings(model, real_tensor, device=device, batch_size=batch_size)
    fake_embeddings = _extract_embeddings(model, fake_tensor, device=device, batch_size=batch_size)
    fake_embeddings = fake_embeddings[: real_embeddings.shape[0]]

    return _compute_fid(real_embeddings, fake_embeddings)
