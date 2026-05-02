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
    "glucose": "vae_glucose",
    "glucose_daily": "vae_glucose_daily",
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


def _resolve_ckpt_path(dataset, ckpt_root=None, ckpt_name="best.pt"):
    ckpt_root = _resolve_ckpt_root(ckpt_root)
    if ckpt_root is None:
        raise FileNotFoundError(
            "Could not find fid_vae_ckpts root. Set FID_VAE_CKPT_ROOT or pass vae_ckpt_root explicitly."
        )

    # 1) Prefer direct folder naming: <ckpt_root>/<dataset>/<ckpt_name>
    direct_path = os.path.join(ckpt_root, dataset, ckpt_name)
    if os.path.exists(direct_path):
        return direct_path

    # 2) Backward-compatible mapped folder naming
    ckpt_dir_name = _DATASET_TO_CKPT_DIR.get(dataset)
    if ckpt_dir_name is not None:
        mapped_path = os.path.join(ckpt_root, ckpt_dir_name, ckpt_name)
        if os.path.exists(mapped_path):
            return mapped_path

    raise FileNotFoundError(
        f"Could not find FID-VAE checkpoint for dataset '{dataset}'. "
        f"Tried: {direct_path}"
        + (
            f", {os.path.join(ckpt_root, ckpt_dir_name, ckpt_name)}"
            if ckpt_dir_name is not None
            else ""
        )
    )


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


def _load_model_state_dict(ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    return state


def _infer_ckpt_input_dim(state_dict):
    # encoder.conv.0.weight shape: (hidden_size//2, input_dim, kernel)
    return int(state_dict["encoder.conv.0.weight"].shape[1])


def _infer_vae_hparams_from_state(state_dict):
    # encoder.to_mu.weight: (latent_dim, hidden_size)
    to_mu_w = state_dict["encoder.to_mu.weight"]
    latent_dim = int(to_mu_w.shape[0])
    hidden_size = int(to_mu_w.shape[1])

    # Infer encoder depth from encoder.layers.<idx>.*
    layer_indices = []
    for key in state_dict.keys():
        if key.startswith("encoder.layers."):
            parts = key.split(".")
            if len(parts) > 2 and parts[2].isdigit():
                layer_indices.append(int(parts[2]))
    num_layers = (max(layer_indices) + 1) if layer_indices else 2

    # Heuristic for heads: prefer 8 when divisible, then 4/2/1
    if hidden_size % 8 == 0:
        num_heads = 8
    elif hidden_size % 4 == 0:
        num_heads = 4
    elif hidden_size % 2 == 0:
        num_heads = 2
    else:
        num_heads = 1

    # decoder.fc.weight: (hidden_size * (seq_len // 4), latent_dim)
    fc_w = state_dict["decoder.fc.weight"]
    seq_len = int((fc_w.shape[0] // hidden_size) * 4)
    return {
        "latent_dim": latent_dim,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "seq_len": seq_len,
    }


def _load_model_state(model, state_dict):
    model.load_state_dict(state_dict, strict=True)
    return model


def VAE_FID(
    ori_data,
    generated_data,
    dataset,
    device,
    vae_ckpt_root=None,
    vae_ckpt_name="best.pt",
    batch_size=128,
):
    real_tensor = _to_bct(ori_data)
    fake_tensor = _to_bct(generated_data)
    channels = real_tensor.shape[1]

    dataset_candidates = [dataset]
    if channels == 1 and not str(dataset).endswith("_one_channel"):
        dataset_candidates.insert(0, f"{dataset}_one_channel")

    selected = None
    tried_msgs = []
    for dataset_name in dataset_candidates:
        try:
            ckpt_path = _resolve_ckpt_path(dataset_name, vae_ckpt_root, ckpt_name=vae_ckpt_name)
        except FileNotFoundError as exc:
            tried_msgs.append(str(exc))
            continue

        state_dict = _load_model_state_dict(ckpt_path, device)
        ckpt_in_dim = _infer_ckpt_input_dim(state_dict)
        if ckpt_in_dim != channels:
            tried_msgs.append(
                f"Found checkpoint at {ckpt_path}, but input_dim={ckpt_in_dim} "
                f"does not match evaluated channels={channels}."
            )
            continue
        selected = (dataset_name, ckpt_path, state_dict)
        break

    if selected is None:
        raise RuntimeError(
            "No compatible FID-VAE checkpoint was found for evaluation.\n"
            + "\n".join(tried_msgs)
        )

    _, ckpt_path, state_dict = selected
    hp = _infer_vae_hparams_from_state(state_dict)

    model = FIDVAE(
        input_dim=channels,
        output_dim=channels,
        seq_len=hp["seq_len"],
        hidden_size=hp["hidden_size"],
        num_layers=hp["num_layers"],
        num_heads=hp["num_heads"],
        latent_dim=hp["latent_dim"],
    ).to(device).eval()
    model = _load_model_state(model, state_dict)
    print("real_tensor size:", real_tensor.shape)
    print("fake_tensor size:", fake_tensor.shape)
    real_embeddings = _extract_embeddings(model, real_tensor, device=device, batch_size=batch_size)
    fake_embeddings = _extract_embeddings(model, fake_tensor, device=device, batch_size=batch_size)
    fake_embeddings = fake_embeddings[: real_embeddings.shape[0]]

    return _compute_fid(real_embeddings, fake_embeddings)
