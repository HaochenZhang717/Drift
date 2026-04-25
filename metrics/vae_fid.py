import importlib.util
import os
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import sqrtm
from torch.utils.data import DataLoader, TensorDataset, random_split


def _load_fid_vae_class():
    try:
        from fid_vae import FIDVAE

        return FIDVAE
    except ImportError:
        pass

    repo_root = Path(__file__).resolve().parents[1]
    candidates = (
        repo_root / "fid_vae.py",
        repo_root.parent / "ts_baselines" / "VerbalTS" / "models" / "vae" / "fid_vae.py",
        repo_root.parent / "ts_baselines" / "ImagenTime" / "metrics" / "fid_vae.py",
    )
    for module_path in candidates:
        if not module_path.exists():
            continue
        spec = importlib.util.spec_from_file_location("fid_vae", module_path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.FIDVAE

    raise ImportError(
        "Unable to load FIDVAE. Add fid_vae.py to PYTHONPATH or place it in the repository root."
    )


FIDVAE = _load_fid_vae_class()

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
    return "./fid_vae_ckpts"


def _dataset_to_ckpt_dir(dataset):
    ckpt_dir_name = _DATASET_TO_CKPT_DIR.get(dataset)
    if ckpt_dir_name is not None:
        return ckpt_dir_name
    safe_dataset = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(dataset))
    return f"vae_{safe_dataset}"


def _resolve_ckpt_path(dataset, ckpt_root=None, require_exists=True):
    ckpt_root = _resolve_ckpt_root(ckpt_root)
    ckpt_dir_name = _dataset_to_ckpt_dir(dataset)
    ckpt_path = os.path.join(ckpt_root, ckpt_dir_name, "best.pt")
    if require_exists and not os.path.exists(ckpt_path):
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


def _make_fid_vae(channels, seq_len, device):
    return FIDVAE(
        input_dim=channels,
        output_dim=channels,
        seq_len=seq_len,
        hidden_size=128,
        num_layers=2,
        num_heads=8,
        latent_dim=64,
    ).to(device)


def _vae_loss(model, x, out):
    if hasattr(model, "loss_function"):
        return model.loss_function(x, out["recon"], out["mu"], out["logvar"])["loss"]
    recon_loss = torch.nn.functional.mse_loss(out["recon"], x)
    kl = -0.5 * (1 + out["logvar"] - out["mu"].pow(2) - out["logvar"].exp())
    beta = getattr(model, "beta", 0.001)
    return recon_loss + beta * kl.mean()


def _validate_vae(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for (x,) in dataloader:
            x = x.to(device)
            total_loss += _vae_loss(model, x, model(x)).item()
    return total_loss / max(len(dataloader), 1)


def _train_vae_checkpoint(
    real_tensor,
    dataset,
    device,
    ckpt_path,
    epochs=50,
    batch_size=64,
    lr=1e-3,
):
    channels, seq_len = real_tensor.shape[1], real_tensor.shape[2]
    model = _make_fid_vae(channels, seq_len, device)

    dataset_obj = TensorDataset(real_tensor)
    if len(dataset_obj) > 1:
        val_size = max(1, int(0.1 * len(dataset_obj)))
        train_size = len(dataset_obj) - val_size
        generator = torch.Generator().manual_seed(0)
        train_dataset, val_dataset = random_split(
            dataset_obj, [train_size, val_size], generator=generator
        )
    else:
        train_dataset = dataset_obj
        val_dataset = dataset_obj

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    ckpt_dir = os.path.dirname(ckpt_path)
    os.makedirs(ckpt_dir, exist_ok=True)
    last_path = os.path.join(ckpt_dir, "last.pt")
    best_val_loss = float("inf")

    print(
        f"FID-VAE checkpoint not found for dataset '{dataset}'. "
        f"Training on real data and saving to {ckpt_path}."
    )
    for epoch in range(epochs):
        model.train()
        for (x,) in DataLoader(train_dataset, batch_size=batch_size, shuffle=True):
            x = x.to(device)
            out = model(x)
            loss = _vae_loss(model, x, out)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        val_loss = _validate_vae(model, val_loader, device)
        torch.save(model.state_dict(), last_path)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)
        print(f"FID-VAE epoch {epoch + 1}/{epochs} | val_loss: {val_loss:.6f}")

    return model


def VAE_FID(
    ori_data,
    generated_data,
    dataset,
    device,
    vae_ckpt_root=None,
    batch_size=128,
    auto_train=True,
    train_epochs=None,
    train_batch_size=64,
    train_lr=1e-3,
):
    real_tensor = _to_bct(ori_data)
    fake_tensor = _to_bct(generated_data)

    channels, seq_len = real_tensor.shape[1], real_tensor.shape[2]
    ckpt_path = _resolve_ckpt_path(dataset, vae_ckpt_root, require_exists=False)

    model = _make_fid_vae(channels, seq_len, device).eval()
    if os.path.exists(ckpt_path):
        model = _load_model_state(model, ckpt_path, device)
    elif auto_train:
        if train_epochs is None:
            train_epochs = int(os.getenv("FID_VAE_TRAIN_EPOCHS", "50"))
        model = _train_vae_checkpoint(
            real_tensor,
            dataset,
            device,
            ckpt_path,
            epochs=train_epochs,
            batch_size=train_batch_size,
            lr=train_lr,
        ).eval()
    else:
        raise FileNotFoundError(
            f"Expected FID-VAE checkpoint at {ckpt_path} for dataset '{dataset}'."
        )

    real_embeddings = _extract_embeddings(model, real_tensor, device=device, batch_size=batch_size)
    fake_embeddings = _extract_embeddings(model, fake_tensor, device=device, batch_size=batch_size)
    fake_embeddings = fake_embeddings[: real_embeddings.shape[0]]

    return _compute_fid(real_embeddings, fake_embeddings)
