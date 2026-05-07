import argparse
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from fid_vae import FIDVAE
from utils.utils_dataset import AIREADIModalityImputationDataset


DEFAULT_GROUP_NAMES = {
    0: "healthy",
    1: "pre_diabetes_lifestyle_controlled",
    2: "oral_medication_and_or_non_insulin_injectable_medication_controlled",
    3: "insulin_treated",
}


def get_args():
    parser = argparse.ArgumentParser(
        description="Visualize daily glucose FID-VAE embeddings by AI-READI study group."
    )
    parser.add_argument("--data_root", type=str, default="./AI-READI")
    parser.add_argument("--participants_tsv_path", type=str, default=None)
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="./fid_vae_ckpts/vae_glucose_daily/best.pt",
        help="Path to the trained daily glucose FID-VAE checkpoint.",
    )
    parser.add_argument("--output_dir", type=str, default="./outputs/fid_vae_study_group")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--ts_seq_len", type=int, default=288)
    parser.add_argument("--daily_min_events", type=int, default=288)
    parser.add_argument("--max_anchor_gap_minutes", type=float, default=10.0)
    parser.add_argument("--max_window_span_hours", type=float, default=24.0)
    parser.add_argument("--anchor_sampling_minutes", type=float, default=5.0)
    parser.add_argument("--anchor_sampling_tolerance_seconds", type=float, default=2.0)
    parser.add_argument(
        "--raw_glucose",
        action="store_true",
        help="Use raw mg/dL glucose values. Leave off for the default normalized VAE.",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--max_points_per_group",
        type=int,
        default=1500,
        help="Balanced number of points per group used for the 2D plots. All embeddings are still saved.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_tsne", action="store_true")
    return parser.parse_args()


class DailyGlucoseWithGroupDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        x = item["target"].detach().to(torch.float32).permute(1, 0).contiguous()
        label = int(item.get("study_group_label", -1))
        group = item.get("study_group", "")
        patient_id = item.get("patient_id", "")
        return {
            "x": x,
            "study_group_label": torch.tensor(label, dtype=torch.long),
            "study_group": group,
            "patient_id": patient_id,
        }


def make_dataset(args):
    return AIREADIModalityImputationDataset(
        root=args.data_root,
        split=args.split,
        modalities=["glucose"],
        anchor_modality="glucose",
        target_modality="glucose",
        window_size=args.ts_seq_len,
        window_stride=args.ts_seq_len,
        window_mode="daily",
        daily_min_events=args.daily_min_events,
        max_events_per_modality={"glucose": args.ts_seq_len},
        min_events_per_modality={"glucose": args.daily_min_events},
        normalize=not args.raw_glucose,
        max_anchor_gap_minutes=args.max_anchor_gap_minutes,
        max_window_span_hours=args.max_window_span_hours,
        anchor_sampling_minutes=args.anchor_sampling_minutes,
        anchor_sampling_tolerance_seconds=args.anchor_sampling_tolerance_seconds,
        participants_tsv_path=args.participants_tsv_path,
        include_clinical_static=False,
        include_participant_metadata=args.participants_tsv_path is not None,
        include_study_group=True,
        include_clinical_site=False,
        pad=True,
        return_dict=True,
    )


def infer_model_kwargs(state_dict, seq_len):
    if "encoder.stem.weight" in state_dict:
        input_dim = int(state_dict["encoder.stem.weight"].shape[1])
        hidden_size = int(state_dict["encoder.stem.weight"].shape[0])
        latent_dim = int(state_dict["encoder.to_mu.weight"].shape[0])
        layer_ids = set()
        for key in state_dict:
            prefix = "encoder.layers."
            if key.startswith(prefix):
                rest = key[len(prefix):]
                layer_ids.add(int(rest.split(".", 1)[0]))
        num_layers = max(layer_ids) + 1 if layer_ids else 2
        latent_downsample = int(state_dict.get("decoder.latent_downsample_buffer", torch.tensor(8)).item())
        decoder_upsample_rate = int(state_dict.get("decoder.decoder_upsample_rate_buffer", torch.tensor(4)).item())
        if "decoder.seq_len_buffer" in state_dict:
            seq_len = int(state_dict["decoder.seq_len_buffer"].item())
        return {
            "input_dim": input_dim,
            "output_dim": input_dim,
            "seq_len": seq_len,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "latent_dim": latent_dim,
            "latent_downsample": latent_downsample,
            "decoder_upsample_rate": decoder_upsample_rate,
        }

    if "encoder.conv.0.weight" not in state_dict or "encoder.conv.2.weight" not in state_dict:
        raise ValueError("Checkpoint does not look like a FIDVAE state_dict.")
    input_dim = int(state_dict["encoder.conv.0.weight"].shape[1])
    hidden_size = int(state_dict["encoder.conv.2.weight"].shape[0])
    latent_dim = int(state_dict["encoder.to_mu.weight"].shape[0])
    layer_ids = set()
    for key in state_dict:
        prefix = "encoder.layers."
        if key.startswith(prefix):
            rest = key[len(prefix):]
            layer_ids.add(int(rest.split(".", 1)[0]))
    num_layers = max(layer_ids) + 1 if layer_ids else 2
    return {
        "input_dim": input_dim,
        "output_dim": input_dim,
        "seq_len": seq_len,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "latent_dim": latent_dim,
    }


def load_fid_vae(ckpt_path, seq_len, device):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    kwargs = infer_model_kwargs(state, seq_len)
    model = FIDVAE(**kwargs).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, kwargs


@torch.no_grad()
def encode_dataset(model, dataloader, device):
    embeddings = []
    labels = []
    groups = []
    patient_ids = []
    for batch in tqdm(dataloader, desc="Encoding train set"):
        x = batch["x"].to(device)
        out = model(x)
        embeddings.append(out["mu"].detach().cpu().numpy())
        labels.append(batch["study_group_label"].detach().cpu().numpy())
        groups.extend(batch["study_group"])
        patient_ids.extend(batch["patient_id"])
    return (
        np.concatenate(embeddings, axis=0),
        np.concatenate(labels, axis=0),
        np.asarray(groups, dtype=object),
        np.asarray(patient_ids, dtype=object),
    )


def balanced_plot_indices(labels, max_points_per_group, seed):
    rng = np.random.default_rng(seed)
    selected = []
    for label in sorted(x for x in np.unique(labels) if x >= 0):
        idx = np.flatnonzero(labels == label)
        if max_points_per_group > 0 and idx.size > max_points_per_group:
            idx = rng.choice(idx, size=max_points_per_group, replace=False)
        selected.append(idx)
    if not selected:
        return np.arange(labels.shape[0])
    idx = np.concatenate(selected)
    rng.shuffle(idx)
    return idx


def group_display_names(labels, groups):
    names = {}
    for label in sorted(np.unique(labels)):
        if label < 0:
            names[int(label)] = "unknown"
            continue
        group_values = [g for g in groups[labels == label] if g]
        if group_values:
            names[int(label)] = Counter(group_values).most_common(1)[0][0]
        else:
            names[int(label)] = DEFAULT_GROUP_NAMES.get(int(label), f"class_{int(label)}")
    return names


def plot_projection(coords, labels, group_names, title, output_path):
    colors = {
        0: "#187c7a",
        1: "#d95f02",
        2: "#7570b3",
        3: "#e7298a",
        -1: "#666666",
    }
    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    for label in sorted(np.unique(labels)):
        idx = labels == label
        name = group_names.get(int(label), f"class_{int(label)}")
        ax.scatter(
            coords[idx, 0],
            coords[idx, 1],
            s=7,
            alpha=0.55,
            linewidths=0,
            color=colors.get(int(label), None),
            label=f"{name} (n={idx.sum()})",
        )
    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.grid(True, alpha=0.2)
    ax.legend(markerscale=2.0, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_counts(labels, groups, output_path):
    by_label = defaultdict(Counter)
    for label, group in zip(labels, groups):
        by_label[int(label)][str(group)] += 1
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("study_group_label,study_group,count\n")
        for label in sorted(by_label):
            for group, count in by_label[label].most_common():
                f.write(f"{label},{group},{count}\n")


def main():
    args = get_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"FID-VAE checkpoint not found: {ckpt_path}. "
            "Train it with train_fid_vae_daily_glucose.py or pass --ckpt_path."
        )

    device = args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    base_dataset = make_dataset(args)
    dataset = DailyGlucoseWithGroupDataset(base_dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    sample = dataset[0]["x"]
    _, seq_len = sample.shape
    model, model_kwargs = load_fid_vae(ckpt_path, seq_len=seq_len, device=device)
    print(f"Loaded FIDVAE from {ckpt_path}")
    print(f"Model kwargs: {model_kwargs}")
    print(f"Encoding {len(dataset)} {args.split} windows on {device}")

    embeddings, labels, groups, patient_ids = encode_dataset(model, dataloader, device)
    group_names = group_display_names(labels, groups)
    save_counts(labels, groups, output_dir / "study_group_counts.csv")

    np.savez(
        output_dir / f"fid_vae_{args.split}_embeddings_by_study_group.npz",
        embeddings=embeddings,
        study_group_label=labels,
        study_group=groups,
        patient_id=patient_ids,
    )

    plot_idx = balanced_plot_indices(labels, args.max_points_per_group, args.seed)
    x_plot = embeddings[plot_idx]
    y_plot = labels[plot_idx]

    pca = PCA(n_components=2, random_state=args.seed)
    pca_coords = pca.fit_transform(x_plot)
    plot_projection(
        pca_coords,
        y_plot,
        group_names,
        f"FID-VAE daily glucose embeddings by study group ({args.split}, PCA)",
        output_dir / f"fid_vae_{args.split}_study_group_pca.png",
    )

    if not args.skip_tsne:
        perplexity = min(30, max(5, (len(plot_idx) - 1) // 3))
        tsne = TSNE(
            n_components=2,
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
            random_state=args.seed,
        )
        tsne_coords = tsne.fit_transform(x_plot)
        plot_projection(
            tsne_coords,
            y_plot,
            group_names,
            f"FID-VAE daily glucose embeddings by study group ({args.split}, t-SNE)",
            output_dir / f"fid_vae_{args.split}_study_group_tsne.png",
        )

    print(f"Saved outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
