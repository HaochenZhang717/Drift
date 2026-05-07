import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.cls_cond_model import DriftDiT_models
from models.evaluation_models.classifier import Simple1DCNNClassifier
from utils.utils_dataset import AIREADIModalityImputationDataset, AI_READI_STUDY_GROUPS
from ts_quality_eval import delay_images_to_series


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_real_dataset(args: argparse.Namespace) -> AIREADIModalityImputationDataset:
    return AIREADIModalityImputationDataset(
        root=args.data_root,
        split=args.real_split,
        modalities=["glucose"],
        anchor_modality="glucose",
        target_modality="glucose",
        window_size=args.ts_seq_len,
        window_stride=args.window_stride,
        window_mode=args.window_mode,
        daily_min_events=args.daily_min_events,
        normalize=True,
        participants_tsv_path=args.participants_tsv_path,
        include_participant_metadata=True,
        include_study_group=True,
        include_clinical_site=False,
        include_clinical_static=False,
        pad=True,
        return_dict=True,
    )


def build_generator_from_ckpt(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    model_fn = DriftDiT_models[args.gen_model]
    gen = model_fn(
        img_size=args.img_size,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        label_dropout=0.0,
    ).to(device)

    ckpt = torch.load(args.gen_ckpt_path, map_location="cpu", weights_only=False)
    if args.use_ema and "ema" in ckpt:
        gen.load_state_dict(ckpt["ema"], strict=True)
        print(f"Loaded generator EMA weights from {args.gen_ckpt_path}")
    else:
        gen.load_state_dict(ckpt["model"], strict=True)
        print(f"Loaded generator model weights from {args.gen_ckpt_path}")

    gen.eval()
    return gen


def build_classifier_from_ckpt(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    clf = Simple1DCNNClassifier(
        in_channels=1,
        num_classes=args.num_classes,
        hidden_channels=args.clf_hidden_channels,
        dropout=args.clf_dropout,
    ).to(device)

    ckpt = torch.load(args.clf_ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    clf.load_state_dict(state, strict=True)
    clf.eval()
    print(f"Loaded classifier from {args.clf_ckpt_path}")
    return clf


@torch.no_grad()
def collect_real_series_per_class(
    dataset: AIREADIModalityImputationDataset,
    num_classes: int,
    n_per_class: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    buffers_x: List[List[torch.Tensor]] = [[] for _ in range(num_classes)]
    counts = [0 for _ in range(num_classes)]

    for batch in loader:
        x = batch["target"].float()
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        y = batch["study_group_label"].long()

        valid = (y >= 0) & (y < num_classes)
        if not valid.any():
            continue
        x = x[valid]
        y = y[valid]

        for c in range(num_classes):
            if counts[c] >= n_per_class:
                continue
            idx = torch.where(y == c)[0]
            if idx.numel() == 0:
                continue
            needed = n_per_class - counts[c]
            take = idx[:needed]
            buffers_x[c].append(x[take].cpu())
            counts[c] += int(take.numel())

        if all(v >= n_per_class for v in counts):
            break

    min_count = min(counts)
    if min_count == 0:
        raise ValueError(f"At least one class has 0 real samples under current filters. counts={counts}")

    if min_count < n_per_class:
        print(f"Warning: not enough real samples for all classes, using min_count={min_count} per class. counts={counts}")
    k = min(n_per_class, min_count)

    xs = []
    ys = []
    for c in range(num_classes):
        x_c = torch.cat(buffers_x[c], dim=0)[:k]
        y_c = torch.full((k,), c, dtype=torch.long)
        xs.append(x_c)
        ys.append(y_c)

    x = torch.cat(xs, dim=0).to(device)
    y = torch.cat(ys, dim=0).to(device)
    return x, y


@torch.no_grad()
def generate_series_per_class(
    gen_model: torch.nn.Module,
    num_classes: int,
    n_per_class: int,
    config: Dict[str, Any],
    device: torch.device,
    batch_size: int,
    cfg_alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xs = []
    ys = []
    for c in range(num_classes):
        n_done = 0
        while n_done < n_per_class:
            cur = min(batch_size, n_per_class - n_done)
            z = torch.randn(
                cur,
                config["in_channels"],
                config["img_size"],
                config["img_size"],
                device=device,
            )
            labels = torch.full((cur,), c, dtype=torch.long, device=device)
            imgs = gen_model.forward_with_cfg(z, labels, alpha=cfg_alpha)
            series = delay_images_to_series(imgs, config=config, device=device)
            xs.append(series)
            ys.append(labels)
            n_done += cur

    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    return x, y


@torch.no_grad()
def evaluate_with_classifier(
    clf: torch.nn.Module,
    x: torch.Tensor,
    y_true: torch.Tensor,
    num_classes: int,
) -> Dict[str, Any]:
    # classifier expects (B, C, T)
    x_in = x.transpose(1, 2)
    logits = clf(x_in)
    probs = F.softmax(logits, dim=-1)
    pred = logits.argmax(dim=-1)

    conf = torch.zeros((num_classes, num_classes), dtype=torch.long, device=x.device)
    for t, p in zip(y_true, pred):
        conf[t.long(), p.long()] += 1

    total = int(conf.sum().item())
    acc = float((pred == y_true).float().mean().item())

    per_class = {}
    recalls = []
    precisions = []
    f1s = []
    for c in range(num_classes):
        tp = float(conf[c, c].item())
        fn = float(conf[c, :].sum().item() - tp)
        fp = float(conf[:, c].sum().item() - tp)

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)

        name = AI_READI_STUDY_GROUPS[c] if c < len(AI_READI_STUDY_GROUPS) else f"class_{c}"
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(conf[c, :].sum().item()),
        }

    # distribution-level gaps
    pred_hist = torch.bincount(pred, minlength=num_classes).float()
    true_hist = torch.bincount(y_true, minlength=num_classes).float()
    pred_dist = pred_hist / pred_hist.sum().clamp_min(1.0)
    true_dist = true_hist / true_hist.sum().clamp_min(1.0)

    eps = 1e-8
    kl_true_pred = float((true_dist * torch.log((true_dist + eps) / (pred_dist + eps))).sum().item())
    tv_true_pred = float(0.5 * torch.abs(true_dist - pred_dist).sum().item())

    return {
        "n": total,
        "acc": acc,
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "per_class": per_class,
        "confusion_matrix": conf.detach().cpu().tolist(),
        "pred_distribution": pred_dist.detach().cpu().tolist(),
        "true_distribution": true_dist.detach().cpu().tolist(),
        "kl_true_vs_pred": kl_true_pred,
        "tv_true_vs_pred": tv_true_pred,
        "avg_confidence": float(probs.max(dim=-1).values.mean().item()),
    }


def summarize_repeat_metrics(repeat_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    metric_keys = [
        ("real", "acc"),
        ("real", "macro_f1"),
        ("real", "macro_precision"),
        ("real", "macro_recall"),
        ("real", "kl_true_vs_pred"),
        ("real", "tv_true_vs_pred"),
        ("real", "avg_confidence"),
        ("generated", "acc"),
        ("generated", "macro_f1"),
        ("generated", "macro_precision"),
        ("generated", "macro_recall"),
        ("generated", "kl_true_vs_pred"),
        ("generated", "tv_true_vs_pred"),
        ("generated", "avg_confidence"),
        ("gap", "delta_acc_gen_minus_real"),
        ("gap", "delta_macro_f1_gen_minus_real"),
        ("gap", "delta_macro_recall_gen_minus_real"),
        ("gap", "delta_avg_confidence_gen_minus_real"),
        ("gap", "delta_kl_true_vs_pred_gen_minus_real"),
        ("gap", "delta_tv_true_vs_pred_gen_minus_real"),
    ]

    agg: Dict[str, Any] = {}
    for section, key in metric_keys:
        vals = np.asarray([float(item[section][key]) for item in repeat_summaries], dtype=np.float64)
        agg[f"{section}.{key}"] = {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
        }
    return agg


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    gen_config = {
        "ts_seq_len": args.ts_seq_len,
        "ts_delay": args.ts_delay,
        "ts_embedding": args.ts_embedding,
        "img_size": args.img_size,
        "in_channels": args.in_channels,
    }

    dataset = build_real_dataset(args)
    gen_model = build_generator_from_ckpt(args, device)
    clf = build_classifier_from_ckpt(args, device)

    repeat_summaries: List[Dict[str, Any]] = []
    for repeat_idx in range(args.num_repeats):
        repeat_seed = args.seed + repeat_idx
        set_seed(repeat_seed)
        print(f"\n[Repeat {repeat_idx + 1}/{args.num_repeats}] seed={repeat_seed}")

        x_real, y_real = collect_real_series_per_class(
            dataset=dataset,
            num_classes=args.num_classes,
            n_per_class=args.samples_per_class,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        k_real = int(y_real.numel() // args.num_classes)
        print(f"Collected real samples: total={y_real.numel()} | per_class={k_real}")

        x_gen, y_gen = generate_series_per_class(
            gen_model=gen_model,
            num_classes=args.num_classes,
            n_per_class=k_real,
            config=gen_config,
            device=device,
            batch_size=args.gen_batch_size,
            cfg_alpha=args.cfg_alpha,
        )
        print(f"Generated samples: total={y_gen.numel()} | per_class={k_real}")

        real_metrics = evaluate_with_classifier(clf, x_real, y_real, args.num_classes)
        gen_metrics = evaluate_with_classifier(clf, x_gen, y_gen, args.num_classes)

        gap = {
            "delta_acc_gen_minus_real": gen_metrics["acc"] - real_metrics["acc"],
            "delta_macro_f1_gen_minus_real": gen_metrics["macro_f1"] - real_metrics["macro_f1"],
            "delta_macro_recall_gen_minus_real": gen_metrics["macro_recall"] - real_metrics["macro_recall"],
            "delta_avg_confidence_gen_minus_real": gen_metrics["avg_confidence"] - real_metrics["avg_confidence"],
            "delta_kl_true_vs_pred_gen_minus_real": gen_metrics["kl_true_vs_pred"] - real_metrics["kl_true_vs_pred"],
            "delta_tv_true_vs_pred_gen_minus_real": gen_metrics["tv_true_vs_pred"] - real_metrics["tv_true_vs_pred"],
        }
        repeat_summaries.append(
            {
                "repeat_idx": repeat_idx,
                "seed": repeat_seed,
                "samples_per_class": k_real,
                "real": real_metrics,
                "generated": gen_metrics,
                "gap": gap,
            }
        )
        print(
            f"repeat={repeat_idx + 1} | real.acc={real_metrics['acc']:.4f} | "
            f"gen.acc={gen_metrics['acc']:.4f} | delta={gap['delta_acc_gen_minus_real']:+.4f}"
        )

    aggregate = summarize_repeat_metrics(repeat_summaries)
    final_samples_per_class = int(repeat_summaries[-1]["samples_per_class"])

    summary = {
        "settings": {
            "gen_ckpt_path": args.gen_ckpt_path,
            "clf_ckpt_path": args.clf_ckpt_path,
            "real_split": args.real_split,
            "samples_per_class": final_samples_per_class,
            "num_classes": args.num_classes,
            "cfg_alpha": args.cfg_alpha,
            "window_mode": args.window_mode,
            "daily_min_events": args.daily_min_events,
            "num_repeats": args.num_repeats,
            "seed": args.seed,
        },
        "repeats": repeat_summaries,
        "aggregate": aggregate,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.output_json
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n=== Repeated Classifier Quality Gap Summary ===")
    print(
        f"real.acc mean±std = {aggregate['real.acc']['mean']:.4f} ± {aggregate['real.acc']['std']:.4f} | "
        f"gen.acc mean±std = {aggregate['generated.acc']['mean']:.4f} ± {aggregate['generated.acc']['std']:.4f} | "
        f"delta mean±std = {aggregate['gap.delta_acc_gen_minus_real']['mean']:+.4f} ± {aggregate['gap.delta_acc_gen_minus_real']['std']:.4f}"
    )
    print(
        f"real.macro_f1 mean±std = {aggregate['real.macro_f1']['mean']:.4f} ± {aggregate['real.macro_f1']['std']:.4f} | "
        f"gen.macro_f1 mean±std = {aggregate['generated.macro_f1']['mean']:.4f} ± {aggregate['generated.macro_f1']['std']:.4f} | "
        f"delta mean±std = {aggregate['gap.delta_macro_f1_gen_minus_real']['mean']:+.4f} ± {aggregate['gap.delta_macro_f1_gen_minus_real']['std']:.4f}"
    )
    print(f"Saved detailed report to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate generated-vs-real AI-READI data quality using a trained evaluator classifier")

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--participants_tsv_path", type=str, required=True)
    parser.add_argument("--real_split", type=str, default="test", choices=["train", "valid", "test"])

    parser.add_argument("--gen_ckpt_path", type=str, required=True)
    parser.add_argument("--gen_model", type=str, default="DriftDiT-Tiny", choices=sorted(DriftDiT_models.keys()))
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--cfg_alpha", type=float, default=1.0)

    parser.add_argument("--clf_ckpt_path", type=str, required=True)
    parser.add_argument("--clf_hidden_channels", type=int, default=32)
    parser.add_argument("--clf_dropout", type=float, default=0.1)

    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--samples_per_class", type=int, default=1000)

    parser.add_argument("--ts_seq_len", type=int, default=288)
    parser.add_argument("--ts_delay", type=int, default=18)
    parser.add_argument("--ts_embedding", type=int, default=18)
    parser.add_argument("--img_size", type=int, default=18)
    parser.add_argument("--in_channels", type=int, default=1)

    parser.add_argument("--window_mode", type=str, default="daily", choices=["daily", "sliding"])
    parser.add_argument("--window_stride", type=int, default=128)
    parser.add_argument("--daily_min_events", type=int, default=288)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gen_batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_repeats", type=int, default=1)

    parser.add_argument("--output_dir", type=str, default="./outputs/aireadi_eval_classifier")
    parser.add_argument("--output_json", type=str, default="generated_vs_real_classifier_gap.json")

    main(parser.parse_args())
