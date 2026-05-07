import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.evaluation_models.classifier import Simple1DCNNClassifier
from models.multimodal_jit.denoiser import Denoiser
from ts_quality_eval import delay_images_to_series
from utils.utils_dataset import AI_READI_STUDY_GROUPS, AIREADIModalityImputationDataset

MODALITY_KEYS = {
    "heart_rate": "heart_rate_aligned_to_glucose",
    "calorie": "calorie_aligned_to_glucose",
    "physical_activity": "physical_activity_aligned_to_glucose",
    "respiratory_rate": "respiratory_rate_aligned_to_glucose",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _select_modality_pack(batch: Dict[str, Any], modality: str) -> Dict[str, torch.Tensor]:
    packs = batch["modalities"]
    preferred_key = MODALITY_KEYS[modality]
    if preferred_key in packs:
        return packs[preferred_key]
    if modality in packs:
        return packs[modality]
    raise KeyError(f"Missing modality pack for {modality}; checked keys: {preferred_key}, {modality}")


def build_dataset(args: argparse.Namespace) -> AIREADIModalityImputationDataset:
    modalities = ["glucose", "heart_rate", "calorie", "physical_activity", "respiratory_rate"]
    max_events = {m: args.ts_seq_len for m in modalities}
    min_events = {"glucose": args.daily_min_events}

    return AIREADIModalityImputationDataset(
        root=args.data_root,
        split=args.real_split,
        modalities=modalities,
        anchor_modality="glucose",
        target_modality="glucose",
        window_size=args.ts_seq_len,
        window_stride=args.window_stride,
        window_mode=args.window_mode,
        daily_min_events=args.daily_min_events,
        max_events_per_modality=max_events,
        min_events_per_modality=min_events,
        normalize=True,
        max_anchor_gap_minutes=args.max_anchor_gap_minutes,
        max_window_span_hours=args.max_window_span_hours,
        anchor_sampling_minutes=args.anchor_sampling_minutes,
        anchor_sampling_tolerance_seconds=args.anchor_sampling_tolerance_seconds,
        participants_tsv_path=args.participants_tsv_path,
        include_clinical_static=False,
        include_participant_metadata=True,
        include_study_group=True,
        include_clinical_site=False,
        pad=True,
        return_dict=True,
    )


def build_mm_model_from_ckpt(args: argparse.Namespace, device: torch.device) -> Denoiser:
    ckpt = torch.load(args.mm_ckpt_path, map_location="cpu", weights_only=False)
    ckpt_args = dict(ckpt.get("args", {}))
    if not ckpt_args:
        raise ValueError("Checkpoint missing 'args'; cannot reconstruct multimodal model.")

    # Ensure current ckpt paths are used if explicitly provided.
    if args.ckpt_heart_rate:
        ckpt_args["ckpt_heart_rate"] = args.ckpt_heart_rate
    if args.ckpt_calorie:
        ckpt_args["ckpt_calorie"] = args.ckpt_calorie
    if args.ckpt_physical_activity:
        ckpt_args["ckpt_physical_activity"] = args.ckpt_physical_activity
    if args.ckpt_respiratory_rate:
        ckpt_args["ckpt_respiratory_rate"] = args.ckpt_respiratory_rate

    model_args = SimpleNamespace(
        num_tokens_per_modality=ckpt_args["num_tokens_per_modality"],
        mm_dim_in=ckpt_args["mm_dim_in"],
        hidden_channels=ckpt_args["hidden_channels"],
        mm_n_heads=ckpt_args["mm_n_heads"],
        ae_input_dim=1,
        ae_d_model=ckpt_args["ae_d_model"],
        ae_max_len=ckpt_args["ae_max_len"],
        ae_nheads=ckpt_args["ae_nheads"],
        ae_num_layers=ckpt_args["ae_num_layers"],
        mm_missing_ratio_threshold=ckpt_args["max_missing_ratio"],
        ae_cpt_paths={
            "heart_rate": ckpt_args["ckpt_heart_rate"],
            "calorie": ckpt_args["ckpt_calorie"],
            "physical_activity": ckpt_args["ckpt_physical_activity"],
            "respiratory_rate": ckpt_args["ckpt_respiratory_rate"],
        },
        img_size=ckpt_args["img_size"],
        patch_size=ckpt_args["patch_size"],
        in_channels=ckpt_args["in_channels"],
        depth=ckpt_args["depth"],
        num_heads=ckpt_args["num_heads"],
        attn_dropout=ckpt_args["attn_dropout"],
        proj_dropout=ckpt_args["proj_dropout"],
        bottleneck_dim=ckpt_args["bottleneck_dim"],
        in_context_start=ckpt_args["in_context_start"],
        label_drop_prob=ckpt_args["label_drop_prob"],
        P_mean=ckpt_args["P_mean"],
        P_std=ckpt_args["P_std"],
        t_eps=ckpt_args["t_eps"],
        noise_scale=ckpt_args["noise_scale"],
        ema_decay1=ckpt_args["ema_decay1"],
        ema_decay2=ckpt_args["ema_decay2"],
        sampling_method=ckpt_args["sampling_method"],
        num_sampling_steps=ckpt_args["num_sampling_steps"],
        cfg=ckpt_args["cfg"],
        interval_min=ckpt_args["interval_min"],
        interval_max=ckpt_args["interval_max"],
        num_classes=ckpt_args["num_classes"],
    )

    model = Denoiser(model_args).to(device)
    which = args.mm_weights
    if which not in {"model", "ema1", "ema2"}:
        raise ValueError("--mm_weights must be one of: model, ema1, ema2")
    if which not in ckpt:
        raise KeyError(f"Checkpoint does not contain key '{which}'")
    model.load_state_dict(ckpt[which], strict=True)
    model.eval()
    print(f"Loaded multimodal model weights '{which}' from {args.mm_ckpt_path}")
    return model


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
def collect_real_and_cond_pool(
    dataset: AIREADIModalityImputationDataset,
    num_classes: int,
    n_per_class: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, Dict[str, torch.Tensor]]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    real_x: List[List[torch.Tensor]] = [[] for _ in range(num_classes)]
    cond_pool: Dict[int, Dict[str, List[torch.Tensor]]] = {
        c: {
            "heart_rate": [],
            "calorie": [],
            "physical_activity": [],
            "respiratory_rate": [],
            "heart_rate_observed_mask": [],
            "calorie_observed_mask": [],
            "physical_activity_observed_mask": [],
            "respiratory_rate_observed_mask": [],
        }
        for c in range(num_classes)
    }
    counts = [0 for _ in range(num_classes)]

    for batch in loader:
        labels = batch["study_group_label"].long()
        valid = (labels >= 0) & (labels < num_classes)
        if not valid.any():
            continue

        x = batch["target"][valid].float()
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        y = labels[valid]

        hr = _select_modality_pack(batch, "heart_rate")
        cal = _select_modality_pack(batch, "calorie")
        pa = _select_modality_pack(batch, "physical_activity")
        rr = _select_modality_pack(batch, "respiratory_rate")

        for c in range(num_classes):
            if counts[c] >= n_per_class:
                continue
            idx = torch.where(y == c)[0]
            if idx.numel() == 0:
                continue
            need = n_per_class - counts[c]
            take = idx[:need]

            real_x[c].append(x[take].cpu())
            cond_pool[c]["heart_rate"].append(hr["values"][valid][take].float().cpu())
            cond_pool[c]["calorie"].append(cal["values"][valid][take].float().cpu())
            cond_pool[c]["physical_activity"].append(pa["values"][valid][take].float().cpu())
            cond_pool[c]["respiratory_rate"].append(rr["values"][valid][take].float().cpu())
            cond_pool[c]["heart_rate_observed_mask"].append(hr["mask"][valid][take].float().cpu())
            cond_pool[c]["calorie_observed_mask"].append(cal["mask"][valid][take].float().cpu())
            cond_pool[c]["physical_activity_observed_mask"].append(pa["mask"][valid][take].float().cpu())
            cond_pool[c]["respiratory_rate_observed_mask"].append(rr["mask"][valid][take].float().cpu())
            counts[c] += int(take.numel())

        if all(v >= n_per_class for v in counts):
            break

    min_count = min(counts)
    if min_count == 0:
        raise ValueError(f"At least one class has 0 valid windows under current filters. counts={counts}")
    if min_count < n_per_class:
        print(f"Warning: not enough windows for all classes, using min_count={min_count}. counts={counts}")
    k = min(n_per_class, min_count)

    xs = []
    ys = []
    cond_final: Dict[int, Dict[str, torch.Tensor]] = {}
    for c in range(num_classes):
        x_c = torch.cat(real_x[c], dim=0)[:k]
        xs.append(x_c)
        ys.append(torch.full((k,), c, dtype=torch.long))

        cond_final[c] = {}
        for key, parts in cond_pool[c].items():
            cond_final[c][key] = torch.cat(parts, dim=0)[:k]

    x_real = torch.cat(xs, dim=0).to(device)
    y_real = torch.cat(ys, dim=0).to(device)
    return x_real, y_real, cond_final


@torch.no_grad()
def _velocity(
    model: Denoiser,
    z: torch.Tensor,
    t: torch.Tensor,
    labels: torch.Tensor,
    cond_tokens: torch.Tensor,
) -> torch.Tensor:
    x_pred = model.net(z, t.flatten(), labels, cond_tokens)
    return (x_pred - z) / (1.0 - t).clamp_min(model.t_eps)


@torch.no_grad()
def sample_mm_images(
    model: Denoiser,
    labels: torch.Tensor,
    cond: Dict[str, torch.Tensor],
) -> torch.Tensor:
    device = labels.device
    bsz = labels.shape[0]

    cond_tokens = model.mm_encoder.forward(
        heart_rate=cond["heart_rate"],
        calorie=cond["calorie"],
        physical_activity=cond["physical_activity"],
        respiratory_rate=cond["respiratory_rate"],
        heart_rate_observed_mask=cond["heart_rate_observed_mask"],
        calorie_observed_mask=cond["calorie_observed_mask"],
        physical_activity_observed_mask=cond["physical_activity_observed_mask"],
        respiratory_rate_observed_mask=cond["respiratory_rate_observed_mask"],
    )

    z = model.noise_scale * torch.randn(bsz, model.net.in_channels, model.img_size, model.img_size, device=device)
    tvals = torch.linspace(0.0, 1.0, model.steps + 1, device=device)

    for i in range(model.steps):
        t = tvals[i].view(1, 1, 1, 1).expand(bsz, 1, 1, 1)
        t_next = tvals[i + 1].view(1, 1, 1, 1).expand(bsz, 1, 1, 1)

        if model.method == "heun" and i < model.steps - 1:
            v_t = _velocity(model, z, t, labels, cond_tokens)
            z_euler = z + (t_next - t) * v_t
            v_next = _velocity(model, z_euler, t_next, labels, cond_tokens)
            v = 0.5 * (v_t + v_next)
            z = z + (t_next - t) * v
        else:
            v = _velocity(model, z, t, labels, cond_tokens)
            z = z + (t_next - t) * v

    return z


@torch.no_grad()
def generate_series_per_class(
    model: Denoiser,
    cond_pool: Dict[int, Dict[str, torch.Tensor]],
    num_classes: int,
    n_per_class: int,
    device: torch.device,
    gen_batch_size: int,
    ts_config: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    xs = []
    ys = []

    for c in range(num_classes):
        pool_size = cond_pool[c]["heart_rate"].shape[0]
        if pool_size == 0:
            raise ValueError(f"Condition pool for class {c} is empty.")

        n_done = 0
        while n_done < n_per_class:
            cur = min(gen_batch_size, n_per_class - n_done)
            idx = torch.randint(0, pool_size, (cur,))
            cond_batch = {k: v[idx].to(device) for k, v in cond_pool[c].items()}
            labels = torch.full((cur,), c, dtype=torch.long, device=device)

            imgs = sample_mm_images(model, labels, cond_batch)
            series = delay_images_to_series(imgs, config=ts_config, device=device)
            xs.append(series)
            ys.append(labels)
            n_done += cur

    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


@torch.no_grad()
def evaluate_with_classifier(
    clf: torch.nn.Module,
    x: torch.Tensor,
    y_true: torch.Tensor,
    num_classes: int,
) -> Dict[str, Any]:
    logits = clf(x.transpose(1, 2))
    probs = F.softmax(logits, dim=-1)
    pred = logits.argmax(dim=-1)

    conf = torch.zeros((num_classes, num_classes), dtype=torch.long, device=x.device)
    for t, p in zip(y_true, pred):
        conf[t.long(), p.long()] += 1

    per_class = {}
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []

    for c in range(num_classes):
        tp = float(conf[c, c].item())
        fn = float(conf[c, :].sum().item() - tp)
        fp = float(conf[:, c].sum().item() - tp)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        name = AI_READI_STUDY_GROUPS[c] if c < len(AI_READI_STUDY_GROUPS) else f"class_{c}"
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(conf[c, :].sum().item()),
        }

    pred_hist = torch.bincount(pred, minlength=num_classes).float()
    true_hist = torch.bincount(y_true, minlength=num_classes).float()
    pred_dist = pred_hist / pred_hist.sum().clamp_min(1.0)
    true_dist = true_hist / true_hist.sum().clamp_min(1.0)

    eps = 1e-8
    kl_true_pred = float((true_dist * torch.log((true_dist + eps) / (pred_dist + eps))).sum().item())
    tv_true_pred = float(0.5 * torch.abs(true_dist - pred_dist).sum().item())

    return {
        "n": int(conf.sum().item()),
        "acc": float((pred == y_true).float().mean().item()),
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


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = build_dataset(args)
    mm_model = build_mm_model_from_ckpt(args, device)
    clf = build_classifier_from_ckpt(args, device)

    x_real, y_real, cond_pool = collect_real_and_cond_pool(
        dataset=dataset,
        num_classes=args.num_classes,
        n_per_class=args.samples_per_class,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    k_real = int(y_real.numel() // args.num_classes)
    print(f"Collected real+cond: total={y_real.numel()} | per_class={k_real}")

    ts_config = {
        "ts_seq_len": args.ts_seq_len,
        "ts_delay": args.ts_delay,
        "ts_embedding": args.ts_embedding,
    }
    x_gen, y_gen = generate_series_per_class(
        model=mm_model,
        cond_pool=cond_pool,
        num_classes=args.num_classes,
        n_per_class=k_real,
        device=device,
        gen_batch_size=args.gen_batch_size,
        ts_config=ts_config,
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

    summary = {
        "settings": {
            "mm_ckpt_path": args.mm_ckpt_path,
            "mm_weights": args.mm_weights,
            "clf_ckpt_path": args.clf_ckpt_path,
            "real_split": args.real_split,
            "samples_per_class": k_real,
            "num_classes": args.num_classes,
            "window_mode": args.window_mode,
            "daily_min_events": args.daily_min_events,
            "sampling_method": mm_model.method,
            "num_sampling_steps": mm_model.steps,
        },
        "real": real_metrics,
        "generated": gen_metrics,
        "gap": gap,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.output_json
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n=== MM Classifier Quality Gap Summary ===")
    print(f"real.acc = {real_metrics['acc']:.4f} | gen.acc = {gen_metrics['acc']:.4f} | delta = {gap['delta_acc_gen_minus_real']:+.4f}")
    print(f"real.macro_f1 = {real_metrics['macro_f1']:.4f} | gen.macro_f1 = {gen_metrics['macro_f1']:.4f} | delta = {gap['delta_macro_f1_gen_minus_real']:+.4f}")
    print(f"real.kl = {real_metrics['kl_true_vs_pred']:.4f} | gen.kl = {gen_metrics['kl_true_vs_pred']:.4f} | delta = {gap['delta_kl_true_vs_pred_gen_minus_real']:+.4f}")
    print(f"Saved detailed report to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate multimodal-JiT generated-vs-real quality with evaluator classifier")

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--participants_tsv_path", type=str, required=True)
    parser.add_argument("--real_split", type=str, default="test", choices=["train", "valid", "test"])

    parser.add_argument("--mm_ckpt_path", type=str, required=True)
    parser.add_argument("--mm_weights", type=str, default="ema2", choices=["model", "ema1", "ema2"])
    parser.add_argument("--ckpt_heart_rate", type=str, default=None)
    parser.add_argument("--ckpt_calorie", type=str, default=None)
    parser.add_argument("--ckpt_physical_activity", type=str, default=None)
    parser.add_argument("--ckpt_respiratory_rate", type=str, default=None)

    parser.add_argument("--clf_ckpt_path", type=str, required=True)
    parser.add_argument("--clf_hidden_channels", type=int, default=32)
    parser.add_argument("--clf_dropout", type=float, default=0.1)

    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--samples_per_class", type=int, default=1000)

    parser.add_argument("--ts_seq_len", type=int, default=288)
    parser.add_argument("--ts_delay", type=int, default=18)
    parser.add_argument("--ts_embedding", type=int, default=18)

    parser.add_argument("--window_mode", type=str, default="daily", choices=["daily", "sliding"])
    parser.add_argument("--window_stride", type=int, default=288)
    parser.add_argument("--daily_min_events", type=int, default=288)
    parser.add_argument("--max_anchor_gap_minutes", type=float, default=10.0)
    parser.add_argument("--max_window_span_hours", type=float, default=24.0)
    parser.add_argument("--anchor_sampling_minutes", type=float, default=5.0)
    parser.add_argument("--anchor_sampling_tolerance_seconds", type=float, default=2.0)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gen_batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--output_dir", type=str, default="./outputs/aireadi_eval_classifier")
    parser.add_argument("--output_json", type=str, default="mm_generated_vs_real_classifier_gap.json")

    main(parser.parse_args())
