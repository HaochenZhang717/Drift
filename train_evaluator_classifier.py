import argparse
import random
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.evaluation_models.classifier import Simple1DCNNClassifier
from utils.utils_dataset import AIREADIModalityImputationDataset


AI_READI_STUDY_GROUPS = [
    "healthy",
    "pre_diabetes_lifestyle_controlled",
    "oral_medication_and_or_non_insulin_injectable_medication_controlled",
    "insulin_dependent",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataset(args: argparse.Namespace, split: str) -> AIREADIModalityImputationDataset:
    return AIREADIModalityImputationDataset(
        root=args.data_root,
        split=split,
        modalities=["glucose"],
        anchor_modality="glucose",
        target_modality="glucose",
        window_size=args.seq_len,
        window_stride=args.window_stride,
        window_mode=args.window_mode,
        daily_min_events=args.daily_min_events,
        normalize=True,
        include_participant_metadata=True,
        include_study_group=True,
        include_clinical_site=False,
        include_clinical_static=False,
        participants_tsv_path=args.participants_tsv_path,
        pad=True,
        return_dict=True,
    )


def extract_batch(batch: Dict[str, Any], device: torch.device, num_classes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    x = batch["target"].float()
    if x.ndim == 2:
        x = x.unsqueeze(-1)
    y = batch["study_group_label"].long()

    valid = (y >= 0) & (y < num_classes)
    if not valid.any():
        return torch.empty(0, device=device), torch.empty(0, dtype=torch.long, device=device)

    x = x[valid].to(device)
    y = y[valid].to(device)

    # x: (B, T, C) -> (B, C, T)
    x = x.transpose(1, 2)
    return x, y


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    num_classes: int,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch in loader:
        x, y = extract_batch(batch, device, num_classes)
        if x.numel() == 0:
            continue

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = criterion(logits, y)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item()) * y.size(0)
        total_correct += int((logits.argmax(dim=-1) == y).sum().item())
        total_count += int(y.size(0))

    if total_count == 0:
        return {"loss": 0.0, "acc": 0.0, "count": 0.0}

    return {
        "loss": total_loss / total_count,
        "acc": total_correct / total_count,
        "count": float(total_count),
    }


def evaluate_per_class(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> Dict[str, float]:
    model.eval()
    correct = torch.zeros(num_classes, dtype=torch.long)
    total = torch.zeros(num_classes, dtype=torch.long)

    with torch.no_grad():
        for batch in loader:
            x, y = extract_batch(batch, device, num_classes)
            if x.numel() == 0:
                continue
            pred = model(x).argmax(dim=-1)
            for c in range(num_classes):
                mask = y == c
                total[c] += int(mask.sum().item())
                correct[c] += int(((pred == c) & mask).sum().item())

    metrics: Dict[str, float] = {}
    for c, name in enumerate(AI_READI_STUDY_GROUPS[:num_classes]):
        denom = int(total[c].item())
        acc = float(correct[c].item()) / denom if denom > 0 else 0.0
        metrics[f"acc_{name}"] = acc
        metrics[f"n_{name}"] = float(denom)
    return metrics


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = build_dataset(args, args.train_split)
    val_dataset = build_dataset(args, args.val_split)
    test_dataset = build_dataset(args, args.test_split)

    print(f"Dataset windows | train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = Simple1DCNNClassifier(
        in_channels=1,
        num_classes=args.num_classes,
        hidden_channels=args.hidden_channels,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best_classifier.pt"
    last_path = out_dir / "last_classifier.pt"

    best_val_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, criterion, optimizer, args.num_classes)
        val_metrics = run_epoch(model, val_loader, device, criterion, None, args.num_classes)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.6f} train_acc={train_metrics['acc']:.4f} | "
            f"val_loss={val_metrics['loss']:.6f} val_acc={val_metrics['acc']:.4f} | "
            f"train_n={int(train_metrics['count'])} val_n={int(val_metrics['count'])}"
        )

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "val_acc": val_metrics["acc"],
        }
        torch.save(state, last_path)

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            torch.save(state, best_path)
            print(f"Saved new best checkpoint to {best_path} (val_acc={best_val_acc:.4f})")

    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded best checkpoint from epoch {ckpt['epoch']} with val_acc={ckpt['val_acc']:.4f}")

    test_metrics = run_epoch(model, test_loader, device, criterion, None, args.num_classes)
    test_class_metrics = evaluate_per_class(model, test_loader, device, args.num_classes)

    print(
        f"Test | loss={test_metrics['loss']:.6f} acc={test_metrics['acc']:.4f} "
        f"n={int(test_metrics['count'])}"
    )
    for k, v in test_class_metrics.items():
        if k.startswith("n_"):
            print(f"{k}: {int(v)}")
        else:
            print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a 4-class 1D-CNN evaluator on AI-READI study groups")
    parser.add_argument("--data_root", type=str, default="./AI-READI-processed")
    parser.add_argument("--participants_tsv_path", type=str, required=True)

    parser.add_argument("--train_split", type=str, default="train", choices=["train", "valid", "test"])
    parser.add_argument("--val_split", type=str, default="valid", choices=["train", "valid", "test"])
    parser.add_argument("--test_split", type=str, default="test", choices=["train", "valid", "test"])

    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--window_stride", type=int, default=128)
    parser.add_argument("--window_mode", type=str, default="daily", choices=["daily", "sliding"])
    parser.add_argument("--daily_min_events", type=int, default=96)

    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--output_dir", type=str, default="./outputs/aireadi_eval_classifier")

    args = parser.parse_args()
    train(args)
