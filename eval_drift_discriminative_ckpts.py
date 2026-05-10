import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from benchmarking_drift import (
    DelayEmbeddingImageDataset,
    MinMaxNormalizedTimeSeriesDataset,
    _fit_minmax_stats,
)
from data_provider.data_provider import get_test, get_train
from metrics.discriminative_torch import discriminative_score_metrics
from models.unconditional_model import DriftDiT_models
from ts_quality_eval import collect_real_time_series, generate_time_series_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate all drift-model checkpoints with discriminative score (10 runs by default)."
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory containing checkpoint_*.pt / checkpoint_final.pt files.",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default="discriminative_ckpt_results.jsonl",
        help="Output jsonl filename (saved inside checkpoint_dir if relative path).",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=10,
        help="Number of repeated runs per checkpoint for averaging.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size used in generation/real-data collection.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Num workers used in real-data loader.",
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Load EMA weights from checkpoint (if available).",
    )
    return parser.parse_args()


def _extract_epoch_for_sort(path: Path) -> Tuple[int, str]:
    stem = path.stem
    if "epoch" in stem:
        try:
            suffix = stem.split("epoch", 1)[1]
            digits = "".join(ch for ch in suffix if ch.isdigit())
            if digits:
                return int(digits), stem
        except Exception:
            pass
    if "final" in stem:
        return 10**9, stem
    if "best" in stem:
        return 10**9 - 1, stem
    return -1, stem


def discover_checkpoints(ckpt_dir: Path) -> List[Path]:
    candidates = sorted(ckpt_dir.glob("checkpoint*.pt"))
    if not candidates:
        candidates = sorted(ckpt_dir.glob("*.pt"))
    candidates = [p for p in candidates if p.is_file()]
    return sorted(candidates, key=_extract_epoch_for_sort)


def build_datasets_from_config(config: Dict) -> Tuple[torch.utils.data.Dataset, Dict]:
    effective_stride = config.get("window_stride")
    if effective_stride is None:
        effective_stride = config.get("ts_stride")
    if effective_stride is None:
        effective_stride = config.get("stride")

    dataset_config = {
        "name": config["dataset_name"],
        "data": config["data"],
        "datasets_dir": config["datasets_dir"],
        "rel_path": config["rel_path"],
        "seq_len": config["ts_seq_len"],
        "flag": "train",
    }
    if config.get("rel_path_train") is not None:
        dataset_config["rel_path_train"] = config["rel_path_train"]
    if config.get("rel_path_valid") is not None:
        dataset_config["rel_path_valid"] = config["rel_path_valid"]
    if effective_stride is not None:
        dataset_config["window_stride"] = int(effective_stride)
        dataset_config["ts_stride"] = int(effective_stride)
        dataset_config["stride"] = int(effective_stride)

    train_base = get_train(dataset_config.copy())
    test_base = get_test(dataset_config.copy())

    one_channel = bool(config.get("one_channel", False))
    data_min, data_max = _fit_minmax_stats(train_base, one_channel=one_channel)

    test_norm = MinMaxNormalizedTimeSeriesDataset(
        test_base,
        data_min=data_min,
        data_max=data_max,
        one_channel=one_channel,
    )
    test_delay_img = DelayEmbeddingImageDataset(test_norm, config)
    return test_delay_img, dataset_config


def build_model_from_config(config: Dict, device: torch.device) -> torch.nn.Module:
    model_name = config["model"]
    if model_name not in DriftDiT_models:
        raise ValueError(f"Unsupported model '{model_name}'. Available: {list(DriftDiT_models.keys())}")
    model = DriftDiT_models[model_name](
        img_size=config["img_size"],
        in_channels=config["in_channels"],
    ).to(device)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    ckpt_dir = Path(args.checkpoint_dir).expanduser().resolve()
    if not ckpt_dir.exists() or not ckpt_dir.is_dir():
        raise FileNotFoundError(f"checkpoint_dir does not exist or is not a directory: {ckpt_dir}")

    output_path = Path(args.output_jsonl)
    if not output_path.is_absolute():
        output_path = ckpt_dir / output_path

    checkpoints = discover_checkpoints(ckpt_dir)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Found {len(checkpoints)} checkpoints under {ckpt_dir}")

    rows = []
    for ckpt_path in checkpoints:
        print(f"\n[Eval] {ckpt_path.name}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "config" not in ckpt:
            raise KeyError(f"Checkpoint missing 'config': {ckpt_path}")
        config = ckpt["config"]

        test_dataset, _ = build_datasets_from_config(config)
        n_test = len(test_dataset)
        print(f"Test size: {n_test}")

        model = build_model_from_config(config, device)
        if args.use_ema and "ema" in ckpt:
            model.load_state_dict(ckpt["ema"], strict=True)
            weight_source = "ema"
        else:
            model.load_state_dict(ckpt["model"], strict=True)
            weight_source = "model"

        real_sig = collect_real_time_series(
            test_dataset,
            config,
            device,
            num_samples=n_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        ).numpy().astype(np.float32)

        run_scores: List[float] = []
        for run_idx in range(args.num_runs):
            gen_sig = generate_time_series_samples(
                model,
                config,
                device,
                num_samples=n_test,
                batch_size=args.batch_size,
            ).numpy().astype(np.float32)

            score = float(discriminative_score_metrics(real_sig, gen_sig, device))
            run_scores.append(score)
            print(f"  run {run_idx + 1:02d}/{args.num_runs}: disc={score:.6f}")

        mean_score = float(np.mean(run_scores))
        std_score = float(np.std(run_scores))

        row = {
            "checkpoint": str(ckpt_path),
            "checkpoint_name": ckpt_path.name,
            "weight_source": weight_source,
            "num_test_samples": int(n_test),
            "num_runs": int(args.num_runs),
            "disc_mean": mean_score,
            "disc_std": std_score,
            "disc_runs": run_scores,
        }
        rows.append(row)
        print(f"  => mean={mean_score:.6f}, std={std_score:.6f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
