import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

from data_provider.data_provider import get_test
from metrics.discriminative_torch import discriminative_score_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate ImagenTime eval_samples with discriminative score, "
            "and write all results into a single JSONL file."
        )
    )
    parser.add_argument(
        "--imagentime_root",
        type=str,
        required=True,
        help="Root directory like .../baselines/ImagenTime/ImagenTime",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        required=True,
        help="Output JSONL path for all evaluations.",
    )
    parser.add_argument(
        "--config_root",
        type=str,
        default="baselines/ImagenFew/configs/ImagenTime",
        help="Config directory containing <run_name>.yaml files.",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=10,
        help="Repeated runs per eval_samples file for averaging.",
    )
    parser.add_argument(
        "--expected_count_strict",
        action="store_true",
        help="Raise an error when sampled count != inferred test-set count.",
    )
    return parser.parse_args()


def discover_eval_sample_files(imagentime_root: Path) -> List[Tuple[str, Path]]:
    files: List[Tuple[str, Path]] = []
    for run_dir in sorted([p for p in imagentime_root.iterdir() if p.is_dir()]):
        eval_dir = run_dir / "eval_samples"
        if not eval_dir.is_dir():
            continue
        for pt in sorted(eval_dir.glob("*.pt")):
            if pt.is_file():
                files.append((run_dir.name, pt))
    return files


def _to_ntc(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:
        return x.unsqueeze(-1)
    if x.ndim != 3:
        raise ValueError(f"Expected tensor ndim in {{2,3}}, got shape={tuple(x.shape)}")
    return x


def load_eval_pair(pt_path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    payload = torch.load(pt_path, map_location="cpu")
    if "real_ts" not in payload or "sampled_ts" not in payload:
        raise KeyError(f"Missing real_ts/sampled_ts in {pt_path}")

    real_ts = _to_ntc(payload["real_ts"].detach().cpu().to(torch.float32))
    sampled_ts = _to_ntc(payload["sampled_ts"].detach().cpu().to(torch.float32))

    n_use = min(real_ts.shape[0], sampled_ts.shape[0])
    real_ts = real_ts[:n_use]
    sampled_ts = sampled_ts[:n_use]

    meta = {
        "dataset_in_file": payload.get("dataset"),
        "epoch": int(payload.get("epoch", -1)),
        "real_shape": list(real_ts.shape),
        "sampled_shape": list(sampled_ts.shape),
    }
    return real_ts.numpy(), sampled_ts.numpy(), meta


def infer_test_count(run_name: str, config_root: Path) -> Optional[int]:
    cfg_path = config_root / f"{run_name}.yaml"
    if not cfg_path.is_file():
        return None

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if "datasets" not in cfg or not cfg["datasets"]:
        return None

    dataset_entry = dict(cfg["datasets"][0])
    dataset_config = {
        "name": dataset_entry["name"],
        "data": dataset_entry["data"],
        "datasets_dir": cfg["datasets_dir"],
        "seq_len": int(cfg["seq_len"]),
        "flag": "test",
    }

    for key in ["rel_path", "rel_path_train", "rel_path_valid", "window_stride", "ts_stride", "stride"]:
        if key in dataset_entry:
            dataset_config[key] = dataset_entry[key]

    test_ds = get_test(dataset_config)
    return int(len(test_ds))


def main() -> None:
    args = parse_args()

    imagentime_root = Path(args.imagentime_root).expanduser().resolve()
    output_path = Path(args.output_jsonl).expanduser().resolve()
    config_root = Path(args.config_root).expanduser().resolve()

    if not imagentime_root.is_dir():
        raise FileNotFoundError(f"imagentime_root is not a directory: {imagentime_root}")

    file_items = discover_eval_sample_files(imagentime_root)
    if not file_items:
        raise FileNotFoundError(f"No eval_samples/*.pt found under: {imagentime_root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Found {len(file_items)} eval sample files")

    rows = []
    for run_name, pt_path in file_items:
        print(f"\n[Eval] run={run_name} file={pt_path.name}")

        real_sig, gen_sig, meta = load_eval_pair(pt_path)
        n_eval = int(real_sig.shape[0])

        expected_test_count = infer_test_count(run_name, config_root)
        count_match = None if expected_test_count is None else (n_eval == expected_test_count)

        if expected_test_count is None:
            print("  warning: config not found or unreadable, skip test-size check")
        else:
            print(f"  sample_count={n_eval}, expected_test_count={expected_test_count}, match={count_match}")
            if args.expected_count_strict and not count_match:
                raise ValueError(
                    f"Sample count mismatch for {pt_path}: sampled={n_eval}, expected_test={expected_test_count}"
                )

        run_scores: List[float] = []
        for run_idx in range(args.num_runs):
            # Re-sample a new paired evaluation set each run so multi-run stats include
            # both discriminator randomness and sample-selection randomness.
            sample_idx = np.random.choice(n_eval, size=n_eval, replace=True)
            run_real = real_sig[sample_idx]
            run_gen = gen_sig[sample_idx]

            score = float(discriminative_score_metrics(run_real, run_gen, device))
            run_scores.append(score)
            print(f"  run {run_idx + 1:02d}/{args.num_runs}: disc={score:.6f}")

        row = {
            "baseline": "ImagenTime",
            "run_name": run_name,
            "eval_pt": str(pt_path),
            "eval_pt_name": pt_path.name,
            "dataset": meta["dataset_in_file"],
            "epoch": meta["epoch"],
            "num_eval_samples": n_eval,
            "expected_test_samples": expected_test_count,
            "sample_count_match_test": count_match,
            "num_runs": int(args.num_runs),
            "disc_mean": float(np.mean(run_scores)),
            "disc_std": float(np.std(run_scores)),
            "disc_runs": run_scores,
            "real_shape": meta["real_shape"],
            "sampled_shape": meta["sampled_shape"],
        }
        rows.append(row)

        print(f"  => mean={row['disc_mean']:.6f}, std={row['disc_std']:.6f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(rows)} rows to: {output_path}")


if __name__ == "__main__":
    main()
