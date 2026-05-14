import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from baselines.JITT1.denoiser import Denoiser
from data_provider.data_provider import get_test
from metrics.discriminative_torch import discriminative_score_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate JITT1 checkpoints with discriminative score using EMA-generated samples."
    )
    parser.add_argument("--run_dir", type=str, required=True, help="Directory containing JITT1 checkpoints (*.pt).")
    parser.add_argument("--output_jsonl", type=str, default="jitt1_discriminative_ckpts.jsonl")
    parser.add_argument("--num_runs", type=int, default=10)
    parser.add_argument("--gen_batch_size", type=int, default=256)
    parser.add_argument("--real_batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda"])
    return parser.parse_args()


def checkpoint_sort_key(path: Path) -> Tuple[int, str]:
    stem = path.stem
    if stem.startswith("epoch_"):
        try:
            return int(stem.split("epoch_")[1]), stem
        except Exception:
            return (10**8, stem)
    if stem == "best":
        return (10**9 - 1, stem)
    if stem == "last":
        return (10**9, stem)
    return (10**8, stem)


def discover_checkpoints(run_dir: Path) -> List[Path]:
    ckpts = [p for p in run_dir.glob("*.pt") if p.is_file()]
    return sorted(ckpts, key=checkpoint_sort_key)


def _to_tensor_batch(batch):
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch


def load_real_test_data(ckpt_args: Dict, seq_len: int, target_channels: int, num_workers: int, batch_size: int) -> np.ndarray:
    train_datasets = ckpt_args.get("train_on_datasets", [])
    if not train_datasets:
        raise ValueError("Checkpoint args missing train_on_datasets.")
    if len(train_datasets) != 1:
        raise ValueError(
            "This evaluator expects one dataset per run directory. "
            f"Found train_on_datasets={train_datasets}"
        )

    ds_name = train_datasets[0]
    if ds_name == "GlucoseSliding":
        ds_data = "GlucoseSliding"
        rel_path = ckpt_args.get("glucose_rel_path", "glucose_{split}.parquet")
        stride = int(ckpt_args.get("glucose_stride", 1))
    elif ds_name == "ErcotData":
        ds_data = "ErcotData"
        rel_path = ckpt_args.get("ercot_rel_path", "ERCOT_merged.csv")
        stride = int(ckpt_args.get("ercot_stride", 1))
    elif ds_name == "HouseholdData":
        ds_data = "HouseholdData"
        rel_path = ckpt_args.get("household_rel_path", "HouseHold_6.csv")
        stride = int(ckpt_args.get("household_stride", 1))
    else:
        raise ValueError(f"Unsupported dataset name in checkpoint args: {ds_name}")

    ds_cfg = {
        "name": ds_name,
        "data": ds_data,
        "datasets_dir": ckpt_args["datasets_dir"],
        "rel_path": rel_path,
        "seq_len": int(seq_len),
        "flag": "val",
        "window_stride": stride,
        "ts_stride": stride,
        "stride": stride,
    }

    test_ds = get_test(ds_cfg)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    chunks: List[torch.Tensor] = []
    for batch in test_loader:
        x = _to_tensor_batch(batch).to(torch.float32)
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        if x.size(-1) < target_channels:
            pad = torch.zeros(x.size(0), x.size(1), target_channels - x.size(-1), dtype=x.dtype)
            x = torch.cat([x, pad], dim=-1)
        elif x.size(-1) > target_channels:
            x = x[..., :target_channels]
        chunks.append(x)

    if not chunks:
        raise ValueError("No validation samples loaded.")

    real = torch.cat(chunks, dim=0).numpy().astype(np.float32)
    return real


def generate_samples(model: Denoiser, n_samples: int, batch_size: int, device: torch.device) -> np.ndarray:
    generated: List[torch.Tensor] = []
    remaining = n_samples
    model.eval()
    with torch.no_grad():
        while remaining > 0:
            bs = min(batch_size, remaining)
            x = model.generate(batch_size=bs, device=device).detach().cpu().to(torch.float32)
            generated.append(x)
            remaining -= bs
    return torch.cat(generated, dim=0).numpy().astype(np.float32)


def get_ema_state_dicts(ckpt: Dict) -> Dict[str, Dict[str, torch.Tensor]]:
    if "models" in ckpt and isinstance(ckpt["models"], dict):
        return {k: v for k, v in ckpt["models"].items() if str(k).startswith("ema_")}
    if "ema_models" in ckpt and isinstance(ckpt["ema_models"], dict):
        return ckpt["ema_models"]
    return {}


def main() -> None:
    args = parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run_dir is not a directory: {run_dir}")

    output_path = Path(args.output_jsonl)
    if not output_path.is_absolute():
        output_path = run_dir / output_path

    ckpts = discover_checkpoints(run_dir)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {run_dir}")

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")
    print(f"Found {len(ckpts)} checkpoints")

    rows: List[Dict] = []

    for ckpt_path in ckpts:
        print(f"\n[Checkpoint] {ckpt_path.name}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        if "model_args" not in ckpt or "args" not in ckpt:
            print("  skip: missing model_args/args")
            continue

        model_args = ckpt["model_args"]
        train_args = ckpt["args"]

        denoiser = Denoiser(argparse.Namespace(**model_args)).to(device)
        target_channels = int(model_args["enc_in"])
        seq_len = int(model_args["seq_len"])

        real_data = load_real_test_data(
            ckpt_args=train_args,
            seq_len=seq_len,
            target_channels=target_channels,
            num_workers=args.num_workers,
            batch_size=args.real_batch_size,
        )

        if args.max_eval_samples is not None:
            real_data = real_data[: args.max_eval_samples]

        n_eval = int(real_data.shape[0])
        print(f"  eval samples: {n_eval}, shape={list(real_data.shape)}")

        ema_states = get_ema_state_dicts(ckpt)
        if not ema_states:
            print("  skip: no ema models found in checkpoint")
            continue

        for ema_name, ema_state in ema_states.items():
            print(f"  [EMA] {ema_name}")
            denoiser.load_state_dict(ema_state, strict=True)
            denoiser.eval()

            run_scores: List[float] = []
            for run_idx in range(args.num_runs):
                gen_data = generate_samples(
                    model=denoiser,
                    n_samples=n_eval,
                    batch_size=args.gen_batch_size,
                    device=device,
                )
                score = float(discriminative_score_metrics(real_data, gen_data, device))
                run_scores.append(score)
                print(f"    run {run_idx + 1:02d}/{args.num_runs}: disc={score:.6f}")

            row = {
                "run_dir": str(run_dir),
                "checkpoint": str(ckpt_path),
                "checkpoint_name": ckpt_path.name,
                "ema_name": ema_name,
                "num_eval_samples": n_eval,
                "num_runs": int(args.num_runs),
                "disc_mean": float(np.mean(run_scores)),
                "disc_std": float(np.std(run_scores)),
                "disc_runs": run_scores,
                "real_shape": list(real_data.shape),
                "seq_len": seq_len,
                "channels": target_channels,
                "dataset": train_args.get("train_on_datasets", [None])[0],
            }
            rows.append(row)
            print(f"    => mean={row['disc_mean']:.6f}, std={row['disc_std']:.6f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(rows)} rows to: {output_path}")


if __name__ == "__main__":
    main()
