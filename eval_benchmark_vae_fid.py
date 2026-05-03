import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from metrics.vae_fid import VAE_FID


def find_all_eval_pts(run_dir: Path) -> list[Path]:
    eval_dir = run_dir / "eval_samples"
    if not eval_dir.is_dir():
        raise FileNotFoundError(f"Missing eval_samples dir: {eval_dir}")

    files = sorted(eval_dir.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No .pt files found in: {eval_dir}")
    return files


def tensor_stats(x: torch.Tensor) -> Dict[str, float]:
    return {
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
    }


def normalize_to_minus1_1_by_real(
    real_ts: torch.Tensor,
    sampled_ts: torch.Tensor,
    fit_real_ts: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    if real_ts.ndim != 3 or sampled_ts.ndim != 3:
        raise ValueError(f"Expected (N, T, C), got real={tuple(real_ts.shape)}, sampled={tuple(sampled_ts.shape)}")

    # Match train_fid_vae_benchmark.py:
    # 1) fit per-channel min/max on a reference real set
    # 2) clamp normalized values to [0, 1]
    # 3) map to [-1, 1]
    real_min = fit_real_ts.amin(dim=(0, 1), keepdim=True)
    real_max = fit_real_ts.amax(dim=(0, 1), keepdim=True)
    denom = (real_max - real_min).clamp_min(1e-6)

    real_u = (real_ts - real_min) / denom
    sampled_u = (sampled_ts - real_min) / denom
    real_norm = torch.clamp(real_u, 0.0, 1.0)
    sampled_norm = torch.clamp(sampled_u, 0.0, 1.0)
    real_norm = real_norm * 2.0 - 1.0
    sampled_norm = sampled_norm * 2.0 - 1.0
    debug = {
        "fit_real_min": float(real_min.min().item()),
        "fit_real_max": float(real_max.max().item()),
        "fit_range_min": float((real_max - real_min).min().item()),
        "fit_range_max": float((real_max - real_min).max().item()),
        "real_u_lt0_ratio": float((real_u < 0.0).to(torch.float32).mean().item()),
        "real_u_gt1_ratio": float((real_u > 1.0).to(torch.float32).mean().item()),
        "sampled_u_lt0_ratio": float((sampled_u < 0.0).to(torch.float32).mean().item()),
        "sampled_u_gt1_ratio": float((sampled_u > 1.0).to(torch.float32).mean().item()),
    }
    return real_norm, sampled_norm, debug


def sample_pair(real_ts: torch.Tensor, sampled_ts: torch.Tensor, n: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    total = min(real_ts.shape[0], sampled_ts.shape[0])
    if total == 0:
        raise ValueError("Empty tensors: cannot sample.")

    if n < 0:
        return real_ts[:total], sampled_ts[:total]

    n_use = min(n, total)
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    idx = torch.randperm(total, generator=g)[:n_use]
    return real_ts[idx], sampled_ts[idx]


def evaluate_one(
    dataset_name: str,
    pt_path: Path,
    device: torch.device,
    num_samples: int,
    seed: int,
    vae_ckpt_root: str,
    vae_ckpt_name: str,
    print_stats: bool = False,
) -> Dict[str, float]:
    payload = torch.load(pt_path, map_location="cpu")

    real_ts = payload["real_ts"].to(torch.float32)
    sampled_ts = payload["sampled_ts"].to(torch.float32)

    real_sub, sampled_sub = sample_pair(real_ts, sampled_ts, n=num_samples, seed=seed)
    real_norm, sampled_norm, norm_debug = normalize_to_minus1_1_by_real(
        real_sub, sampled_sub, fit_real_ts=real_ts
    )

    if print_stats:
        print(f"[stats] file={pt_path.name}, seed={seed}")
        print(f"[stats] raw_real     {tensor_stats(real_sub)}")
        print(f"[stats] raw_sampled  {tensor_stats(sampled_sub)}")
        print(f"[stats] norm_real    {tensor_stats(real_norm)}")
        print(f"[stats] norm_sampled {tensor_stats(sampled_norm)}")
        print(f"[stats] norm_debug   {norm_debug}")

    fid = VAE_FID(
        ori_data=real_norm,
        generated_data=sampled_norm,
        dataset=dataset_name,
        device=device,
        vae_ckpt_root=vae_ckpt_root,
        vae_ckpt_name=vae_ckpt_name,
    )

    return {
        "fid": float(fid),
        "num_samples": int(real_norm.shape[0]),
        "pt_path": str(pt_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VAE-FID on benchmark generated samples.")
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--num_repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--print_stats", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--vae_ckpt_root", type=str, default=None)
    parser.add_argument("--vae_ckpt_name", type=str, default="best.pt")
    parser.add_argument("--single_dataset_name", type=str, default=None)
    parser.add_argument("--single_run_dir", type=str, default=None)

    parser.add_argument(
        "--ettm2_dir",
        type=str,
        default="/playpen-shared/haochenz/ImagenFew/logs/ImagenTime/ETTm2/c51e88cc-807e-44f5-9570-e0ba34934f03",
    )
    parser.add_argument(
        "--etth2_dir",
        type=str,
        default="/playpen-shared/haochenz/ImagenFew/logs/ImagenTime/ETTh2/39d8773e-f43b-464f-b15f-4728bb590971",
    )
    parser.add_argument(
        "--weather_dir",
        type=str,
        default="/playpen-shared/haochenz/ImagenFew/logs/ImagenTime/Weather/c3412176-555b-430d-8f86-6972c8348b28",
    )
    parser.add_argument(
        "--airquality_dir",
        type=str,
        default="/playpen-shared/haochenz/ImagenFew/logs/ImagenTime/AirQuality/a02ed1a0-5603-495e-8306-5d5dc0f661bc",
    )

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable, fallback to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    if args.single_dataset_name is not None or args.single_run_dir is not None:
        if not args.single_dataset_name or not args.single_run_dir:
            raise ValueError(
                "When using single-dataset mode, both --single_dataset_name and --single_run_dir are required."
            )
        dataset_to_dir = {
            args.single_dataset_name: Path(args.single_run_dir),
        }
    else:
        dataset_to_dir = {
            "ETTm2": Path(args.ettm2_dir),
            "ETTh2": Path(args.etth2_dir),
            "Weather": Path(args.weather_dir),
            "AirQuality": Path(args.airquality_dir),
        }

    print(f"device={device}")
    print(f"num_samples={args.num_samples}, num_repeats={args.num_repeats}, seed={args.seed}")

    all_results = {}
    for dataset_name, run_dir in dataset_to_dir.items():
        print("=" * 80)
        print(f"Evaluating {dataset_name} | run_dir={run_dir}")
        pt_files = find_all_eval_pts(run_dir)
        dataset_results = []
        for pt_path in pt_files:
            repeat_fids = []
            for repeat_idx in range(args.num_repeats):
                result = evaluate_one(
                    dataset_name=dataset_name,
                    pt_path=pt_path,
                    device=device,
                    num_samples=args.num_samples,
                    seed=args.seed + repeat_idx,
                    vae_ckpt_root=args.vae_ckpt_root,
                    vae_ckpt_name=args.vae_ckpt_name,
                    print_stats=args.print_stats,
                )
                repeat_fids.append(result["fid"])

            repeat_fids = np.asarray(repeat_fids, dtype=np.float64)
            result = {
                "pt_path": str(pt_path),
                "num_samples": int(args.num_samples),
                "num_repeats": int(args.num_repeats),
                "fid_mean": float(repeat_fids.mean()),
                "fid_std": float(repeat_fids.std()),
                "fid_var": float(repeat_fids.var()),
                "fid_values": [float(x) for x in repeat_fids.tolist()],
            }
            dataset_results.append(result)
            print(
                f"{dataset_name}: vae_fid_mean={result['fid_mean']:.6f}, "
                f"std={result['fid_std']:.6f}, var={result['fid_var']:.6f}, "
                f"n={result['num_samples']}, repeats={result['num_repeats']}, file={result['pt_path']}"
            )
        all_results[dataset_name] = dataset_results

    print("=" * 80)
    print("Summary")
    for dataset_name, results in all_results.items():
        for result in results:
            print(
                f"{dataset_name}\t{Path(result['pt_path']).name}\t"
                f"{result['fid_mean']:.6f}\t{result['fid_std']:.6f}\t{result['fid_var']:.6f}\t"
                f"{result['num_samples']}\t{result['num_repeats']}"
            )


if __name__ == "__main__":
    main()
