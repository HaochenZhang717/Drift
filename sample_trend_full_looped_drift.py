"""Sample from a two-loop trend/full Drift checkpoint."""

import argparse
from pathlib import Path

import torch

from benchmarking_drift import save_time_series_grid
from models.trend_full_looped_unconditional_model import TrendFullLoopedDriftDiT_models
from ts_quality_eval import delay_images_to_series
from utils.trend_filter import gaussian_smooth_series


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    *,
    use_ema: bool = True,
) -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model_fn = TrendFullLoopedDriftDiT_models[config["model"]]
    model = model_fn(
        img_size=config["img_size"],
        in_channels=config["in_channels"],
        loop_depth=config.get("tf_loop_depth", 3),
        num_loops=config.get("tf_num_loops", 2),
    ).to(device)
    state_key = "ema" if use_ema and "ema" in checkpoint else "model"
    model.load_state_dict(checkpoint[state_key], strict=True)
    model.eval()
    return model, config


@torch.no_grad()
def sample(
    checkpoint_path: str,
    output_dir: str,
    *,
    num_samples: int = 80,
    output_component: str = "full",
    save_all_components: bool = False,
    seed: int = 42,
    use_ema: bool = True,
):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_model_from_checkpoint(checkpoint_path, device, use_ema=use_ema)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    noise = torch.randn(
        num_samples,
        config["in_channels"],
        config["img_size"],
        config["img_size"],
        device=device,
    )
    outputs = model(noise)

    if output_component not in outputs:
        raise ValueError(
            f"output_component must be one of {sorted(outputs.keys())}, got {output_component!r}"
        )

    selected = outputs[output_component]
    selected_series = delay_images_to_series(selected, config, device)
    save_time_series_grid(
        selected_series,
        str(output_dir / f"samples_{output_component}.png"),
        ncol=8,
    )

    if save_all_components:
        for component, samples in outputs.items():
            series = delay_images_to_series(samples, config, device)
            save_time_series_grid(
                series,
                str(output_dir / f"samples_{component}.png"),
                ncol=8,
            )

        full_series = delay_images_to_series(outputs["full"], config, device)
        smooth_full = gaussian_smooth_series(
            full_series,
            input_layout="btc",
            kernel_size=int(config.get("tf_trend_kernel_size", 15)),
            sigma=float(config.get("tf_trend_sigma", 3.0)),
        )
        save_time_series_grid(
            smooth_full,
            str(output_dir / "samples_smooth_full.png"),
            ncol=8,
        )

    print(f"Saved samples to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Sample Trend/Full Looped Drift")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=80)
    parser.add_argument("--output_component", type=str, default="full", choices=["trend", "full"])
    parser.add_argument("--save_all_components", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_ema", action="store_true")
    args = parser.parse_args()

    sample(
        args.checkpoint,
        args.output_dir,
        num_samples=args.num_samples,
        output_component=args.output_component,
        save_all_components=args.save_all_components,
        seed=args.seed,
        use_ema=not args.no_ema,
    )


if __name__ == "__main__":
    main()
