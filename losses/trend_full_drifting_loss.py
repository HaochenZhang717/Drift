"""Trend/full drifting loss for two-loop time-series Drift models."""

import math
from typing import Any

import torch
import torch.nn.functional as F

from drifting import compute_V
from ts_quality_eval import delay_images_to_series
from utils.trend_filter import gaussian_smooth_series


def _lerp(start: float, end: float, s: float) -> float:
    return start * (1.0 - s) + end * s


def get_component_weights(
    step: int,
    total_steps: int,
    *,
    trend_start: float = 1.0,
    trend_end: float = 0.2,
    full_start: float = 0.2,
    full_end: float = 1.0,
    schedule: str = "cosine",
) -> dict[str, float]:
    """Return scheduled trend/full loss weights."""
    if total_steps <= 0:
        progress = 1.0
    else:
        progress = min(max(step / float(total_steps), 0.0), 1.0)

    if schedule == "cosine":
        s = 0.5 - 0.5 * math.cos(math.pi * progress)
    elif schedule == "linear":
        s = progress
    elif schedule == "constant":
        s = 0.0
    else:
        raise ValueError(f"Unsupported weight schedule {schedule!r}")

    return {
        "trend": _lerp(trend_start, trend_end, s),
        "full": _lerp(full_start, full_end, s),
    }


def _flatten_feature(series: torch.Tensor) -> torch.Tensor:
    return series.flatten(start_dim=1)


def _compute_component_drift_loss(
    feat_gen: torch.Tensor,
    feat_pos: torch.Tensor,
    temperatures: list[float],
) -> tuple[torch.Tensor, dict[str, float]]:
    feat_neg = feat_gen
    total_loss = torch.tensor(0.0, device=feat_gen.device)
    total_drift_norm = 0.0
    total_v_norm = 0.0
    total_true_v_norm = 0.0

    V_total = torch.zeros_like(feat_gen)
    V_total_raw = torch.zeros_like(feat_gen)

    for tau in temperatures:
        V_tau = compute_V(
            feat_gen,
            feat_pos,
            feat_neg,
            tau,
            mask_self=True,
        )
        V_total_raw = V_total_raw + V_tau
        v_norm = torch.sqrt(torch.mean(V_tau ** 2) + 1e-8)
        V_tau = V_tau / (v_norm + 1e-8)
        V_total = V_total + V_tau
        total_v_norm += v_norm.item()

    true_v_norm = torch.sqrt(torch.mean(V_total_raw ** 2) + 1e-8)
    total_true_v_norm += true_v_norm.item()
    target = (feat_gen + V_total).detach()
    total_loss = total_loss + F.mse_loss(feat_gen, target)
    total_drift_norm += torch.sqrt(torch.mean(V_total ** 2) + 1e-8).item()

    return total_loss, {
        "drift_norm": total_drift_norm,
        "v_norm": total_v_norm,
        "true_v_norm": total_true_v_norm,
    }


def _get_config_value(config: dict[str, Any], key: str, default: Any) -> Any:
    return config[key] if key in config and config[key] is not None else default


def compute_trend_full_drifting_loss(
    x_outputs: dict[str, torch.Tensor],
    x_pos: torch.Tensor,
    *,
    temperatures: list[float],
    config: dict[str, Any],
    step: int,
    total_steps: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute scheduled trend/full drifting loss.

    x_outputs must contain:
        trend: first-loop delay-image output
        full: second-loop delay-image output
    x_pos is a batch of real delay images sampled from the queue.
    """
    if "trend" not in x_outputs or "full" not in x_outputs:
        raise KeyError("x_outputs must contain 'trend' and 'full'")

    device = x_outputs["full"].device
    rep_trend_gen = delay_images_to_series(x_outputs["trend"], config, device)
    rep_full_gen = delay_images_to_series(x_outputs["full"], config, device)
    rep_full_pos = delay_images_to_series(x_pos, config, device)

    kernel_size = int(_get_config_value(config, "tf_trend_kernel_size", 15))
    sigma = float(_get_config_value(config, "tf_trend_sigma", 3.0))
    rep_trend_pos = gaussian_smooth_series(
        rep_full_pos,
        input_layout="btc",
        kernel_size=kernel_size,
        sigma=sigma,
    )

    feat_trend_gen = _flatten_feature(rep_trend_gen)
    feat_trend_pos = _flatten_feature(rep_trend_pos)
    feat_full_gen = _flatten_feature(rep_full_gen)
    feat_full_pos = _flatten_feature(rep_full_pos)

    loss_trend, info_trend = _compute_component_drift_loss(
        feat_trend_gen,
        feat_trend_pos,
        temperatures,
    )
    loss_full, info_full = _compute_component_drift_loss(
        feat_full_gen,
        feat_full_pos,
        temperatures,
    )

    weights = get_component_weights(
        step,
        total_steps,
        trend_start=float(_get_config_value(config, "tf_trend_start", 1.0)),
        trend_end=float(_get_config_value(config, "tf_trend_end", 0.2)),
        full_start=float(_get_config_value(config, "tf_full_start", 0.2)),
        full_end=float(_get_config_value(config, "tf_full_end", 1.0)),
        schedule=str(_get_config_value(config, "tf_weight_schedule", "cosine")),
    )

    loss = weights["trend"] * loss_trend + weights["full"] * loss_full

    consistency_weight = float(_get_config_value(config, "tf_consistency_weight", 0.0))
    consistency_loss = torch.tensor(0.0, device=device)
    if consistency_weight > 0:
        full_trend = gaussian_smooth_series(
            rep_full_gen,
            input_layout="btc",
            kernel_size=kernel_size,
            sigma=sigma,
        ).detach()
        consistency_loss = F.mse_loss(rep_trend_gen, full_trend)
        loss = loss + consistency_weight * consistency_loss

    info = {
        "loss": loss.item(),
        "loss_trend": loss_trend.item(),
        "loss_full": loss_full.item(),
        "loss_consistency": consistency_loss.item(),
        "weight_trend": weights["trend"],
        "weight_full": weights["full"],
        "drift_norm": (
            weights["trend"] * info_trend["drift_norm"]
            + weights["full"] * info_full["drift_norm"]
        ),
        "drift_norm_trend": info_trend["drift_norm"],
        "drift_norm_full": info_full["drift_norm"],
        "v_norm": info_trend["v_norm"] + info_full["v_norm"],
        "v_norm_trend": info_trend["v_norm"],
        "v_norm_full": info_full["v_norm"],
        "true_v_norm": info_trend["true_v_norm"] + info_full["true_v_norm"],
        "true_v_norm_trend": info_trend["true_v_norm"],
        "true_v_norm_full": info_full["true_v_norm"],
    }

    return loss, info
