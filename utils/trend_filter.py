"""Differentiable trend filters for time-series tensors."""

import torch
import torch.nn.functional as F


def _to_bct(series: torch.Tensor, input_layout: str) -> tuple[torch.Tensor, str]:
    if input_layout not in {"btc", "bct"}:
        raise ValueError(f"input_layout must be 'btc' or 'bct', got {input_layout!r}")
    if series.ndim != 3:
        raise ValueError(f"Expected a 3D tensor, got shape {tuple(series.shape)}")
    if input_layout == "btc":
        return series.transpose(1, 2).contiguous(), input_layout
    return series, input_layout


def _from_bct(series: torch.Tensor, output_layout: str) -> torch.Tensor:
    if output_layout == "btc":
        return series.transpose(1, 2).contiguous()
    return series


def gaussian_kernel1d(
    kernel_size: int,
    sigma: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create a normalized 1D Gaussian kernel."""
    if kernel_size <= 0:
        raise ValueError("kernel_size must be positive")
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd for symmetric padding")
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    radius = kernel_size // 2
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def gaussian_smooth_series(
    series: torch.Tensor,
    *,
    input_layout: str = "btc",
    kernel_size: int = 15,
    sigma: float = 3.0,
    padding_mode: str = "reflect",
) -> torch.Tensor:
    """
    Smooth a time series along the time dimension with a depthwise Gaussian filter.

    Args:
        series: Tensor with shape (B, T, C) if input_layout="btc", or (B, C, T)
            if input_layout="bct".
        input_layout: Layout of the input tensor.
        kernel_size: Odd Gaussian kernel size.
        sigma: Gaussian standard deviation in time steps.
        padding_mode: Padding mode passed to torch.nn.functional.pad.

    Returns:
        Smoothed tensor in the same layout as the input.
    """
    series_bct, original_layout = _to_bct(series, input_layout)
    _, channels, _ = series_bct.shape
    kernel = gaussian_kernel1d(
        kernel_size,
        sigma,
        device=series_bct.device,
        dtype=series_bct.dtype,
    )
    weight = kernel.view(1, 1, kernel_size).repeat(channels, 1, 1)
    pad = kernel_size // 2
    if padding_mode == "reflect" and pad >= series_bct.shape[-1]:
        padding_mode = "replicate"
    padded = F.pad(series_bct, (pad, pad), mode=padding_mode)
    smoothed = F.conv1d(padded, weight, groups=channels)
    return _from_bct(smoothed, original_layout)
