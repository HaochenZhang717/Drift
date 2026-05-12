# Trend-Full Looped Drifting: Implementation Plan

## Goal

This document outlines a code implementation plan for a new time-series Drift variant without modifying the existing Drift training path.

The first version should be intentionally simple:

```text
model(noise) -> (x_trend, x_full)
```

For each real time series `y`, we construct:

```text
y -> (y_trend, y_full)
```

The trend component is obtained with a differentiable Gaussian filter. The full component is the original time series representation.

The training loss applies drifting independently to the two components:

```text
L =
  w_trend(t) * Drift(x_trend, y_trend)
+ w_full(t)  * Drift(x_full,  y_full)
```

The weights can change gradually during training. Early training can emphasize the trend component; later training can shift weight toward the full-resolution component.

## Design Principles

- Do not modify the original Drift implementation path.
- Add new files for the new method.
- Keep the first implementation minimal and debuggable.
- Avoid a complicated general multi-resolution utility for this version.
- Avoid relying on a pretrained self-supervised time-series encoder.
- Make both outputs easy to inspect visually and numerically.

## Proposed New Files

```text
utils/trend_filter.py
losses/trend_full_drifting_loss.py
models/trend_full_looped_unconditional_model.py
train_trend_full_looped_drift.py
sample_trend_full_looped_drift.py
```

The original files should remain unchanged:

```text
benchmarking_drift.py
train_ts_uncond_daily.py
models/unconditional_model.py
drifting.py
```

If later we want to share helper code between the old and new training scripts, we can refactor after the first prototype works.

## Step 1: Gaussian Trend Filter Utility

Create:

```text
utils/trend_filter.py
```

Responsibilities:

- Apply a differentiable Gaussian smoothing filter along the time dimension.
- Preserve the channel dimension.
- Support tensors shaped either `(B, T, C)` or `(B, C, T)`, with explicit layout handling.
- Return the trend component in the same layout as the input.

Initial API:

```python
def gaussian_smooth_series(
    series: torch.Tensor,
    *,
    input_layout: str = "btc",
    kernel_size: int = 15,
    sigma: float = 3.0,
    padding_mode: str = "reflect",
) -> torch.Tensor:
    ...
```

Implementation details:

- Convert to `(B, C, T)` internally.
- Build a 1D Gaussian kernel on the same device and dtype as the input.
- Use depthwise `F.conv1d` with `groups=C`.
- Use reflection padding before convolution to reduce boundary artifacts.
- Convert back to the original layout before returning.

Definitions:

```text
y_full  = y
y_trend = gaussian_smooth_series(y)
```

For generated outputs:

```text
x_full  = model output after loop 2
x_trend = model output after loop 1
```

The generated `x_trend` is supervised against the Gaussian trend of real series. It does not need to be computed by smoothing `x_full`, though an optional consistency loss can later encourage that relationship.

## Step 2: Two-Output Looped Generator

Create:

```text
models/trend_full_looped_unconditional_model.py
```

Start from the existing unconditional `DriftDiT` design, but implement a two-loop transformer core.

Target forward behavior:

```python
outputs = model(noise)

outputs = {
    "trend": x_trend,
    "full": x_full,
}
```

Each output should be a delay image tensor compatible with the current pipeline:

```text
(B, C, H, W)
```

The loss function can then convert these delay images back to time series using the existing `delay_images_to_series` helper.

Suggested architecture:

```text
patch_embed
register tokens
loop block repeated 2 times
output head after loop 1 -> x_trend
output head after loop 2 -> x_full
```

For the first prototype:

- Use `num_loops = 2`.
- Use a loop block with multiple transformer layers, not a single layer.
- Use one shared `final_layer` initially unless this proves unstable.
- Return both outputs during training.
- During sampling/evaluation, use `outputs["full"]` as the final generated sample.

Possible later variants:

- Separate lightweight heads for trend and full outputs.
- More than two loops.
- ELT-style stochastic intermediate loop sampling.
- ELT-style self-distillation from `x_full` to `x_trend`.

## Step 3: Trend-Full Drifting Loss

Create:

```text
losses/trend_full_drifting_loss.py
```

Responsibilities:

- Convert generated delay-image outputs into time-series representations.
- Convert real delay-image samples into time-series representations.
- Construct `y_trend` by applying the Gaussian trend filter to real series.
- Use `x_trend` directly as the generated trend representation.
- Use `x_full` directly as the generated full representation.
- Compute drifting loss separately for trend and full components.
- Apply scheduled component weights.
- Return detailed logging information.

Initial API:

```python
def compute_trend_full_drifting_loss(
    x_outputs: dict[str, torch.Tensor],
    x_pos: torch.Tensor,
    *,
    temperatures: list[float],
    config: dict,
    step: int,
    total_steps: int,
) -> tuple[torch.Tensor, dict]:
    ...
```

Important detail:

Do not concatenate trend and full features into one giant vector for the first version. Compute separate drift losses:

```python
loss_trend = drift(flatten(x_trend), flatten(y_trend))
loss_full  = drift(flatten(x_full),  flatten(y_full))
```

Then combine them explicitly:

```python
loss = w_trend * loss_trend + w_full * loss_full
```

This avoids the full-resolution component dominating pairwise distances due to dimensionality or magnitude.

## Step 4: Scheduled Component Weights

Implement a schedule function in:

```text
losses/trend_full_drifting_loss.py
```

Initial API:

```python
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
    ...
```

Recommended first schedule:

```text
trend: 1.0 -> 0.2
full:  0.2 -> 1.0
```

Use cosine interpolation:

```python
s = 0.5 - 0.5 * cos(pi * progress)
```

Log these values during training:

```text
weight/trend
weight/full
```

## Step 5: Optional Trend Consistency Loss

The first prototype can omit this, but the design should leave room for it.

Risk:

The model may learn two unrelated outputs whose marginal distributions match the real data:

```text
x_trend matches y_trend distribution
x_full matches y_full distribution
```

but `x_trend` may not be the actual trend of `x_full` for the same noise seed.

Possible consistency loss:

```text
x_trend should match gaussian_smooth_series(x_full)
```

Example:

```python
L_cons = mse(
    x_trend_series,
    stopgrad(gaussian_smooth_series(x_full_series)),
)
```

Potential total loss:

```text
L_total =
  L_trend_full_drift
+ mu_cons(t) * L_cons
```

Recommendation:

- Do not enable this in the very first run.
- Add it only if visualizations show that `x_trend` and the smoothed version of `x_full` are not coherent.

## Step 6: New Training Script

Create:

```text
train_trend_full_looped_drift.py
```

Start by copying the structure of `benchmarking_drift.py`, then replace only the model and loss path.

Main differences:

```python
from models.trend_full_looped_unconditional_model import TrendFullLoopedDriftDiT
from losses.trend_full_drifting_loss import compute_trend_full_drifting_loss
```

Training step:

```python
noise = torch.randn(...)
x_outputs = model(noise)
x_pos = queue.sample(n_pos, device)

loss, info = compute_trend_full_drifting_loss(
    x_outputs,
    x_pos,
    temperatures=temperatures,
    config=config,
    step=global_step,
    total_steps=total_steps,
)
```

Evaluation and sampling:

```python
x_gen = x_outputs["full"]
```

Add CLI arguments:

```text
--tf_num_loops
--tf_loop_depth
--tf_trend_kernel_size
--tf_trend_sigma
--tf_weight_schedule
--tf_trend_start
--tf_trend_end
--tf_full_start
--tf_full_end
--tf_consistency_weight
```

For the first version:

```text
--tf_num_loops should default to 2
--tf_consistency_weight should default to 0.0
```

## Step 7: New Sampling Script

Create:

```text
sample_trend_full_looped_drift.py
```

It should support:

```text
--output_component full
--save_all_components
```

When `--save_all_components` is enabled, save visualizations for:

```text
trend
full
gaussian_smooth(full)
```

This is important for diagnosing whether the first loop output behaves like the trend of the final output.

## Step 8: Logging

The new training script should log:

```text
loss/total
loss/trend
loss/full
weight/trend
weight/full
drift_norm/trend
drift_norm/full
true_v_norm/trend
true_v_norm/full
grad_norm
```

If consistency is enabled:

```text
loss/consistency
```

For debugging, periodically save generated samples for both components.

## Risks To Watch

### 1. Trend And Full Outputs May Become Independent

Matching marginal distributions at each component does not guarantee sample-wise consistency.

Mitigation:

- Add trend consistency loss if needed.
- Visualize `x_trend`, `x_full`, and `gaussian_smooth(x_full)` from the same noise seed.

### 2. Trend Loss May Cause Mode Averaging

Trend representations can hide important multimodal structure.

Mitigation:

- Use scheduled weights.
- Keep full-resolution loss active from the beginning.
- Monitor diversity metrics.

### 3. Drift Normalization Makes Component Weights Very Important

The current Drift implementation normalizes each temperature-level drift field. This means explicit component weights will strongly control the contribution of each component.

Mitigation:

- Log per-component losses and weights.
- Sweep component schedules.

### 4. Gaussian Smoothing Can Remove Important Events

The trend component may remove spikes, rare events, or lagged relationships.

Mitigation:

- Use the trend component only as an auxiliary curriculum signal.
- Keep full-resolution drift active throughout training.
- Sweep `kernel_size` and `sigma`.

### 5. Two-Component Drift Increases Compute

Computing drift for trend and full components is more expensive than full-only drift.

Mitigation:

- Profile before scaling to long sequences.
- Consider fewer temperatures for trend if needed.

## Suggested First Prototype

Implement the smallest useful version:

```text
num_loops = 2
outputs = trend, full
y_trend = gaussian_smooth_series(y_full)
y_full = original series
weights = cosine schedule:
  trend: 1.0 -> 0.2
  full:  0.2 -> 1.0
consistency_weight = 0.0
```

Success criterion for the first prototype:

```text
The code trains without touching the original Drift path, logs per-component losses, saves trend/full samples, and allows a direct comparison against the existing baseline.
```
