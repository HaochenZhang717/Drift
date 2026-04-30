import torch
from torch.utils.data import DataLoader

from utils.utils_dataset import get_dataset


DATA_ROOT = "./AI-READI"
CFG = {
    "ts_seq_len": 128,
    "ts_delay": 12,
    "ts_embedding": 12,
    "ts_stride": 128,
    "max_activity_events": 128,
    "require_activity": True,
}


def assert_item_shapes(item, condition_as_image=False):
    assert item["target"].shape == (1, CFG["ts_embedding"], CFG["ts_embedding"])
    if condition_as_image:
        assert item["condition"].shape == (1, CFG["ts_embedding"], CFG["ts_embedding"])
    else:
        assert item["condition"].shape == (CFG["max_activity_events"], 1)

    assert item["condition_mask"].shape == (CFG["max_activity_events"], 1)
    assert item["activity_calorie"].shape == (CFG["max_activity_events"], 1)
    assert item["activity_time_local"].shape == (CFG["max_activity_events"],)
    assert item["activity_mask"].shape == (CFG["max_activity_events"], 1)
    assert item["glucose"].shape == (CFG["ts_seq_len"], 1)
    assert item["glucose_time_local"].shape == (CFG["ts_seq_len"],)
    assert item["activity_length"].ndim == 0
    assert item["raw_activity_length"].ndim == 0
    assert item["activity_interpolated"].ndim == 0


def assert_item_values(item):
    assert torch.isfinite(item["target"]).all()
    assert torch.isfinite(item["condition"]).all()
    assert torch.isfinite(item["glucose"]).all()
    assert torch.isfinite(item["activity_calorie"]).all()

    assert item["target"].min() >= -1.0 and item["target"].max() <= 1.0
    assert item["glucose"].min() >= -1.0 and item["glucose"].max() <= 1.0

    mask = item["activity_mask"].bool().squeeze(-1)
    assert mask.any()
    assert item["activity_calorie"][mask].min() >= -1.0
    assert item["activity_calorie"][mask].max() <= 1.0

    assert torch.all(item["glucose_time_local"][1:] >= item["glucose_time_local"][:-1])
    activity_times = item["activity_time_local"][mask]
    assert torch.all(activity_times[1:] >= activity_times[:-1])
    assert activity_times.min() >= item["glucose_time_local"][0]
    assert activity_times.max() <= item["glucose_time_local"][-1]
    if bool(item["activity_interpolated"]):
        assert int(item["raw_activity_length"].item()) > CFG["max_activity_events"]
        assert torch.equal(activity_times, item["glucose_time_local"][: activity_times.numel()])
    else:
        assert int(item["raw_activity_length"].item()) == int(item["activity_length"].item())


def test_basic_dataset():
    train_dataset, test_dataset = get_dataset("glucose_imputation", CFG, root=DATA_ROOT)
    assert len(train_dataset) > 0
    assert len(test_dataset) > 0

    item = train_dataset[0]
    assert_item_shapes(item)
    assert_item_values(item)

    return train_dataset, test_dataset


def test_dataloader(train_dataset):
    loader = DataLoader(train_dataset, batch_size=8, shuffle=False)
    batch = next(iter(loader))

    assert batch["target"].shape == (8, 1, CFG["ts_embedding"], CFG["ts_embedding"])
    assert batch["condition"].shape == (8, CFG["max_activity_events"], 1)
    assert batch["condition_mask"].shape == (8, CFG["max_activity_events"], 1)
    assert batch["activity_calorie"].shape == (8, CFG["max_activity_events"], 1)
    assert batch["activity_time_local"].shape == (8, CFG["max_activity_events"])
    assert batch["glucose"].shape == (8, CFG["ts_seq_len"], 1)
    assert batch["glucose_time_local"].shape == (8, CFG["ts_seq_len"])
    assert batch["raw_activity_length"].shape == (8,)
    assert batch["activity_interpolated"].shape == (8,)


def test_condition_as_image():
    cfg = dict(CFG)
    cfg["condition_as_image"] = True
    train_dataset, _ = get_dataset("glucose_calorie_imputation", cfg, root=DATA_ROOT)
    item = train_dataset[0]
    assert_item_shapes(item, condition_as_image=True)
    assert_item_values(item)


def test_local_time_window_activity(train_dataset, max_items=64):
    checked = 0
    max_events = 0
    interpolated = 0

    for idx in range(min(max_items, len(train_dataset))):
        item = train_dataset[idx]
        mask = item["activity_mask"].bool().squeeze(-1)
        activity_times = item["activity_time_local"][mask]

        assert activity_times.numel() == int(item["activity_length"].item())
        assert activity_times.numel() > 0
        assert activity_times.min() >= item["glucose_time_local"][0]
        assert activity_times.max() <= item["glucose_time_local"][-1]

        max_events = max(max_events, activity_times.numel())
        interpolated += int(bool(item["activity_interpolated"]))
        checked += 1

    print(
        f"checked {checked} windows; max activity events in condition = {max_events}; "
        f"interpolated windows = {interpolated}"
    )


def _pearsonr(x, y):
    x = x.float()
    y = y.float()
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt((x.square().sum() * y.square().sum()).clamp_min(1e-12))
    return float((x * y).sum() / denom)


def _rankdata(x):
    order = torch.argsort(x)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(x.numel(), dtype=torch.float32)
    return ranks


def analyze_glucose_activity_mean_correlation(dataset, batch_size=256, max_batches=None):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    glucose_means = []
    activity_means = []
    activity_counts = []
    raw_activity_counts = []
    interpolation_flags = []

    for batch_idx, batch in enumerate(loader):
        glucose_mean = batch["glucose"].mean(dim=(1, 2))

        activity_mask = batch["activity_mask"]
        activity_sum = (batch["activity_calorie"] * activity_mask).sum(dim=(1, 2))
        activity_count = activity_mask.sum(dim=(1, 2))
        valid = activity_count > 0

        if valid.any():
            glucose_means.append(glucose_mean[valid])
            activity_means.append(activity_sum[valid] / activity_count[valid])
            activity_counts.append(activity_count[valid])
            raw_activity_counts.append(batch["raw_activity_length"][valid])
            interpolation_flags.append(batch["activity_interpolated"][valid])

        if max_batches is not None and batch_idx + 1 >= max_batches:
            break

    if not glucose_means:
        print("no windows with activity events; cannot compute correlation")
        return

    glucose_means = torch.cat(glucose_means)
    activity_means = torch.cat(activity_means)
    activity_counts = torch.cat(activity_counts)
    raw_activity_counts = torch.cat(raw_activity_counts)
    interpolation_flags = torch.cat(interpolation_flags).bool()

    pearson = _pearsonr(glucose_means, activity_means)
    spearman = _pearsonr(_rankdata(glucose_means), _rankdata(activity_means))

    print("glucose/activity masked mean correlation")
    print(f"  windows used: {glucose_means.numel()}")
    print(f"  pearson r: {pearson:.4f}")
    print(f"  spearman rho: {spearman:.4f}")
    print(
        "  glucose mean normalized: "
        f"mean={glucose_means.mean().item():.4f}, std={glucose_means.std(unbiased=False).item():.4f}"
    )
    print(
        "  activity mean normalized: "
        f"mean={activity_means.mean().item():.4f}, std={activity_means.std(unbiased=False).item():.4f}"
    )
    print(
        "  activity events per window: "
        f"mean={activity_counts.float().mean().item():.2f}, "
        f"min={int(activity_counts.min().item())}, max={int(activity_counts.max().item())}"
    )
    print(
        "  raw activity events per glucose window: "
        f"mean={raw_activity_counts.float().mean().item():.2f}, "
        f"min={int(raw_activity_counts.min().item())}, max={int(raw_activity_counts.max().item())}"
    )
    print(f"  interpolated windows: {int(interpolation_flags.sum().item())}")


def main():
    train_dataset, test_dataset = test_basic_dataset()
    test_dataloader(train_dataset)
    test_condition_as_image()
    test_local_time_window_activity(train_dataset)
    analyze_glucose_activity_mean_correlation(train_dataset)

    print(f"train windows: {len(train_dataset)}")
    print(f"test windows: {len(test_dataset)}")
    print("all glucose imputation dataset checks passed")


if __name__ == "__main__":
    main()
