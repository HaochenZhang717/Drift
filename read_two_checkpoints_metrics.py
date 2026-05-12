import json
from pathlib import Path
from statistics import mean

# 需要读取的 checkpoint / 汇总结果文件
CHECKPOINT_FILES = [
    Path('/Users/zhc/Documents/clinical_results/cls_cond_jit_ckpt_sweep/cls_cond_jit_gap_checkpoint_epoch800_ema2.json'),
    Path('/Users/zhc/Documents/clinical_results/mm_ckpt_sweep/mm_gap_checkpoint_epoch600_ema2.json'),
    Path('/Users/zhc/Documents/clinical_results/cls_cond_drift_ckpt/generated_vs_real_classifier_gap.json'),
]


def load_json(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def get_metric(aggregate: dict, key: str):
    item = aggregate.get(key)
    if not item:
        return None
    return item.get('mean')


def collect_metrics(data: dict):
    agg = data.get('aggregate', {})
    return {
        'generated_accuracy': get_metric(agg, 'generated.acc'),
        'generated_f1': get_metric(agg, 'generated.macro_f1'),
        'generated_recall': get_metric(agg, 'generated.macro_recall'),
        'generated_precision': get_metric(agg, 'generated.macro_precision'),
        'real_accuracy': get_metric(agg, 'real.acc'),
        'real_f1': get_metric(agg, 'real.macro_f1'),
        'real_recall': get_metric(agg, 'real.macro_recall'),
        'real_precision': get_metric(agg, 'real.macro_precision'),
    }


def safe_mean(values):
    vals = [v for v in values if v is not None]
    return mean(vals) if vals else None


def fmt(v):
    return 'N/A' if v is None else f'{v:.6f}'


def main():
    per_file = []

    for fp in CHECKPOINT_FILES:
        data = load_json(fp)
        metrics = collect_metrics(data)
        per_file.append((fp, metrics))

    # 输出每个 checkpoint 的结果
    print('=== Per-checkpoint Metrics (mean) ===')
    for fp, m in per_file:
        print(f'\n[{fp}]')
        print(f"generated: accuracy={fmt(m['generated_accuracy'])}, f1={fmt(m['generated_f1'])}, recall={fmt(m['generated_recall'])}, precision={fmt(m['generated_precision'])}")
        print(f"real     : accuracy={fmt(m['real_accuracy'])}, f1={fmt(m['real_f1'])}, recall={fmt(m['real_recall'])}, precision={fmt(m['real_precision'])}")

    # 计算所有文件的平均值
    keys = list(per_file[0][1].keys()) if per_file else []
    avg_metrics = {k: safe_mean([m[k] for _, m in per_file]) for k in keys}

    print(f'\n=== Average of the {len(per_file)} files (mean of means) ===')
    print(f"generated: accuracy={fmt(avg_metrics['generated_accuracy'])}, f1={fmt(avg_metrics['generated_f1'])}, recall={fmt(avg_metrics['generated_recall'])}, precision={fmt(avg_metrics['generated_precision'])}")
    print(f"real     : accuracy={fmt(avg_metrics['real_accuracy'])}, f1={fmt(avg_metrics['real_f1'])}, recall={fmt(avg_metrics['real_recall'])}, precision={fmt(avg_metrics['real_precision'])}")


if __name__ == '__main__':
    main()
