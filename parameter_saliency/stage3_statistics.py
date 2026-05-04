"""Stage3 pipeline module.

Reads stage2 outputs and emits type-wise histograms and summary CSV.
"""

import csv
import glob
import os

import numpy as np
import torch


TYPE_INFO = [
    ('tp', 'TP'),
    ('fn', 'FN'),
    ('fp_a', 'FP-A'),
    ('fp_b', 'FP-B'),
]


def _collect_saliencies(stage2_paths: list, type_name: str):
    values = []
    for path in stage2_paths:
        data = torch.load(path, map_location='cpu', weights_only=False)
        for rec in data.get(type_name, []):
            sal = rec.get('saliency_norm')
            if sal is not None:
                values.append(sal.float().numpy().ravel())
    if not values:
        return np.array([], dtype=np.float32)
    return np.concatenate(values, axis=0)


def _compute_stats(values: np.ndarray) -> dict:
    if values.size == 0:
        return {
            'n': 0,
            'mean': float('nan'),
            'std': float('nan'),
            'median': float('nan'),
            'p90': float('nan'),
        }
    return {
        'n': int(values.size),
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'median': float(np.median(values)),
        'p90': float(np.percentile(values, 90)),
    }


def _save_histogram(values: np.ndarray, type_label: str, clip: float, bins: int, out_path: str):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        clipped = np.clip(values, -clip, clip) if values.size > 0 else values
        fig, ax = plt.subplots(figsize=(6, 4))
        if clipped.size > 0:
            ax.hist(clipped, bins=bins, density=True, color='steelblue', alpha=0.8, edgecolor='white')
        ax.set_xlabel('Normalized saliency $\\hat{s}_{k,f}$')
        ax.set_ylabel('Density')
        ax.set_title(f'{type_label}  (n={values.size:,})')
        ax.set_xlim(-clip, clip)
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
    except Exception as exc:
        print(f'  [WARN] histogram save failed for {type_label}: {exc}')


def run_stage3_statistics(args):
    if not args.run_dir:
        raise ValueError('--run_dir is required for pipeline_stage=stage3')

    stage2_paths = sorted(glob.glob(os.path.join(args.run_dir, '*', 'stage2_normalized_saliency.pth')))
    if not stage2_paths:
        raise FileNotFoundError(
            f'No stage2_normalized_saliency.pth found under {args.run_dir}. Run Stage2 first.'
        )

    out_dir = os.path.join(args.run_dir, 'stage3_stats')
    os.makedirs(out_dir, exist_ok=True)

    print(f'[Stage3] Found {len(stage2_paths)} stage2 files in {args.run_dir}')
    all_stats = []
    for type_key, type_label in TYPE_INFO:
        values = _collect_saliencies(stage2_paths, type_key)
        stats = _compute_stats(values)
        all_stats.append({'type': type_label, **stats})
        print(
            f'  {type_label:5s}: n={stats["n"]:>8,}  '
            f'mean={stats["mean"]:+.4f}  std={stats["std"]:.4f}  '
            f'median={stats["median"]:+.4f}  p90={stats["p90"]:+.4f}'
        )
        _save_histogram(
            values,
            type_label,
            args.stage3_hist_clip,
            args.stage3_hist_bins,
            os.path.join(out_dir, f'hist_{type_key}.png'),
        )

    csv_path = os.path.join(out_dir, 'summary_stats.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['type', 'n', 'mean', 'std', 'median', 'p90'])
        writer.writeheader()
        writer.writerows(all_stats)
    print(f'[Stage3] Summary CSV -> {csv_path}')
