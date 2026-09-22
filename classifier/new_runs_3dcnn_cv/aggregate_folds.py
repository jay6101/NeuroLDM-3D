"""
Combine per-fold test results into a single all_folds_metrics.json.

For a given variant folder (e.g. runs/runs_500/real_500_syn_25) this reads every
    fold_*/fold_metrics.json
takes each fold's `test_metrics`, and writes, at the variant-folder level,
    all_folds_metrics.json
in the SAME format as Diffusion_paper/classifier/new_runs/.../all_folds_metrics.json:

    {
        "overall_results": {
            "<metric>": {"mean": <float>, "std": <float>},
            ...
        }
    }

It also writes a richer run_summary.json (config-free: per-fold test metrics +
overall mean/std) next to it for convenience.

Usage
-----
# one variant folder
python aggregate_folds.py runs/runs_500/real_500_syn_25

# a whole group (aggregates every variant folder found underneath)
python aggregate_folds.py runs/runs_500

# everything
python aggregate_folds.py runs

# default path is ./runs
python aggregate_folds.py
"""

import argparse
import json
import os

import numpy as np

METRICS = [
    'accuracy', 'auc_roc', 'auc_pr', 'ppv',
    'sensitivity', 'specificity', 'precision', 'recall', 'f1_score',
]


def calculate_overall_metrics(all_folds_metrics):
    """Mean/std of each test metric across folds.

    Identical output to utils.calculate_overall_metrics, but kept dependency-light
    (numpy only) so this aggregator doesn't need torch.
    """
    overall_results = {}
    for metric in METRICS:
        values = [float(fold['test_metrics'][metric])
                  for fold in all_folds_metrics['fold_metrics']
                  if metric in fold['test_metrics']]
        if not values:
            continue
        overall_results[metric] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
        }
    return overall_results


def _fold_number(fold_dir_name):
    """Return the integer in 'fold_<k>' (for numeric sorting); inf if missing."""
    try:
        return int(fold_dir_name.split('_')[-1])
    except (ValueError, IndexError):
        return float('inf')


def find_fold_dirs(variant_dir):
    """Return fold_* subdirectories (with a fold_metrics.json), sorted by fold no."""
    fold_dirs = []
    for name in os.listdir(variant_dir):
        full = os.path.join(variant_dir, name)
        if os.path.isdir(full) and name.startswith('fold_') and \
                os.path.exists(os.path.join(full, 'fold_metrics.json')):
            fold_dirs.append(full)
    return sorted(fold_dirs, key=lambda p: _fold_number(os.path.basename(p)))


def is_variant_dir(path):
    return len(find_fold_dirs(path)) > 0


def aggregate_variant(variant_dir, verbose=True):
    """Aggregate all test folds in one variant folder; write all_folds_metrics.json."""
    fold_dirs = find_fold_dirs(variant_dir)

    fold_metrics_list = []
    per_fold = []
    for fold_dir in fold_dirs:
        with open(os.path.join(fold_dir, 'fold_metrics.json')) as f:
            fm = json.load(f)
        test_metrics = fm.get('test_metrics')
        if not test_metrics:
            if verbose:
                print(f"  [skip] {os.path.basename(fold_dir)}: no test_metrics yet")
            continue
        fold_metrics_list.append({'test_metrics': test_metrics})
        per_fold.append({
            'fold': _fold_number(os.path.basename(fold_dir)),
            'test_metrics': {m: float(test_metrics[m]) for m in METRICS if m in test_metrics},
        })

    if not fold_metrics_list:
        print(f"[!] {variant_dir}: no completed folds with test_metrics -> skipping")
        return None

    overall_results = calculate_overall_metrics({'fold_metrics': fold_metrics_list})

    # reference-format file
    out_path = os.path.join(variant_dir, 'all_folds_metrics.json')
    with open(out_path, 'w') as f:
        json.dump({'overall_results': overall_results}, f, indent=4)

    # richer companion file
    summary = {
        'variant': os.path.basename(variant_dir.rstrip('/')),
        'n_folds_aggregated': len(fold_metrics_list),
        'per_fold': per_fold,
        'overall_results': overall_results,
    }
    with open(os.path.join(variant_dir, 'run_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)

    if verbose:
        auc = overall_results['auc_roc']
        acc = overall_results['accuracy']
        print(f"[ok] {variant_dir}  ({len(fold_metrics_list)} folds)  "
              f"AUC {auc['mean']:.4f}+/-{auc['std']:.4f}  "
              f"Acc {acc['mean']:.4f}+/-{acc['std']:.4f}  -> all_folds_metrics.json")
    return overall_results


def aggregate_path(path):
    """Aggregate a single variant folder, or recurse to find variant folders."""
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        raise SystemExit(f"Not a directory: {path}")

    if is_variant_dir(path):
        aggregate_variant(path)
        return

    # recurse: aggregate every variant folder found underneath
    found = 0
    for root, dirs, _ in os.walk(path):
        if is_variant_dir(root):
            aggregate_variant(root)
            found += 1
            # don't descend into the fold_* subdirs of a variant
            dirs[:] = [d for d in dirs if not d.startswith('fold_')]
    if found == 0:
        print(f"No variant folders (with fold_*/fold_metrics.json) found under {path}")
    else:
        print(f"\nAggregated {found} variant folder(s).")


def main():
    default_runs = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs')
    parser = argparse.ArgumentParser(description="Combine per-fold test results.")
    parser.add_argument('path', nargs='?', default=default_runs,
                        help="variant folder, group folder, or runs root (default: ./runs)")
    args = parser.parse_args()
    aggregate_path(args.path)


if __name__ == "__main__":
    main()
