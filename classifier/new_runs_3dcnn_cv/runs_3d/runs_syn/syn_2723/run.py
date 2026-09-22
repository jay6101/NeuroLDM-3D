"""
Driver for the 5-fold CV experiments (3D or 2D CNN).

Each variant trains the chosen architecture (--model 3d|2d) with 5-fold
cross-validation over the combined val+test pool (fixed train set). The
training recipe is identical for both architectures; only the model and the
dataset input shape differ. 3D results go under runs/, 2D under runs_2d/:
    <runs-dir>/<group>/<variant>/
        parameters.json
        all_folds_metrics.json          (per-fold test metrics + overall mean/std)
        fold_1/ ... fold_5/
            train.csv  val.csv  test.csv
            best_model.pth
            fold_metrics.json
            individual_test_performance.csv

Examples
--------
# List every variant that would run
python run.py --list

# Quick end-to-end smoke test (1 epoch, 1 fold, tiny synth) on one variant
python run.py --variants real_500 --epochs 1 --max-folds 1 --device cuda:0

# Train the 2D EfficientNetV2 instead of the 3D CNN (-> runs_2d/)
python run.py --model 2d --groups runs_1000 --device cuda:0 --skip-existing

# Run a whole group on GPU 0, skipping variants already finished
python run.py --groups runs_1000 --device cuda:0 --skip-existing

# Run everything (default = 3d)
python run.py --device cuda:0
"""

import argparse
import os
import traceback

import torch

import config
from train import train_model


def select_variants(args):
    variants = config.VARIANTS
    if args.variants:
        wanted = {v.strip() for v in args.variants.split(",")}
        variants = [v for v in variants if v['name'] in wanted]
        missing = wanted - {v['name'] for v in variants}
        if missing:
            raise SystemExit(f"Unknown variant(s): {sorted(missing)}\n"
                             f"Available: {config.get_variant_names()}")
    elif args.groups:
        wanted = {g.strip() for g in args.groups.split(",")}
        variants = [v for v in variants if v['group'] in wanted]
        missing = wanted - {v['group'] for v in variants}
        if missing:
            raise SystemExit(f"Unknown group(s): {sorted(missing)}\n"
                             f"Available: {config.get_groups()}")
    return variants


def build_hyperparams(variant, args):
    hp = config.build_base_hyperparams(args.model)
    hp['device'] = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    hp['num_samples'] = variant['num_samples']
    hp['num_synth_samples'] = variant['num_synth_samples']
    hp['variant_name'] = variant['name']
    hp['group'] = variant['group']
    hp['run_dir'] = os.path.join(args.runs_dir, variant['group'], variant['name'])

    # CLI overrides (handy for quick tests)
    if args.epochs is not None:
        hp['num_epochs'] = args.epochs
    if args.n_folds is not None:
        hp['n_folds'] = args.n_folds
    if args.max_folds is not None:
        hp['max_folds'] = args.max_folds
    if args.patience is not None:
        hp['early_stopping_patience'] = args.patience
    if args.num_workers is not None:
        hp['num_workers'] = args.num_workers
    return hp


def main():
    parser = argparse.ArgumentParser(description="5-fold CV runner (3D or 2D CNN)")
    parser.add_argument('--model', default=config.DEFAULT_MODEL, choices=config.model_choices(),
                        help="architecture: 3d (cnn_ben_3d) or 2d (efficientNetV2). "
                             "Training recipe is identical for both.")
    parser.add_argument('--device', default='cuda:0', help="cuda:0 / cuda:1 / cpu")
    parser.add_argument('--groups', default=None,
                        help="comma-separated groups (e.g. runs_500,runs_1000)")
    parser.add_argument('--variants', default=None,
                        help="comma-separated variant names (overrides --groups)")
    parser.add_argument('--runs-dir', default=None,
                        help="output directory root (default: runs/ for 3d, runs_2d/ for 2d)")
    parser.add_argument('--epochs', type=int, default=None, help="override num_epochs")
    parser.add_argument('--n-folds', type=int, default=None,
                        help="override n_folds (StratifiedKFold splits, >=2)")
    parser.add_argument('--max-folds', type=int, default=None,
                        help="train only the first N folds (quick tests; geometry unchanged)")
    parser.add_argument('--patience', type=int, default=None,
                        help="override early_stopping_patience")
    parser.add_argument('--num-workers', type=int, default=None, help="override num_workers")
    parser.add_argument('--skip-existing', action='store_true',
                        help="skip variants that already have all_folds_metrics.json")
    parser.add_argument('--list', action='store_true', help="list variants and exit")
    parser.add_argument('--dry-run', action='store_true',
                        help="print what would run without training")
    args = parser.parse_args()

    if args.runs_dir is None:
        args.runs_dir = config.default_runs_dir(args.model)

    variants = select_variants(args)

    if args.list:
        print(f"{'GROUP':12s} {'VARIANT':22s} {'num_samples':>12s} {'num_synth':>10s}")
        for v in variants:
            print(f"{v['group']:12s} {v['name']:22s} "
                  f"{str(v['num_samples']):>12s} {str(v['num_synth_samples']):>10s}")
        print(f"\nTotal: {len(variants)} variants")
        return

    print(f"Model: {args.model} ({config.MODELS[args.model]['model_name']})")
    print(f"Selected {len(variants)} variant(s); output -> {args.runs_dir}")

    results = {}
    for v in variants:
        hp = build_hyperparams(v, args)
        done_marker = os.path.join(hp['run_dir'], 'all_folds_metrics.json')
        if args.skip_existing and os.path.exists(done_marker):
            print(f"[skip] {v['group']}/{v['name']} (already has all_folds_metrics.json)")
            continue

        if args.dry_run:
            print(f"[dry-run] would train {v['group']}/{v['name']} "
                  f"(num_samples={v['num_samples']}, num_synth_samples={v['num_synth_samples']}) "
                  f"-> {hp['run_dir']}")
            continue

        try:
            summary = train_model(hp, hp['model_name'])
            results[v['name']] = summary['overall_results']['auc_roc']['mean']
        except Exception:
            print(f"[ERROR] variant {v['group']}/{v['name']} failed:")
            traceback.print_exc()

    if results:
        print("\n================ AUC (mean over folds) ================")
        for name, auc in results.items():
            print(f"{name:24s}: {auc:.4f}")


if __name__ == "__main__":
    main()
