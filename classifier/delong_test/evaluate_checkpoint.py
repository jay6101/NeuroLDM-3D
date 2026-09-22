#!/usr/bin/env python3
"""
Evaluate a trained classifier checkpoint on a given test CSV and regenerate
``individual_test_performance.csv`` and ``fold_metrics.json``.

This reuses the project's own ``dataset.MRIDataset`` and ``utils.validate`` so the
preprocessing, scoring (sigmoid), threshold (0.5), metrics and output columns are
identical to what ``train.py`` produced during training -- only the test set
changes.

Existing output files are backed up (``*.bak_<tag>``) before being overwritten.

Example
-------
python delong_test/evaluate_checkpoint.py \
    --checkpoint new_runs/runs_syn/syn_2723/fold_1/best_model.pth \
    --test_csv   /space/mcdonald-syn01/1/projects/jsawant/Diffusion_paper/data_csvs/test.csv \
    --output_dir new_runs/runs_syn/syn_2723/fold_1 \
    --dropout_rate 0.5
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Make the classifier package importable regardless of CWD.
CLASSIFIER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
if CLASSIFIER_DIR not in sys.path:
    sys.path.insert(0, CLASSIFIER_DIR)

from dataset import MRIDataset                         # noqa: E402
from utils import validate, save_fold_metrics          # noqa: E402
from model.efficientNetV2 import MRIClassifier          # noqa: E402

LABEL_COL = "HC_vs_LTLE_vs_RTLE_string"
KEEP_CLASSES = ["right", "left", "HC"]


def _load_state_dict(path, map_location="cpu"):
    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # older torch without weights_only kwarg
        ckpt = torch.load(path, map_location=map_location)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    return ckpt


def _backup(path, tag):
    if os.path.exists(path):
        bak = f"{path}.bak_{tag}"
        if not os.path.exists(bak):           # never clobber the true original
            shutil.copy2(path, bak)
            print(f"  backed up {os.path.basename(path)} -> {os.path.basename(bak)}")
        else:
            print(f"  backup already exists, leaving it intact: {os.path.basename(bak)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--output_dir", required=True,
                    help="Where to write individual_test_performance.csv / fold_metrics.json")
    ap.add_argument("--dropout_rate", type=float, default=0.5)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--backup_tag", default=None,
                    help="Suffix for backups of overwritten files (default: timestamp)")
    args = ap.parse_args()

    tag = args.backup_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    hyperparams = {"device": device, "batch_size": args.batch_size,
                   "num_workers": args.num_workers}

    print(f"Device         : {device}")
    print(f"Checkpoint     : {args.checkpoint}")
    print(f"Test CSV       : {args.test_csv}")

    # Build the test dataset/loader (train=False -> no augmentation, no synthetic).
    test_df = pd.read_csv(args.test_csv)
    test_df = test_df.loc[test_df[LABEL_COL].isin(KEEP_CLASSES)]
    test_dataset = MRIDataset(test_df, hyperparams, train=False)
    print(f"Test samples   : {len(test_dataset)} "
          f"(HC={int((test_df[LABEL_COL]=='HC').sum())}, "
          f"left={int((test_df[LABEL_COL]=='left').sum())}, "
          f"right={int((test_df[LABEL_COL]=='right').sum())})")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # Load model + checkpoint.
    print("Loading model (efficientNetV2) ...")
    model = MRIClassifier(dropout_rate=args.dropout_rate).to(device)
    model.load_state_dict(_load_state_dict(args.checkpoint, map_location="cpu"))
    model.eval()

    # Inference (identical code path to train.py's test phase).
    print("Running inference ...")
    criterion = nn.BCEWithLogitsLoss()
    test_metrics, individual = validate(
        model, test_loader, criterion, hyperparams,
        fixed_threshold=args.threshold, return_individual_results=True,
    )

    # Performance labels (TP/FP/FN/TN), exactly as train.py builds them.
    perf = []
    for t, p in zip(individual["labels"], individual["predictions"]):
        if t == 1 and p == 1:
            perf.append("TP")
        elif t == 0 and p == 1:
            perf.append("FP")
        elif t == 1 and p == 0:
            perf.append("FN")
        else:
            perf.append("TN")

    results_df = pd.DataFrame({
        "img_path": individual["paths"],
        "predicted_score": individual["scores"],
        "original_label": individual["labels"],
        "predicted_label": individual["predictions"],
        "performance_type": perf,
    })

    csv_path = os.path.join(args.output_dir, "individual_test_performance.csv")
    _backup(csv_path, tag)
    results_df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}  ({len(results_df)} rows)")

    # fold_metrics.json: preserve existing training history if present, only
    # refresh the test_metrics block (computed on the new test set).
    test_metrics["threshold_used"] = args.threshold
    test_metrics["test_csv"] = os.path.abspath(args.test_csv)

    fm_path = os.path.join(args.output_dir, "fold_metrics.json")
    if os.path.exists(fm_path):
        with open(fm_path) as fh:
            fold_metrics = json.load(fh)
        _backup(fm_path, tag)
    else:
        fold_metrics = {}
    fold_metrics["test_metrics"] = test_metrics
    save_fold_metrics(fold_metrics, args.output_dir)
    print(f"Wrote {fm_path}")

    print("\nTest metrics on this test set:")
    for k in ["accuracy", "auc_roc", "auc_pr", "sensitivity", "specificity", "f1_score"]:
        if k in test_metrics:
            print(f"  {k:12s}: {test_metrics[k]:.4f}")


if __name__ == "__main__":
    main()
