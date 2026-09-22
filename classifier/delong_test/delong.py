#!/usr/bin/env python3
"""
DeLong analysis of ROC AUCs over already-saved classifier predictions.

For every ``individual_test_performance.csv`` found under a runs directory this
script:

  1. Computes the ROC AUC and a DeLong confidence interval for each model
     (equivalent to R: ``pROC::ci.auc(roc_obj, method = "delong")``).
  2. Groups models that were evaluated on the *exact same* test set (matched by
     image path) and runs the paired DeLong test between every pair of models in
     a group (equivalent to R:
     ``pROC::roc.test(roc1, roc2, method = "delong", paired = TRUE)``).

Only the per-sample scores already stored on disk are used, so no GPU / model
re-evaluation is required. This captures variability in *testing* (the DeLong
asymptotic theory), not the variability of the whole training process.

The fast DeLong covariance follows Sun & Xu (2014), "Fast Implementation of
DeLong's Algorithm for Comparing the Areas Under Correlated Receiver Operating
Characteristic Curves", IEEE SPL. The midrank implementation mirrors the widely
used reference at https://github.com/yandexdataschool/roc_comparison (MIT).

Outputs (written to ``--output_dir``):
  single_auc_delong_ci.csv        one row per model (AUC + DeLong CI)
  pairwise_delong_tests.csv       one row per comparable model pair
  pairwise_pvalues__<group>.csv   square matrix of two-sided p-values
  pairwise_zstat__<group>.csv     square matrix of DeLong z statistics
  pairwise_aucdiff__<group>.csv   square matrix of AUC differences (row - col)
  groups.json                     which models share an identical test set
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from glob import glob
from itertools import combinations
from statistics import NormalDist

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

_NORM = NormalDist()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RUNS = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "new_runs"))
DEFAULT_OUT = os.path.join(SCRIPT_DIR, "results")


# --------------------------------------------------------------------------- #
# Fast DeLong (Sun & Xu, 2014)
# --------------------------------------------------------------------------- #
def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Midranks of ``x`` (ties receive the average rank), 1-indexed."""
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    ranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def _fast_delong(predictions_sorted: np.ndarray, m: int):
    """Core DeLong computation.

    Parameters
    ----------
    predictions_sorted : array of shape [k, N]
        Scores for ``k`` classifiers on ``N`` samples, with the ``m`` positive
        samples occupying the first ``m`` columns and negatives the rest.
    m : int
        Number of positive samples.

    Returns
    -------
    aucs : array [k]
    cov : array [k, k]
        DeLong covariance matrix of the AUC estimates.
    """
    n = predictions_sorted.shape[1] - m
    k = predictions_sorted.shape[0]
    positive = predictions_sorted[:, :m]
    negative = predictions_sorted[:, m:]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(positive[r, :])
        ty[r, :] = _compute_midrank(negative[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n          # placement values on positives
    v10 = 1.0 - (tz[:, m:] - ty) / m    # placement values on negatives
    sx = np.cov(v01)
    sy = np.cov(v10)
    cov = sx / m + sy / n
    return aucs, np.atleast_2d(cov)


def delong_auc_cov(y: np.ndarray, scores: np.ndarray):
    """AUCs and DeLong covariance for ``scores`` [k, N] given labels ``y`` [N]."""
    order = (-y).argsort(kind="mergesort")   # positives (label 1) first, stable
    m = int(y.sum())
    return _fast_delong(scores[:, order], m)


# --------------------------------------------------------------------------- #
# Statistics helpers
# --------------------------------------------------------------------------- #
def auc_delong_ci(auc: float, var: float, conf_level: float = 0.95):
    """DeLong CI for a single AUC, clamped to [0, 1] like pROC::ci.auc."""
    se = float(np.sqrt(max(var, 0.0)))
    alpha = 1.0 - conf_level
    z_lo = _NORM.inv_cdf(alpha / 2.0)
    z_hi = _NORM.inv_cdf(1.0 - alpha / 2.0)
    lower = min(max(auc + z_lo * se, 0.0), 1.0)
    upper = min(max(auc + z_hi * se, 0.0), 1.0)
    return se, lower, upper


def paired_delong(auc1, auc2, var1, var2, cov12):
    """Paired DeLong test statistic and two-sided p-value (pROC::roc.test)."""
    var_diff = var1 + var2 - 2.0 * cov12
    if var_diff <= 0:
        if np.isclose(auc1, auc2):
            return 0.0, 1.0
        return float(np.sign(auc1 - auc2) * np.inf), 0.0
    z = (auc1 - auc2) / np.sqrt(var_diff)
    p = 2.0 * _NORM.cdf(-abs(z))
    return float(z), float(p)


def _benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(n) + 1.0)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(adj, 0.0, 1.0)
    return out


def _holm(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * (n - np.arange(n))
    adj = np.maximum.accumulate(adj)
    out = np.empty(n)
    out[order] = np.clip(adj, 0.0, 1.0)
    return out


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def _path_to_stem(path):
    """Filename stem of an image path (basename without .nii / .nii.gz)."""
    base = os.path.basename(str(path))
    for ext in (".nii.gz", ".nii"):
        if base.endswith(ext):
            return base[: -len(ext)]
    return os.path.splitext(base)[0]


def discover_models(runs_dir, csv_name, score_col, label_col, id_col, positive_label,
                    exclude_stems=None):
    csv_paths = sorted(glob(os.path.join(runs_dir, "**", csv_name), recursive=True))
    exclude_stems = exclude_stems or set()
    models = []
    for csv_path in csv_paths:
        fold_dir = os.path.dirname(csv_path)        # .../<condition>/fold_x
        cond_dir = os.path.dirname(fold_dir)        # .../<condition>
        model_id = os.path.relpath(cond_dir, runs_dir).replace(os.sep, "/")

        df = pd.read_csv(csv_path)
        missing = {score_col, label_col, id_col} - set(df.columns)
        if missing:
            print(f"  [skip] {model_id}: missing columns {sorted(missing)}")
            continue

        df = df[[id_col, score_col, label_col]].dropna()
        n_before = len(df)
        df = df.drop_duplicates(subset=id_col, keep="first")
        n_dupes = n_before - len(df)
        if n_dupes:
            print(f"  [warn] {model_id}: dropped {n_dupes} duplicate '{id_col}' row(s)")

        if exclude_stems:
            keep = ~df[id_col].astype(str).map(_path_to_stem).isin(exclude_stems)
            n_excl = int((~keep).sum())
            df = df[keep]
            if n_excl:
                print(f"  [excl] {model_id}: removed {n_excl} coincidence example(s)")

        ids = df[id_col].astype(str).to_numpy()
        scores = df[score_col].astype(float).to_numpy()
        labels = (df[label_col].astype(float) == float(positive_label)).astype(int).to_numpy()

        n_pos, n_neg = int(labels.sum()), int((labels == 0).sum())
        if n_pos == 0 or n_neg == 0:
            print(f"  [skip] {model_id}: only one class present (pos={n_pos}, neg={n_neg})")
            continue

        models.append({
            "model_id": model_id,
            "csv_path": csv_path,
            "score_by_id": dict(zip(ids, scores)),
            "label_by_id": dict(zip(ids, labels)),
            "id_set": frozenset(ids),
            "n": len(ids),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "auc_sklearn": float(roc_auc_score(labels, scores)),
        })
    return models


# --------------------------------------------------------------------------- #
# Main analysis
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs_dir", default=DEFAULT_RUNS,
                    help=f"Directory searched recursively for the CSVs (default: {DEFAULT_RUNS})")
    ap.add_argument("--output_dir", default=DEFAULT_OUT,
                    help=f"Where results are written (default: {DEFAULT_OUT})")
    ap.add_argument("--csv_name", default="individual_test_performance.csv")
    ap.add_argument("--score_col", default="predicted_score")
    ap.add_argument("--label_col", default="original_label")
    ap.add_argument("--id_col", default="img_path")
    ap.add_argument("--positive_label", type=float, default=1.0)
    ap.add_argument("--conf_level", type=float, default=0.95)
    # Optional: drop test examples that coincide with training data.
    ap.add_argument("--exclude_csv", default=None,
                    help="CSV listing test examples to drop (e.g. test_train_coincidence.csv).")
    ap.add_argument("--exclude_stem_col", default="test_stem",
                    help="Column in --exclude_csv holding the image filename stem.")
    ap.add_argument("--exclude_mse_col", default="nearest_mse",
                    help="Numeric column used with --exclude_mse_threshold.")
    ap.add_argument("--exclude_mse_threshold", type=float, default=1e-4,
                    help="Drop rows of --exclude_csv with <col> below this value.")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Build the set of image stems to exclude (test/train coincidences).
    exclude_stems = set()
    if args.exclude_csv:
        ex = pd.read_csv(args.exclude_csv)
        sel = ex[ex[args.exclude_mse_col] < args.exclude_mse_threshold]
        exclude_stems = set(sel[args.exclude_stem_col].astype(str))
        print(f"Excluding {len(exclude_stems)} example(s) with "
              f"{args.exclude_mse_col} < {args.exclude_mse_threshold:g} "
              f"(from {os.path.basename(args.exclude_csv)})")
        sel.sort_values(args.exclude_mse_col).to_csv(
            os.path.join(args.output_dir, "excluded_examples.csv"), index=False)

    print(f"Scanning {args.runs_dir} for '{args.csv_name}' ...")
    models = discover_models(args.runs_dir, args.csv_name, args.score_col,
                             args.label_col, args.id_col, args.positive_label,
                             exclude_stems=exclude_stems)
    if not models:
        print("No usable prediction CSVs found. Nothing to do.")
        return
    print(f"Loaded {len(models)} models.\n")

    # Group models that share an identical test set (matched by image path).
    by_testset = defaultdict(list)
    for mdl in models:
        by_testset[mdl["id_set"]].append(mdl)
    groups = sorted(by_testset.values(), key=lambda g: (-len(next(iter(g))["id_set"]), -len(g)))

    single_rows = []
    pairwise_rows = []
    groups_meta = []

    for gi, group in enumerate(groups, start=1):
        common_ids = sorted(group[0]["id_set"])
        label_vec = np.array([group[0]["label_by_id"][i] for i in common_ids], dtype=int)
        n_samples = len(common_ids)
        n_pos, n_neg = int(label_vec.sum()), int((label_vec == 0).sum())
        group_label = f"group{gi:02d}_n{n_samples}"

        # Verify label consistency across models in the group.
        for mdl in group[1:]:
            lv = np.array([mdl["label_by_id"][i] for i in common_ids], dtype=int)
            if not np.array_equal(lv, label_vec):
                print(f"  [warn] {group_label}: label mismatch for model "
                      f"{mdl['model_id']}; using first model's labels.")

        score_mat = np.vstack([
            np.array([mdl["score_by_id"][i] for i in common_ids], dtype=float)
            for mdl in group
        ])
        aucs, cov = delong_auc_cov(label_vec, score_mat)

        # Single-AUC DeLong confidence intervals.
        for idx, mdl in enumerate(group):
            var = float(cov[idx, idx])
            se, lo, hi = auc_delong_ci(float(aucs[idx]), var, args.conf_level)
            if abs(aucs[idx] - mdl["auc_sklearn"]) > 1e-6:
                print(f"  [warn] {mdl['model_id']}: DeLong AUC {aucs[idx]:.6f} != "
                      f"sklearn AUC {mdl['auc_sklearn']:.6f}")
            single_rows.append({
                "model": mdl["model_id"],
                "group": group_label,
                "n": mdl["n"],
                "n_pos": mdl["n_pos"],
                "n_neg": mdl["n_neg"],
                "auc": float(aucs[idx]),
                "delong_se": se,
                "ci_lower": lo,
                "ci_upper": hi,
                "conf_level": args.conf_level,
            })

        groups_meta.append({
            "group": group_label,
            "n_samples": n_samples,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "n_models": len(group),
            "models": [m["model_id"] for m in group],
            "example_ids": common_ids[:3],
        })

        # Pairwise paired DeLong tests within the group.
        k = len(group)
        if k < 2:
            print(f"{group_label}: {k} model, {n_samples} samples "
                  f"(pos={n_pos}, neg={n_neg}) - no pair to compare.")
            continue
        print(f"{group_label}: {k} models, {n_samples} samples "
              f"(pos={n_pos}, neg={n_neg}) -> {k*(k-1)//2} pairwise test(s).")

        labels_idx = [m["model_id"] for m in group]
        pmat = pd.DataFrame(np.eye(k) * 0.0 + np.nan, index=labels_idx, columns=labels_idx)
        zmat = pmat.copy()
        dmat = pmat.copy()
        np.fill_diagonal(pmat.values, 1.0)
        np.fill_diagonal(zmat.values, 0.0)
        np.fill_diagonal(dmat.values, 0.0)

        group_pairs = []
        for a, b in combinations(range(k), 2):
            z, p = paired_delong(float(aucs[a]), float(aucs[b]),
                                 float(cov[a, a]), float(cov[b, b]), float(cov[a, b]))
            diff = float(aucs[a] - aucs[b])
            ma, mb = labels_idx[a], labels_idx[b]
            pmat.loc[ma, mb] = p
            pmat.loc[mb, ma] = p
            zmat.loc[ma, mb] = z
            zmat.loc[mb, ma] = -z
            dmat.loc[ma, mb] = diff
            dmat.loc[mb, ma] = -diff
            group_pairs.append({
                "group": group_label,
                "model_a": ma,
                "model_b": mb,
                "auc_a": float(aucs[a]),
                "auc_b": float(aucs[b]),
                "auc_diff": diff,
                "delong_z": z,
                "p_value": p,
                "n_common": n_samples,
            })

        # Multiple-comparison correction within this family of tests.
        praw = np.array([gp["p_value"] for gp in group_pairs])
        p_holm = _holm(praw)
        p_bh = _benjamini_hochberg(praw)
        for gp, ph, pb in zip(group_pairs, p_holm, p_bh):
            gp["p_holm"] = float(ph)
            gp["p_fdr_bh"] = float(pb)
        pairwise_rows.extend(group_pairs)

        safe = group_label
        pmat.to_csv(os.path.join(args.output_dir, f"pairwise_pvalues__{safe}.csv"))
        zmat.to_csv(os.path.join(args.output_dir, f"pairwise_zstat__{safe}.csv"))
        dmat.to_csv(os.path.join(args.output_dir, f"pairwise_aucdiff__{safe}.csv"))

    # Write top-level tables.
    single_df = pd.DataFrame(single_rows).sort_values(
        ["group", "auc"], ascending=[True, False]).reset_index(drop=True)
    single_df.to_csv(os.path.join(args.output_dir, "single_auc_delong_ci.csv"), index=False)

    if pairwise_rows:
        cols = ["group", "model_a", "model_b", "auc_a", "auc_b", "auc_diff",
                "delong_z", "p_value", "p_holm", "p_fdr_bh", "n_common"]
        pairwise_df = pd.DataFrame(pairwise_rows)[cols].sort_values(
            ["group", "p_value"]).reset_index(drop=True)
    else:
        pairwise_df = pd.DataFrame(columns=["group", "model_a", "model_b"])
    pairwise_df.to_csv(os.path.join(args.output_dir, "pairwise_delong_tests.csv"), index=False)

    with open(os.path.join(args.output_dir, "groups.json"), "w") as fh:
        json.dump({"conf_level": args.conf_level, "groups": groups_meta}, fh, indent=2)

    print("\n" + "=" * 70)
    print(f"Models analysed       : {len(models)}")
    print(f"Comparable test groups: {len(groups)}")
    print(f"Pairwise tests        : {len(pairwise_rows)}")
    print(f"Results written to    : {args.output_dir}")
    print("=" * 70)
    print("\nTop AUCs (DeLong CI):")
    show = single_df.head(10)
    for _, r in show.iterrows():
        print(f"  {r['model']:<45s} AUC={r['auc']:.4f} "
              f"[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]  ({r['group']})")


if __name__ == "__main__":
    main()
