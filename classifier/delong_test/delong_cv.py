#!/usr/bin/env python3
"""
DeLong analysis over AGGREGATED cross-validation folds (``new_runs_3dcnn_cv``).

The 3D-CNN CV runs differ from ``new_runs``: instead of a single hold-out test
set, every variant is evaluated with 5 test folds

    <group>/<variant>/fold_k/individual_test_performance.csv        (k = 1..5)

The 5 folds *partition* the ~481-sample evaluation pool (each sample is in the
test fold of exactly one split, scored by the model trained for that split), so
concatenating the 5 folds yields one **out-of-fold prediction per sample** over
the whole pool -- i.e. a single pooled score vector of length ~481 per variant.

Because the fold partition is fixed (same ``split_seed``) across every variant,
all variants share the *identical* aggregated test set (same img_paths, same
labels). That is exactly the setting a paired DeLong test needs, so this script:

  1. Concatenates each variant's 5 folds into one pooled prediction set.
  2. Aligns all variants on their shared samples (intersection by img_path).
  3. Computes each variant's pooled AUC + DeLong CI (Sun & Xu 2014), and runs
     the paired DeLong test between every pair of variants
     (pROC::roc.test(method="delong", paired=TRUE)-equivalent).

The DeLong math is imported unchanged from ``delong.py``.

Note on interpretation: the pooled score vector for a variant is produced by 5
different fold checkpoints (one per held-out fold). Pooling out-of-fold
predictions and computing one ROC/AUC over them is the standard CV approach; the
DeLong CI then quantifies test-sampling uncertainty of that pooled AUC. Pairing
across variants is by sample (each sample is scored by the same fold index in
every variant).

Outputs (mirrors delong.py; written to ``--output_dir``):
  single_auc_delong_ci.csv        one row per variant (pooled AUC + DeLong CI)
  fold_auc_summary.csv            per-variant per-fold AUC + mean/std vs pooled
  pairwise_delong_tests.csv       one row per variant pair (paired DeLong)
  pairwise_pvalues__<group>.csv   square matrix of two-sided p-values
  pairwise_zstat__<group>.csv     square matrix of DeLong z statistics
  pairwise_aucdiff__<group>.csv   square matrix of AUC differences (row - col)
  groups.json                     shared-test-set metadata
  aggregated/<group>__<variant>.csv   the pooled per-sample predictions used
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from glob import glob
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from delong import (  # noqa: E402  (local module, see sys.path above)
    delong_auc_cov, auc_delong_ci, paired_delong,
    _holm, _benjamini_hochberg, _path_to_stem,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RUNS = os.path.normpath(
    os.path.join(SCRIPT_DIR, "..", "new_runs_3dcnn_cv", "runs"))
DEFAULT_OUT = os.path.join(SCRIPT_DIR, "results_3dcnn_cv")


# --------------------------------------------------------------------------- #
# Fold discovery / aggregation
# --------------------------------------------------------------------------- #
def find_variant_dirs(runs_dir, csv_name):
    """Return sorted variant dirs containing at least one fold_*/<csv_name>."""
    hits = glob(os.path.join(runs_dir, "**", "fold_*", csv_name), recursive=True)
    variant_dirs = set()
    for csv_path in hits:
        fold_dir = os.path.dirname(csv_path)     # .../<variant>/fold_k
        variant_dir = os.path.dirname(fold_dir)  # .../<variant>
        variant_dirs.add(variant_dir)
    return sorted(variant_dirs)


def _fold_index(fold_dir_name):
    try:
        return int(fold_dir_name.split("_")[-1])
    except (ValueError, IndexError):
        return 1 << 30


def aggregate_variant(variant_dir, csv_name, score_col, label_col, id_col):
    """Concatenate a variant's fold CSVs into one pooled prediction frame.

    Returns (pooled_df, fold_info) where pooled_df has columns
    [id_col, score_col, label_col, fold] and fold_info is a list of
    {fold, n, n_pos, n_neg, auc} dicts (per-fold AUC on that fold alone).
    """
    fold_csvs = sorted(
        glob(os.path.join(variant_dir, "fold_*", csv_name)),
        key=lambda p: _fold_index(os.path.basename(os.path.dirname(p))),
    )
    frames = []
    fold_info = []
    for csv_path in fold_csvs:
        fold_name = os.path.basename(os.path.dirname(csv_path))
        fold_k = _fold_index(fold_name)
        df = pd.read_csv(csv_path)
        missing = {score_col, label_col, id_col} - set(df.columns)
        if missing:
            print(f"    [skip fold] {csv_path}: missing columns {sorted(missing)}")
            continue
        df = df[[id_col, score_col, label_col]].dropna().copy()
        df["fold"] = fold_k
        frames.append(df)

        y = df[label_col].astype(float).to_numpy()
        s = df[score_col].astype(float).to_numpy()
        yb = (y == 1.0).astype(int)
        npos, nneg = int(yb.sum()), int((yb == 0).sum())
        fold_info.append({
            "fold": fold_k, "n": len(df), "n_pos": npos, "n_neg": nneg,
            "auc": float(roc_auc_score(yb, s)) if npos and nneg else float("nan"),
        })

    if not frames:
        return None, []
    pooled = pd.concat(frames, ignore_index=True)
    return pooled, fold_info


def discover_cv_models(runs_dir, csv_name, score_col, label_col, id_col,
                       positive_label, exclude_stems=None):
    """One model per variant, built from pooled out-of-fold predictions."""
    exclude_stems = exclude_stems or set()
    variant_dirs = find_variant_dirs(runs_dir, csv_name)
    models = []
    for variant_dir in variant_dirs:
        model_id = os.path.relpath(variant_dir, runs_dir).replace(os.sep, "/")
        pooled, fold_info = aggregate_variant(
            variant_dir, csv_name, score_col, label_col, id_col)
        if pooled is None:
            print(f"  [skip] {model_id}: no usable fold CSVs")
            continue

        n_folds = pooled["fold"].nunique()

        # Duplicate img_paths across folds would violate the partition assumption.
        n_before = len(pooled)
        pooled = pooled.drop_duplicates(subset=id_col, keep="first")
        n_dupes = n_before - len(pooled)
        if n_dupes:
            print(f"  [warn] {model_id}: dropped {n_dupes} duplicate '{id_col}' "
                  f"row(s) across folds (folds should partition the pool)")

        if exclude_stems:
            keep = ~pooled[id_col].astype(str).map(_path_to_stem).isin(exclude_stems)
            n_excl = int((~keep).sum())
            pooled = pooled[keep]
            if n_excl:
                print(f"  [excl] {model_id}: removed {n_excl} coincidence example(s)")

        ids = pooled[id_col].astype(str).to_numpy()
        scores = pooled[score_col].astype(float).to_numpy()
        labels = (pooled[label_col].astype(float) == float(positive_label)).astype(int).to_numpy()

        n_pos, n_neg = int(labels.sum()), int((labels == 0).sum())
        if n_pos == 0 or n_neg == 0:
            print(f"  [skip] {model_id}: only one class present (pos={n_pos}, neg={n_neg})")
            continue

        models.append({
            "model_id": model_id,
            "variant_dir": variant_dir,
            "n_folds": int(n_folds),
            "fold_info": fold_info,
            "pooled_df": pooled,
            "score_by_id": dict(zip(ids, scores)),
            "label_by_id": dict(zip(ids, labels)),
            "id_set": frozenset(ids),
            "n": len(ids),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "auc_sklearn": float(roc_auc_score(labels, scores)),
        })
        fold_aucs = [fi["auc"] for fi in fold_info]
        print(f"  [ok] {model_id}: {n_folds} folds -> pooled n={len(ids)} "
              f"(pos={n_pos}, neg={n_neg}) pooled AUC={models[-1]['auc_sklearn']:.4f} "
              f"| per-fold AUC mean={np.nanmean(fold_aucs):.4f}")
    return models


# --------------------------------------------------------------------------- #
# Analysis (paired DeLong on the shared aggregated test set)
# --------------------------------------------------------------------------- #
def run_analysis(models, output_dir, conf_level):
    # Align every model on the shared samples (intersection by img_path).
    common = set(models[0]["id_set"])
    for m in models[1:]:
        common &= set(m["id_set"])
    common_ids = sorted(common)
    n_samples = len(common_ids)
    if n_samples < 2:
        print("Models do not share a common test set; cannot run paired stats.")
        return None, None

    # Report any variant that is missing samples relative to the shared set.
    union = set()
    for m in models:
        union |= set(m["id_set"])
    if len(union) != n_samples:
        print(f"  [note] variants differ in coverage: union={len(union)} "
              f"shared={n_samples}. Using the shared {n_samples} samples "
              f"(some variant is missing folds/samples).")
        for m in models:
            miss = len(union) - len(m["id_set"])
            if miss:
                print(f"         {m['model_id']}: missing {miss} sample(s) "
                      f"(n={m['n']}, folds={m['n_folds']})")

    label_vec = np.array([models[0]["label_by_id"][i] for i in common_ids], dtype=int)
    for m in models[1:]:
        lv = np.array([m["label_by_id"][i] for i in common_ids], dtype=int)
        if not np.array_equal(lv, label_vec):
            print(f"  [warn] label mismatch for {m['model_id']}; "
                  f"using first model's labels.")
    n_pos, n_neg = int(label_vec.sum()), int((label_vec == 0).sum())

    score_mat = np.vstack([
        np.array([m["score_by_id"][i] for i in common_ids], dtype=float)
        for m in models
    ])
    aucs, cov = delong_auc_cov(label_vec, score_mat)

    group_label = f"aggregated_n{n_samples}"
    labels_idx = [m["model_id"] for m in models]

    # ---- single-AUC DeLong CIs ----
    single_rows = []
    for idx, m in enumerate(models):
        var = float(cov[idx, idx])
        se, lo, hi = auc_delong_ci(float(aucs[idx]), var, conf_level)
        if abs(aucs[idx] - m["auc_sklearn"]) > 1e-6:
            print(f"  [warn] {m['model_id']}: DeLong AUC {aucs[idx]:.6f} != "
                  f"sklearn AUC {m['auc_sklearn']:.6f}")
        fold_aucs = [fi["auc"] for fi in m["fold_info"]]
        single_rows.append({
            "model": m["model_id"],
            "group": group_label,
            "n_folds": m["n_folds"],
            "n": n_samples,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "auc": float(aucs[idx]),
            "delong_se": se,
            "ci_lower": lo,
            "ci_upper": hi,
            "conf_level": conf_level,
            "fold_auc_mean": float(np.nanmean(fold_aucs)),
            "fold_auc_std": float(np.nanstd(fold_aucs)),
        })

    # ---- pairwise paired DeLong ----
    k = len(models)
    pmat = pd.DataFrame(np.full((k, k), np.nan), index=labels_idx, columns=labels_idx)
    zmat = pmat.copy()
    dmat = pmat.copy()
    np.fill_diagonal(pmat.values, 1.0)
    np.fill_diagonal(zmat.values, 0.0)
    np.fill_diagonal(dmat.values, 0.0)

    pairwise_rows = []
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
        pairwise_rows.append({
            "group": group_label, "model_a": ma, "model_b": mb,
            "auc_a": float(aucs[a]), "auc_b": float(aucs[b]),
            "auc_diff": diff, "delong_z": z, "p_value": p, "n_common": n_samples,
        })

    # Multiple-comparison correction across all variant pairs.
    if pairwise_rows:
        praw = np.array([gp["p_value"] for gp in pairwise_rows])
        p_holm = _holm(praw)
        p_bh = _benjamini_hochberg(praw)
        for gp, ph, pb in zip(pairwise_rows, p_holm, p_bh):
            gp["p_holm"] = float(ph)
            gp["p_fdr_bh"] = float(pb)

    # ---- write outputs ----
    single_df = pd.DataFrame(single_rows).sort_values("auc", ascending=False).reset_index(drop=True)
    single_df.to_csv(os.path.join(output_dir, "single_auc_delong_ci.csv"), index=False)

    if pairwise_rows:
        cols = ["group", "model_a", "model_b", "auc_a", "auc_b", "auc_diff",
                "delong_z", "p_value", "p_holm", "p_fdr_bh", "n_common"]
        pairwise_df = pd.DataFrame(pairwise_rows)[cols].sort_values("p_value").reset_index(drop=True)
    else:
        pairwise_df = pd.DataFrame(columns=["group", "model_a", "model_b"])
    pairwise_df.to_csv(os.path.join(output_dir, "pairwise_delong_tests.csv"), index=False)

    pmat.to_csv(os.path.join(output_dir, f"pairwise_pvalues__{group_label}.csv"))
    zmat.to_csv(os.path.join(output_dir, f"pairwise_zstat__{group_label}.csv"))
    dmat.to_csv(os.path.join(output_dir, f"pairwise_aucdiff__{group_label}.csv"))

    groups_meta = [{
        "group": group_label,
        "n_samples": n_samples,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_models": k,
        "models": labels_idx,
        "example_ids": common_ids[:3],
    }]
    with open(os.path.join(output_dir, "groups.json"), "w") as fh:
        json.dump({"conf_level": conf_level, "groups": groups_meta}, fh, indent=2)

    return single_df, pairwise_df


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs_dir", default=DEFAULT_RUNS,
                    help=f"CV runs directory (default: {DEFAULT_RUNS})")
    ap.add_argument("--output_dir", default=DEFAULT_OUT,
                    help=f"Where results are written (default: {DEFAULT_OUT})")
    ap.add_argument("--csv_name", default="individual_test_performance.csv")
    ap.add_argument("--score_col", default="predicted_score")
    ap.add_argument("--label_col", default="original_label")
    ap.add_argument("--id_col", default="img_path")
    ap.add_argument("--positive_label", type=float, default=1.0)
    ap.add_argument("--conf_level", type=float, default=0.95)
    ap.add_argument("--save_aggregated", action="store_true", default=True,
                    help="dump per-variant pooled predictions to <out>/aggregated/")
    ap.add_argument("--no_save_aggregated", dest="save_aggregated",
                    action="store_false")
    # Optional coincidence filter (same semantics as delong.py / paper_stats.py).
    ap.add_argument("--exclude_csv", default=None,
                    help="CSV listing test examples to drop (test/train coincidences).")
    ap.add_argument("--exclude_stem_col", default="test_stem")
    ap.add_argument("--exclude_mse_col", default="nearest_mse")
    ap.add_argument("--exclude_mse_threshold", type=float, default=1e-4)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    exclude_stems = set()
    if args.exclude_csv:
        ex = pd.read_csv(args.exclude_csv)
        sel = ex[ex[args.exclude_mse_col] < args.exclude_mse_threshold]
        exclude_stems = set(sel[args.exclude_stem_col].astype(str))
        print(f"Excluding {len(exclude_stems)} coincidence example(s) "
              f"({args.exclude_mse_col} < {args.exclude_mse_threshold:g}).")
        sel.sort_values(args.exclude_mse_col).to_csv(
            os.path.join(args.output_dir, "excluded_examples.csv"), index=False)

    print(f"Scanning {args.runs_dir} for variants with 'fold_*/{args.csv_name}' ...")
    models = discover_cv_models(
        args.runs_dir, args.csv_name, args.score_col, args.label_col,
        args.id_col, args.positive_label, exclude_stems=exclude_stems)
    if not models:
        print("No usable CV variants found. Nothing to do.")
        return
    print(f"\nLoaded {len(models)} variant(s).")

    # Per-fold AUC summary (per variant) + optional pooled-prediction dumps.
    fold_rows = []
    if args.save_aggregated:
        agg_dir = os.path.join(args.output_dir, "aggregated")
        os.makedirs(agg_dir, exist_ok=True)
    for m in models:
        fold_aucs = [fi["auc"] for fi in m["fold_info"]]
        for fi in m["fold_info"]:
            fold_rows.append({
                "model": m["model_id"], "fold": fi["fold"], "n": fi["n"],
                "n_pos": fi["n_pos"], "n_neg": fi["n_neg"], "auc": fi["auc"],
            })
        fold_rows.append({
            "model": m["model_id"], "fold": "pooled", "n": m["n"],
            "n_pos": m["n_pos"], "n_neg": m["n_neg"], "auc": m["auc_sklearn"],
        })
        fold_rows.append({
            "model": m["model_id"], "fold": "mean_of_folds", "n": m["n"],
            "n_pos": m["n_pos"], "n_neg": m["n_neg"],
            "auc": float(np.nanmean(fold_aucs)),
        })
        if args.save_aggregated:
            safe = m["model_id"].replace("/", "__")
            m["pooled_df"].to_csv(os.path.join(agg_dir, f"{safe}.csv"), index=False)
    pd.DataFrame(fold_rows).to_csv(
        os.path.join(args.output_dir, "fold_auc_summary.csv"), index=False)

    single_df, pairwise_df = run_analysis(models, args.output_dir, args.conf_level)
    if single_df is None:
        return

    print("\n" + "=" * 74)
    print(f"Variants analysed : {len(models)}")
    print(f"Shared pooled test set : {single_df['n'].iloc[0]} samples "
          f"(pos={single_df['n_pos'].iloc[0]}, neg={single_df['n_neg'].iloc[0]})")
    print(f"Pairwise paired DeLong tests : {len(pairwise_df)}")
    print(f"Results written to : {args.output_dir}")
    print("=" * 74)
    print("\nPooled AUC (DeLong 95% CI)  [mean+/-std over folds]:")
    for _, r in single_df.iterrows():
        print(f"  {r['model']:<26s} AUC={r['auc']:.4f} "
              f"[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]   "
              f"folds {r['fold_auc_mean']:.4f}+/-{r['fold_auc_std']:.4f}")

    if len(pairwise_df):
        sig = pairwise_df[pairwise_df["p_holm"] < 0.05]
        print(f"\nSignificant pairs after Holm (p<0.05): {len(sig)}")
        for _, r in sig.head(15).iterrows():
            print(f"  {r['model_a']:<24s} vs {r['model_b']:<24s} "
                  f"dAUC={r['auc_diff']:+.4f}  z={r['delong_z']:+.2f}  "
                  f"p={r['p_value']:.1e}  Holm={r['p_holm']:.1e}")


if __name__ == "__main__":
    main()
