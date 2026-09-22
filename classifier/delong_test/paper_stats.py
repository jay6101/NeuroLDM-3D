#!/usr/bin/env python3
"""
Publication-oriented significance analysis for a *fixed* set of classifier
checkpoints evaluated on a single hold-out test set.

Only the per-sample scores already on disk are used (no retraining, no new
data). Everything quantifies *test-set sampling* uncertainty for fixed models,
which is the standard inference target for a deployed clinical classifier
(cf. DeLong 1988; Sun & Xu 2014; pROC).

It complements ``delong.py`` with the tools a reviewer asking for "statistical
significance" on a small test set actually wants:

  1. Paired, class-stratified bootstrap CIs for each AUC and for every
     pre-specified AUC *difference* (Efron & Tibshirani 1993).
  2. Equivalence (two one-sided tests, TOST; Schuirmann 1987; Lakens 2017) and
     one-sided non-inferiority tests against a pre-specified AUC margin -- so a
     "no significant difference" result becomes a *positive* equivalence claim.
  3. McNemar's exact paired test (McNemar 1947) on the classifiers' decisions at
     a fixed operating threshold.
  4. Power / minimum-detectable-effect (MDE) for every comparison, to state
     honestly what the test set can and cannot resolve.

Comparisons are restricted to a small, hypothesis-driven *primary family* (not
all pairs), which keeps the multiple-comparison penalty mild. Holm and
Benjamini-Hochberg adjustments are reported within each family.

Outputs (``--output_dir``):
  primary_comparisons.csv   one row per pre-specified comparison (all stats)
  bootstrap_auc_ci.csv      per-model AUC with DeLong and bootstrap CIs
  power_summary.csv         MDE and achieved power per comparison
  forest_auc.png            AUC + 95% DeLong CI, all models
  equivalence_deltas.png    delta-AUC + CI vs the equivalence margin
  paper_stats_summary.json  machine-readable digest
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from math import comb, sqrt
from statistics import NormalDist

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from delong import (  # noqa: E402  (local module, see sys.path above)
    discover_models, delong_auc_cov, auc_delong_ci, _holm, _benjamini_hochberg,
)

_NORM = NormalDist()
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RUNS = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "new_runs"))
DEFAULT_OUT = os.path.join(SCRIPT_DIR, "results_no_coincidence", "paper_significance")
DEFAULT_EXCLUDE = ("/space/mcdonald-syn01/1/projects/jsawant/Diffusion_paper/"
                   "diffusion/nn_eval/results/test_train_coincidence.csv")

REAL_BUDGETS = [500, 1000, 2000, 2723]
SYNTH_PCTS = [25, 50, 75, 100]
SYNTH_ONLY = [500, 1000, 2000, 2723, 5446]


# --------------------------------------------------------------------------- #
# Model id helpers (match the folder layout under new_runs/)
# --------------------------------------------------------------------------- #
def real_only(n):
    return f"runs_{n}/real_{n}"


def mix(n, pct):
    return f"runs_{n}/real_{n}_syn_{pct}"


def synth_only(m):
    return f"runs_syn/syn_{m}"


def pretty(model_id):
    tail = model_id.split("/", 1)[-1]
    if tail.startswith("run_") and "efficientNetV2" in tail:
        return "real2723 (baseline)"
    if tail.startswith("real_") and "_syn_" in tail:
        n, pct = tail.replace("real_", "").split("_syn_")
        return f"real{n}+syn{pct}%"
    if tail.startswith("real_"):
        return f"real{tail.replace('real_', '')}"
    if tail.startswith("syn_"):
        return f"syn{tail.replace('syn_', '')}"
    return tail


def kind(model_id):
    tail = model_id.split("/", 1)[-1]
    if "efficientNetV2" in tail:
        return "baseline"
    if "_syn_" in tail:
        return "mixed"
    if tail.startswith("syn_"):
        return "synthetic-only"
    return "real-only"


# --------------------------------------------------------------------------- #
# Fast AUC for the bootstrap (paired: same resample applied to every model)
# --------------------------------------------------------------------------- #
def auc_rows(y, S):
    """AUC of each row of ``S`` (m x n) against shared labels ``y`` (n,).

    Mann-Whitney form with midranks for ties.
    """
    m, n = S.shape
    npos = int(y.sum())
    nneg = n - npos
    if npos == 0 or nneg == 0:
        return np.full(m, np.nan)
    order = np.argsort(S, axis=1, kind="mergesort")
    S_sorted = np.take_along_axis(S, order, axis=1)
    y_sorted = np.take_along_axis(np.broadcast_to(y, S.shape), order, axis=1)
    ranks = np.broadcast_to(np.arange(1, n + 1, dtype=float), S.shape).copy()
    # Tie correction only where needed (continuous scores rarely tie).
    same = np.zeros((m, n), dtype=bool)
    same[:, 1:] = S_sorted[:, 1:] == S_sorted[:, :-1]
    for r in np.where(same.any(axis=1))[0]:
        s_row, rk = S_sorted[r], ranks[r]
        i = 0
        while i < n:
            j = i
            while j + 1 < n and s_row[j + 1] == s_row[i]:
                j += 1
            if j > i:
                rk[i:j + 1] = 0.5 * (i + j) + 1.0
            i = j + 1
    r_pos = (ranks * (y_sorted == 1)).sum(axis=1)
    return (r_pos - npos * (npos + 1) / 2.0) / (npos * nneg)


def paired_bootstrap(label_vec, score_mat, n_boot, seed):
    """Return (B, m) bootstrap AUCs; resampling is stratified by class and
    shared across models so AUC *differences* are correctly paired."""
    rng = np.random.default_rng(seed)
    pos_idx = np.where(label_vec == 1)[0]
    neg_idx = np.where(label_vec == 0)[0]
    m = score_mat.shape[0]
    out = np.empty((n_boot, m), dtype=float)
    for b in range(n_boot):
        ip = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        ineg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([ip, ineg])
        out[b] = auc_rows(label_vec[idx], score_mat[:, idx])
    return out


# --------------------------------------------------------------------------- #
# Inference helpers
# --------------------------------------------------------------------------- #
def se_diff(cov, i, j):
    return sqrt(max(cov[i, i] + cov[j, j] - 2.0 * cov[i, j], 1e-18))


def tost(delta, se, margin):
    """Two one-sided tests for equivalence within +/- margin.
    Returns (p_tost, p_lower, p_upper). Equivalent if p_tost < alpha."""
    z_lower = (delta + margin) / se       # H0: delta <= -margin
    z_upper = (delta - margin) / se       # H0: delta >= +margin
    p_lower = 1.0 - _NORM.cdf(z_lower)
    p_upper = _NORM.cdf(z_upper)
    return max(p_lower, p_upper), p_lower, p_upper


def non_inferiority(delta, se, margin):
    """One-sided: model A not worse than B by more than margin (delta>-margin)."""
    z = (delta + margin) / se
    return 1.0 - _NORM.cdf(z)


def mcnemar_exact(correct_a, correct_b):
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1))
    return b, c, float(min(1.0, 2.0 * tail * (0.5 ** n)))


def mde(se, alpha, power):
    z_a = _NORM.inv_cdf(1.0 - alpha / 2.0)
    z_b = _NORM.inv_cdf(power)
    return (z_a + z_b) * se


def power_at(delta_true, se, alpha):
    z_a = _NORM.inv_cdf(1.0 - alpha / 2.0)
    return float(_NORM.cdf(abs(delta_true) / se - z_a))


# --------------------------------------------------------------------------- #
# Primary, hypothesis-driven comparison families
# --------------------------------------------------------------------------- #
def build_families(present):
    """present: set of available model_ids. Returns list of comparison dicts.

    direction convention: delta = AUC(a) - AUC(b); 'a' is the model whose
    superiority / equivalence / non-inferiority we want to assert.
    """
    fams = []

    # A) Augmentation: does adding synthetic data to a fixed real budget hurt?
    #    a = real+synthetic, b = real-only baseline. Claim: equivalence / NI.
    for n in REAL_BUDGETS:
        b = real_only(n)
        for pct in SYNTH_PCTS:
            a = mix(n, pct)
            if a in present and b in present:
                fams.append(dict(family="A_augmentation", a=a, b=b,
                                 test="equivalence"))

    # B) Synthetic-only vs matched real budget. Claim: equivalence.
    for n in REAL_BUDGETS:
        a, b = synth_only(n), real_only(n)
        if a in present and b in present:
            fams.append(dict(family="B_synthetic_vs_real", a=a, b=b,
                             test="equivalence"))

    # C) Synthetic-only utility scales with generated volume. Claim: superiority
    #    of larger synthetic set over the smallest one.
    base = synth_only(min(SYNTH_ONLY))
    for m in SYNTH_ONLY:
        if m == min(SYNTH_ONLY):
            continue
        a = synth_only(m)
        if a in present and base in present:
            fams.append(dict(family="C_synthetic_scaling", a=a, b=base,
                             test="superiority"))

    # D) How much real data is synthetic-only worth? The maximal synthetic-only
    #    model (most generated data) vs each real budget. Claim: equivalence to
    #    some real budget. Pre-specified to use the largest synthetic set.
    a = synth_only(max(SYNTH_ONLY))
    for n in REAL_BUDGETS:
        b = real_only(n)
        if a in present and b in present:
            fams.append(dict(family="D_maxsynthetic_vs_real", a=a, b=b,
                             test="equivalence"))
    return fams


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
COLORS = {"real-only": "#1f77b4", "mixed": "#ff7f0e",
          "synthetic-only": "#2ca02c", "baseline": "#7f7f7f"}


def forest_plot(auc_df, path, conf_level):
    d = auc_df.sort_values("auc").reset_index(drop=True)
    n = len(d)
    fig, ax = plt.subplots(figsize=(8.2, max(5.5, 0.32 * n)))
    y = np.arange(n)
    xerr = np.vstack([d["auc"] - d["ci_lower"], d["ci_upper"] - d["auc"]])
    ax.errorbar(d["auc"], y, xerr=xerr, fmt="none", ecolor="#999999",
                elinewidth=1.2, capsize=2.5, zorder=1)
    ax.scatter(d["auc"], y, c=[COLORS[k] for k in d["kind"]], s=42,
               zorder=2, edgecolor="white", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([pretty(m) for m in d["model"]], fontsize=8)
    ax.set_xlabel(f"AUC-ROC ({int(conf_level * 100)}% DeLong CI)")
    ax.set_title("Per-model AUC with test-set confidence intervals")
    ax.grid(axis="x", alpha=0.3)
    handles = [plt.Line2D([0], [0], marker="o", color="w", label=k,
                          markerfacecolor=c, markersize=8)
               for k, c in COLORS.items()]
    ax.legend(handles=handles, fontsize=8, loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def equivalence_plot(rows, margin, path):
    fam_titles = {"A_augmentation": "A) Augmented vs real-only (does synthetic hurt?)",
                  "B_synthetic_vs_real": "B) Synthetic-only vs matched real budget",
                  "D_maxsynthetic_vs_real": "C) Max synthetic-only vs real budget"}
    fams = [f for f in fam_titles if any(r["family"] == f for r in rows)]
    if not fams:
        return
    fig, axes = plt.subplots(1, len(fams), figsize=(5.4 * len(fams), 5.4),
                             squeeze=False)
    for ax, fam in zip(axes[0], fams):
        sub = [r for r in rows if r["family"] == fam]
        y = np.arange(len(sub))
        delta = np.array([r["delta"] for r in sub])
        lo = np.array([r["ci90_lower"] for r in sub])   # 90% CI: matches TOST @ alpha
        hi = np.array([r["ci90_upper"] for r in sub])
        ax.axvspan(-margin, margin, color="#2ca02c", alpha=0.12,
                   label=f"equivalence margin (+/-{margin:g})")
        ax.axvline(0.0, color="black", lw=1)
        ax.axvline(-margin, color="#2ca02c", ls="--", lw=1)
        ax.axvline(margin, color="#2ca02c", ls="--", lw=1)
        xerr = np.vstack([delta - lo, hi - delta])
        cols = []
        for r in sub:
            if r["equivalent"]:
                cols.append("#2ca02c")           # equivalent (TOST)
            elif r["noninferior"]:
                cols.append("#1f77b4")           # non-inferior (>= baseline, maybe better)
            else:
                cols.append("#d62728")           # inferiority not excluded
        ax.errorbar(delta, y, xerr=xerr, fmt="o", ecolor="#666666",
                    elinewidth=1.3, capsize=3, mfc="white", mec="#666",
                    zorder=1)
        ax.scatter(delta, y, c=cols, s=46, zorder=2,
                   edgecolor="white", linewidth=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{pretty(r['a'])} vs {pretty(r['b'])}"
                            for r in sub], fontsize=8)
        ax.set_xlabel("delta AUC (a - b), 90% bootstrap CI")
        ax.set_title(fam_titles[fam], fontsize=10)
        ax.grid(axis="x", alpha=0.3)
        cleg = [plt.Line2D([0], [0], marker="o", color="w", markersize=8,
                           markerfacecolor=c, label=lbl)
                for c, lbl in [("#2ca02c", "equivalent"),
                               ("#1f77b4", "non-inferior (maybe better)"),
                               ("#d62728", "inferiority not excluded")]]
        ax.legend(handles=cleg, fontsize=7, loc="best", frameon=True)
    fig.suptitle("Equivalence / non-inferiority of synthetic data "
                 "(green CI fully inside margin = equivalent)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs_dir", default=DEFAULT_RUNS)
    ap.add_argument("--output_dir", default=DEFAULT_OUT)
    ap.add_argument("--csv_name", default="individual_test_performance.csv")
    ap.add_argument("--score_col", default="predicted_score")
    ap.add_argument("--label_col", default="original_label")
    ap.add_argument("--id_col", default="img_path")
    ap.add_argument("--positive_label", type=float, default=1.0)
    ap.add_argument("--conf_level", type=float, default=0.95)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--margin", type=float, default=0.05,
                    help="Equivalence / non-inferiority margin on AUC.")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Operating threshold for McNemar (score >= thr -> pos).")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=12345)
    # Same coincidence filter as the main DeLong run -> identical 274-sample set.
    ap.add_argument("--exclude_csv", default=DEFAULT_EXCLUDE)
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

    models = discover_models(args.runs_dir, args.csv_name, args.score_col,
                             args.label_col, args.id_col, args.positive_label,
                             exclude_stems=exclude_stems)
    if not models:
        print("No usable prediction CSVs found.")
        return

    # Align every model onto the shared test samples.
    common = set(models[0]["id_set"])
    for m in models[1:]:
        common &= set(m["id_set"])
    common_ids = sorted(common)
    if len(common_ids) < 2:
        print("Models do not share a common test set; cannot run paired stats.")
        return
    label_vec = np.array([models[0]["label_by_id"][i] for i in common_ids], int)
    score_mat = np.vstack([[m["score_by_id"][i] for i in common_ids]
                           for m in models]).astype(float)
    idx_of = {m["model_id"]: r for r, m in enumerate(models)}
    n_samples = len(common_ids)
    n_pos, n_neg = int(label_vec.sum()), int((label_vec == 0).sum())
    print(f"Aligned {len(models)} models on {n_samples} samples "
          f"(pos={n_pos}, neg={n_neg}).")

    # Analytic DeLong AUCs + full covariance (one shot for all models).
    aucs, cov = delong_auc_cov(label_vec, score_mat)

    # Paired, class-stratified bootstrap.
    print(f"Bootstrapping ({args.n_boot} resamples) ...")
    boot = paired_bootstrap(label_vec, score_mat, args.n_boot, args.seed)
    lo_q, hi_q = 100 * (1 - args.conf_level) / 2, 100 * (1 + args.conf_level) / 2

    # ---- per-model AUC table (DeLong + bootstrap CI) ----
    auc_rows_out = []
    for mdl in models:
        i = idx_of[mdl["model_id"]]
        se, lo, hi = auc_delong_ci(float(aucs[i]), float(cov[i, i]), args.conf_level)
        b_lo, b_hi = np.percentile(boot[:, i], [lo_q, hi_q])
        auc_rows_out.append(dict(
            model=mdl["model_id"], pretty=pretty(mdl["model_id"]),
            kind=kind(mdl["model_id"]), auc=float(aucs[i]), delong_se=se,
            ci_lower=lo, ci_upper=hi,
            boot_ci_lower=float(b_lo), boot_ci_upper=float(b_hi),
            n=n_samples, n_pos=n_pos, n_neg=n_neg))
    auc_df = pd.DataFrame(auc_rows_out).sort_values("auc", ascending=False)
    auc_df.to_csv(os.path.join(args.output_dir, "bootstrap_auc_ci.csv"), index=False)

    # Per-sample correctness for McNemar at the operating threshold.
    preds = (score_mat >= args.threshold).astype(int)
    correct = (preds == label_vec[None, :])

    # ---- primary comparisons ----
    fams = build_families(set(idx_of))
    rows = []
    z_a = _NORM.inv_cdf(1 - args.alpha / 2.0)
    for cmp in fams:
        i, j = idx_of[cmp["a"]], idx_of[cmp["b"]]
        a_auc, b_auc = float(aucs[i]), float(aucs[j])
        delta = a_auc - b_auc
        se = se_diff(cov, i, j)
        z = delta / se
        p_two = 2.0 * _NORM.cdf(-abs(z))
        p_sup = 1.0 - _NORM.cdf(z)                       # one-sided: a > b
        p_tost, p_lo, p_hi = tost(delta, se, args.margin)
        p_ni = non_inferiority(delta, se, args.margin)
        bd = boot[:, i] - boot[:, j]
        b95 = np.percentile(bd, [2.5, 97.5])
        b90 = np.percentile(bd, [5.0, 95.0])
        mb, mc, p_mc = mcnemar_exact(correct[i], correct[j])
        primary_p = p_sup if cmp["test"] == "superiority" else p_tost
        rows.append(dict(
            family=cmp["family"], test=cmp["test"],
            a=cmp["a"], b=cmp["b"],
            a_label=pretty(cmp["a"]), b_label=pretty(cmp["b"]),
            auc_a=a_auc, auc_b=b_auc, delta=delta, se_diff=se, z=z,
            p_superiority_2sided=p_two, p_superiority_1sided=p_sup,
            ci95_lower=float(b95[0]), ci95_upper=float(b95[1]),
            ci90_lower=float(b90[0]), ci90_upper=float(b90[1]),
            margin=args.margin, p_tost=p_tost,
            equivalent=bool(p_tost < args.alpha),
            p_noninferiority=p_ni, noninferior=bool(p_ni < args.alpha),
            mcnemar_b=mb, mcnemar_c=mc, p_mcnemar=p_mc,
            mde_power80=mde(se, args.alpha, 0.80),
            mde_power90=mde(se, args.alpha, 0.90),
            power_at_margin=power_at(args.margin, se, args.alpha),
            primary_p=primary_p))

    # Multiplicity correction *within* each primary family.
    df = pd.DataFrame(rows)
    df["p_holm"] = np.nan
    df["p_fdr_bh"] = np.nan
    for fam, sub in df.groupby("family"):
        pv = sub["primary_p"].to_numpy()
        df.loc[sub.index, "p_holm"] = _holm(pv)
        df.loc[sub.index, "p_fdr_bh"] = _benjamini_hochberg(pv)
    df.to_csv(os.path.join(args.output_dir, "primary_comparisons.csv"), index=False)

    power_cols = ["family", "a_label", "b_label", "delta", "se_diff",
                  "mde_power80", "mde_power90", "power_at_margin"]
    df[power_cols].to_csv(os.path.join(args.output_dir, "power_summary.csv"),
                          index=False)

    # ---- figures ----
    forest_plot(auc_df, os.path.join(args.output_dir, "forest_auc.png"),
                args.conf_level)
    equivalence_plot(df.to_dict("records"), args.margin,
                     os.path.join(args.output_dir, "equivalence_deltas.png"))

    # representative MDE across the primary family
    mde80_med = float(df["mde_power80"].median())

    summary = dict(
        n_samples=n_samples, n_pos=n_pos, n_neg=n_neg, n_models=len(models),
        margin=args.margin, alpha=args.alpha, threshold=args.threshold,
        n_boot=args.n_boot, median_mde_power80=mde80_med,
        n_primary_comparisons=int(len(df)))
    with open(os.path.join(args.output_dir, "paper_stats_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    # ---- console digest ----
    print("\n" + "=" * 78)
    print(f"Primary comparisons: {len(df)}  |  margin +/-{args.margin:g} AUC  "
          f"|  alpha={args.alpha}")
    print(f"Median MDE (80% power, two-sided): delta AUC = {mde80_med:.3f}")
    print("=" * 78)
    for fam, sub in df.groupby("family"):
        print(f"\n### {fam}")
        for _, r in sub.iterrows():
            if r["test"] == "superiority":
                verdict = ("SUPERIOR (Holm)" if r["p_holm"] < args.alpha
                           else "n.s.")
                print(f"  {r['a_label']:>16s} > {r['b_label']:<16s} "
                      f"dAUC={r['delta']:+.3f} [{r['ci95_lower']:+.3f},"
                      f"{r['ci95_upper']:+.3f}]  p={r['primary_p']:.1e} "
                      f"Holm={r['p_holm']:.1e}  -> {verdict}")
            else:
                verdict = ("EQUIVALENT" if r["equivalent"] else
                           ("non-inferior" if r["noninferior"] else "inconclusive"))
                print(f"  {r['a_label']:>16s} ~ {r['b_label']:<16s} "
                      f"dAUC={r['delta']:+.3f} [{r['ci95_lower']:+.3f},"
                      f"{r['ci95_upper']:+.3f}]  TOST p={r['p_tost']:.3f} "
                      f"NI p={r['p_noninferiority']:.3f}  -> {verdict}")
    print(f"\nResults written to {args.output_dir}")


if __name__ == "__main__":
    main()
