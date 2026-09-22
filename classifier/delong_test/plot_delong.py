#!/usr/bin/env python3
"""
Figure-4-equivalent plots built from the DeLong results.

DeLong's method gives a confidence interval for the *AUC* only, so these plots
reproduce the AUC-ROC panels of the original figure, with 95% DeLong CIs drawn
as error bars:

  Panel A: AUC-ROC vs training sample size, for "Real only" and "Synthetic only".
  Panel B: AUC-ROC vs synthetic-data percentage, for each real-data regime.

Input : results/single_auc_delong_ci.csv  (produced by delong.py)
Output: results/delong_auc_vs_samplesize.png
        results/delong_auc_vs_synthetic_pct.png
        results/delong_auc_panels.png   (both panels side by side)
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS = os.path.join(SCRIPT_DIR, "results")

SAMPLE_SIZES = [500, 1000, 2000, 2723, 5446]
SYNTH_PCTS = [0, 25, 50, 75, 100]
REAL_REGIMES = [500, 1000, 2000, 2723]
REGIME_COLORS = {500: "#e377c2", 1000: "#bcbd22", 2000: "#2ca02c", 2723: "#17becf"}


def real_only_id(n):
    return f"runs_{n}/real_{n}"


def synth_only_id(n):
    return f"runs_syn/syn_{n}"


def mix_id(real_n, pct):
    return f"runs_{real_n}/real_{real_n}" if pct == 0 else f"runs_{real_n}/real_{real_n}_syn_{pct}"


def yerr(rows):
    """Asymmetric error bars from CI bounds."""
    lo = [r["auc"] - r["ci_lower"] for r in rows]
    hi = [r["ci_upper"] - r["auc"] for r in rows]
    return [lo, hi]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", default=DEFAULT_RESULTS)
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    csv_path = os.path.join(args.results_dir, "single_auc_delong_ci.csv")
    df = pd.read_csv(csv_path)
    lut = {row["model"]: row for _, row in df.iterrows()}

    def collect(ids_with_x):
        xs, ys, rows = [], [], []
        for x, mid in ids_with_x:
            if mid in lut:
                r = lut[mid]
                xs.append(x)
                ys.append(r["auc"])
                rows.append(r)
            else:
                print(f"  [warn] missing model in CSV: {mid}")
        return xs, ys, rows

    conf = float(df["conf_level"].iloc[0]) if "conf_level" in df else 0.95
    ci_pct = int(round(conf * 100))

    # ---------------- Panel A: AUC vs sample size ----------------
    figA, axA = plt.subplots(figsize=(6.4, 5.0))
    xs, ys, rows = collect([(n, real_only_id(n)) for n in REAL_REGIMES])
    axA.errorbar(xs, ys, yerr=yerr(rows), marker="o", color="tab:blue",
                 capsize=4, lw=2, label="Real only")
    xs, ys, rows = collect([(n, synth_only_id(n)) for n in SAMPLE_SIZES])
    axA.errorbar(xs, ys, yerr=yerr(rows), marker="s", color="tab:red",
                 capsize=4, lw=2, label="Synthetic only")
    axA.set_xlabel("Sample Size")
    axA.set_ylabel(f"AUC-ROC ({ci_pct}% DeLong CI)")
    axA.set_title("AUC-ROC vs Sample Size")
    axA.set_xticks(SAMPLE_SIZES)
    axA.grid(True, alpha=0.3)
    axA.legend()
    figA.tight_layout()
    pathA = os.path.join(args.results_dir, "delong_auc_vs_samplesize.png")
    figA.savefig(pathA, dpi=args.dpi)

    # ---------------- Panel B: AUC vs synthetic % ----------------
    figB, axB = plt.subplots(figsize=(6.4, 5.0))
    for R in REAL_REGIMES:
        xs, ys, rows = collect([(p, mix_id(R, p)) for p in SYNTH_PCTS])
        axB.errorbar(xs, ys, yerr=yerr(rows), marker="o", color=REGIME_COLORS[R],
                     capsize=4, lw=2, label=f"{R} real samples")
    axB.set_xlabel("Synthetic Data Percentage (%)")
    axB.set_ylabel(f"AUC-ROC ({ci_pct}% DeLong CI)")
    axB.set_title("AUC-ROC vs Synthetic Data Percentage")
    axB.set_xticks(SYNTH_PCTS)
    axB.grid(True, alpha=0.3)
    axB.legend()
    figB.tight_layout()
    pathB = os.path.join(args.results_dir, "delong_auc_vs_synthetic_pct.png")
    figB.savefig(pathB, dpi=args.dpi)

    # ---------------- Combined ----------------
    figC, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.0))
    xs, ys, rows = collect([(n, real_only_id(n)) for n in REAL_REGIMES])
    ax1.errorbar(xs, ys, yerr=yerr(rows), marker="o", color="tab:blue",
                 capsize=4, lw=2, label="Real only")
    xs, ys, rows = collect([(n, synth_only_id(n)) for n in SAMPLE_SIZES])
    ax1.errorbar(xs, ys, yerr=yerr(rows), marker="s", color="tab:red",
                 capsize=4, lw=2, label="Synthetic only")
    ax1.set_xlabel("Sample Size")
    ax1.set_ylabel(f"AUC-ROC ({ci_pct}% DeLong CI)")
    ax1.set_title("A) AUC-ROC vs Sample Size")
    ax1.set_xticks(SAMPLE_SIZES)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    for R in REAL_REGIMES:
        xs, ys, rows = collect([(p, mix_id(R, p)) for p in SYNTH_PCTS])
        ax2.errorbar(xs, ys, yerr=yerr(rows), marker="o", color=REGIME_COLORS[R],
                     capsize=4, lw=2, label=f"{R} real samples")
    ax2.set_xlabel("Synthetic Data Percentage (%)")
    ax2.set_ylabel(f"AUC-ROC ({ci_pct}% DeLong CI)")
    ax2.set_title("B) AUC-ROC vs Synthetic Data Percentage")
    ax2.set_xticks(SYNTH_PCTS)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    figC.tight_layout()
    pathC = os.path.join(args.results_dir, "delong_auc_panels.png")
    figC.savefig(pathC, dpi=args.dpi)

    print("Wrote:")
    for p in (pathA, pathB, pathC):
        print(f"  {p}")


if __name__ == "__main__":
    main()
