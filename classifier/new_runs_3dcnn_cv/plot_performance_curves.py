#!/usr/bin/env python3
"""
Performance curves (accuracy, AUC-ROC and F1) built from the cross-validated 3D CNN runs.

Reads every variant's all_folds_metrics.json under a runs root and emits three
figures, each row carrying an accuracy panel, an AUC-ROC panel and an F1 panel:

  Figure 1: metric vs training sample size, for "Real Only" and "Synthetic Only".
  Figure 2: metric vs synthetic-data percentage, one line per real-data regime.
  Figure 3: both of the above stacked as panels A) and B).

Input : <runs_root>/runs_<N>/real_<N>[_syn_<pct>]/all_folds_metrics.json
        <runs_root>/runs_syn/syn_<N>/all_folds_metrics.json
Output: <out_dir>/metrics_vs_sample_size.{png,pdf}
        <out_dir>/metrics_vs_synthetic_pct.{png,pdf}
        <out_dir>/metrics_combined.{png,pdf}
        <out_dir>/aggregated_metrics.csv

Usage
-----
python plot_performance_curves.py                       # defaults to ./runs_3d
python plot_performance_curves.py --runs_root runs_2d
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SAMPLE_SIZES = [500, 1000, 2000, 2723, 5446]
REAL_REGIMES = [500, 1000, 2000, 2723]
SYNTH_PCTS = [0, 25, 50, 75, 100]

METRICS = [("accuracy", "Accuracy"), ("auc_roc", "AUC-ROC"), ("f1_score", "F1 Score")]

PANEL_SIZE = (4.8, 4.3)

REAL_COLOR = "#1a1af0"
SYNTH_COLOR = "#ff1622"
REGIME_COLORS = {500: "#ff6c83", 1000: "#b8912f", 2000: "#2eaa3b", 2723: "#00a79c"}

AXES_FACECOLOR = "#f7f8f9"
SPINE_COLOR = "#b8b8b8"


def set_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": AXES_FACECOLOR,
        "axes.edgecolor": SPINE_COLOR,
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "grid.color": "white",
        "grid.linewidth": 0.9,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "font.family": "DejaVu Sans",
        "savefig.facecolor": "white",
    })


def real_only_id(n):
    return f"runs_{n}/real_{n}"


def synth_only_id(n):
    return f"runs_syn/syn_{n}"


def mix_id(real_n, pct):
    return f"runs_{real_n}/real_{real_n}" if pct == 0 else f"runs_{real_n}/real_{real_n}_syn_{pct}"


def load_metrics(runs_root):
    """Map variant id -> {metric: {'mean': float, 'std': float}}."""
    table = {}
    for group in sorted(os.listdir(runs_root)):
        group_dir = os.path.join(runs_root, group)
        if not (os.path.isdir(group_dir) and group.startswith("runs_")):
            continue
        for variant in sorted(os.listdir(group_dir)):
            path = os.path.join(group_dir, variant, "all_folds_metrics.json")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                table[f"{group}/{variant}"] = json.load(f)["overall_results"]
    return table


def series(table, ids_with_x, metric):
    """Return (xs, means) for the variants that exist, warning about the rest."""
    xs, means = [], []
    for x, vid in ids_with_x:
        if vid not in table:
            print(f"  [warn] missing variant: {vid}")
            continue
        xs.append(x)
        means.append(table[vid][metric]["mean"])
    return xs, means


def draw(ax, xs, ys, color, marker, label):
    ax.plot(xs, ys, color=color, marker=marker, markersize=6.5, lw=2.0,
            markeredgecolor="#5a5a5a", markeredgewidth=0.8, label=label,
            zorder=3, clip_on=False)


def style_axes(ax, xlabel, title, log_x):
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_axisbelow(True)
    if log_x:
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(SAMPLE_SIZES))
        ax.xaxis.set_minor_locator(FixedLocator([]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    else:
        ax.set_xticks(SYNTH_PCTS)


def legend(ax, n_entries):
    """Legend in the lower right, on blank space reserved beneath the curves."""
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - (0.06 + 0.04 * n_entries) * (hi - lo), hi)
    ax.legend(loc="lower right", frameon=True, framealpha=1.0, facecolor="white",
              edgecolor="#d5d5d5", borderpad=0.5)


def panel_grid_size(rows=1):
    w, h = PANEL_SIZE
    return (w * len(METRICS), h * rows)


def panel_tag(ax, text):
    """Panel letter above the row's first axes, left of the y-axis label."""
    ax.annotate(text, xy=(0, 1), xycoords="axes fraction",
                xytext=(-52, 14), textcoords="offset points",
                fontsize=16, fontweight="bold", va="bottom", ha="left")


def row_vs_sample_size(axes, table):
    for ax, (metric, nice) in zip(axes, METRICS):
        xs, ys = series(table, [(n, real_only_id(n)) for n in REAL_REGIMES], metric)
        draw(ax, xs, ys, REAL_COLOR, "o", "Real Only")
        xs, ys = series(table, [(n, synth_only_id(n)) for n in SAMPLE_SIZES], metric)
        draw(ax, xs, ys, SYNTH_COLOR, "s", "Synthetic Only")
        ax.set_ylabel(nice)
        style_axes(ax, "Sample Size", f"{nice} vs Sample Size", log_x=True)
        legend(ax, 2)


def row_vs_synthetic_pct(axes, table):
    for ax, (metric, nice) in zip(axes, METRICS):
        for regime in REAL_REGIMES:
            xs, ys = series(table, [(p, mix_id(regime, p)) for p in SYNTH_PCTS], metric)
            draw(ax, xs, ys, REGIME_COLORS[regime], "o", f"{regime} real samples")
        ax.set_ylabel(nice)
        style_axes(ax, "Synthetic Data Percentage (%)",
                   f"{nice} vs Synthetic Data Percentage", log_x=False)
        legend(ax, 4)


def figure_vs_sample_size(table):
    fig, axes = plt.subplots(1, len(METRICS), figsize=panel_grid_size())
    row_vs_sample_size(axes, table)
    fig.tight_layout()
    return fig


def figure_vs_synthetic_pct(table):
    fig, axes = plt.subplots(1, len(METRICS), figsize=panel_grid_size())
    row_vs_synthetic_pct(axes, table)
    fig.tight_layout()
    return fig


def figure_combined(table):
    fig, axes = plt.subplots(2, len(METRICS), figsize=panel_grid_size(rows=2))
    row_vs_sample_size(axes[0], table)
    row_vs_synthetic_pct(axes[1], table)
    fig.tight_layout()
    for ax, tag in zip(axes[:, 0], ("A)", "B)")):
        panel_tag(ax, tag)
    return fig


def write_csv(table, out_path):
    rows = []
    for metric, nice in METRICS:
        for n in REAL_REGIMES:
            rows.append(("real_only", n, 0, metric, real_only_id(n)))
        for n in SAMPLE_SIZES:
            rows.append(("synthetic_only", n, 100, metric, synth_only_id(n)))
        for n in REAL_REGIMES:
            for p in SYNTH_PCTS:
                rows.append(("mixed", n, p, metric, mix_id(n, p)))

    records = []
    for kind, n, pct, metric, vid in rows:
        if vid not in table:
            continue
        records.append({
            "setting": kind, "real_samples": n, "synthetic_pct": pct,
            "variant": vid, "metric": metric,
            "mean": table[vid][metric]["mean"], "std": table[vid][metric]["std"],
        })
    df = pd.DataFrame(records).drop_duplicates(subset=["variant", "metric", "setting"])
    df.to_csv(out_path, index=False)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs_root", default=os.path.join(SCRIPT_DIR, "runs_3d"))
    ap.add_argument("--out_dir", default=None,
                    help="default: <runs_root>/figures")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    runs_root = os.path.abspath(args.runs_root)
    out_dir = args.out_dir or os.path.join(runs_root, "figures")
    os.makedirs(out_dir, exist_ok=True)

    table = load_metrics(runs_root)
    print(f"Loaded {len(table)} variants from {runs_root}")

    set_style()
    figs = {
        "metrics_vs_sample_size": figure_vs_sample_size(table),
        "metrics_vs_synthetic_pct": figure_vs_synthetic_pct(table),
        "metrics_combined": figure_combined(table),
    }

    written = []
    for name, fig in figs.items():
        for ext in ("png", "pdf"):
            path = os.path.join(out_dir, f"{name}.{ext}")
            fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
            written.append(path)
        plt.close(fig)

    csv_path = os.path.join(out_dir, "aggregated_metrics.csv")
    write_csv(table, csv_path)
    written.append(csv_path)

    print("Wrote:")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
