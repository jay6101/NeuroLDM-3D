"""Re-derive the memorization report after dropping test/train coincidences.

Reads the already-computed per-sample distances (results/distances.npz) plus the
coincidence report (results/test_train_coincidence.csv), drops real-test samples
whose nearest-train MSE is <= --threshold, and rewrites the stats/plots for the
surviving set. No GPU work and no re-encoding: the saved distances are exactly
what the original run plotted.

Optionally subsamples the synthetic queries down to the real-test count so both
arms have equal n (class-stratified by default, so per-class n matches too).

Run:
  PY=/home/jsawant/.conda/envs/jay2/bin/python
  $PY filter_coincidence.py --threshold 0.05
  $PY filter_coincidence.py --threshold 1e-4 --no-match-n   # original variant
"""
import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # type: ignore

# Matches the VAE ROI figures (VAE/recon_eval/roi_analysis/roi_mse_analysis.py) so every
# distribution figure in the paper reads the same way.
PLOT_STYLE = {
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 10,
    "figure.titlesize": 18,
}
ARM_COLORS = {"Real test": "#4682B4", "Synthetic": "#FF6347"}  # Steel Blue / Tomato
# paths.CLASS_NAMES calls class 1 "epilepsy"; the paper figures say TLE.
GROUP_LABELS = {0: "HC", 1: "TLE"}


def describe(x):
    x = np.asarray(x, np.float64)
    return {"n": int(x.size), "mean": float(x.mean()), "std": float(x.std()),
            "min": float(x.min()), "p5": float(np.percentile(x, 5)),
            "q1": float(np.percentile(x, 25)), "median": float(np.median(x)),
            "q3": float(np.percentile(x, 75)), "max": float(x.max())}


def compare(dr, df):
    ks = stats.ks_2samp(df, dr)
    mwu = stats.mannwhitneyu(df, dr, alternative="two-sided")
    mwu_less = stats.mannwhitneyu(df, dr, alternative="less")
    return {"ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue),
            "mwu_two_sided_p": float(mwu.pvalue), "mwu_fake_less_p": float(mwu_less.pvalue),
            "median_ratio_fake_over_real": float(np.median(df) / np.median(dr)),
            "frac_fake_below_real_min": float(np.mean(df < np.min(dr))),
            "frac_fake_below_real_p5": float(np.mean(df < np.percentile(dr, 5))),
            "frac_fake_below_real_p1": float(np.mean(df < np.percentile(dr, 1))),
            "n_fake_below_real_min": int(np.sum(df < np.min(dr)))}


def plot_hist(dr, df, path, title):
    plt.figure(figsize=(8, 5))
    hi = max(np.percentile(dr, 99.5), np.percentile(df, 99.5))
    bins = np.linspace(min(dr.min(), df.min()), hi, 50)
    plt.hist(dr, bins=bins, alpha=0.55, density=True, color="tab:blue",
             label=f"real test (n={dr.size})")
    plt.hist(df, bins=bins, alpha=0.55, density=True, color="tab:red",
             label=f"synthetic (n={df.size})")
    plt.axvline(np.median(dr), color="tab:blue", ls="--", lw=1)
    plt.axvline(np.median(df), color="tab:red", ls="--", lw=1)
    plt.xlabel("MSE to nearest TRAIN latent"); plt.ylabel("density")
    plt.title(title); plt.legend(); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()


def plot_cdf(dr, df, path):
    plt.figure(figsize=(8, 5))
    for arr, lab, c in [(dr, "real test", "tab:blue"), (df, "synthetic", "tab:red")]:
        xs = np.sort(arr)
        plt.plot(xs, np.arange(1, xs.size + 1) / xs.size, label=lab, color=c)
    plt.xlabel("MSE to nearest TRAIN latent"); plt.ylabel("empirical CDF")
    plt.title("CDF of nearest-train distances (left shift = memorization)")
    plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_box(dr, df, path):
    plt.figure(figsize=(6, 5))
    plt.boxplot([dr, df], tick_labels=["real test", "synthetic"], showfliers=True)
    plt.ylabel("MSE to nearest TRAIN latent")
    plt.title("Nearest-train distance distributions")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def draw_boxes(ax, data, positions, width, color):
    bp = ax.boxplot(data, positions=positions, widths=width, patch_artist=True,
                    showfliers=True,
                    medianprops=dict(color="black", linewidth=2),
                    flierprops=dict(marker="o", markersize=2.5, markerfacecolor="0.35",
                                    markeredgecolor="none", alpha=0.5))
    for box in bp["boxes"]:
        box.set_facecolor(color)
        box.set_alpha(0.75)
        box.set_edgecolor("black")
        box.set_linewidth(1)
    for key in ("whiskers", "caps"):
        for artist in bp[key]:
            artist.set_color("black")
            artist.set_linewidth(1)
    return bp


def draw_violins(ax, data, positions, width, color):
    parts = ax.violinplot(data, positions=positions, widths=width,
                          showmeans=True, showmedians=True)
    for body, values in zip(parts["bodies"], data):
        body.set_facecolor(color)
        body.set_alpha(0.7)
        body.set_edgecolor("black")
        body.set_linewidth(1)
        # trim the kernel density to the observed range
        verts = body.get_paths()[0].vertices
        verts[:, 1] = np.clip(verts[:, 1], values.min(), values.max())
    parts["cmeans"].set_color("red")
    parts["cmeans"].set_linewidth(3)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(2)
    for key in ("cbars", "cmins", "cmaxes"):
        parts[key].set_color("black")
        parts[key].set_linewidth(1)
    return parts


def upper_whisker(x):
    """Largest observation within 1.5 IQR of Q3, i.e. where matplotlib caps the whisker."""
    q1, q3 = np.percentile(x, [25, 75])
    return float(x[x <= q3 + 1.5 * (q3 - q1)].max())


# Boxes only draw fliers past the whisker cap, so that cap is the natural axis limit;
# violin bodies run all the way to the observed max, so they need a percentile instead.
KINDS = {
    "box": (draw_boxes, 0.2, upper_whisker),
    "violin": (draw_violins, 0.26, lambda x: float(np.percentile(x, 97.5))),
}


def plot_by_group(d_real, real_lab, d_fake, fake_lab, path, title, kind="box"):
    """Real vs synthetic paired inside each diagnosis group, as boxes or violins."""
    draw, width, cap = KINDS[kind]
    classes = sorted(set(real_lab.tolist()) | set(fake_lab.tolist()))
    arms = [("Real test", d_real, real_lab, -0.125), ("Synthetic", d_fake, fake_lab, 0.125)]

    with plt.style.context(["seaborn-v0_8", PLOT_STYLE]):
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        drawn, per_arm_n = [], {}
        for name, d, lab, off in arms:
            data = [d[lab == cls] for cls in classes]
            draw(ax, data, np.arange(len(classes)) + off, width, ARM_COLORS[name])
            per_arm_n[name] = [v.size for v in data]
            drawn += data

        # A handful of far real-test samples would otherwise flatten every distribution. The
        # low tail, which is the side that would signal memorization, is always shown in full.
        lo = min(v.min() for v in drawn)
        top = max(cap(v) for v in drawn)
        rng = top - lo
        hi_lim = top + 0.22 * rng  # headroom keeps the surviving fliers clear of the note
        ax.set_ylim(lo - 0.08 * rng, hi_lim)
        n_hidden = int(sum((v > hi_lim).sum() for v in drawn))
        if n_hidden:
            ax.text(0.99, 0.985, f"{n_hidden} of {sum(v.size for v in drawn)} values above axis limit",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8, color="0.35",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.85))

        ticks = []
        for k, cls in enumerate(classes):
            nr, nf = per_arm_n["Real test"][k], per_arm_n["Synthetic"][k]
            n_txt = f"n = {nr}" if nr == nf else f"real n = {nr}, syn n = {nf}"
            ticks.append(f"{GROUP_LABELS.get(cls, paths.CLASS_NAMES.get(cls, str(cls)))}\n({n_txt})")
        ax.set_xticks(np.arange(len(classes)))
        ax.set_xticklabels(ticks)
        ax.set_xlim(-0.55, len(classes) - 0.45)

        handles = [plt.Line2D([0], [0], color=c, lw=4, label=n) for n, c in ARM_COLORS.items()]
        ax.legend(handles=handles, loc="upper left", framealpha=0.9)

        ax.set_xlabel("Diagnosis", fontsize=14, fontweight="bold")
        ax.set_ylabel("MSE to nearest TRAIN latent", fontsize=14, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--", color="gray")
        ax.set_axisbelow(True)
        ax.set_facecolor("#f8f9fa")
        for spine in ax.spines.values():
            spine.set_edgecolor("gray")
            spine.set_linewidth(1.5)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(f"{path}.{ext}", dpi=300 if ext == "png" else None,
                        bbox_inches="tight", facecolor="white")
        plt.close(fig)


def match_fake_n(real_lab, fake_lab, seed, stratified=True):
    """Indices of a random synthetic subset with the same n as the real set."""
    rng = np.random.default_rng(seed)
    if not stratified:
        return np.sort(rng.choice(fake_lab.size, size=real_lab.size, replace=False))
    picked = []
    for cls in np.unique(fake_lab):
        pool = np.flatnonzero(fake_lab == cls)
        want = int((real_lab == cls).sum())
        if want > pool.size:
            raise ValueError(f"class {cls}: need {want} synthetic samples, only {pool.size} "
                             f"available -- use --match-strategy overall")
        picked.append(rng.choice(pool, size=want, replace=False))
    return np.sort(np.concatenate(picked))


def default_tag(thr):
    return f"{thr:g}".replace(".", "p").replace("-", "m")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.05,
                    help="drop real-test samples with nearest-train MSE <= this")
    ap.add_argument("--match-n", action=argparse.BooleanOptionalAction, default=True,
                    help="subsample synthetic queries to the surviving real-test count")
    ap.add_argument("--match-strategy", choices=["stratified", "overall"], default="stratified",
                    help="stratified keeps per-class counts equal too")
    ap.add_argument("--seed", type=int, default=0, help="seed for the synthetic subsample")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    thr = args.threshold
    out = args.outdir or os.path.join(paths.RESULTS_DIR, f"excl_coincidence_{default_tag(thr)}")
    os.makedirs(out, exist_ok=True)

    coin = pd.read_csv(os.path.join(paths.RESULTS_DIR, "test_train_coincidence.csv"))
    exclude = set(coin.loc[coin["nearest_mse"] <= thr, "test_stem"])
    coin.loc[coin["nearest_mse"] > thr].to_csv(
        os.path.join(out, "test_train_coincidence.csv"), index=False)

    rt = np.load(os.path.join(paths.REAL_TEST_DIR, paths.LATENTS_FILE))
    stems = rt["stems"].astype(str)
    keep = np.array([s not in exclude for s in stems])

    d = np.load(os.path.join(paths.RESULTS_DIR, "distances.npz"))
    d_real, real_lab = d["d_real"][keep], d["real_lab"][keep]
    nn_real = d["nn_real_idx"][keep]
    d_fake, fake_lab, nn_fake = d["d_fake"], d["fake_lab"], d["nn_fake_idx"]

    if args.match_n:
        sel = match_fake_n(real_lab, fake_lab, args.seed, args.match_strategy == "stratified")
        d_fake, fake_lab, nn_fake = d_fake[sel], fake_lab[sel], nn_fake[sel]
        sampling = {"matched_to_real_n": True, "strategy": args.match_strategy,
                    "seed": args.seed, "n_synthetic_pool": int(d["d_fake"].size),
                    "n_synthetic_used": int(d_fake.size),
                    "selected_indices": sel.tolist()}
    else:
        sampling = {"matched_to_real_n": False, "n_synthetic_used": int(d_fake.size)}

    results = {
        "exclusion": {"threshold": thr, "n_excluded": int(len(exclude)),
                      "n_kept_real": int(keep.sum()), "excluded_stems": sorted(exclude)},
        "synthetic_sampling": sampling,
        "note": "Filtered from saved distances.npz; mode_collapse not recomputed",
        "stats_d_real": describe(d_real), "stats_d_fake": describe(d_fake),
        "comparison": compare(d_real, d_fake), "per_class": {},
        "coverage": {"unique_train_nn_of_real": int(np.unique(nn_real).size),
                     "unique_train_nn_of_fake": int(np.unique(nn_fake).size),
                     "n_queries": int(d_real.size)},
    }

    for cls in sorted(set(real_lab.tolist()) | set(fake_lab.tolist())):
        dr, df = d_real[real_lab == cls], d_fake[fake_lab == cls]
        if dr.size and df.size:
            name = paths.CLASS_NAMES.get(cls, str(cls))
            results["per_class"][name] = {"stats_d_real": describe(dr),
                                          "stats_d_fake": describe(df),
                                          "comparison": compare(dr, df)}
            plot_hist(dr, df, os.path.join(out, f"hist_{name}.png"),
                      f"Nearest-train MSE — {name} (excl. coincidence ≤ {thr:g})")

    plot_hist(d_real, d_fake, os.path.join(out, "hist_overall.png"),
              f"Nearest-train MSE: real test vs synthetic (excl. coincidence ≤ {thr:g})")
    plot_cdf(d_real, d_fake, os.path.join(out, "cdf_overall.png"))
    plot_box(d_real, d_fake, os.path.join(out, "box_overall.png"))
    for kind in KINDS:
        plot_by_group(d_real, real_lab, d_fake, fake_lab,
                      os.path.join(out, f"{kind}_by_group"),
                      "Nearest-train MSE: real test vs synthetic by diagnosis\n"
                      f"(excl. coincidence ≤ {thr:g})", kind=kind)

    np.savez(os.path.join(out, "distances.npz"), d_real=d_real, d_fake=d_fake,
             real_lab=real_lab, fake_lab=fake_lab, nn_real_idx=nn_real, nn_fake_idx=nn_fake)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    c = results["comparison"]
    print(f"Wrote filtered results -> {out}")
    print(f"Excluded {len(exclude)} test samples (nearest_mse <= {thr:g})")
    print(f"d_real  n={d_real.size}  median={np.median(d_real):.4f}  min={d_real.min():.4f}")
    print(f"d_fake  n={d_fake.size}  median={np.median(d_fake):.4f}  min={d_fake.min():.4f}")
    for name, blk in results["per_class"].items():
        print(f"  {name:<9} real n={blk['stats_d_real']['n']:<4} fake n={blk['stats_d_fake']['n']}")
    print(f"KS p={c['ks_p']:.3g} | MWU(fake<real) p={c['mwu_fake_less_p']:.3g}")
    print(f"median ratio fake/real = {c['median_ratio_fake_over_real']:.3f}")
    print(f"fakes below real-min = {c['n_fake_below_real_min']} "
          f"({100 * c['frac_fake_below_real_min']:.1f}%)")


if __name__ == "__main__":
    main()
