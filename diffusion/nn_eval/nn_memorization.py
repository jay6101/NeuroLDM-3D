"""Stage 4: nearest-neighbor memorization / leakage / mode-collapse analysis.

All in VAE-latent space (dim = 4*28*34*28 = 106,624). Loads the three latent
sets produced by the earlier stages:

  TRAIN  : latents/real_train/latents.npz   (mu)  -> reference DB
  REAL   : latents/real_test/latents.npz    (mu)  -> held-out real query
  FAKE   : latents/synthetic/latents.npz    (z)   -> generated query

Core test (Meehan et al. 2020, data-copying):
  d_real = MSE(real, nearest train),  d_fake = MSE(fake, nearest train)
  d_fake shifted SMALLER than d_real => memorization/copying.
  d_real ~ d_fake               => generalizes like held-out real data.
Distance = MSE = mean over elements of (a-b)^2  (== L2^2 / D).

Also reports: KS + Mann-Whitney tests, left-tail copy counts, per-class
breakdown, within-set diversity (mode collapse), and train coverage.

Run:
  /home/jsawant/.conda/envs/jay2/bin/python \
      /space/mcdonald-syn01/1/projects/jsawant/Diffusion_paper/diffusion/nn_eval/nn_memorization.py \
      --device cuda:0
"""
import os
import json
import argparse

import numpy as np
import torch
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import paths


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _load(npz_dir):
    p = os.path.join(npz_dir, paths.LATENTS_FILE)
    if not os.path.exists(p):
        raise FileNotFoundError(f"missing {p} -- run the earlier stage first")
    return np.load(p, allow_pickle=True)


def load_real(npz_dir, reparam=False, seed=0, use_rt=False):
    d = _load(npz_dir)
    key = "latents_rt" if use_rt else "latents"
    if key not in d.files:
        raise KeyError(f"'{key}' not in {npz_dir}; run roundtrip.py --set all first")
    mu = d[key].astype(np.float32)
    labels = d["labels"]
    logvar = d["logvars"].astype(np.float32) if "logvars" in d.files else None
    mean_std = float(np.exp(0.5 * logvar).mean()) if logvar is not None else float("nan")
    if reparam and not use_rt and logvar is not None:
        rng = np.random.default_rng(seed)
        mu = mu + np.exp(0.5 * logvar) * rng.standard_normal(mu.shape).astype(np.float32)
    return mu.reshape(len(mu), -1), labels, mean_std


def load_fake(npz_dir):
    """Fakes are always the roundtripped mu = encode(decode(z))."""
    d = _load(npz_dir)
    if "latents_rt" not in d.files:
        raise KeyError("'latents_rt' not found; run roundtrip.py --set fakes (or all) first")
    lat = d["latents_rt"].astype(np.float32)
    return lat.reshape(len(lat), -1), d["labels"]


# --------------------------------------------------------------------------- #
# Nearest-neighbor distances (MSE = squared-L2 / D)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def nn_mse(query, ref, device, chunk=64, exclude_self=False):
    D = query.shape[1]
    ref_t = torch.from_numpy(np.ascontiguousarray(ref)).to(device)
    ref_norm = (ref_t ** 2).sum(1)
    min_mse, arg = [], []
    for i in range(0, query.shape[0], chunk):
        q = torch.from_numpy(np.ascontiguousarray(query[i:i + chunk])).to(device)
        d2 = (q ** 2).sum(1, keepdim=True) + ref_norm.unsqueeze(0) - 2.0 * (q @ ref_t.T)
        d2.clamp_(min=0)
        if exclude_self:
            idx = torch.arange(q.shape[0], device=device)
            d2[idx, i + idx] = float("inf")
        mn, ar = d2.min(1)
        min_mse.append((mn / D).cpu().numpy())
        arg.append(ar.cpu().numpy())
    return np.concatenate(min_mse), np.concatenate(arg)


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #
def describe(x):
    x = np.asarray(x, np.float64)
    return {"n": int(x.size), "mean": float(x.mean()), "std": float(x.std()),
            "min": float(x.min()), "p5": float(np.percentile(x, 5)),
            "q1": float(np.percentile(x, 25)), "median": float(np.median(x)),
            "q3": float(np.percentile(x, 75)), "max": float(x.max())}


def compare(d_real, d_fake):
    ks = stats.ks_2samp(d_fake, d_real)
    mwu = stats.mannwhitneyu(d_fake, d_real, alternative="two-sided")
    mwu_less = stats.mannwhitneyu(d_fake, d_real, alternative="less")
    return {"ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue),
            "mwu_two_sided_p": float(mwu.pvalue), "mwu_fake_less_p": float(mwu_less.pvalue),
            "median_ratio_fake_over_real": float(np.median(d_fake) / np.median(d_real)),
            "frac_fake_below_real_min": float(np.mean(d_fake < np.min(d_real))),
            "frac_fake_below_real_p5": float(np.mean(d_fake < np.percentile(d_real, 5))),
            "frac_fake_below_real_p1": float(np.mean(d_fake < np.percentile(d_real, 1))),
            "n_fake_below_real_min": int(np.sum(d_fake < np.min(d_real)))}


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_hist(d_real, d_fake, path, title):
    plt.figure(figsize=(8, 5))
    hi = max(np.percentile(d_real, 99.5), np.percentile(d_fake, 99.5))
    bins = np.linspace(min(d_real.min(), d_fake.min()), hi, 50)
    plt.hist(d_real, bins=bins, alpha=0.55, density=True, color="tab:blue",
             label=f"real test (n={d_real.size})")
    plt.hist(d_fake, bins=bins, alpha=0.55, density=True, color="tab:red",
             label=f"synthetic (n={d_fake.size})")
    plt.axvline(np.median(d_real), color="tab:blue", ls="--", lw=1)
    plt.axvline(np.median(d_fake), color="tab:red", ls="--", lw=1)
    plt.xlabel("MSE to nearest TRAIN latent"); plt.ylabel("density")
    plt.title(title); plt.legend(); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()


def plot_cdf(d_real, d_fake, path):
    plt.figure(figsize=(8, 5))
    for d, lab, c in [(d_real, "real test", "tab:blue"), (d_fake, "synthetic", "tab:red")]:
        xs = np.sort(d)
        plt.plot(xs, np.arange(1, xs.size + 1) / xs.size, label=lab, color=c)
    plt.xlabel("MSE to nearest TRAIN latent"); plt.ylabel("empirical CDF")
    plt.title("CDF of nearest-train distances (left shift = memorization)")
    plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_box(d_real, d_fake, path):
    plt.figure(figsize=(6, 5))
    plt.boxplot([d_real, d_fake], tick_labels=["real test", "synthetic"], showfliers=True)
    plt.ylabel("MSE to nearest TRAIN latent")
    plt.title("Nearest-train distance distributions")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--roundtrip-reals", action=argparse.BooleanOptionalAction, default=True,
                    help="represent reals as encode(decode(mu)) to match the fakes' "
                         "decode+encode pass (symmetric, correct). --no-roundtrip-reals "
                         "uses raw mu (asymmetric).")
    ap.add_argument("--real_as_sample", action="store_true",
                    help="reparameterize real-test latents (only used with --no-roundtrip-reals)")
    ap.add_argument("--outdir", default=paths.RESULTS_DIR)
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    os.makedirs(args.outdir, exist_ok=True)
    print(f"Device: {device}")

    rt = args.roundtrip_reals
    real, real_lab, real_std = load_real(paths.REAL_TEST_DIR, reparam=args.real_as_sample, use_rt=rt)
    train, train_lab, _ = load_real(paths.REAL_TRAIN_DIR, use_rt=rt)
    fake, fake_lab = load_fake(paths.SYNTH_DIR)
    real_repr = "roundtrip" if rt else ("reparam_sample" if args.real_as_sample else "mu")
    print(f"representations -> real/train: {real_repr} | fake: roundtrip")
    D = train.shape[1]
    print(f"shapes: real={real.shape}, fake={fake.shape}, train={train.shape} (D={D})")
    print(f"mean real-test posterior std = {real_std:.4e} "
          f"(large => z carries real noise; roundtrip removes it for all sets)")

    d_real, nn_real = nn_mse(real, train, device, args.chunk)
    d_fake, nn_fake = nn_mse(fake, train, device, args.chunk)

    results = {
        "n_train": int(train.shape[0]), "latent_dim": int(D),
        "real_representation": real_repr,
        "fake_representation": "roundtrip",
        "mean_real_posterior_std": real_std,
        "stats_d_real": describe(d_real), "stats_d_fake": describe(d_fake),
        "comparison": compare(d_real, d_fake), "per_class": {},
    }

    for cls in sorted(set(real_lab.tolist()) | set(fake_lab.tolist())):
        dr, df = d_real[real_lab == cls], d_fake[fake_lab == cls]
        if dr.size and df.size:
            name = paths.CLASS_NAMES.get(cls, str(cls))
            results["per_class"][name] = {"stats_d_real": describe(dr),
                                          "stats_d_fake": describe(df),
                                          "comparison": compare(dr, df)}
            plot_hist(dr, df, os.path.join(args.outdir, f"hist_{name}.png"),
                      f"Nearest-train MSE — {name}")

    d_real_self, _ = nn_mse(real, real, device, args.chunk, exclude_self=True)
    d_fake_self, _ = nn_mse(fake, fake, device, args.chunk, exclude_self=True)
    results["mode_collapse"] = {
        "within_real_nn": describe(d_real_self), "within_fake_nn": describe(d_fake_self),
        "diversity_ratio_fake_over_real": float(np.median(d_fake_self) / np.median(d_real_self)),
        "note": "ratio << 1 => synthetic samples cluster together => mode collapse"}

    results["coverage"] = {
        "unique_train_nn_of_real": int(np.unique(nn_real).size),
        "unique_train_nn_of_fake": int(np.unique(nn_fake).size),
        "n_queries": int(d_real.size)}

    plot_hist(d_real, d_fake, os.path.join(args.outdir, "hist_overall.png"),
              "Nearest-train MSE: real test vs synthetic")
    plot_cdf(d_real, d_fake, os.path.join(args.outdir, "cdf_overall.png"))
    plot_box(d_real, d_fake, os.path.join(args.outdir, "box_overall.png"))

    np.savez(os.path.join(args.outdir, "distances.npz"),
             d_real=d_real, d_fake=d_fake, real_lab=real_lab, fake_lab=fake_lab,
             nn_real_idx=nn_real, nn_fake_idx=nn_fake)
    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    c, mc, cov = results["comparison"], results["mode_collapse"], results["coverage"]
    print("\n================ SUMMARY ================")
    print(f"real representation : {results['real_representation']}")
    print(f"d_real  median={np.median(d_real):.4f}  mean={d_real.mean():.4f}  min={d_real.min():.4f}")
    print(f"d_fake  median={np.median(d_fake):.4f}  mean={d_fake.mean():.4f}  min={d_fake.min():.4f}")
    print(f"KS p={c['ks_p']:.3g} | MWU(2-sided) p={c['mwu_two_sided_p']:.3g} | "
          f"MWU(fake<real) p={c['mwu_fake_less_p']:.3g}")
    print(f"median ratio fake/real = {c['median_ratio_fake_over_real']:.3f}")
    print(f"fakes below real-min = {c['n_fake_below_real_min']} "
          f"({100*c['frac_fake_below_real_min']:.1f}%), below real-p5 = "
          f"{100*c['frac_fake_below_real_p5']:.1f}%")
    print(f"within-set NN median: real={mc['within_real_nn']['median']:.4f} "
          f"fake={mc['within_fake_nn']['median']:.4f} "
          f"(diversity ratio={mc['diversity_ratio_fake_over_real']:.3f})")
    print(f"coverage: fakes hit {cov['unique_train_nn_of_fake']} distinct train subjects, "
          f"reals hit {cov['unique_train_nn_of_real']} (of {results['n_train']})")
    print(f"\nWrote results to {args.outdir}")


if __name__ == "__main__":
    main()
