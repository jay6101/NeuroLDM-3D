"""Export good/bad reconstruction examples (HC and TLE) as NIfTI volumes and slice PNGs.

Reads results/per_subject_metrics.csv, picks four held-out subjects, and writes
<out-dir>/<GROUP>_<quality>_<subject>/ with the preprocessed real volume,
its reconstruction, the absolute error map, middle-slice PNGs and the metrics.

Two ways to pick the four subjects (--selection):
  extreme     the best and worst subject per group (default)
  percentile  the subjects sitting at a low/high MSE percentile per group, so the
              examples are typical rather than outliers, e.g.
              --selection percentile --percentiles 5 95 --out-dir results/examples_p5_p95

Run with --figures-only to redraw the PNGs from an existing export (no GPU needed).
"""

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from common import (  # noqa: E402
    CSV_DIR,
    RUN_FOLDER,
    LPIPS3D,
    MultiCSVMRIDataset,
    load_vae,
    preprocessed_affine,
    reconstruct,
    voxel_metrics,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
EX_DIR = os.path.join(RESULTS_DIR, "examples")

# (axis to slice through, plane name, in-plane orientation after np.rot90)
PLANES = [
    (0, "sagittal", "superior up, anterior right"),
    (1, "coronal", "superior up, subject-left on image right"),
    (2, "axial", "anterior up, subject-left on image right"),
]

# Fixed |error| display range, shared by every example so the error maps and their
# colorbars are directly comparable across subjects. Values above are clipped.
ERROR_VMIN, ERROR_VMAX = 0.0, 1.0

EXAMPLE_ORDER = [("HC", "good"), ("HC", "bad"), ("TLE", "good"), ("TLE", "bad")]


def pick_extremes(df):
    """Best/worst subject per group, ranked on MSE and LPIPS percentiles jointly."""
    picks = []
    for group in ("HC", "TLE"):
        sub = df[df["group"] == group].copy()
        sub["score"] = (sub["mse"].rank(pct=True) + sub["lpips"].rank(pct=True)) / 2
        sub = sub.sort_values("score")
        for quality, row in (("good", sub.iloc[0]), ("bad", sub.iloc[-1])):
            picks.append((group, quality, row))
    return picks


def pick_mse_percentiles(df, low, high):
    """Subject closest to the low/high MSE percentile of its own group.

    Percentiles are taken over the group's held-out MSE distribution, then realised by
    the subject whose MSE is nearest the interpolated target, so the pick is an actual
    scan rather than a quantile value.
    """
    picks = []
    for group in ("HC", "TLE"):
        sub = df[df["group"] == group]
        for quality, pct in (("good", low), ("bad", high)):
            target = np.percentile(sub["mse"], pct)
            picks.append((group, quality, sub.loc[(sub["mse"] - target).abs().idxmin()]))
    return picks


def selection_label(selection):
    """How the four subjects were chosen, phrased for a figure title."""
    if selection["mode"] == "percentile":
        low, high = selection["percentiles"]
        return f"held-out examples at the {low:g}th and {high:g}th MSE percentile per group"
    return "best and worst held-out examples per group"


def save_slice_png(arr, path, vmin, vmax, upscale=4):
    """Write one slice as a single-channel 8-bit grayscale PNG, nearest-neighbour upscaled."""
    norm = np.clip((arr - vmin) / max(vmax - vmin, 1e-8), 0, 1)
    img = Image.fromarray(np.round(norm * 255).astype(np.uint8), mode="L")
    h, w = arr.shape
    img.resize((w * upscale, h * upscale), Image.NEAREST).save(path)


def mid_slice(vol, axis):
    idx = vol.shape[axis] // 2
    return np.rot90(np.take(vol, idx, axis=axis)), idx


def comparison_figure(real, recon, vmin, vmax, title, path):
    """3 planes x (real, reconstruction, |error|), every panel drawn at the same size.

    Row heights follow each plane's slice aspect ratio so the images fill their cells
    exactly, and the colorbars get their own axes next to the error panel: passing
    `ax=` to `colorbar` instead takes the space out of that panel and renders the
    error map smaller than the two grayscale panels beside it.
    """
    rows = []
    for axis, plane, _ in PLANES:
        rs, idx = mid_slice(real, axis)
        cs, _ = mid_slice(recon, axis)
        rows.append((plane, idx, rs, cs, np.abs(rs - cs)))

    left, right, top, bottom, wspace, hspace = 0.045, 0.90, 0.90, 0.015, 0.02, 0.05
    fig_w = 9.5
    height_ratios = [rs.shape[0] / rs.shape[1] for _, _, rs, _, _ in rows]
    cell_w = fig_w * (right - left) / (3 + 2 * wspace)
    fig_h = cell_w * sum(height_ratios) * (3 + 2 * hspace) / 3 / (top - bottom)

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(3, 3, height_ratios=height_ratios, left=left, right=right,
                          top=top, bottom=bottom, wspace=wspace, hspace=hspace)

    error_panels = []
    for r, (plane, idx, rs, cs, ds) in enumerate(rows):
        for c, (img, lab, cmap, lo, hi) in enumerate(
            [(rs, "real", "gray", vmin, vmax),
             (cs, "reconstruction", "gray", vmin, vmax),
             (ds, "|error|", "inferno", ERROR_VMIN, ERROR_VMAX)]
        ):
            ax = fig.add_subplot(gs[r, c])
            im = ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
            ax.set_axis_off()
            if r == 0:
                ax.set_title(lab, fontsize=12)
            if c == 0:
                ax.text(-0.06, 0.5, f"{plane}\n(index {idx})", transform=ax.transAxes,
                        rotation=90, va="center", ha="center", fontsize=10)
            if c == 2:
                error_panels.append((ax, im))

    fig.canvas.draw()  # settles the aspect-adjusted panel boxes the colorbars align to
    for ax, im in error_panels:
        box = ax.get_position()
        cax = fig.add_axes([box.x1 + 0.012, box.y0, 0.017, box.height])
        cb = fig.colorbar(im, cax=cax, extend="max")
        cb.set_ticks(np.linspace(ERROR_VMIN, ERROR_VMAX, 6))
        cb.ax.tick_params(labelsize=9)

    fig.suptitle(title, fontsize=12)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def tile_comparisons(folders, path, pad=40):
    """Paste the four per-subject comparison.png files into one 2x2 image."""
    tiles = [Image.open(os.path.join(f, "comparison.png")).convert("RGB") for f in folders]
    cell_w, cell_h = max(t.width for t in tiles), max(t.height for t in tiles)
    sheet = Image.new("RGB", (2 * cell_w + 3 * pad, 2 * cell_h + 3 * pad), "white")
    for i, tile in enumerate(tiles):
        x = pad + (i % 2) * (cell_w + pad) + (cell_w - tile.width) // 2
        y = pad + (i // 2) * (cell_h + pad) + (cell_h - tile.height) // 2
        sheet.paste(tile, (x, y))
    sheet.save(path)


def write_slice_pngs(real, recon, vmin, vmax, out):
    """Mid-slice PNGs per plane: real, reconstruction and the |error| map."""
    os.makedirs(os.path.join(out, "slices_png"), exist_ok=True)
    for axis, plane, _ in PLANES:
        rs, idx = mid_slice(real, axis)
        cs, _ = mid_slice(recon, axis)
        tag = f"{plane}_axis{axis}_idx{idx:03d}"
        save_slice_png(rs, os.path.join(out, "slices_png", f"real_{tag}.png"), vmin, vmax)
        save_slice_png(cs, os.path.join(out, "slices_png", f"recon_{tag}.png"), vmin, vmax)
        save_slice_png(np.abs(rs - cs), os.path.join(out, "slices_png", f"abserr_{tag}.png"),
                       ERROR_VMIN, ERROR_VMAX)


def figure_title(m):
    return (f"{m['group']} / {m['quality']} reconstruction - {m['subject']} ({m['diagnosis']})\n"
            f"MSE = {m['mse']:.4f}   LPIPS = {m['lpips']:.4f}   "
            f"MAE = {m['mae']:.4f}   PSNR = {m['psnr_db']:.2f} dB")


def overview_figure(entries, path, label="best and worst held-out examples per group"):
    """One axial + coronal + sagittal row per example, real vs reconstruction."""
    fig, axes = plt.subplots(len(entries), 6, figsize=(15, 2.7 * len(entries)))
    for r, e in enumerate(entries):
        real, recon = e["real"], e["recon"]
        vmin, vmax = e["vmin"], e["vmax"]
        col = 0
        for axis, plane, _ in PLANES:
            rs, _ = mid_slice(real, axis)
            cs, _ = mid_slice(recon, axis)
            for img, lab in ((rs, "real"), (cs, "recon")):
                ax = axes[r, col]
                ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
                ax.set_axis_off()
                if r == 0:
                    ax.set_title(f"{plane}\n{lab}", fontsize=10)
                col += 1
        axes[r, 0].text(-0.12, 0.5,
                        f"{e['group']} {e['quality']}\n{e['subject']}\n"
                        f"MSE {e['mse']:.4f}\nLPIPS {e['lpips']:.4f}",
                        transform=axes[r, 0].transAxes, rotation=0, va="center",
                        ha="right", fontsize=9)
    fig.suptitle(f"VAE reconstructions: {label}", fontsize=13)
    fig.tight_layout(rect=[0.06, 0, 1, 0.94])
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def rebuild_figures(ex_dir):
    """Redraw the slice PNGs and figures from already exported volumes, without the model."""
    by_key, label = {}, "best and worst held-out examples per group"
    for mpath in glob.glob(os.path.join(ex_dir, "*", "metrics.json")):
        out = os.path.dirname(mpath)
        with open(mpath) as f:
            m = json.load(f)
        real = nib.load(os.path.join(out, "real_preprocessed.nii.gz")).get_fdata().astype(np.float32)
        recon = nib.load(os.path.join(out, "reconstruction.nii.gz")).get_fdata().astype(np.float32)
        vmin, vmax = m["volume"]["display_window"]

        m["volume"]["error_display_window"] = [ERROR_VMIN, ERROR_VMAX]
        with open(mpath, "w") as f:
            json.dump(m, f, indent=2)
        if "selection" in m:
            label = selection_label(m["selection"])

        write_slice_pngs(real, recon, vmin, vmax, out)
        comparison_figure(real, recon, vmin, vmax, figure_title(m),
                          os.path.join(out, "comparison.png"))
        by_key[(m["group"], m["quality"])] = {
            "real": real, "recon": recon, "vmin": vmin, "vmax": vmax,
            "group": m["group"], "quality": m["quality"], "subject": m["subject"],
            "mse": m["mse"], "lpips": m["lpips"], "folder": out,
        }
        print(f"redrew {out}")

    entries = [by_key[k] for k in EXAMPLE_ORDER]
    overview_figure(entries, os.path.join(ex_dir, "all_examples_overview.png"), label)
    tile_comparisons([e["folder"] for e in entries], os.path.join(ex_dir, "all_comparisons.png"))
    print(f"\nFigures rebuilt in {ex_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--figures-only", action="store_true",
                    help="redraw figures from the exported NIfTIs instead of rerunning the VAE")
    ap.add_argument("--out-dir", default=EX_DIR, help="where to write the example folders")
    ap.add_argument("--selection", choices=("extreme", "percentile"), default="extreme",
                    help="pick the best/worst subject per group, or one at a given MSE percentile")
    ap.add_argument("--percentiles", nargs=2, type=float, default=(5.0, 95.0),
                    metavar=("GOOD", "BAD"),
                    help="MSE percentiles for the good/bad pick with --selection percentile")
    args = ap.parse_args()

    ex_dir = os.path.abspath(args.out_dir)
    if args.figures_only:
        rebuild_figures(ex_dir)
        return

    low, high = args.percentiles
    selection = ({"mode": "percentile", "percentiles": [low, high],
                  "metric": "mse", "cohort": "own group, held-out"}
                 if args.selection == "percentile" else
                 {"mode": "extreme", "metric": "mean of mse and lpips percentile ranks",
                  "cohort": "own group, held-out"})

    df = pd.read_csv(os.path.join(RESULTS_DIR, "per_subject_metrics.csv"))
    held = df[df["split"] == "test"].reset_index(drop=True)

    ds = MultiCSVMRIDataset([os.path.join(CSV_DIR, "val.csv"), os.path.join(CSV_DIR, "test.csv")], train=False)
    path_to_idx = {p: i for i, (p, _) in enumerate(ds.samples)}

    vae = load_vae(RUN_FOLDER, args.device)
    lpips_fn = LPIPS3D("alex", chunk=64).to(args.device).eval()
    for p in lpips_fn.parameters():
        p.requires_grad = False

    os.makedirs(ex_dir, exist_ok=True)
    entries, summary_rows = [], []

    picks = (pick_mse_percentiles(held, low, high) if args.selection == "percentile"
             else pick_extremes(held))
    for group, quality, row in picks:
        src = row["file"]
        image, _, _ = ds[path_to_idx[src]]
        x = image.unsqueeze(0).float().to(args.device)
        with torch.no_grad():
            recon_t, _, _ = reconstruct(vae, x, stochastic=False)
            mse = voxel_metrics(x, recon_t)[0].item()
            lp = lpips_fn(recon_t, x)[0].item()
        assert abs(mse - row["mse"]) < 1e-5 and abs(lp - row["lpips"]) < 1e-5, "metrics drifted"

        real = x[0, 0].cpu().numpy().astype(np.float32)
        recon = recon_t[0, 0].cpu().numpy().astype(np.float32)
        diff = np.abs(real - recon)

        subject = row["sbj_nme"]
        out = os.path.join(ex_dir, f"{group}_{quality}_{subject}")
        os.makedirs(out, exist_ok=True)

        orig = nib.load(src)
        affine = preprocessed_affine(orig.affine, orig.shape, real.shape, ds.SLICE_START)
        for vol, fname in ((real, "real_preprocessed"), (recon, "reconstruction"), (diff, "abs_error")):
            img = nib.Nifti1Image(vol, affine)
            img.header.set_xyzt_units("mm")
            nib.save(img, os.path.join(out, f"{fname}.nii.gz"))

        vmin, vmax = np.percentile(real, [0.5, 99.5])
        write_slice_pngs(real, recon, vmin, vmax, out)

        metrics = {
            "subject": subject,
            "group": group,
            "diagnosis": row["diagnosis"],
            "quality": quality,
            "split": "held-out (val.csv U test.csv)",
            "selection": selection,
            "in_test_csv": bool(row["in_test_csv"]),
            "mse": mse,
            "lpips": lp,
            "lpips_norm11": float(row["lpips_norm11"]),
            "mae": float(row["mae"]),
            "psnr_db": float(row["psnr"]),
            "rank_within_group": {
                "n_in_group": int((held["group"] == group).sum()),
                "mse_percentile": float((held[held["group"] == group]["mse"] < mse).mean() * 100),
                "lpips_percentile": float((held[held["group"] == group]["lpips"] < lp).mean() * 100),
            },
            "volume": {
                "shape": list(real.shape),
                "axes": "axis0 = L-R (sagittal), axis1 = P-A (coronal), axis2 = I-S (axial)",
                "intensity": "per-volume z-scored, identical scale for real and reconstruction",
                "display_window": [float(vmin), float(vmax)],
                "error_display_window": [ERROR_VMIN, ERROR_VMAX],
            },
            "source_image": src,
        }
        with open(os.path.join(out, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        comparison_figure(real, recon, vmin, vmax, figure_title(metrics),
                          os.path.join(out, "comparison.png"))

        entries.append({"real": real, "recon": recon, "vmin": vmin, "vmax": vmax,
                        "group": group, "quality": quality, "subject": subject,
                        "mse": mse, "lpips": lp, "folder": out})
        summary_rows.append({k: v for k, v in metrics.items()
                             if k not in ("selection", "rank_within_group", "volume")}
                            | {"mse_percentile_in_group": metrics["rank_within_group"]["mse_percentile"],
                               "folder": os.path.basename(out)})
        print(f"{group:>3s} {quality:>4s}  {subject}  MSE={mse:.4f}  LPIPS={lp:.4f}  "
              f"MSE pct in {group}={metrics['rank_within_group']['mse_percentile']:.1f}  -> {out}")

    pd.DataFrame(summary_rows).to_csv(os.path.join(ex_dir, "examples_summary.csv"), index=False)
    label = selection_label(selection)
    overview_figure(entries, os.path.join(ex_dir, "all_examples_overview.png"), label)
    tile_comparisons([e["folder"] for e in entries], os.path.join(ex_dir, "all_comparisons.png"))
    print(f"\nExamples written to {ex_dir}")


if __name__ == "__main__":
    main()
