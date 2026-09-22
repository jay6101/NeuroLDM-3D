"""Grid figure of the four exported examples: real vs reconstruction in all three planes.

Columns are the good/bad HC and TLE examples, rows alternate real and reconstruction
for the coronal, sagittal and axial mid-slices, with MSE/LPIPS printed per column.
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

EX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "examples")
COLUMN_ORDER = [("HC", "good"), ("HC", "bad"), ("TLE", "good"), ("TLE", "bad")]
PLANES = [(1, "Coronal"), (0, "Sagittal"), (2, "Axial")]
IN_PLANE_AXES = {0: (1, 2), 1: (0, 2), 2: (0, 1)}


def load_examples(ex_dir):
    by_key = {}
    for folder in sorted(glob.glob(os.path.join(ex_dir, "*", "metrics.json"))):
        m = json.load(open(folder))
        d = os.path.dirname(folder)
        m["real"] = nib.load(os.path.join(d, "real_preprocessed.nii.gz")).get_fdata()
        m["recon"] = nib.load(os.path.join(d, "reconstruction.nii.gz")).get_fdata()
        by_key[(m["group"], m["quality"])] = m
    return [by_key[k] for k in COLUMN_ORDER]


def brain_bbox(vols, margin=2):
    """Bounding box of non-background voxels across all volumes, so every panel is framed alike."""
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for v in vols:
        mask = v > v.min() + 0.02 * (v.max() - v.min())
        for ax in range(3):
            idx = np.where(mask.any(axis=tuple(i for i in range(3) if i != ax)))[0]
            lo[ax], hi[ax] = min(lo[ax], idx[0]), max(hi[ax], idx[-1])
    lo = np.maximum(lo - margin, 0).astype(int)
    hi = np.minimum(hi + margin + 1, vols[0].shape).astype(int)
    return lo, hi


def mid_slice(vol, axis, bbox=None):
    """Middle slice of the full volume, optionally cropped in-plane to the shared bounding box."""
    sl = np.take(vol, vol.shape[axis] // 2, axis=axis)
    if bbox is not None:
        lo, hi = bbox
        a, b = IN_PLANE_AXES[axis]
        sl = sl[lo[a] : hi[a], lo[b] : hi[b]]
    return np.rot90(sl)


def column_label(examples):
    """How export_examples.py chose these four subjects, for the figure title."""
    selection = examples[0].get("selection", {})
    if selection.get("mode") == "percentile":
        low, high = selection["percentiles"]
        return f"{low:g}th and {high:g}th MSE percentile per group"
    return "best and worst example per group"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples-dir", default=EX_DIR, help="folder written by export_examples.py")
    ap.add_argument("--out", help="output path without extension (default <examples-dir>/examples_grid)")
    args = ap.parse_args()

    ex_dir = os.path.abspath(args.examples_dir)
    out_base = args.out or os.path.join(ex_dir, "examples_grid")

    examples = load_examples(ex_dir)
    bbox = brain_bbox([e["real"] for e in examples] + [e["recon"] for e in examples])
    shapes = [mid_slice(examples[0]["real"], axis, bbox).shape for axis, _ in PLANES]
    # equal column widths, so each row's height follows its own slice aspect ratio
    height_ratios = [r for h, w in shapes for r in (h / w, h / w)]

    fig = plt.figure(figsize=(8.6, 12.2))
    gs = fig.add_gridspec(6, 4, height_ratios=height_ratios,
                          hspace=0.04, wspace=0.03,
                          left=0.085, right=0.99, top=0.895, bottom=0.075)

    axes = np.empty((6, 4), dtype=object)
    for r, (axis, plane) in enumerate(PLANES):
        for half, kind in enumerate(("Real", "Reconstruction")):
            row = 2 * r + half
            for c, ex in enumerate(examples):
                ax = fig.add_subplot(gs[row, c])
                axes[row, c] = ax
                vol = ex["real"] if kind == "Real" else ex["recon"]
                vmin, vmax = ex["volume"]["display_window"]
                ax.imshow(mid_slice(vol, axis, bbox), cmap="gray", vmin=vmin, vmax=vmax,
                          interpolation="nearest", aspect="equal")
                ax.set_xticks([])
                ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
                if c == 0:
                    ax.set_ylabel(f"{plane}\n({kind})", fontsize=9.5, fontweight="bold", labelpad=8)
                if row == 0:
                    # split across two text objects so only the group/quality line is bold
                    lat = "" if ex["diagnosis"] == "HC" else f", {ex['diagnosis']}"
                    ax.text(0.5, 1.02, f"{ex['subject']}{lat}", transform=ax.transAxes,
                            ha="center", va="bottom", fontsize=9.5)
                    ax.set_title(f"{ex['group']} \u2014 {ex['quality']}",
                                 fontsize=10.5, fontweight="bold", pad=17)

    for c, ex in enumerate(examples):
        box = axes[5, c].get_position()
        fig.text(box.x0 + box.width / 2, box.y0 - 0.017,
                 f"MSE = {ex['mse']:.4f}\nLPIPS = {ex['lpips']:.4f}",
                 ha="center", va="top", fontsize=9.5)

    fig.suptitle(f"VAE reconstruction quality on held-out subjects: {column_label(examples)}",
                 fontsize=12.5, fontweight="bold", y=0.982, wrap=True)

    for ext in ("png", "pdf"):
        path = f"{out_base}.{ext}"
        fig.savefig(path, dpi=300 if ext == "png" else None, facecolor="white")
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
