"""Per-ROI reconstruction MSE for the VAE, over the held-out (val + test) subjects.

Atlas labels are resampled into the preprocessed grid, grouped into the TLE-relevant
structures, and used to mask the squared error of each reconstruction. Outputs
per-subject/per-ROI values, a summary table, QC overlays and violin plots.

--atlas selects the parcellation: Harvard-Oxford maxprob (default, -> results/) or
AAL3v2 (-> results_aal3/). Both produce the same seven structures, the same file names
and the same figures; only the voxel definitions differ.
"""

import argparse
import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (  # noqa: E402
    CSV_DIR,
    RUN_FOLDER,
    MultiCSVMRIDataset,
    load_vae,
    preprocessed_affine,
    reconstruct,
)

HERE = os.path.dirname(os.path.abspath(__file__))

# Each ROI is the union of the listed Harvard-Oxford maxprob labels.
HARVARD_OXFORD_ROI_DEFS = [
    ("Hippocampus", "sub", ["Left Hippocampus", "Right Hippocampus"]),
    ("Parahippocampal gyrus", "cort", ["Parahippocampal Gyrus, anterior division",
                                       "Parahippocampal Gyrus, posterior division"]),
    ("Amygdala", "sub", ["Left Amygdala", "Right Amygdala"]),
    ("Thalamus", "sub", ["Left Thalamus", "Right Thalamus"]),
    ("Insula", "cort", ["Insular Cortex"]),
    ("Cingulate cortex", "cort", ["Cingulate Gyrus, anterior division",
                                  "Cingulate Gyrus, posterior division"]),
    ("Anterior temporal cortex", "cort", ["Temporal Pole",
                                          "Superior Temporal Gyrus, anterior division",
                                          "Middle Temporal Gyrus, anterior division",
                                          "Inferior Temporal Gyrus, anterior division"]),
]

# AAL3 replaced the single thalamus label of AAL1/2 with 15 nuclei per hemisphere,
# so the thalamus ROI is their union.
AAL3_THALAMIC_NUCLEI = [f"Thal_{nucleus}_{side}"
                        for nucleus in ("AV", "LP", "VA", "VL", "VPL", "IL", "Re", "MDm",
                                        "MDl", "LGN", "MGN", "PuI", "PuM", "PuA", "PuL")
                        for side in ("L", "R")]

# The AAL3 counterparts of the Harvard-Oxford groupings above. AAL3 splits the anterior
# cingulate into subgenual, pregenual and supracallosal parts, and has no anterior/
# posterior subdivision of the temporal gyri, so the anterior temporal ROI is the
# superior and middle temporal poles.
AAL3_ROI_DEFS = [
    ("Hippocampus", "aal3", ["Hippocampus_L", "Hippocampus_R"]),
    ("Parahippocampal gyrus", "aal3", ["ParaHippocampal_L", "ParaHippocampal_R"]),
    ("Amygdala", "aal3", ["Amygdala_L", "Amygdala_R"]),
    ("Thalamus", "aal3", AAL3_THALAMIC_NUCLEI),
    ("Insula", "aal3", ["Insula_L", "Insula_R"]),
    ("Cingulate cortex", "aal3", ["ACC_sub_L", "ACC_sub_R",
                                  "ACC_pre_L", "ACC_pre_R",
                                  "ACC_sup_L", "ACC_sup_R",
                                  "Cingulate_Mid_L", "Cingulate_Mid_R",
                                  "Cingulate_Post_L", "Cingulate_Post_R"]),
    ("Anterior temporal cortex", "aal3", ["Temporal_Pole_Sup_L", "Temporal_Pole_Sup_R",
                                          "Temporal_Pole_Mid_L", "Temporal_Pole_Mid_R"]),
]

ATLASES = {
    "harvard-oxford": {"roi_defs": HARVARD_OXFORD_ROI_DEFS,
                       "results_dir": "results", "display": "Harvard-Oxford"},
    "aal3": {"roi_defs": AAL3_ROI_DEFS,
             "results_dir": "results_aal3", "display": "AAL3"},
}
DEFAULT_ATLAS = "harvard-oxford"

# Both atlases resolve to the same seven structures, so the plots stay comparable.
ROI_NAMES = [n for n, _, _ in HARVARD_OXFORD_ROI_DEFS]
GM_REFERENCE = "Whole brain (non-background)"
NATIVE = " [mwp1]"

# Keyed on ROI name rather than plot position, so a structure keeps the same colour
# in the QC overlay and the violin plots however the violins end up sorted.
ROI_COLORS = {n: plt.get_cmap("tab10")(i) for i, n in enumerate(ROI_NAMES)}
ROI_COLORS[GM_REFERENCE] = "#999999"

# Matches the FID comparison figures (fid_results_ignite/create_clean_comparison_plots.py)
# so every distribution figure in the paper reads the same way.
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
GROUP_COLORS = {"HC": "#4682B4", "TLE": "#FF6347"}  # Steel Blue / Tomato, as in the FID figures

PREP_SHAPE = (112, 136, 112)
ORIG_SHAPE = (113, 137, 113)
ORIG_AFFINE = np.array([[-1.5, 0, 0, 84.0], [0, 1.5, 0, -120.0], [0, 0, 1.5, -72.0], [0, 0, 0, 1.0]])
SLICE_START = 1


def resample_labels(atlas_img, prep_affine, shape):
    """Nearest-neighbour pull of an atlas label volume onto the preprocessed grid."""
    data = np.asarray(atlas_img.dataobj).astype(np.int16)
    ijk = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"), -1).reshape(-1, 3)
    world = nib.affines.apply_affine(prep_affine, ijk)
    src = np.rint(nib.affines.apply_affine(np.linalg.inv(atlas_img.affine), world)).astype(int)
    ok = np.all((src >= 0) & (src < np.array(data.shape)), axis=1)
    out = np.zeros(len(ijk), np.int16)
    out[ok] = data[tuple(src[ok].T)]
    return out.reshape(shape)


def atlas_sources(atlas_key):
    """Label volumes for one atlas, as source -> (image, label name -> voxel value, description)."""
    from nilearn import datasets

    if atlas_key == "harvard-oxford":
        sources = {}
        for src in ("cort", "sub"):
            version = f"{src}-maxprob-thr25-1mm"
            a = datasets.fetch_atlas_harvard_oxford(version)
            img = nib.load(a.filename) if isinstance(a.maps, str) else a.maps
            # Harvard-Oxford voxel values are positions in the label list
            sources[src] = (img, {label: i for i, label in enumerate(a.labels)},
                            f"Harvard-Oxford {version}")
        return sources

    a = datasets.fetch_atlas_aal(version="3v2")
    # The fetcher returns the 2 mm map; the 1 mm volume shipped alongside it matches the
    # Harvard-Oxford run's resolution, so the nearest-neighbour pull loses less detail.
    maps = os.path.join(os.path.dirname(a.maps), "AAL3v1_1mm.nii")
    resolution = "AAL3v1_1mm"
    if not os.path.exists(maps):
        maps, resolution = a.maps, os.path.basename(a.maps)
    # AAL3 voxel values are not consecutive and do not index the label list
    values = {label: int(index) for label, index in zip(a.labels, a.indices)}
    return {"aal3": (nib.load(maps), values, f"AAL3v2 ({resolution})")}


def build_masks(prep_affine, roi_defs, sources):
    labelled = {k: resample_labels(img, prep_affine, PREP_SHAPE)
                for k, (img, _, _) in sources.items()}

    masks, provenance = {}, {}
    for name, src, wanted in roi_defs:
        _, values, description = sources[src]
        codes = [values[w] for w in wanted]
        masks[name] = np.isin(labelled[src], codes)
        provenance[name] = {"atlas": description, "labels": wanted, "label_values": codes}
    return masks, provenance, labelled


def hemisphere_map(prep_affine, shape):
    """World-x sign per voxel; negative x is the left hemisphere in MNI convention."""
    ijk = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"), -1).reshape(-1, 3)
    x = nib.affines.apply_affine(prep_affine, ijk)[:, 0].reshape(shape)
    return x < 0


def mean_real_volume(ds, n, seed=0):
    idx = np.random.default_rng(seed).choice(len(ds), size=min(n, len(ds)), replace=False)
    acc = np.zeros(PREP_SHAPE, np.float64)
    for i in idx:
        acc += ds[int(i)][0][0].numpy()
    return (acc / len(idx)).astype(np.float32)


def qc_figure(mean_vol, masks, gm_mask, path, atlas_display):
    """Overlay each ROI on the group-mean grey-matter map to eyeball atlas alignment."""
    planes = [(1, "Coronal"), (0, "Sagittal"), (2, "Axial")]
    # centre each view on the hippocampus so the temporal ROIs are visible
    hipp = np.argwhere(masks["Hippocampus"])
    centre = hipp.mean(0).round().astype(int)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    vmin, vmax = np.percentile(mean_vol, [0.5, 99.5])
    for ax, (axis, plane) in zip(axes, planes):
        bg = np.rot90(np.take(mean_vol, centre[axis], axis=axis))
        ax.imshow(bg, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        for name in masks:
            sl = np.rot90(np.take(masks[name], centre[axis], axis=axis))
            if sl.any():
                rgba = np.zeros(sl.shape + (4,))
                rgba[sl] = ROI_COLORS[name]
                rgba[..., 3] = sl * 0.55
                ax.imshow(rgba, interpolation="nearest")
        outline = np.rot90(np.take(gm_mask, centre[axis], axis=axis))
        ax.contour(outline, levels=[0.5], colors="w", linewidths=0.5, alpha=0.6)
        ax.set_title(f"{plane} (index {centre[axis]})", fontsize=11, fontweight="bold")
        ax.set_axis_off()
    handles = [mpatches.Patch(color=ROI_COLORS[n], label=f"{n} ({int(masks[n].sum())} vox)") for n in ROI_NAMES]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.suptitle(f"Atlas QC: {atlas_display} ROIs on the group-mean preprocessed GM map",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0.13, 1, 0.95])
    fig.savefig(path, dpi=200)
    plt.close(fig)


@torch.no_grad()
def compute_roi_mse(ds, masks, order, vae, device, batch_size, num_workers):
    flat = torch.stack([torch.from_numpy(masks[n].reshape(-1)).float() for n in order]).to(device)
    counts = flat.sum(1)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    rows, t0 = [], time.time()
    for i, (images, _, paths) in enumerate(loader):
        images = images.float().to(device, non_blocking=True)
        recon, _, _ = reconstruct(vae, images, stochastic=False)
        se = (recon - images).pow(2).flatten(1)
        roi = (se @ flat.T) / counts
        for j, path in enumerate(paths):
            rows.append({"file": path, "mse_all_voxels": se[j].mean().item(),
                         **{n: roi[j, k].item() for k, n in enumerate(order)}})
        if (i + 1) % 30 == 0 or len(rows) == len(ds):
            el = time.time() - t0
            print(f"  {len(rows)}/{len(ds)}  {el:.0f}s", flush=True)
    return pd.DataFrame(rows)


def order_by_hc_median(df, names, suffix=""):
    """ROI order, lowest to highest HC median, so the violins rise left to right."""
    hc = df[df.group == "HC"]
    return sorted(names, key=lambda n: hc[n + suffix].median())


def draw_violins(ax, data, positions, width, colors):
    parts = ax.violinplot(data, positions=positions, widths=width,
                          showmeans=True, showmedians=True)
    for body, values, color in zip(parts["bodies"], data, colors):
        body.set_facecolor(color)
        body.set_alpha(0.7)
        body.set_edgecolor("black")
        body.set_linewidth(1)
        # trim the kernel density to the observed range; MSE cannot be negative
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


def violin_plot(df, order, path, title, ylabel, suffix="", by_group=False):
    with plt.style.context(["seaborn-v0_8", PLOT_STYLE]):
        fig, ax = plt.subplots(figsize=(13, 6.5))
        drawn = []
        if by_group:
            for grp, off in (("HC", -0.2), ("TLE", 0.2)):
                data = [df.loc[df.group == grp, n + suffix].to_numpy() for n in order]
                draw_violins(ax, data, np.arange(len(order)) + off, 0.34,
                             [GROUP_COLORS[grp]] * len(order))
                drawn += data
            handles = [plt.Line2D([0], [0], color=c, lw=4,
                                  label=f"{g} (n = {int((df.group == g).sum())})")
                       for g, c in GROUP_COLORS.items()]
            ax.legend(handles=handles, loc="upper left", title="Diagnosis",
                      title_fontsize=11, framealpha=0.9)
        else:
            drawn = [df[n + suffix].to_numpy() for n in order]
            draw_violins(ax, drawn, np.arange(len(order)), 0.7, [ROI_COLORS[n] for n in order])
            for k, v in enumerate(drawn):
                ax.scatter(np.full(len(v), k) + np.random.default_rng(0).normal(0, 0.055, len(v)),
                           v, s=3, color="black", alpha=0.15, zorder=3, linewidths=0)

        # a handful of extreme subjects would otherwise flatten every violin
        top = max(np.percentile(v, 97.5) for v in drawn) * 1.15
        n_hidden = int(sum((v > top).sum() for v in drawn))
        if n_hidden:
            ax.text(0.99, 0.955, f"{n_hidden} of {sum(v.size for v in drawn)} subject values above axis limit",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8, color="0.35",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.85))
        ax.set_ylim(0, top)

        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels([n.replace(" (non-background)", "\n(non-background)") for n in order],
                           rotation=45, ha="right")
        ax.set_xlabel("Region of interest", fontsize=14, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=14, fontweight="bold")
        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
        ax.grid(True, alpha=0.3, linestyle="--", color="gray")
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


def write_violin_plots(df, names, results_dir):
    # file names are kept from the earlier boxplot version so existing references still resolve
    n_sub = len(df)
    violin_plot(df, order_by_hc_median(df, names), os.path.join(results_dir, "roi_mse_boxplot"),
                f"VAE reconstruction MSE by ROI (held-out val + test, n = {n_sub})",
                "MSE, per-volume z-scored units")
    violin_plot(df, order_by_hc_median(df, names, NATIVE), os.path.join(results_dir, "roi_mse_boxplot_native"),
                f"VAE reconstruction MSE by ROI, native mwp1 units (held-out val + test, n = {n_sub})",
                "MSE, native mwp1 grey-matter units", suffix=NATIVE)
    violin_plot(df, order_by_hc_median(df, names), os.path.join(results_dir, "roi_mse_boxplot_by_group"),
                "VAE reconstruction MSE by ROI and diagnosis (held-out val + test)",
                "MSE, per-volume z-scored units", by_group=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", default=DEFAULT_ATLAS, choices=sorted(ATLASES),
                    help="parcellation to group into the seven ROIs")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=10)
    ap.add_argument("--qc-only", action="store_true")
    ap.add_argument("--plots-only", action="store_true",
                    help="redraw the violin plots from an existing per_subject_roi_mse.csv")
    args = ap.parse_args()

    atlas = ATLASES[args.atlas]
    roi_defs = atlas["roi_defs"]
    results_dir = os.path.join(HERE, atlas["results_dir"])
    os.makedirs(results_dir, exist_ok=True)
    print(f"atlas: {atlas['display']}  ->  {results_dir}")

    if args.plots_only:
        df = pd.read_csv(os.path.join(results_dir, "per_subject_roi_mse.csv"))
        write_violin_plots(df, ROI_NAMES + [GM_REFERENCE], results_dir)
        print(f"violin plots redrawn from per_subject_roi_mse.csv (n = {len(df)}) in {results_dir}")
        return

    prep_affine = preprocessed_affine(ORIG_AFFINE, ORIG_SHAPE, PREP_SHAPE, SLICE_START)

    ds = MultiCSVMRIDataset([os.path.join(CSV_DIR, "val.csv"), os.path.join(CSV_DIR, "test.csv")], train=False)
    print(f"held-out volumes: {len(ds)}")

    masks, provenance, labelled = build_masks(prep_affine, roi_defs, atlas_sources(args.atlas))
    print("building group-mean GM map for the GM mask and QC ...")
    mean_vol = mean_real_volume(ds, n=60)
    # Brain mask: everything above the constant background level, as a whole-brain reference
    gm_mask = mean_vol > mean_vol.min() + 0.10 * (mean_vol.max() - mean_vol.min())
    masks[GM_REFERENCE] = gm_mask
    provenance[GM_REFERENCE] = {"atlas": "data-driven",
                                "labels": ["group-mean intensity > background + 10% of range, 60 held-out subjects"],
                                "label_values": []}
    order = ROI_NAMES + [GM_REFERENCE]

    left = hemisphere_map(prep_affine, PREP_SHAPE)
    stats = []
    for n in order:
        m = masks[n]
        stats.append({"roi": n, "n_voxels": int(m.sum()),
                      "volume_ml": float(m.sum() * np.abs(np.linalg.det(prep_affine[:3, :3])) / 1000),
                      "n_left": int((m & left).sum()), "n_right": int((m & ~left).sum()),
                      "mean_gm_in_roi": float(mean_vol[m].mean())})
    print(pd.DataFrame(stats).round(3).to_string(index=False))

    label_vol = np.zeros(PREP_SHAPE, np.int16)
    for k, n in enumerate(ROI_NAMES, start=1):
        label_vol[masks[n]] = k
    nib.save(nib.Nifti1Image(label_vol, prep_affine), os.path.join(results_dir, "roi_labels.nii.gz"))
    with open(os.path.join(results_dir, "roi_definitions.json"), "w") as f:
        json.dump({"grid": {"shape": list(PREP_SHAPE), "affine": prep_affine.tolist()},
                   "label_index": {str(k): n for k, n in enumerate(ROI_NAMES, start=1)},
                   "rois": provenance, "voxel_stats": stats}, f, indent=2)

    qc_figure(mean_vol, {n: masks[n] for n in ROI_NAMES}, gm_mask,
              os.path.join(results_dir, "atlas_qc_overlay.png"), atlas["display"])
    print(f"QC overlay -> {os.path.join(results_dir, 'atlas_qc_overlay.png')}")
    if args.qc_only:
        return

    vae = load_vae(RUN_FOLDER, args.device)
    print("computing per-ROI MSE ...")
    df = compute_roi_mse(ds, masks, order, vae, args.device, args.batch_size, args.num_workers)

    meta = pd.read_csv(os.path.join(os.path.dirname(HERE), "results", "per_subject_metrics.csv"))
    meta = meta[meta.split == "test"][["file", "sbj_nme", "diagnosis", "group", "in_test_csv",
                                       "mse", "norm_std"]]
    df = meta.merge(df, on="file").rename(columns={"mse": "mse_whole_volume"})
    # the model works on per-volume z-scored data; undo it to get native mwp1 units
    for n in order:
        df[n + NATIVE] = df[n] * df["norm_std"] ** 2
    df.to_csv(os.path.join(results_dir, "per_subject_roi_mse.csv"), index=False)

    summary = pd.DataFrame([
        {"roi": n, "n_subjects": len(sub), "group": g, "scale": scale,
         "mean": sub[col].mean(), "std": sub[col].std(ddof=1), "median": sub[col].median(),
         "q1": sub[col].quantile(0.25), "q3": sub[col].quantile(0.75),
         "min": sub[col].min(), "max": sub[col].max()}
        for n in order
        for scale, col in [("zscored", n), ("native_mwp1", n + NATIVE)]
        for g, sub in [("all", df), ("HC", df[df.group == "HC"]), ("TLE", df[df.group == "TLE"])]
    ])
    summary.to_csv(os.path.join(results_dir, "roi_mse_summary.csv"), index=False)

    write_violin_plots(df, order, results_dir)

    print("\n=== ROI MSE, all held-out subjects ===")
    for n in order:
        z = summary[(summary.roi == n) & (summary.group == "all") & (summary.scale == "zscored")].iloc[0]
        nat = summary[(summary.roi == n) & (summary.group == "all") & (summary.scale == "native_mwp1")].iloc[0]
        print(f"  {n:30s} z-scored {z['mean']:.4f} +/- {z['std']:.4f}   "
              f"native {nat['mean']:.2e} +/- {nat['std']:.1e}")
    print(f"\nWritten to {results_dir}")


if __name__ == "__main__":
    main()
