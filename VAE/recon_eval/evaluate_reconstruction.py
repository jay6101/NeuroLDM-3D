"""Per-volume MSE / LPIPS for the trained VAE on the train and held-out splits.

Writes per-subject metrics plus mean/std summaries into results/.
"""

import argparse
import json
import os
import time

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from common import (
    CSV_DIR,
    RUN_FOLDER,
    LPIPS3D,
    MultiCSVMRIDataset,
    load_vae,
    reconstruct,
    to_unit_range,
    voxel_metrics,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
METRIC_COLS = ["mse", "mse_native", "rmse_native", "nmse", "lpips", "lpips_norm11",
               "mae", "mae_native", "psnr", "mse_sampled", "lpips_sampled"]


class NormStatsDataset(Dataset):
    """Recovers the per-volume mean/std that the preprocessing z-scoring divided by.

    Needed to express errors in native mwp1 units: the crop and resize are linear, so
    de-standardising commutes with them and MSE_native = MSE_zscored * std**2.
    """

    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        vol = nib.load(self.paths[i]).get_fdata()
        return i, float(vol.mean()), float(vol.std())


def compute_norm_stats(paths, num_workers):
    loader = DataLoader(NormStatsDataset(paths), batch_size=8, num_workers=num_workers)
    mean, std = np.zeros(len(paths)), np.zeros(len(paths))
    for idx, m, s in loader:
        mean[idx.numpy()] = m.numpy()
        std[idx.numpy()] = s.numpy()
    return mean, std


def build_metadata():
    """file path -> subject id, diagnosis and split membership, taken from the CSVs."""
    meta = {}
    for split in ("train", "val", "test"):
        df = pd.read_csv(os.path.join(CSV_DIR, f"{split}.csv"))
        df = df.loc[df["HC_vs_LTLE_vs_RTLE_string"].isin(["right", "left", "HC"])]
        for _, row in df.iterrows():
            rec = meta.setdefault(
                row["file"],
                {
                    "sbj_nme": row["sbj_nme"],
                    "diagnosis": row["HC_vs_LTLE_vs_RTLE_string"],
                    "in_train_csv": False,
                    "in_val_csv": False,
                    "in_test_csv": False,
                },
            )
            rec[f"in_{split}_csv"] = True
    for rec in meta.values():
        rec["group"] = "HC" if rec["diagnosis"] == "HC" else "TLE"
    return meta


@torch.no_grad()
def evaluate_split(name, csv_files, vae, lpips_fn, meta, device, batch_size, num_workers, seed):
    ds = MultiCSVMRIDataset([os.path.join(CSV_DIR, c) for c in csv_files], train=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    print(f"\n[{name}] {len(ds)} volumes")

    torch.manual_seed(seed)
    rows, t0 = [], time.time()
    for i, (images, _, paths) in enumerate(loader):
        images = images.float().to(device, non_blocking=True)

        recon, _, _ = reconstruct(vae, images, stochastic=False)
        mse, mae, psnr = voxel_metrics(images, recon)
        lp = lpips_fn(recon, images)
        real_n, recon_n = to_unit_range(images, recon)
        lp_n = lpips_fn(recon_n, real_n)

        recon_s, _, _ = reconstruct(vae, images, stochastic=True)
        mse_s = voxel_metrics(images, recon_s)[0]
        lp_s = lpips_fn(recon_s, images)
        real_var = images.flatten(1).var(dim=1)

        for j, path in enumerate(paths):
            rows.append(
                {
                    "split": name,
                    **meta[path],
                    "file": path,
                    "mse": mse[j].item(),
                    "real_var": real_var[j].item(),
                    "lpips": lp[j].item(),
                    "lpips_norm11": lp_n[j].item(),
                    "mae": mae[j].item(),
                    "psnr": psnr[j].item(),
                    "mse_sampled": mse_s[j].item(),
                    "lpips_sampled": lp_s[j].item(),
                }
            )

        done = len(rows)
        if (i + 1) % 50 == 0 or done == len(ds):
            el = time.time() - t0
            print(f"  {done}/{len(ds)}  {el:.0f}s elapsed, eta {el/done*(len(ds)-done):.0f}s", flush=True)

    return pd.DataFrame(rows)


def summarise(df, split_label, group_label):
    out = {"split": split_label, "group": group_label, "n": len(df)}
    for col in METRIC_COLS:
        v = df[col].to_numpy()
        out[f"{col}_mean"] = float(v.mean())
        out[f"{col}_std"] = float(v.std(ddof=1))
        out[f"{col}_sem"] = float(v.std(ddof=1) / np.sqrt(len(v)))
        out[f"{col}_median"] = float(np.median(v))
        out[f"{col}_min"] = float(v.min())
        out[f"{col}_max"] = float(v.max())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    meta = build_metadata()

    vae = load_vae(RUN_FOLDER, args.device)
    lpips_fn = LPIPS3D("alex", chunk=64).to(args.device).eval()
    for p in lpips_fn.parameters():
        p.requires_grad = False

    train_df = evaluate_split("train", ["train.csv"], vae, lpips_fn, meta,
                              args.device, args.batch_size, args.num_workers, args.seed)
    held_df = evaluate_split("test", ["val.csv", "test.csv"], vae, lpips_fn, meta,
                             args.device, args.batch_size, args.num_workers, args.seed)

    per_subject = pd.concat([train_df, held_df], ignore_index=True)

    print("\nrecovering per-volume normalisation constants ...")
    norm_mean, norm_std = compute_norm_stats(per_subject["file"].tolist(), args.num_workers)
    per_subject["norm_mean"] = norm_mean
    per_subject["norm_std"] = norm_std
    per_subject["mse_native"] = per_subject["mse"] * norm_std**2
    per_subject["rmse_native"] = np.sqrt(per_subject["mse_native"])
    per_subject["mae_native"] = per_subject["mae"] * norm_std
    per_subject["nmse"] = per_subject["mse"] / per_subject["real_var"]
    per_subject.to_csv(os.path.join(RESULTS_DIR, "per_subject_metrics.csv"), index=False)

    train_df = per_subject[per_subject["split"] == "train"]
    held_df = per_subject[per_subject["split"] == "test"]

    # Headline splits, plus the two nested subsets of the held-out pool for reference.
    subsets = [
        ("train", train_df),
        ("test", held_df),
        ("test_csv_only", held_df[held_df["in_test_csv"]]),
        ("val_csv_only", held_df[held_df["in_val_csv"] & ~held_df["in_test_csv"]]),
    ]
    summary = [
        summarise(sub if grp == "all" else sub[sub["group"] == grp], label, grp)
        for label, sub in subsets
        for grp in ("all", "HC", "TLE")
    ]
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "summary_metrics.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, "summary_metrics.json"), "w") as f:
        json.dump(
            {
                "checkpoint": os.path.join(RUN_FOLDER, "best_vae.pth"),
                "reconstruction": "deterministic (decode posterior mean); *_sampled use the reparameterised latent",
                "lpips": "alex backbone, mean over every slice of all three axes (matches VAE/model/lpips3D.py)",
                "note": "test = union of val.csv and test.csv, de-duplicated (test.csv is a strict subset of val.csv)",
                "units": {
                    "mse / mae / mse_sampled": "per-volume z-scored intensities, i.e. the scale the model "
                                               "is trained and evaluated on. Because each volume is divided "
                                               "by its own std, this is already a variance-normalised error.",
                    "mse_native / rmse_native / mae_native": "native mwp1 grey-matter units, obtained by "
                                                             "undoing the z-scoring. Crop and resize are "
                                                             "linear, so MSE_native = MSE_zscored * norm_std**2 "
                                                             "and MAE_native = MAE_zscored * norm_std exactly.",
                    "nmse": "mse divided by the variance of the preprocessed real volume; equals 1 - R^2. "
                            "Scale-free and unaffected by the normalisation choice.",
                    "psnr": "invariant to the standardisation, since the data range rescales with the error.",
                    "norm_mean / norm_std": "mean and std of the full original volume, used by the z-scoring step.",
                },
                "summary": summary,
            },
            f,
            indent=2,
        )

    pd.set_option("display.width", 200)
    print("\n=== mean +/- std ===")
    for r in summary:
        print(f"{r['split']:>14s} {r['group']:>4s} n={r['n']:5d}  "
              f"MSE(z) {r['mse_mean']:.4f} +/- {r['mse_std']:.4f}   "
              f"MSE(mwp1) {r['mse_native_mean']:.2e} +/- {r['mse_native_std']:.1e}   "
              f"RMSE(mwp1) {r['rmse_native_mean']:.4f} +/- {r['rmse_native_std']:.4f}   "
              f"LPIPS {r['lpips_mean']:.4f} +/- {r['lpips_std']:.4f}")
    print(f"\nWritten to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
