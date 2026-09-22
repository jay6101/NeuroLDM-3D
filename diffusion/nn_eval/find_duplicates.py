"""Detect test images that coincide with train images.

Two independent notions of "coincide", both reported:
  (A) same scan id  -> exact overlap of `stems` between test and train
  (B) identical image -> near-zero nearest-train MSE on the deterministic
      posterior mean `mu` (key `latents`). Identical input -> identical mu,
      so a true duplicate shows MSE ~ 0 even if the filename/id differs.

Usage:
  PY=/home/jsawant/.conda/envs/jay2/bin/python
  $PY find_duplicates.py --device cuda:0
"""
import os
import sys
import csv
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # type: ignore


def load_set(set_dir, key="latents"):
    npz = os.path.join(set_dir, paths.LATENTS_FILE)
    d = np.load(npz, allow_pickle=False)
    lat = d[key].astype(np.float32)
    lat = lat.reshape(len(lat), -1)
    stems = d["stems"].astype(str)
    labels = d["labels"]
    return lat, stems, labels


@torch.no_grad()
def nearest_train(test, train, device, chunk=64):
    """For each test row: (min MSE to any train row, argmin index)."""
    D = test.shape[1]
    T = torch.from_numpy(train).to(device)
    t_norm = (T * T).sum(1)  # (M,)
    best_val = np.empty(len(test), np.float64)
    best_idx = np.empty(len(test), np.int64)
    for i in range(0, len(test), chunk):
        Xc = torch.from_numpy(test[i:i + chunk]).to(device)
        x_norm = (Xc * Xc).sum(1, keepdim=True)            # (b,1)
        d2 = x_norm + t_norm.unsqueeze(0) - 2.0 * (Xc @ T.T)  # (b,M)
        d2.clamp_(min=0)
        mse = d2 / D
        vmin, imin = mse.min(dim=1)
        best_val[i:i + chunk] = vmin.double().cpu().numpy()
        best_idx[i:i + chunk] = imin.cpu().numpy()
    return best_val, best_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--key", default="latents",
                    help="latents (mu, default) or latents_rt")
    ap.add_argument("--chunk", type=int, default=64)
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    print(f"Device: {device} | key: {args.key}")
    train, train_stems, train_labels = load_set(paths.REAL_TRAIN_DIR, args.key)
    test, test_stems, test_labels = load_set(paths.REAL_TEST_DIR, args.key)
    print(f"train: {train.shape}  test: {test.shape}  (D={test.shape[1]})")

    # --- (A) exact scan-id overlap -------------------------------------------
    train_set = set(train_stems.tolist())
    shared = [s for s in test_stems.tolist() if s in train_set]
    print("\n================ (A) SAME SCAN ID (stem overlap) ================")
    print(f"test stems            : {len(test_stems)} ({len(set(test_stems.tolist()))} unique)")
    print(f"train stems           : {len(train_stems)} ({len(train_set)} unique)")
    print(f"test stems also in train: {len(shared)}")
    for s in shared[:20]:
        print(f"   {s}")
    if len(shared) > 20:
        print(f"   ... (+{len(shared) - 20} more)")

    # --- (B) identical latent (near-zero nearest-train MSE) -------------------
    val, idx = nearest_train(test, train, device, args.chunk)
    thresholds = [0.0, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2]
    print("\n========= (B) IDENTICAL LATENT (nearest-train MSE on mu) =========")
    print(f"min nearest-train MSE : {val.min():.3e}")
    print(f"median                : {np.median(val):.4f}")
    for thr in thresholds:
        n = int((val <= thr).sum())
        print(f"  test with MSE <= {thr:>7.0e} : {n}")

    order = np.argsort(val)
    print("\n--- 20 closest test->train pairs ---")
    print(f"{'test_stem':<28} {'nearest_train_stem':<28} {'MSE':>10}  same_id")
    for k in order[:20]:
        ts = test_stems[k]
        trs = train_stems[idx[k]]
        print(f"{ts:<28} {trs:<28} {val[k]:>10.3e}  {str(ts == trs)}")

    # --- save full report ----------------------------------------------------
    os.makedirs(paths.RESULTS_DIR, exist_ok=True)
    out_csv = os.path.join(paths.RESULTS_DIR, "test_train_coincidence.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["test_stem", "nearest_train_stem", "nearest_mse",
                    "same_scan_id", "test_label"])
        for k in order:
            ts = test_stems[k]
            trs = train_stems[idx[k]]
            w.writerow([ts, trs, f"{val[k]:.6e}", int(ts == trs), int(test_labels[k])])
    print(f"\nsaved per-test report -> {out_csv}")

    # headline number: union of (A) and (B at a tight threshold)
    dup_by_latent = {test_stems[k] for k in range(len(test)) if val[k] <= 1e-4}
    dup_union = set(shared) | dup_by_latent
    print("\n================ HEADLINE ================")
    print(f"test images coinciding with train:")
    print(f"  by scan id            : {len(shared)}")
    print(f"  by identical latent(<=1e-4): {len(dup_by_latent)}")
    print(f"  union (either)        : {len(dup_union)}  of {len(test_stems)}")


if __name__ == "__main__":
    main()
