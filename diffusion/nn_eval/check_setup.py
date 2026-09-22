"""Pre-flight checks for the nearest-neighbor memorization analysis.

Verifies, using only the standard library (no torch needed, fast):
  1. Number/labels of test samples in the classifier test CSV.
  2. Overlap between the test set and the diffusion train/val latents
     (critical: if test images are in the diffusion TRAIN set, the
     real-test nearest-neighbor distances would be biased toward 0).
  3. Whether best_vae.pth and best_vae_gan.pth are the same checkpoint
     (the train latents + diffusion model live in the latent space of
     whichever VAE created latent_data/, per save_latent.py).
"""
import os
import csv
import hashlib
from collections import Counter

TEST_CSV = "/space/mcdonald-syn01/1/projects/jsawant/Diffusion_paper/classifier/new_runs/runs_500/real_500_syn_50/fold_1/individual_test_performance.csv"
TRAIN_DIR = "/space/mcdonald-syn01/1/projects/jsawant/Diffusion_paper/diffusion/latent_data/train"
VAL_DIR = "/space/mcdonald-syn01/1/projects/jsawant/Diffusion_paper/diffusion/latent_data/val"
VAE_DIR = "/space/mcdonald-syn01/1/projects/jsawant/Diffusion_paper/VAE/best_runs/vae_run_20250226_161525"


def stem_from_nii(path):
    # Matches save_latent.py: os.path.basename(path).split('.')[0]
    return os.path.basename(path).split(".")[0]


def latent_stems(d):
    out = set()
    if not os.path.isdir(d):
        return out
    for fn in os.listdir(d):
        if fn.endswith("_latent.pkl"):
            out.add(fn[: -len("_latent.pkl")])
    return out


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    test_stems, test_labels = [], {}
    with open(TEST_CSV) as f:
        for row in csv.DictReader(f):
            s = stem_from_nii(row["img_path"])
            test_stems.append(s)
            test_labels[s] = row["original_label"]
    test_set = set(test_stems)
    print(f"n_test rows           : {len(test_stems)}")
    print(f"n_test unique stems   : {len(test_set)}")
    print(f"test label dist       : {dict(Counter(test_labels.values()))}")

    train_set = latent_stems(TRAIN_DIR)
    val_set = latent_stems(VAL_DIR)
    print(f"n_train latents       : {len(train_set)}")
    print(f"n_val latents         : {len(val_set)}")

    print(f"overlap test & train  : {len(test_set & train_set)}")
    print(f"overlap test & val    : {len(test_set & val_set)}")
    print(f"test in NEITHER       : {len(test_set - train_set - val_set)}")
    print(f"overlap train & val   : {len(train_set & val_set)}")

    print(f"example test stems    : {sorted(test_set)[:2]}")
    print(f"example train stems   : {sorted(train_set)[:2]}")

    for name in ["best_vae.pth", "best_vae_gan.pth"]:
        p = os.path.join(VAE_DIR, name)
        if os.path.exists(p):
            print(f"{name:16s}      : size={os.path.getsize(p):,} md5={md5(p)}")
        else:
            print(f"{name:16s}      : MISSING")


if __name__ == "__main__":
    main()
