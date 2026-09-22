"""Stage 1: build train/val/test sub-CSVs from the master pickle_prep CSV.

We recover the diffusion train/val SPLIT from the filenames in latent_data/
(NOT their stored latent values), match each stem back to its row in
pickle_prep (the source of truth for real `file` paths + labels), and write
fresh MRIDataset-compatible CSVs:

    data_csvs/train.csv   (reference DB = images the diffusion was trained on)
    data_csvs/val.csv
    data_csvs/test.csv    (303 classifier test images; subset of val)

These CSVs carry all pickle_prep columns, so VAE/dataset.py:MRIDataset can
consume them directly (it needs `file` + `HC_vs_LTLE_vs_RTLE_string`).

Run:
    /home/jsawant/.conda/envs/jay2/bin/python \
        /space/mcdonald-syn01/1/projects/jsawant/Diffusion_paper/diffusion/nn_eval/build_csvs.py
"""
import os
import csv
from collections import Counter

import paths


def stem_of(path):
    return os.path.basename(path).split(".")[0]


def diag_to_label(diag):
    return 0 if str(diag).strip() == "HC" else 1


def load_pickle_prep():
    """stem -> row dict, plus the header field order."""
    by_stem = {}
    dup = 0
    with open(paths.PICKLE_PREP, newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        for row in reader:
            fpath = (row.get("file") or "").strip()
            if not fpath:
                continue
            s = stem_of(fpath)
            if s in by_stem:
                dup += 1
                continue
            by_stem[s] = row
    print(f"pickle_prep: {len(by_stem)} unique stems indexed (skipped {dup} dup basenames)")
    return by_stem, header


def latent_stems(d):
    if not os.path.isdir(d):
        print(f"WARNING: {d} not found")
        return []
    return sorted(
        fn[: -len("_latent.pkl")] for fn in os.listdir(d) if fn.endswith("_latent.pkl")
    )


def test_stems_and_labels():
    stems, orig = [], {}
    with open(paths.TEST_PERF_CSV, newline="") as f:
        for row in csv.DictReader(f):
            s = stem_of(row["img_path"])
            stems.append(s)
            orig[s] = int(float(row["original_label"]))
    return stems, orig


def write_split(name, stems, by_stem, header, out_csv, orig_labels=None):
    rows, matched, missing = [], [], []
    for s in stems:
        if s in by_stem:
            rows.append(by_stem[s])
            matched.append(s)
        else:
            missing.append(s)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)

    labels = [diag_to_label(r.get("HC_vs_LTLE_vs_RTLE_string")) for r in rows]
    dist = {paths.CLASS_NAMES[k]: v for k, v in sorted(Counter(labels).items())}
    print(f"\n[{name}] wrote {len(rows)} rows -> {out_csv}")
    print(f"        matched={len(matched)}  missing={len(missing)}  label_dist={dist}")
    if missing:
        print(f"        e.g. missing stems: {missing[:3]}")

    if orig_labels is not None:  # cross-check derived vs classifier original_label
        mism = [s for s, r in zip(matched, rows)
                if diag_to_label(r.get("HC_vs_LTLE_vs_RTLE_string")) != orig_labels[s]]
        print(f"        label cross-check vs original_label: {len(mism)} mismatches")
        if mism:
            print(f"        e.g. mismatched stems: {mism[:5]}")


def main():
    by_stem, header = load_pickle_prep()

    train = latent_stems(paths.LATENT_TRAIN_DIR)
    val = latent_stems(paths.LATENT_VAL_DIR)
    test, orig = test_stems_and_labels()
    print(f"split sizes from filenames: train={len(train)} val={len(val)} test={len(test)}")

    write_split("train", train, by_stem, header, paths.TRAIN_CSV)
    write_split("val", val, by_stem, header, paths.VAL_CSV)
    write_split("test", test, by_stem, header, paths.TEST_CSV, orig_labels=orig)

    # sanity: test should be disjoint from train (memorization-test requirement)
    inter = set(train) & set(test)
    print(f"\nsanity: |train ∩ test| = {len(inter)} (must be 0)")


if __name__ == "__main__":
    main()
