"""Per-ROI reconstruction MSE for the four exported example subjects.

Joins <examples-dir>/examples_summary.csv against the held-out per-subject ROI table
produced by roi_mse_analysis.py, and reports each example's ROI MSE together with its
percentile rank in the held-out cohort (all subjects and its own diagnosis group).

Writes a tidy table to <examples-dir>/examples_roi_mse.csv and a roi_mse.json into
each example folder, alongside the metrics.json that export_examples.py already writes.

--atlas picks which roi_mse_analysis.py run to read. Non-default atlases get an
_<atlas> suffix on every output name, so the two parcellations can live side by side
in the same example folders.
"""

import argparse
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EX_DIR = os.path.join(os.path.dirname(HERE), "results", "examples")

# results dir written by roi_mse_analysis.py for each --atlas choice
ATLAS_RESULTS_DIRS = {"harvard-oxford": "results", "aal3": "results_aal3"}
DEFAULT_ATLAS = "harvard-oxford"

ROIS = [
    "Hippocampus",
    "Parahippocampal gyrus",
    "Amygdala",
    "Thalamus",
    "Insula",
    "Cingulate cortex",
    "Anterior temporal cortex",
    "Whole brain (non-background)",
]
NATIVE = " [mwp1]"


def percentile(values, x):
    """Share of the cohort strictly below x, in percent."""
    return float((values < x).mean() * 100)


def build_table(roi_df, examples):
    rows = []
    for ex in examples.itertuples():
        subject = roi_df[roi_df.sbj_nme == ex.subject]
        if subject.empty:
            raise SystemExit(f"{ex.subject} is not in per_subject_roi_mse.csv")
        subject = subject.iloc[0]
        group = roi_df[roi_df.group == subject.group]

        for roi in ROIS:
            z, native = subject[roi], subject[roi + NATIVE]
            rows.append({
                "subject": ex.subject,
                "group": ex.group,
                "diagnosis": ex.diagnosis,
                "quality": ex.quality,
                "folder": ex.folder,
                "roi": roi,
                "mse_zscored": z,
                "mse_native_mwp1": native,
                "pct_rank_all": percentile(roi_df[roi], z),
                "pct_rank_within_group": percentile(group[roi], z),
                "cohort_median_zscored": float(roi_df[roi].median()),
                "group_median_zscored": float(group[roi].median()),
                "n_cohort": len(roi_df),
                "n_group": len(group),
            })
    return pd.DataFrame(rows)


def write_per_example_json(table, ex_dir, results_dirname, suffix):
    for folder, sub in table.groupby("folder", sort=False):
        out = os.path.join(ex_dir, folder)
        if not os.path.isdir(out):
            print(f"  skipped {folder} (no such example folder)")
            continue
        first = sub.iloc[0]
        payload = {
            "subject": first.subject,
            "group": first.group,
            "diagnosis": first.diagnosis,
            "quality": first.quality,
            "source": f"roi_analysis/{results_dirname}/per_subject_roi_mse.csv",
            "cohort": {"n_all": int(first.n_cohort), "n_group": int(first.n_group),
                       "split": "held-out (val.csv U test.csv)"},
            "scales": {
                "zscored": "MSE in the per-volume z-scored units the VAE operates on",
                "native_mwp1": "z-scored MSE x norm_std^2, i.e. native mwp1 grey-matter units",
            },
            "rois": {
                r.roi: {
                    "mse_zscored": r.mse_zscored,
                    "mse_native_mwp1": r.mse_native_mwp1,
                    "pct_rank_all": r.pct_rank_all,
                    "pct_rank_within_group": r.pct_rank_within_group,
                    "cohort_median_zscored": r.cohort_median_zscored,
                    "group_median_zscored": r.group_median_zscored,
                }
                for r in sub.itertuples()
            },
        }
        path = os.path.join(out, f"roi_mse{suffix}.json")
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  {path}")


def build_wide(table):
    """One row per ROI, one column block per example: the layout used in the paper tables."""
    key = table.group + "_" + table.quality
    rows = []
    for roi in ROIS:
        sub = table[table.roi == roi]
        row = {"roi": roi}
        for k, r in zip(key[sub.index], sub.itertuples()):
            row[f"{k}_subject"] = r.subject
            row[f"{k}_mse_zscored"] = r.mse_zscored
            row[f"{k}_pct_within_group"] = r.pct_rank_within_group
            row[f"{k}_pct_all"] = r.pct_rank_all
            row[f"{k}_mse_native_mwp1"] = r.mse_native_mwp1
        for g in ("HC", "TLE"):
            match = sub[sub.group == g]
            row[f"{g}_median_zscored"] = float(match.group_median_zscored.iloc[0])
        row["cohort_median_zscored"] = float(sub.cohort_median_zscored.iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def print_table(table):
    for roi in ROIS:
        print(f"\n{roi}")
        for r in table[table.roi == roi].itertuples():
            print(f"  {r.folder:32s} z={r.mse_zscored:.5f}  native={r.mse_native_mwp1:.3e}  "
                  f"pct(all)={r.pct_rank_all:5.1f}  pct({r.group})={r.pct_rank_within_group:5.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", default=DEFAULT_ATLAS, choices=sorted(ATLAS_RESULTS_DIRS),
                    help="which roi_mse_analysis.py run to read")
    ap.add_argument("--no-json", action="store_true",
                    help="only write the CSV, leave the example folders untouched")
    ap.add_argument("--examples-dir", default=EX_DIR,
                    help="folder written by export_examples.py")
    args = ap.parse_args()

    results_dirname = ATLAS_RESULTS_DIRS[args.atlas]
    suffix = "" if args.atlas == DEFAULT_ATLAS else f"_{args.atlas}"

    ex_dir = os.path.abspath(args.examples_dir)
    roi_df = pd.read_csv(os.path.join(HERE, results_dirname, "per_subject_roi_mse.csv"))
    examples = pd.read_csv(os.path.join(ex_dir, "examples_summary.csv"))
    print(f"atlas: {args.atlas} (roi_analysis/{results_dirname})")
    print(f"held-out subjects with ROI metrics: {len(roi_df)}   examples: {len(examples)}")

    table = build_table(roi_df, examples)
    csv_path = os.path.join(ex_dir, f"examples_roi_mse{suffix}.csv")
    table.to_csv(csv_path, index=False)

    wide_path = os.path.join(ex_dir, f"examples_roi_mse_wide{suffix}.csv")
    build_wide(table).to_csv(wide_path, index=False)

    print_table(table)
    print(f"\n{csv_path}\n{wide_path}")
    if not args.no_json:
        write_per_example_json(table, ex_dir, results_dirname, suffix)


if __name__ == "__main__":
    main()
