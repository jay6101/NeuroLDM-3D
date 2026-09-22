"""
End-to-end verification for new_runs_3dcnn_cv.
Run:  conda activate jay2 && python _verify_e2e.py
"""
import copy
import json
import os
import sys
import traceback

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import config
from dataset import MRIDataset, BalancedBatchSampler
from train import build_eval_folds, get_model_class, train_model
from aggregate_folds import aggregate_variant, calculate_overall_metrics

PASS = []
FAIL = []
WARN = []


def ok(msg):
    PASS.append(msg)
    print(f"  PASS  {msg}")


def bad(msg, exc=None):
    FAIL.append(msg)
    print(f"  FAIL  {msg}")
    if exc:
        print(f"        {exc}")


def warn(msg):
    WARN.append(msg)
    print(f"  WARN  {msg}")


def section(title):
    print(f"\n{'='*60}\n{title}\n{'='*60}")


# ------------------------------------------------------------------ 1. Config
section("1. Config / variant definitions")
hp = dict(config.BASE_HYPERPARAMS)
for key in ['train_csv', 'val_csv', 'test_csv', 'synth_hc_folder_path', 'synth_tle_folder_path']:
    if os.path.exists(hp[key]):
        ok(f"path exists: {key}")
    else:
        bad(f"missing path: {key} -> {hp[key]}")

if len(config.VARIANTS) == 25:
    ok(f"25 variants defined")
else:
    bad(f"expected 25 variants, got {len(config.VARIANTS)}")

# spot-check a few variant configs vs original new_runs train.py patterns
checks = {
    'real_500': (500, None),
    'real_2723_syn_50': (2723, 1361),
    'syn_500': (None, 500),
    'syn_5446': (None, 5446),
}
vmap = {v['name']: v for v in config.VARIANTS}
for name, (ns, nss) in checks.items():
    v = vmap.get(name)
    if v and v['num_samples'] == ns and v['num_synth_samples'] == nss:
        ok(f"variant {name}: num_samples={ns}, num_synth_samples={nss}")
    else:
        bad(f"variant {name} mismatch: {v}")


# ------------------------------------------------------------------ 2. Eval folds
section("2. Eval fold splitting (val+test -> 5 folds)")
hp_fold = dict(hp)
hp_fold['n_folds'] = 5
eval_df, folds = build_eval_folds(hp_fold)

if len(eval_df) == 481:
    ok("eval pool = 481 unique volumes (deduped val+test)")
else:
    bad(f"eval pool size {len(eval_df)}, expected 481")

if len(folds) == 5:
    ok("5 folds created")
else:
    bad(f"expected 5 folds, got {len(folds)}")

all_test_files = set()
for i, (vdf, tdf) in enumerate(folds):
    overlap = set(vdf['file']) & set(tdf['file'])
    if overlap:
        bad(f"fold {i+1}: val/test overlap {len(overlap)} files")
    else:
        ok(f"fold {i+1}: val={len(vdf)} test={len(tdf)} no val/test overlap")
    all_test_files.update(tdf['file'].tolist())

if len(all_test_files) == len(eval_df):
    ok("5 test folds partition entire eval pool exactly")
else:
    bad(f"test folds cover {len(all_test_files)}/{len(eval_df)} eval samples")

# train pool disjoint from eval
train_df = pd.read_csv(hp['train_csv'])
train_files = set(train_df['file'])
eval_files = set(eval_df['file'])
if train_files & eval_files:
    bad(f"train/eval leak: {len(train_files & eval_files)} shared files")
else:
    ok("train and eval pools are disjoint")


# ------------------------------------------------------------------ 3. Dataset logic (no heavy IO for synth counts)
section("3. Dataset sample composition (real / real+synth / syn-only)")
full_train = train_df[train_df['HC_vs_LTLE_vs_RTLE_string'].isin(['right', 'left', 'HC'])]

def count_sources(ds):
    real = sum(1 for p, _ in ds.samples if '.pkl' not in p)
    synth = sum(1 for p, _ in ds.samples if '.pkl' in p)
    pos = sum(1 for _, l in ds.samples if l == 1)
    neg = sum(1 for _, l in ds.samples if l == 0)
    return real, synth, pos, neg


# real only
ds_real = MRIDataset(full_train, hp, train=True, num_samples=500, num_synth_samples=None)
r, s, pos, neg = count_sources(ds_real)
if r == 500 and s == 0:
    ok(f"real_500 train: {r} real, {s} synth")
else:
    bad(f"real_500 train: expected 500/0, got {r}/{s}")
if pos > 0 and neg > 0:
    ok(f"real_500 balanced classes: pos={pos} neg={neg}")
else:
    bad(f"real_500 missing a class: pos={pos} neg={neg}")

# real + synth (small synth for speed)
ds_mix = MRIDataset(full_train, hp, train=True, num_samples=500, num_synth_samples=20)
r, s, pos, neg = count_sources(ds_mix)
if r == 500 and s > 0:
    ok(f"real+synth train: {r} real + {s} synth = {len(ds_mix)} total")
else:
    bad(f"real+synth: expected 500 real + synth, got {r}/{s}")
if pos > 0 and neg > 0:
    ok(f"real+synth balanced: pos={pos} neg={neg}")
else:
    bad(f"real+synth missing class")

# syn only
ds_syn = MRIDataset(full_train, hp, train=True, num_samples=None, num_synth_samples=20)
r, s, pos, neg = count_sources(ds_syn)
if r == 0 and s > 0:
    ok(f"syn-only train: {r} real, {s} synth")
else:
    bad(f"syn-only: expected 0 real + synth, got {r}/{s}")
if pos > 0 and neg > 0:
    ok(f"syn-only balanced: pos={pos} neg={neg}")
else:
    bad(f"syn-only missing class: pos={pos} neg={neg}")

# val/test always real
val_df = pd.read_csv(hp['val_csv'])
ds_val = MRIDataset(val_df, hp, train=False)
r, s, _, _ = count_sources(ds_val)
if r == len(ds_val) and s == 0:
    ok(f"val set: {r} real only (no synth)")
else:
    bad(f"val set wrong: real={r} synth={s}")

# deterministic subsampling
ds_a = MRIDataset(full_train, hp, train=True, num_samples=500, num_synth_samples=None)
ds_b = MRIDataset(full_train, hp, train=True, num_samples=500, num_synth_samples=None)
if [p for p, _ in ds_a.samples] == [p for p, _ in ds_b.samples]:
    ok("real subsampling is deterministic (same sample_seed)")
else:
    bad("real subsampling not deterministic")


# ------------------------------------------------------------------ 4. Data loading shapes
section("4. __getitem__ tensor shapes (1 real + 1 synth sample)")
try:
    img_r, lbl_r, path_r = ds_real[0]
    assert img_r.shape == (1, 115, 128, 128), f"real shape {img_r.shape}"
    assert img_r.dtype == torch.float32
    ok(f"real volume shape {tuple(img_r.shape)} dtype={img_r.dtype} label={lbl_r}")

    syn_idx = next(i for i, (p, _) in enumerate(ds_mix.samples) if '.pkl' in p)
    img_s, lbl_s, path_s = ds_mix[syn_idx]
    assert img_s.shape == (1, 115, 128, 128)
    ok(f"synth volume shape {tuple(img_s.shape)} label={lbl_s}")
except Exception as e:
    bad("__getitem__ failed", e)


# ------------------------------------------------------------------ 5. BalancedBatchSampler
section("5. BalancedBatchSampler for all run types")
from torch.utils.data import DataLoader

for tag, ds in [('real', ds_real), ('mix', ds_mix), ('syn', ds_syn)]:
    try:
        sampler = BalancedBatchSampler(ds, batch_size=8)
        loader = DataLoader(ds, batch_sampler=sampler, num_workers=0)
        batch = next(iter(loader))
        images, labels, paths = batch
        assert images.shape[1:] == (1, 115, 128, 128)
        assert len(labels) == 8
        n_pos = (labels == 1).sum().item()
        n_neg = (labels == 0).sum().item()
        if n_pos == 4 and n_neg == 4:
            ok(f"{tag}: batch shape {tuple(images.shape)}, balanced 4/4")
        else:
            warn(f"{tag}: batch not perfectly balanced pos={n_pos} neg={n_neg} (may be ok with tiny sets)")
        if len(sampler) == 0:
            bad(f"{tag}: sampler has 0 batches!")
    except Exception as e:
        bad(f"{tag} BalancedBatchSampler", e)


# ------------------------------------------------------------------ 6. Model forward
section("6. 3D CNN model forward pass")
try:
    Model = get_model_class(config.MODEL_NAME)
    model = Model(dropout_rate=0.2, filters=hp['filters'])
    dummy = torch.randn(2, 1, 115, 128, 128)
    out = model(dummy)
    assert out.shape == (2, 1)
    ok(f"cnn_ben_3d output shape {tuple(out.shape)}")
except Exception as e:
    bad("model forward", e)


# ------------------------------------------------------------------ 7. Mini end-to-end training (3 variant types)
section("7. Mini end-to-end training (1 epoch, 1 fold, tiny samples)")
TEST_ROOT = "/tmp/cv_3dcnn_verify"
os.makedirs(TEST_ROOT, exist_ok=True)

mini_variants = [
    {'group': 'runs_500', 'name': 'real_500', 'num_samples': 40, 'num_synth_samples': None},
    {'group': 'runs_500', 'name': 'real_500_syn_25', 'num_samples': 40, 'num_synth_samples': 10},
    {'group': 'runs_syn', 'name': 'syn_500', 'num_samples': None, 'num_synth_samples': 16},
]

for v in mini_variants:
    print(f"\n  --- training {v['name']} ---")
    try:
        hp_run = copy.deepcopy(hp)
        hp_run.update({
            'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
            'num_samples': v['num_samples'],
            'num_synth_samples': v['num_synth_samples'],
            'variant_name': v['name'],
            'group': v['group'],
            'run_dir': os.path.join(TEST_ROOT, v['group'], v['name']),
            'num_epochs': 1,
            'max_folds': 1,
            'num_workers': 2,
            'early_stopping_patience': 150,
        })
        summary = train_model(hp_run, config.MODEL_NAME)
        run_dir = hp_run['run_dir']
        fold_dir = os.path.join(run_dir, 'fold_1')

        required = [
            'parameters.json', 'all_folds_metrics.json', 'run_summary.json',
            'fold_1/best_model.pth', 'fold_1/fold_metrics.json',
            'fold_1/train.csv', 'fold_1/val.csv', 'fold_1/test.csv',
            'fold_1/individual_test_performance.csv',
        ]
        missing = [f for f in required if not os.path.exists(os.path.join(run_dir, f))]
        if missing:
            bad(f"{v['name']}: missing outputs {missing}")
        else:
            ok(f"{v['name']}: all output artifacts present")

        # check all_folds_metrics.json format
        with open(os.path.join(run_dir, 'all_folds_metrics.json')) as f:
            agg = json.load(f)
        if 'overall_results' in agg and 'auc_roc' in agg['overall_results']:
            ok(f"{v['name']}: all_folds_metrics.json format OK (auc={agg['overall_results']['auc_roc']['mean']:.4f})")
        else:
            bad(f"{v['name']}: bad all_folds_metrics.json format")

        # check train.csv sources
        tcsv = pd.read_csv(os.path.join(fold_dir, 'train.csv'))
        if 'source' in tcsv.columns:
            src = tcsv['source'].value_counts().to_dict()
            if v['num_samples'] is None:
                if src.get('real', 0) == 0 and src.get('synthetic', 0) > 0:
                    ok(f"{v['name']}: train.csv synth-only {src}")
                else:
                    bad(f"{v['name']}: expected synth-only train.csv, got {src}")
            elif v['num_synth_samples'] is None:
                if src.get('synthetic', 0) == 0 and src.get('real', 0) > 0:
                    ok(f"{v['name']}: train.csv real-only {src}")
                else:
                    bad(f"{v['name']}: expected real-only train.csv, got {src}")
            else:
                if src.get('real', 0) > 0 and src.get('synthetic', 0) > 0:
                    ok(f"{v['name']}: train.csv mixed {src}")
                else:
                    bad(f"{v['name']}: expected mixed train.csv, got {src}")
        else:
            bad(f"{v['name']}: train.csv missing 'source' column")

        # test metrics sanity
        tm = json.load(open(os.path.join(fold_dir, 'fold_metrics.json')))['test_metrics']
        for m in ['accuracy', 'auc_roc', 'sensitivity', 'specificity']:
            if m not in tm or not (0 <= tm[m] <= 1 or m == 'auc_roc'):
                bad(f"{v['name']}: bad test metric {m}={tm.get(m)}")
        ok(f"{v['name']}: test metrics in valid range")

        # aggregate_folds.py on same folder
        aggregate_variant(run_dir, verbose=False)
        ok(f"{v['name']}: aggregate_folds.py runs OK")

    except Exception as e:
        bad(f"{v['name']}: training failed", e)
        traceback.print_exc()


# ------------------------------------------------------------------ 8. Edge cases
section("8. Edge cases / potential breakage")
# tiny syn-only with batch_size=8
try:
    ds_tiny = MRIDataset(full_train, hp, train=True, num_samples=None, num_synth_samples=8)
    sampler = BalancedBatchSampler(ds_tiny, batch_size=8)
    if len(sampler) >= 1:
        ok(f"tiny syn-only (8 synth): {len(sampler)} batches")
    else:
        warn("tiny syn-only: 0 batches (may break training with very small synth counts)")
except Exception as e:
    bad("tiny syn-only sampler", e)

# syn count rounding (int(0.423*N)+int(0.577*N) may be < N)
expected = int(0.423 * 125) + int(0.577 * 125)
ds_round = MRIDataset(full_train, hp, train=True, num_samples=500, num_synth_samples=125)
actual_synth = sum(1 for p, _ in ds_round.samples if '.pkl' in p)
if actual_synth == expected:
    ok(f"synth count rounding: requested 125 -> loaded {actual_synth} (matches original formula)")
else:
    warn(f"synth count: requested 125, formula gives {expected}, loaded {actual_synth}")

# checkpoint loadable
try:
    ckpt_path = os.path.join(TEST_ROOT, 'runs_500', 'real_500', 'fold_1', 'best_model.pth')
    ckpt = torch.load(ckpt_path, weights_only=False)
    assert 'model_state_dict' in ckpt
    ok("checkpoint loads and has model_state_dict")
except Exception as e:
    bad("checkpoint load", e)


# ------------------------------------------------------------------ Summary
section("SUMMARY")
print(f"  PASSED : {len(PASS)}")
print(f"  FAILED : {len(FAIL)}")
print(f"  WARNINGS: {len(WARN)}")
if FAIL:
    print("\nFailures:")
    for f in FAIL:
        print(f"  - {f}")
if WARN:
    print("\nWarnings:")
    for w in WARN:
        print(f"  - {w}")

sys.exit(1 if FAIL else 0)
