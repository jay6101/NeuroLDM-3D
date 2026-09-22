"""
Central configuration for the 3D-CNN 5-fold CV experiments.

Defines:
  * BASE_HYPERPARAMS : 3D CNN training hyper-parameters (from
    3D_CNN/3D_supervised_HC_vs_TLE) + dataset paths / seeds from the
    Diffusion classifier runs.
  * VARIANTS         : one entry per run variant, mirroring the runs in
    Diffusion_paper/classifier/new_runs (real_* and syn_*), each with the
    exact (num_samples, num_synth_samples) used originally.
"""

import os

# ---------------------------------------------------------------------------
# Paths (kept identical to the Diffusion classifier runs)
# ---------------------------------------------------------------------------
DATA_DIR = "/space/mcdonald-syn01/1/projects/jsawant/Diffusion_paper/data_csvs"
SYNTH_HC = "/space/mcdonald-syn01/1/projects/jsawant/Diffusion_paper/synthetic_data_pkls_HC"
SYNTH_TLE = "/space/mcdonald-syn01/1/projects/jsawant/Diffusion_paper/synthetic_data_pkls_TLE"

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RUNS_DIR = os.path.join(THIS_DIR, "runs")

MODEL_NAME = "cnn_ben_3d"

# ---------------------------------------------------------------------------
# Hyper-parameters
#   - training / architecture: 3D_CNN/3D_supervised_HC_vs_TLE
#   - dataset seeds: Diffusion classifier new_runs (sample_seed=1000, split_seed=7)
# ---------------------------------------------------------------------------
BASE_HYPERPARAMS = {
    # ----- data -----
    'train_csv': os.path.join(DATA_DIR, "train.csv"),
    'val_csv':   os.path.join(DATA_DIR, "val.csv"),
    'test_csv':  os.path.join(DATA_DIR, "test.csv"),
    'synth_hc_folder_path': SYNTH_HC,
    'synth_tle_folder_path': SYNTH_TLE,

    # ----- cross-validation -----
    'n_folds': 5,
    'split_seed': 7,      # fold split (matches original runs' split_seed)
    'sample_seed': 1000,  # real stratified subsample seed (matches original runs)

    # ----- 3D CNN training hyper-parameters (Ben 3D CNN reference) -----
    'random_seed': 42,
    'batch_size': 8,
    'learning_rate': 1e-3,
    'num_epochs': 1000,
    'dropout_rate': 0.2,
    'filters': [64, 128, 256, 256, 256, 256],
    'early_stopping_patience': 250,
    'scheduler_patience': 20,
    'scheduler_factor': 0.5,
    'scheduler_threshold': 1e-4,
    'min_lr': 1e-8,
    'num_workers': 4,
    'steps_multiplier': 10,
    'use_best_threshold': False,
    'weight_decay': 0,
}


def _build_variants():
    """Mirror Diffusion_paper/classifier/new_runs exactly."""
    variants = []

    # real_* groups: runs_500 / runs_1000 / runs_2000 / runs_2723
    for n in [500, 1000, 2000, 2723]:
        group = f"runs_{n}"
        # real only
        variants.append({
            'group': group, 'name': f"real_{n}",
            'num_samples': n, 'num_synth_samples': None,
        })
        # real + synthetic at 25/50/75/100 %
        for pct, frac in [(25, 0.25), (50, 0.50), (75, 0.75), (100, 1.00)]:
            variants.append({
                'group': group, 'name': f"real_{n}_syn_{pct}",
                'num_samples': n, 'num_synth_samples': int(frac * n),
            })

    # syn_* group: full real train pool + N synthetic volumes
    #   NOTE: the original new_runs/runs_syn/syn_2723 used NO synthetic data
    #   (a leftover MRIDataset(...) call without num_synth_samples). Here it is
    #   made pattern-consistent: full real + 2723 synthetic volumes.
    for n_synth in [500, 1000, 2000, 2723, 5446]:  # 5446 == int(2723*2)
        variants.append({
            'group': "runs_syn", 'name': f"syn_{n_synth}",
            'num_samples': None, 'num_synth_samples': n_synth,
        })

    return variants


VARIANTS = _build_variants()


def get_variant_names():
    return [v['name'] for v in VARIANTS]


def get_groups():
    seen = []
    for v in VARIANTS:
        if v['group'] not in seen:
            seen.append(v['group'])
    return seen
