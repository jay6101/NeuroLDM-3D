"""
Central configuration for the 5-fold CV experiments.

Two model *architectures* can be chosen at run time (``--model``):

  * ``3d`` -> cnn_ben_3d (3D CNN, from 3D_CNN/3D_supervised_HC_vs_TLE).
              Dataset feeds a single-channel volume  [1, D, H, W].
  * ``2d`` -> efficientNetV2 (2D EfficientNetV2-L, from
              Diffusion_paper/classifier/model). Dataset feeds the slices as
              channels  [D, H, W].

IMPORTANT: only the *architecture* changes. The TRAINING ALGORITHM / hyper-
parameters are shared (BASE_HYPERPARAMS, i.e. the 3D_CNN recipe:
BCEWithLogits + Adam + ReduceLROnPlateau + BalancedBatchSampler + capped
steps/epoch + early stopping). Data CSVs, synthetic pools, fold-split seed and
real-subsample seed are shared too, so the 5 folds are identical across models
and variants -> fair comparison.

Public API
----------
BASE_HYPERPARAMS             shared data + training recipe
MODELS                       per-architecture info (model name / input shape)
build_base_hyperparams(m)    -> dict of hyper-parameters for architecture ``m``
default_runs_dir(m)          -> output root for architecture ``m``
model_choices()              -> ['3d', '2d']
VARIANTS / get_variant_names / get_groups
"""

import copy
import os

# ---------------------------------------------------------------------------
# Paths (kept identical to the Diffusion classifier runs)
# ---------------------------------------------------------------------------
DATA_DIR = "/space/mcdonald-syn01/1/projects/jsawant/Diffusion_paper/data_csvs"
SYNTH_HC = "/space/mcdonald-syn01/1/projects/jsawant/Diffusion_paper/synthetic_data_pkls_HC"
SYNTH_TLE = "/space/mcdonald-syn01/1/projects/jsawant/Diffusion_paper/synthetic_data_pkls_TLE"

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Hyper-parameters (SHARED across both architectures)
#   - training / recipe : 3D_CNN/3D_supervised_HC_vs_TLE
#   - dataset seeds      : Diffusion classifier new_runs (sample_seed=1000, split_seed=7)
# `filters` is only consumed by cnn_ben_3d; for the 2D model it is set to None
# in build_base_hyperparams (and therefore not passed to the model).
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

    # ----- training recipe (Ben 3D CNN reference; used for BOTH models) -----
    'random_seed': 42,
    'batch_size': 8,
    'learning_rate': 1e-3,
    'num_epochs': 1000,
    'dropout_rate': 0.2,
    'filters': [64, 128, 256, 256, 256, 256],   # cnn_ben_3d only
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

# ---------------------------------------------------------------------------
# Per-architecture info (NOT training hyper-parameters)
#   volume_3d  -> dataset output shape: True [1,D,H,W] / False [D,H,W]
#   use_filters-> whether the `filters` arg is passed to the model constructor
# ---------------------------------------------------------------------------
MODELS = {
    '3d': {'model_name': 'cnn_ben_3d',    'volume_3d': True,  'use_filters': True},
    '2d': {'model_name': 'efficientNetV2', 'volume_3d': False, 'use_filters': False},
}

# Keep each architecture's outputs in a separate root so they never collide:
#   3D -> runs/      (existing results live here)
#   2D -> runs_2d/
RUNS_DIR_BY_MODEL = {
    '3d': os.path.join(THIS_DIR, 'runs'),
    '2d': os.path.join(THIS_DIR, 'runs_2d'),
}


def model_choices():
    return list(MODELS.keys())


def build_base_hyperparams(model='3d'):
    """Return a fresh hyper-parameter dict: shared recipe + architecture info.

    The training algorithm is identical for every architecture; only the model
    name, the dataset output shape (`volume_3d`) and whether `filters` is used
    depend on the chosen architecture.
    """
    if model not in MODELS:
        raise ValueError(f"Unknown model '{model}'. Choose from {model_choices()}")
    hp = copy.deepcopy(BASE_HYPERPARAMS)
    arch = MODELS[model]
    hp['model_key'] = model
    hp['model_name'] = arch['model_name']
    hp['volume_3d'] = arch['volume_3d']
    if not arch['use_filters']:
        hp['filters'] = None   # efficientNetV2 constructor takes no `filters`
    return hp


def default_runs_dir(model='3d'):
    return RUNS_DIR_BY_MODEL.get(model, os.path.join(THIS_DIR, f'runs_{model}'))


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


# ---------------------------------------------------------------------------
# Backward-compatible defaults (default architecture = 3D CNN)
# ---------------------------------------------------------------------------
DEFAULT_MODEL = '3d'
MODEL_NAME = MODELS[DEFAULT_MODEL]['model_name']   # 'cnn_ben_3d'
DEFAULT_RUNS_DIR = RUNS_DIR_BY_MODEL[DEFAULT_MODEL]
