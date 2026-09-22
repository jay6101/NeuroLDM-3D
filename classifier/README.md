# Classifier Component

This directory contains the implementation of deep learning classifiers for HC (Healthy Controls) vs TLE (Temporal Lobe Epilepsy) classification, along with comprehensive saliency analysis tools for model interpretability.

## Purpose

The classifier component serves multiple roles in our pipeline:
1. **Baseline Performance**: Establish classification performance on real data
2. **Synthetic Data Evaluation**: Assess quality of generated synthetic images
3. **Interpretability**: Generate saliency maps to understand model decisions

## Directory Structure

```
classifier/
├── model/                        # Neural network architectures
├── train.py                     # Main training script with k-fold CV
├── run.py                       # Training orchestration script  
├── dataset.py                   # Dataset handling with synthetic data support
├── utils.py                     # Training utilities and metrics
├── generate_saliency_maps.py    # Comprehensive saliency analysis
├── run_saliency_generation.py   # Saliency generation orchestration
├── vanilla_backprop.py          # Vanilla backpropagation saliency
├── SALIENCY_README.md           # Detailed saliency analysis documentation
├── saliency_requirements.txt    # Saliency-specific dependencies
├── new_runs/                    # Original EfficientNetV2 hold-out runs
├── new_runs_3dcnn_cv/           # 5-fold 3D CNN CV (revision)
│   ├── run.py / aggregate_folds.py / plot_performance_curves.py
│   └── runs_3d/                 # fold metrics + figures (no .pth / CSVs)
└── delong_test/                 # Paired DeLong tests on pooled CV scores
    ├── delong_cv.py / plot_delong.py
    └── results_3dcnn_cv/
```

## Model Architecture

**EfficientNetV2** (`model/efficientNetV2.py`): Binary classifier for HC vs TLE (original pipeline)
- Input: 112×115×112 MRI volumes (We crop the region from the coronal axis where there is no brain information)
- Slices stacked as channels; modified first conv for that input
- Output: Binary classification score

**3D CNN** (`new_runs_3dcnn_cv/models/cnn_ben_3d.py`): same HC vs TLE task, volumetric `[1, D, H, W]` input. Training recipe (Adam, ReduceLROnPlateau, balanced sampler, early stopping) is shared with the 2D option in that folder; `--model 3d|2d` only swaps the architecture.

## Usage

### Usage

**Training**: `python run.py` or `python train.py` (edit hyperparameters in script)

**Synthetic Data Integration**: Set `num_synth_samples` in hyperparams
- Automatically mixes real + synthetic .pkl files
- Balanced batch sampling for stable training

**Training Scenarios** (see `new_runs/` and `new_runs_3dcnn_cv/runs_3d/`):
- `real_X`: Real only
- `real_X_syn_Y`: Real + Y% synthetic (Y is a percentage of the real count)
- `syn_X`: Synthetic only

**3D CNN 5-fold CV**: `python new_runs_3dcnn_cv/run.py --list` then `python new_runs_3dcnn_cv/run.py --device cuda:0`. Folds partition the ~481-sample val+test pool (`split_seed=7`). Aggregate with `python new_runs_3dcnn_cv/aggregate_folds.py`; plots with `python new_runs_3dcnn_cv/plot_performance_curves.py`.

**DeLong tests** on the pooled out-of-fold scores: `python delong_test/delong_cv.py`. Tables and figures are in `delong_test/results_3dcnn_cv/`.

## Training Configuration

**Key Hyperparameters**: dropout=0.5, batch_size=128, lr=1e-3, epochs=100

**Loss**: BCEWithLogitsLoss (binary classification)

**Scheduler**: Cosine annealing with restarts

## Evaluation Metrics

Accuracy, Sensitivity, Specificity, Precision, F1-Score, AUC

## Saliency Analysis

For detailed saliency analysis documentation, see [SALIENCY_README.md](SALIENCY_README.md).

### Quick Start: Saliency Generation

`python run_saliency_generation.py` or `python generate_saliency_maps.py`

Edit paths in script for model checkpoint, parameters.json, and validation CSV.

**Methods**: Integrated Gradients, Saliency Maps, Gradient SHAP, Input×Gradient

## Implementation Details

**Dataset** (`dataset.py`): Loads NIfTI files (real) and .pkl files (synthetic) with balanced sampling

**BalancedBatchSampler**: Ensures equal HC/TLE samples per batch

## Training Pipeline

1. Load train/test CSVs → filter to HC/TLE
2. Create datasets (real + optional synthetic)
3. Train with balanced sampling
4. Evaluate on test set
5. Generate saliency maps

## Output Structure

Training outputs are organized by experiment:

```
new_runs/                         # original EfficientNetV2 hold-out layout
new_runs_3dcnn_cv/runs_3d/        # 3D CNN CV (same variant names)
├── runs_500/                     # Experiments with 500 real samples
│   ├── real_500/                 # Real data only
│   ├── real_500_syn_25/          # Real + 25% synthetic (25% of #real_scans)
│   ├── real_500_syn_50/
│   ├── real_500_syn_75/
│   └── real_500_syn_100/
├── runs_1000/ / runs_2000/ / runs_2723/
└── runs_syn/                     # Synthetic-only (syn_500 … syn_5446)

# Each 3D-CNN variant directory contains:
├── parameters.json / run_summary.json
├── all_folds_metrics.json        # fold-wise mean±std of test metrics
└── fold_k/fold_metrics.json
```

## Integration with Pipeline

Synthetic samples from diffusion model are evaluated via:
1. Classification performance (train with mixed real+synthetic)
2. Saliency analysis (verify anatomical correctness)