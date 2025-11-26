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
└── new_runs/                    # Training run outputs and checkpoints
```

## Model Architecture

**EfficientNetV2** (`model/efficientNetV2.py`): Binary classifier for HC vs TLE
- Input: 112×115×112 MRI volumes (We crop the region from the coronal axis where there is no brain information)
- Modified first conv for single-channel input
- Output: Binary classification score

## Usage

### Usage

**Training**: `python run.py` or `python train.py` (edit hyperparameters in script)

**Synthetic Data Integration**: Set `num_synth_samples` in hyperparams
- Automatically mixes real + synthetic .pkl files
- Balanced batch sampling for stable training

**Training Scenarios** (see `new_runs/`):
- `real_X`: Real only
- `real_X_syn_Y`: Real + Y% synthetic
- `syn_X`: Synthetic only

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
new_runs/
├── runs_500/                     # Experiments with 500 real samples
│   ├── real_500/                 # Real data only
│   ├── real_500_syn_25/          # Real + 25% synthetic (25% corresponds to 25% of #real_scans)
│   ├── real_500_syn_50/          # Real + 50% synthetic
│   ├── real_500_syn_75/          # Real + 75% synthetic
│   └── real_500_syn_100/         # Real + 100% synthetic
├── runs_1000/                    # Experiments with 1000 real samples
├── runs_2000/                    # Experiments with 2000 real samples
├── runs_2723/                    # Experiments with 2723 real samples (full dataset)
└── runs_syn/                     # Synthetic-only experiments
    ├── syn_500/
    ├── syn_1000/
    ├── syn_2000/
    ├── syn_2723/
    └── syn_5446/                 # Double dataset size (synthetic)

# Each experiment directory contains:
├── dataset.py                    # Dataset script used
├── efficientNetV2.py             # Model architecture used
├── train.py                      # Training script used
├── run.py                        # Orchestration script used
├── utils.py                      # Utilities used
├── parameters.json               # Hyperparameters
└── all_folds_metrics.json        # Aggregated metrics across folds
```

## Integration with Pipeline

Synthetic samples from diffusion model are evaluated via:
1. Classification performance (train with mixed real+synthetic)
2. Saliency analysis (verify anatomical correctness)