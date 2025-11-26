# MRI Saliency Map Generation

This directory contains scripts for generating saliency maps from trained MRI classification models using Captum attribution methods.

## Overview

The saliency map generator (`generate_saliency_maps.py`) loads a trained EfficientNetV2 model and generates various types of attribution maps to visualize which regions of the MRI scans the model focuses on when making predictions.

**Purpose**: 
- Understand model decision-making process
- Identify important brain regions for classification
- Validate that the model focuses on anatomically relevant structures
- Compare attribution patterns between real and synthetic data

## Features

- Multiple attribution methods (Integrated Gradients, Saliency, Gradient SHAP, etc.)
- Slice visualizations and multi-slice summaries
- Statistical analysis and comparison plots
- Raw attribution data export (.npy files)

## Installation

1. Install required packages:
```bash
pip install -r saliency_requirements.txt
```

Key dependencies include:
- `captum`: Attribution methods library
- `torch`, `torchvision`: PyTorch framework
- `nibabel`: NIfTI file handling
- `matplotlib`, `numpy`, `pandas`: Data handling and visualization

2. Ensure you have access to:
   - Trained model checkpoint (e.g., `new_runs/runs_*/*/parameters.json` and corresponding `.pth` file)
   - Model parameters file (JSON)
   - Validation/test data CSV with paths to MRI files

## Quick Start

### Option 1: Use the Runner Script (Recommended for Testing)

Edit paths in `run_saliency_generation.py` to match your setup, then run:

```bash
python run_saliency_generation.py
```

This will process a limited number of samples (default: 20) with predefined settings.

### Option 2: Direct Script Usage

Edit configuration directly in `generate_saliency_maps.py`:

```python
# Key configuration variables to update:
model_path = "path/to/your/model.pth"
parameters_path = "path/to/parameters.json"
csv_path = "path/to/val.csv"
output_dir = "path/to/output"
max_samples = 50
key_slices = [30, 40, 50, 60, 70, 80, 90]
methods = ['integrated_gradients', 'saliency', 'gradient_shap']
```

Then run:

```bash
python generate_saliency_maps.py
```

**Example paths** (update to match your directory structure):
- Model: `new_runs/runs_2723/real_2723/parameters.json` and corresponding model checkpoint
- CSV: Path to your validation data CSV file
- Output: `new_runs/runs_2723/real_2723/saliency_maps/`

## Configuration

**Attribution Methods**: integrated_gradients, saliency, gradient_shap, input_x_gradient, deep_lift, noise_tunnel_ig

**Key Parameters** (edit in script):
- `max_samples`: Number of samples to process
- `key_slices`: Slice indices to visualize (e.g., [30,40,50,60,70,80,90])
- `methods`: List of attribution methods
- `output_dir`: Results directory

## Output Structure

```
output_dir/
├── integrated_gradients/subject-*/
│   ├── *_summary.png
│   ├── *_slice_*.png
│   └── *_attributions.npy
├── saliency/
├── gradient_shap/
├── attribution_statistics.csv
└── attribution_summary.png
```

## Visualization

**Individual Slices**: Original MRI + attribution heatmap + overlay
**Summary**: Multi-slice overview with prediction confidence
**Statistics**: Attribution magnitude distributions and comparisons

## Clinical Interpretation

- **TLE**: High attribution in hippocampal/temporal lobe regions
- **HC**: More distributed patterns
- Verify anatomically plausible attributions