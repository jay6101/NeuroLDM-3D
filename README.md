# Diffusion Models for Synthetic Medical Image Generation

This repository contains the implementation of a diffusion model approach for generating synthetic T1-weighted MRI brain images, specifically focusing on Temporal Lobe Epilepsy (TLE) vs Healthy Controls (HC) classification tasks.

## Project Architecture

The project consists of three main components working together in a pipeline:

1. **VAE (Variational Autoencoder)** - Encodes 3D MRI volumes into a lower-dimensional latent space
2. **Diffusion Model** - Generates new samples in the VAE latent space using a 3D DiT (Diffusion Transformer)
3. **Classifier** - Classifies real and synthetic images, with saliency map generation for interpretability

## Repository Structure

```
NeuroLDM-3D/
├── VAE/                           # Variational Autoencoder implementation
│   ├── model/                     # VAE model architectures
│   │   ├── maisi_vae.py          # Main VAE model (MAISI-based)
│   │   └── lpips3D.py            # 3D LPIPS perceptual loss
│   ├── train.py                   # VAE training script
│   ├── infer.py                   # VAE inference script
│   ├── save_latent.py             # Save latent representations
│   ├── generate_synthetic_samples.py  # Generate samples from VAE
│   ├── dataset.py                 # VAE dataset handling
│   ├── utils.py                   # Training utilities
│   ├── visualize_recon.ipynb      # Reconstruction visualization
│   └── best_runs/                 # Best model checkpoints
├── diffusion/                     # Diffusion model implementation
│   ├── model/                     # DiT model architectures
│   │   ├── dit3d_window_attn.py  # 3D DiT with window attention
│   │   ├── maisi_vae.py          # VAE model for loading
│   │   └── efficientNetV2.py     # EfficientNet backbone
│   ├── train.py                   # Diffusion training script
│   ├── save_synth.py              # Generate synthetic samples
│   ├── dataset.py                 # Diffusion dataset handling
│   ├── per_slice_fid_ignite.py    # Per-slice FID evaluation
│   ├── prepare_nii.ipynb          # Data preparation notebook
│   └── viualize_latent.ipynb      # Latent space visualization
├── classifier/                    # Classification and evaluation
│   ├── model/                     # Classifier architectures
│   │   └── efficientNetV2.py     # EfficientNetV2 classifier
│   ├── train.py                   # Classifier training script
│   ├── run.py                     # Training orchestration
│   ├── dataset.py                 # Dataset with synthetic data support
│   ├── utils.py                   # Training utilities
│   ├── generate_saliency_maps.py  # Saliency map generation
│   ├── run_saliency_generation.py # Saliency generation runner
│   ├── SALIENCY_README.md         # Saliency analysis documentation
│   ├── saliency_requirements.txt  # Saliency dependencies
│   └── new_runs/                  # Training outputs and checkpoints
├── samples/                       # Sample visualizations (real vs synthetic)
├── fid_results_ignite/            # FID evaluation results
├── visualize.ipynb                # Visualization utilities
└── data_split.ipynb               # Data splitting utilities
```

**Note**: The actual MRI data and CSV files are typically stored outside this repository in a parent directory (e.g., `Diffusion_paper/data_csvs/`, `Diffusion_paper/synthetic_data_pkls_TLE/`, `Diffusion_paper/synthetic_data_pkls_HC/`). Update paths in training scripts according to your data location.

## Pipeline

1. **Data Preparation**: `data_split.ipynb` - Create train/val splits
2. **VAE Training**: `VAE/train.py` - Learn latent representations
3. **Save Latents**: `VAE/save_latent.py` - Encode dataset to latent space
4. **Diffusion Training**: `diffusion/train.py` - Train generative model
5. **Generate Samples**: `diffusion/save_synth.py` - Create synthetic MRIs
6. **Evaluation**: 
   - FID: `diffusion/per_slice_fid_ignite.py`
   - Classification: `classifier/train.py`
   - Saliency: `classifier/generate_saliency_maps.py`

## Components

### VAE
- 3D VAE with ResNet blocks
- Compresses MRI volumes (112×136×112) to latent space (4×28×34×28)
- Loss: Reconstruction + KL divergence + LPIPS

### Diffusion Model
- 3D DiT (Diffusion Transformer) with window attention
- Generates latent codes conditioned on HC/TLE labels
- DDPM training in latent space (1000 timesteps)

### Classifier
- EfficientNetV2 for HC vs TLE classification
- Supports real + synthetic data training
- Saliency map generation via Captum

## Evaluation

- **FID**: Per-slice FID scores (Inception-based)
- **Classification**: Accuracy, sensitivity, specificity, AUC on real/synthetic/mixed data
- **Saliency**: Attribution maps (InputxGradient, Integrated Gradients, Gradient SHAP) for interpretability

## Medical Context

- **Application**: Temporal Lobe Epilepsy (TLE) vs Healthy Controls (HC) classification
- **Data**: T1-weighted MRI brain scans
- **Goal**: Generate synthetic medical images for data augmentation
