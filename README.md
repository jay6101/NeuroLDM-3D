# Diffusion Models for Synthetic Medical Image Generation

This repository contains the implementation of a diffusion model approach for generating synthetic T1-weighted MRI brain images, specifically focusing on Temporal Lobe Epilepsy (TLE) vs Healthy Controls (HC) classification tasks.

## Project Architecture

The project consists of three main components working together in a pipeline:

1. **VAE (Variational Autoencoder)** - Encodes 3D MRI volumes into a lower-dimensional latent space
2. **Diffusion Model** - Generates new samples in the VAE latent space using a 3D DiT (Diffusion Transformer)
3. **Classifier** - Classifies real and synthetic images, with saliency map generation for interpretability

## Repository Structure

```
Diffusion_paper/
├── VAE/                           # Variational Autoencoder implementation
│   ├── model/                     # VAE model architectures
│   ├── train.py                   # VAE training script
│   ├── infer.py                   # VAE inference
│   └── dataset.py                 # VAE dataset handling
├── diffusion/                     # Diffusion model implementation
│   ├── model/                     # DiT model architectures
│   ├── train.py                   # Diffusion training script
│   ├── save_synth.py             # Generate synthetic samples
│   └── dataset.py                # Diffusion dataset handling
├── classifier/                    # Classification and evaluation
│   ├── model/                     # Classifier architectures
│   ├── train.py                   # Classifier training
│   ├── generate_saliency_maps.py  # Saliency map generation
│   └── SALIENCY_README.md        # Saliency analysis documentation
├── data_csvs/                     # Dataset splits
│   ├── train.csv                  # Training data
│   ├── val.csv                    # Validation data
│   └── pickle_prep_*.csv         # Preprocessed data indices
├── synthetic_data_pkls_TLE/       # Generated TLE synthetic samples
├── synthetic_data_pkls_HC/        # Generated HC synthetic samples
├── fid_results/                   # FID evaluation results
├── visualize.ipynb               # Visualization utilities
└── data_split.ipynb              # Data splitting utilities
```

## Getting Started

### Quick Start

1. **Data Preparation**: Use `data_split.ipynb` to prepare your dataset splits
2. **VAE Training**: Train the VAE to learn latent representations
3. **Diffusion Training**: Train the diffusion model in latent space
4. **Synthetic Generation**: Generate new samples using trained models
5. **Evaluation**: Use the classifier and saliency maps for evaluation

## Model Details

### VAE Component
- **Architecture**: 3D encoder-decoder with ResNet blocks
- **Purpose**: Compress 3D MRI volumes into latent space
- **Training**: Reconstruction loss + KL divergence + optional adversarial loss

### Diffusion Component  
- **Architecture**: 3D DiT (Diffusion Transformer)
- **Purpose**: Generate new latent codes that can be decoded into synthetic MRI volumes
- **Training**: Denoising diffusion probabilistic model (DDPM) in latent space

### Classifier Component
- **Purpose**: Evaluate quality of synthetic images through classification performance
- **Features**: HC vs TLE classification with saliency map generation
- **Evaluation**: Provides interpretability through attention visualization

## Evaluation Metrics

- **FID (Fréchet Inception Distance)**: Measures distributional similarity between real and synthetic images
- **Classification Performance**: Accuracy, sensitivity, specificity on real vs synthetic data
- **Saliency Analysis**: Brain region importance for classification decisions

## Medical Context

This work focuses on:
- **Temporal Lobe Epilepsy (TLE)**: A common form of focal epilepsy
- **T1-weighted MRI**: Structural brain imaging modality
- **Synthetic Data Generation**: Augmenting limited medical datasets while preserving privacy
- **Interpretability**: Understanding which brain regions are important for classification
