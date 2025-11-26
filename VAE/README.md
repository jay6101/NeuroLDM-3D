# VAE (Variational Autoencoder) Component

This directory contains the implementation of a 3D Variational Autoencoder for encoding T1-weighted MRI brain images into a compact latent representation.

## Purpose

The VAE serves as the first stage in our diffusion pipeline:
1. **Encode** 3D MRI volumes into a lower-dimensional latent space
2. **Decode** latent representations back to image space
3. **Enable** efficient diffusion modeling in latent space rather than high-dimensional image space

## Directory Structure

```
VAE/
├── model/
│   ├── __init__.py               # Package initialization
│   ├── maisi_vae.py              # Main VAE architectures (VAE Lite)
│   └── lpips3D.py                # 3D LPIPS perceptual loss
├── train.py                      # Main training script
├── infer.py                      # Inference script for encoding/decoding
├── save_latent.py                # Save latent representations of dataset
├── generate_synthetic_samples.py # Generate samples from VAE prior
├── dataset.py                    # Dataset handling (MRIDataset, BalancedBatchSampler)
├── utils.py                      # Training utilities and helper functions
└── visualize_recon.ipynb         # Reconstruction visualization notebook
```

## Architecture Details

### Key Components

- **`maisi_vae.py`**: VAE and VAE_Lite models (MAISI-based 3D encoder-decoder)
  - Input: 112×136×112 MRI volumes → Latent: 4×28×34×28
  - Components: Encoder, Decoder, ResNet blocks with attention
- **`lpips3D.py`**: 3D LPIPS perceptual loss for better reconstruction quality

## Usage

### Usage

**Training**: Edit hyperparameters in script, then run `python train.py`

**Inference**: Encode/decode MRI volumes with `python infer.py` (update `run_folder` path)

**Save Latents**: Encode dataset for diffusion training with `python save_latent.py`

**Generate Samples**: Sample from VAE prior with `python generate_synthetic_samples.py`

## Training Details

**Loss**: Reconstruction (L1) + KL divergence + LPIPS perceptual loss

**Key Parameters**: Configure in `train.py` (batch_size=2-4, lr=1e-4, latent_channels=4)

**Evaluation**: Reconstruction quality (L1, LPIPS), KL divergence

**Visualization**: Use `visualize_recon.ipynb` for qualitative assessment


