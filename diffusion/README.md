# Diffusion Model Component

This directory contains the implementation of a 3D Diffusion Transformer (DiT) for generating synthetic medical images in the VAE latent space.

## Purpose

The diffusion model serves as the generative core of our pipeline:
1. **Learn** the distribution of VAE-encoded brain MRI latents
2. **Generate** new latent codes that decode to realistic synthetic brain images
3. **Enable** conditional generation for specific medical conditions (HC vs TLE)

## Directory Structure

```
diffusion/
├── model/
│   ├── dit3d_window_attn.py      # 3D DiT with window attention (main model)
│   ├── maisi_vae.py              # VAE model definition (for loading VAE)
│   └── efficientNetV2.py         # EfficientNet classifier (for evaluation)
├── train.py                      # Main diffusion training script
├── save_synth.py                 # Generate synthetic samples
├── dataset.py                    # Dataset handling for latent codes
├── per_slice_fid_ignite.py       # Per-slice FID evaluation script
├── prepare_nii.ipynb             # Data preparation notebook
└── viualize_latent.ipynb         # Latent space visualization notebook
```

**Note**: Model checkpoints and latent data are typically stored during training in directories you specify (not included in the repo by default).

## Architecture Details

### Architecture

**DiT3D with Window Attention** (`dit3d_window_attn.py`):
- 3D patch embedding, window attention, conditional generation (HC/TLE labels)
- Timestep and label embeddings with adaptive layer normalization
- Variants: DiT_XL_2, DiT_B_4, DiT_S_4 (we use DiT_S_4)


### Diffusion Process

**Forward**: Add Gaussian noise over 1000 timesteps
**Reverse**: Denoise conditioned on labels (HC=0, TLE=1)

## Usage

### Usage

**Training**: Edit `parse_args()` in `train.py`, then run `python train.py`
- Key params: img_size=(8,28,34,28), patch_size=(4,2,4), in_chans=4, num_timesteps=1000

**Generate Samples**: Edit paths in `save_synth.py`, then run `python save_synth.py`
- Loads VAE + diffusion model → samples latents → decodes to MRI → saves as .pkl

## Training Details

**Loss**: MSE between predicted and actual noise
**Schedule**: Linear beta schedule (0.0001 to 0.02) over 1000 steps
**Optimization**: Adam with EMA, learning rate warmup

## Implementation Details

**Dataset** (`dataset.py`): Loads pre-computed latent .pkl files with balanced sampling

**Sampling**: DDPM - start from noise, iteratively denoise for 1000 steps with label conditioning

## Evaluation

**FID Scores**: `per_slice_fid_ignite.py` - Computes per-slice FID across all axes
**Classification**: Train classifier on synthetic data (see `../classifier/`)
**Saliency**: Verify anatomically correct activations

## Pipeline Integration

1. Encode training data with VAE → save latents as .pkl
2. Train diffusion on latent codes
3. Generate new latents → decode with VAE → synthetic MRI volumes
4. Use synthetic samples for classifier training (see `../classifier/`)

