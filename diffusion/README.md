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
│   ├── dit3d_window_attn.py      # 3D DiT with window attention
│   ├── dit3d.py                  # Standard 3D DiT implementation
│   ├── maisi_vae.py             # VAE model definition (for loading)
│   ├── efficientNetV2.py        # EfficientNet backbone option
│   └── utils_vit.py             # Vision Transformer utilities
├── train.py                      # Main diffusion training script
├── save_synth.py                # Generate synthetic samples
├── dataset.py                   # Dataset handling for latent codes
├── utils.py                     # Training utilities (currently empty)
├── prepare_nii.ipynb            # Data preparation notebook
├── viualize_latent.ipynb        # Latent space visualization
├── checkpoints_2/               # Model checkpoints
├── latent_data/                 # Stored VAE latent representations
└── per_slice_fid/              # Per-slice FID evaluation results
```

## Architecture Details

### DiT3D with Window Attention (`dit3d_window_attn.py`)

Our main architecture implements a 3D Diffusion Transformer with several key innovations:

#### Core Features
- **3D Patch Embedding**: Divides 3D latent volumes into patches for transformer processing
- **Conditional Generation**: Label conditioning for HC vs TLE generation
- **Timestep Embedding**: Sinusoidal embeddings for diffusion timesteps


### Diffusion Process

#### Forward Process (Noise Addition)
```python
# Gaussian noise schedule
q(x_t | x_0) = N(√(α̅_t) x_0, (1 - α̅_t)I)
```

#### Reverse Process (Denoising)
```python
# Model prediction
p_θ(x_{t-1} | x_t, c) = N(μ_θ(x_t, t, c), Σ_θ(x_t, t, c))
```

Where `c` represents the condition label (HC=0, TLE=1).

## Usage

### Training

```bash
python train.py \
    --data_dir /path/to/latent/data \
    --output_dir ./runs \
    --batch_size 8 \
    --num_epochs 1000 \
    --learning_rate 1e-4 \
    --num_timesteps 1000
```

### Key Training Parameters

```python
# Model configuration
model_config = {
    'input_size': (8, 14, 8),     # Latent spatial dimensions
    'patch_size': (4, 4, 4),      # 3D patch size
    'in_channels': 4,             # VAE latent channels
    'hidden_size': 384,           # Transformer dimension
    'depth': 12,                  # Number of layers
    'num_heads': 6,               # Attention heads
    'window_size': (4, 4, 4),     # Attention window
    'num_classes': 2,             # HC vs TLE
}

# Training configuration
training_config = {
    'batch_size': 8,
    'learning_rate': 1e-4,
    'num_timesteps': 1000,
    'beta_schedule': 'linear',
    'beta_start': 0.0001,
    'beta_end': 0.02,
}
```

### Generating Synthetic Samples

```bash
python save_synth.py \
    --model_checkpoint ./runs/best_model.pth \
    --vae_checkpoint ./VAE/best_vae.pth \
    --num_samples 1000 \
    --condition_label 1 \
    --output_dir ./synthetic_samples
```

## Training Process

### Loss Function

The diffusion model is trained with a simple MSE loss on the predicted noise:

```python
def training_loss(model, x_0, t, label):
    noise = torch.randn_like(x_0)
    x_t = q_sample(x_0, t, noise)  # Add noise
    predicted_noise = model(x_t, t, label)
    loss = F.mse_loss(predicted_noise, noise)
    return loss
```

### Training Schedule

1. **Warmup**: Gradual learning rate increase (optional)
2. **Main Training**: Stable learning rate with cosine annealing
3. **EMA Updates**: Exponential moving average for stable inference

### Noise Schedule

```python
# Linear beta schedule
betas = torch.linspace(beta_start, beta_end, num_timesteps)

# Derived quantities
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
```

## Implementation Details

### Dataset Handling (`dataset.py`)

```python
class MRIDataset:
    """
    Dataset for loading VAE-encoded latent representations
    
    Features:
    - Loads pre-computed latent codes from VAE
    - Balanced sampling across HC/TLE classes
    - Data augmentation in latent space (optional)
    """
```

### Sampling Strategies

#### DDPM Sampling
```python
@torch.no_grad()
def ddpm_sample(model, shape, device, num_steps=1000):
    """Standard DDPM sampling with full denoising steps"""
```

#### DDIM Sampling (Optional)
```python
@torch.no_grad()
def ddim_sample(model, shape, device, num_steps=50):
    """Faster DDIM sampling with fewer steps"""
```

## Evaluation

### Generation Quality Metrics

1. **FID (Fréchet Inception Distance)**: Distributional similarity
2. **IS (Inception Score)**: Sample quality and diversity
3. **LPIPS**: Perceptual similarity to real images
4. **Medical Metrics**: Anatomical correctness, tissue contrast

### Conditional Generation Evaluation

- **Class Consistency**: Generated samples match specified conditions
- **Classifier Performance**: How well classifiers perform on synthetic data
- **Saliency Analysis**: Whether synthetic images activate correct brain regions

## Technical Notes

### Latent Space Properties

- **Dimensionality**: Typically (4, 8, 14, 8) for compressed MRI volumes
- **Normalization**: Latent codes standardized to zero mean, unit variance
- **Interpolation**: Smooth interpolation possible in latent space

### Conditioning Strategy

- **Label Embedding**: Condition labels embedded into transformer
- **Classifier-Free Guidance**: Optional unconditional generation capability
- **Guidance Scale**: Controls conditioning strength during sampling

## Pipeline Integration

### VAE Integration
```python
# Load VAE for encoding/decoding
vae = VAE_Lite.load_from_checkpoint(vae_path)
vae.eval()

# Encode real images to latent space
with torch.no_grad():
    latents = vae.encode(real_images)
    
# Decode generated latents to images
with torch.no_grad():
    synthetic_images = vae.decode(generated_latents)
```

### Classifier Integration
```python
# Use generated samples for classifier training
synthetic_data = diffusion_model.sample(num_samples=1000)
classifier.train(real_data + synthetic_data)
```

