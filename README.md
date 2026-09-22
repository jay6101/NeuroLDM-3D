# Diffusion Models for Synthetic Medical Image Generation

This repository contains the implementation of a diffusion model approach for generating synthetic T1-weighted MRI brain images, specifically focusing on Temporal Lobe Epilepsy (TLE) vs Healthy Controls (HC) classification tasks.

The latest committed results (revision analyses) cover VAE reconstruction quality (including TLE-relevant ROIs), nearest-neighbor memorization checks in latent space, and 5-fold 3D CNN classification with DeLong tests.

## Project Architecture

The project consists of three main components working together in a pipeline:

1. **VAE (Variational Autoencoder)** - Encodes 3D MRI volumes into a lower-dimensional latent space
2. **Diffusion Model** - Generates new samples in the VAE latent space using a 3D DiT (Diffusion Transformer)
3. **Classifier** - Classifies real and synthetic images (2D EfficientNetV2 and 3D CNN), with saliency maps and DeLong tests for interpretability and significance

## Repository Structure

```
NeuroLDM-3D/
├── VAE/                           # Variational Autoencoder
│   ├── model/                     # MAISI-based VAE + 3D LPIPS
│   ├── train.py / infer.py / save_latent.py
│   └── recon_eval/                # Reconstruction metrics, examples, ROI MSE
│       ├── evaluate_reconstruction.py
│       ├── export_examples.py
│       ├── results/               # per-subject + summary metrics, example figures
│       └── roi_analysis/          # AAL3 / Harvard-Oxford ROI MSE
├── diffusion/                     # 3D DiT generative model
│   ├── train.py / save_synth.py
│   ├── per_slice_fid_ignite.py    # Per-slice FID
│   └── nn_eval/                   # Latent nearest-neighbor memorization analysis
│       ├── nn_memorization.py
│       ├── filter_coincidence.py  # Drop near-duplicate test/train pairs
│       └── results/
├── classifier/                    # Classification and evaluation
│   ├── model/efficientNetV2.py    # Original 2D (slice-as-channels) classifier
│   ├── train.py / run.py          # Original hold-out EfficientNetV2 runs
│   ├── generate_saliency_maps.py
│   ├── new_runs/                  # Original training outputs
│   ├── new_runs_3dcnn_cv/         # 5-fold 3D CNN CV (revision)
│   │   ├── run.py / aggregate_folds.py / plot_performance_curves.py
│   │   └── runs_3d/               # Metrics + figures (checkpoints omitted)
│   └── delong_test/               # Paired DeLong AUC tests on CV scores
├── samples/                       # Sample visualizations (real vs synthetic)
├── fid_results_ignite/            # FID evaluation results
├── visualize.ipynb
└── data_split.ipynb
```

**Note**: The actual MRI data and CSV files are typically stored outside this repository (e.g. `data_csvs/`, `synthetic_data_pkls_TLE/`, `synthetic_data_pkls_HC/`). Update paths in the training / eval scripts to match your layout. Large checkpoints (`.pth`) and saliency volumes are not tracked.

## Pipeline

1. **Data Preparation**: `data_split.ipynb` - Create train/val splits
2. **VAE Training**: `VAE/train.py` - Learn latent representations
3. **Save Latents**: `VAE/save_latent.py` - Encode dataset to latent space
4. **Diffusion Training**: `diffusion/train.py` - Train generative model
5. **Generate Samples**: `diffusion/save_synth.py` - Create synthetic MRIs
6. **Evaluation**:
   - Reconstruction: `VAE/recon_eval/evaluate_reconstruction.py`
   - FID: `diffusion/per_slice_fid_ignite.py`
   - Memorization: `diffusion/nn_eval/nn_memorization.py`
   - Classification: `classifier/train.py` (EfficientNetV2) or `classifier/new_runs_3dcnn_cv/run.py` (3D CNN CV)
   - Significance: `classifier/delong_test/delong_cv.py`
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
- **EfficientNetV2** (original): slice stack as channels; real / mixed / synthetic-only training
- **3D CNN** (`new_runs_3dcnn_cv`): 5-fold CV over the combined val+test pool (~481 volumes), same real-budget and synthetic-percentage grid as the original runs
- Saliency map generation via Captum (EfficientNetV2 pipeline)

## Evaluation

- **Reconstruction**: Per-volume MSE / NMSE / PSNR / LPIPS on train and held-out splits; ROI MSE for hippocampus, parahippocampal gyrus, amygdala, thalamus, insula, cingulate, and anterior temporal cortex (`VAE/recon_eval/`)
- **FID**: Per-slice FID scores (Inception-based)
- **Memorization**: Nearest-train MSE for held-out real vs synthetic latents (Meehan et al. data-copying test), with an optional filter that drops test/train coincidences (`diffusion/nn_eval/`)
- **Classification**: Accuracy, sensitivity, specificity, AUC on real / synthetic / mixed data; fold mean±std in `all_folds_metrics.json`
- **DeLong**: Pooled out-of-fold AUCs and paired tests across variants (`classifier/delong_test/results_3dcnn_cv/`)
- **Saliency**: Attribution maps (InputxGradient, Integrated Gradients, Gradient SHAP)

## Revision results (committed)

These files are already in the repo; you do not need checkpoints to inspect them.

| Analysis | Where to look |
|---|---|
| VAE reconstruction summaries | `VAE/recon_eval/results/summary_metrics.csv` |
| Example reconstructions (HC/TLE, 5th/95th percentile MSE) | `VAE/recon_eval/results/examples_p5_p95/` |
| AAL3 ROI MSE | `VAE/recon_eval/roi_analysis/results_aal3/` |
| NN distances after dropping coincidences (threshold 0.05) | `diffusion/nn_eval/results/excl_coincidence_0p05/` |
| 3D CNN CV fold metrics and performance curves | `classifier/new_runs_3dcnn_cv/runs_3d/` |
| DeLong CIs and pairwise tests | `classifier/delong_test/results_3dcnn_cv/` |

Held-out reconstruction (union of val+test, n=481): mean MSE ≈ 0.021, NMSE ≈ 0.022, PSNR ≈ 33.2 dB, LPIPS ≈ 0.019. Train (n=2723) is essentially the same, so the VAE is not collapsing on unseen volumes.

Nearest-neighbor check (round-tripped latents, coincidences with nearest-train MSE ≤ 0.05 removed, n=257 per arm): no synthetic sample sits below the real-test minimum distance; median fake/real distance ratio ≈ 0.87. See `diffusion/nn_eval/summary.json` and `results/excl_coincidence_0p05/summary.json`.

3D CNN, full real training set (`runs_2723/real_2723`): fold-mean AUC-ROC 0.889 ± 0.009. Pooled out-of-fold AUC (n=481) is 0.877 [0.846, 0.907]. Mixing synthetic data at the same budget stays in a similar range (e.g. `real_2723_syn_50` pooled AUC 0.874). Synthetic-only runs are lower (e.g. `syn_2723` fold-mean AUC 0.703). Curves: `classifier/new_runs_3dcnn_cv/runs_3d/figures/`.

## Medical Context

- **Application**: Temporal Lobe Epilepsy (TLE) vs Healthy Controls (HC) classification
- **Data**: T1-weighted MRI brain scans
- **Goal**: Generate synthetic medical images for data augmentation
