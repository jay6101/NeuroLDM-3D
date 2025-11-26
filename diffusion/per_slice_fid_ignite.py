#!/usr/bin/env python3

import os
import pickle
import numpy as np
import math
from glob import glob
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import nibabel as nib
from random import shuffle
import argparse
from sklearn.preprocessing import MinMaxScaler
from skimage.transform import resize
from typing import Union
from packaging.version import Version

# Ignite imports for metrics
from ignite.metrics import FID
import ignite.distributed as idist


class InceptionModel(torch.nn.Module):
    """Inception Model pre-trained on the ImageNet Dataset."""

    def __init__(self, return_features: bool, device: Union[str, torch.device] = "cpu") -> None:
        try:
            import torchvision
            from torchvision import models
        except ImportError:
            raise ModuleNotFoundError("This module requires torchvision to be installed.")
        super(InceptionModel, self).__init__()
        self._device = device
        if Version(torchvision.__version__) < Version("0.13.0"):
            model_kwargs = {"pretrained": True, "transform_input": False}
        else:
            model_kwargs = {"weights": models.Inception_V3_Weights.DEFAULT, "transform_input": False}

        self.model = models.inception_v3(**model_kwargs).to(self._device)

        if return_features:
            self.model.fc = torch.nn.Identity()
        else:
            self.model.fc = torch.nn.Sequential(self.model.fc, torch.nn.Softmax(dim=1))
        self.model.eval()

    @torch.no_grad()
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        if data.dim() != 4:
            raise ValueError(f"Inputs should be a tensor of dim 4, got {data.dim()}")
        if data.shape[1] != 3:
            raise ValueError(f"Inputs should be a tensor with 3 channels, got {data.shape}")
        if data.device != torch.device(self._device):
            data = data.to(self._device)
        return self.model(data)


def preprocess_slice_for_inception(slice_2d, target_height=112, target_width=112, minmax_for_synthetic=False):
    """
    Convert a 2D numpy array (e.g., shape [H, W]) into a (3, target_size, target_size) tensor for Inception:
      1) If minmax_for_synthetic=True, scale slice by (slice - min)/(max - min) then *255.
         Otherwise assume the slice is already in a suitable range (e.g., after standardizing the entire volume).
      2) Convert to PIL and resize to (target_size, target_size).
      3) Replicate 1 channel → 3 channels.
      4) Normalize by ImageNet means & stds.
    """
    # Convert to 0-1 range if needed
    # smin, smax = slice_2d.min(), slice_2d.max()
    # if smax - smin < 1e-10:
    #     slice_2d = np.zeros_like(slice_2d)
    # else:
    #     slice_2d = (slice_2d - smin) / (smax - smin)
    
    slice_2d = (slice_2d * 255).astype(np.uint8)

    pil_img = Image.fromarray(slice_2d)
    transform_steps = transforms.Compose([
        #transforms.Resize((target_height, target_width)),
        transforms.Resize((299, 299)),
        transforms.ToTensor(),  # shape (1, H, W)
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # replicate grayscale -> 3 channels
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    return transform_steps(pil_img)


def load_synthetic_volumes_batch(pkl_files, batch_start, batch_size):
    """Load a batch of synthetic volumes from .pkl files."""
    volumes = []
    batch_end = min(batch_start + batch_size, len(pkl_files))
    
    for idx in range(batch_start, batch_end):
        pkl_file = pkl_files[idx]
        try:
            with open(pkl_file, 'rb') as f:
                data_dict = pickle.load(f)
            
            volume = data_dict["image"]  # shape [D,H,W]
            volume = np.asarray(volume, dtype=np.float32)
            
            # Min-max normalize to 0-1 range
            vol_min, vol_max = volume.min(), volume.max()
            if vol_max - vol_min > 1e-10:
                volume = (volume - vol_min) / (vol_max - vol_min)
            else:
                volume = np.zeros_like(volume)
            
            volumes.append(volume)
        except Exception as e:
            print(f"Error loading synthetic file {pkl_file}: {e}")
    
    return volumes


def load_real_volumes_batch(nii_paths, batch_start, batch_size):
    """Load a batch of real volumes from .nii files."""
    volumes = []
    batch_end = min(batch_start + batch_size, len(nii_paths))
    
    for idx in range(batch_start, batch_end):
        nii_path = nii_paths[idx]
        try:
            if not os.path.exists(nii_path):
                print(f"File not found: {nii_path}")
                continue
            
            # Load NIFTI
            img_nib = nib.load(nii_path)
            volume = img_nib.get_fdata()  # shape [D,H,W]
            
            # Subtract mean and divide by std for entire 3D volume
            volume = (volume - np.mean(volume)) / np.std(volume)
            
            # Crop and resize to match synthetic data dimensions
            volume = volume[:, 1:137, :]
            
            # Resize each slice from 113x113 to 112x112
            current_D, current_H, current_W = volume.shape
            resized_slices = []
            
            resize_transform = transforms.Compose([
                transforms.Resize((112, 112)),
                transforms.ToTensor()
            ])
            
            # Resize slices along axis 1 (coronal slices)
            for h_idx in range(current_H):
                slice_2d = Image.fromarray(volume[:, h_idx, :])  # Shape: (D, W) = (113, 113)
                slice_tensor = resize_transform(slice_2d)
                resized_slices.append(slice_tensor.numpy()[0])
            
            # Reconstruct volume with new dimensions: (112, 136, 112)
            volume = np.stack(resized_slices, axis=1)  # Stack along H dimension
            vol_min, vol_max = volume.min(), volume.max()
            if vol_max - vol_min > 1e-10:
                volume = (volume - vol_min) / (vol_max - vol_min)
            else:
                volume = np.zeros_like(volume)
            volumes.append(volume)
            
        except Exception as e:
            print(f"Error loading real file {nii_path}: {e}")
    
    return volumes


def calculate_per_slice_fid_ignite(config):
    """
    Calculate FID for each slice position using PyTorch Ignite with support for .pkl synthetic data.
    """
    synthetic_folder = config.get('synthetic_folder')
    synthetic_folder_2 = config.get('synthetic_folder_2')
    real_csv = config.get('real_csv')
    device = config.get('device', 'cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = config.get('batch_size', 16)
    max_files = config.get('max_files', 100)
    volume_batch_size = config.get('volume_batch_size', 10)  # Number of volumes to load at once
    
    print(f"Using device: {device}")
    print(f"Processing {volume_batch_size} volumes at a time")
    
    # Get synthetic file lists
    synthetic_files = sorted(glob(os.path.join(synthetic_folder, '*.pkl')))
    if synthetic_folder_2:
        synthetic_files_2 = sorted(glob(os.path.join(synthetic_folder_2, '*.pkl')))
        synthetic_files.extend(synthetic_files_2)
    shuffle(synthetic_files)
    synthetic_files = synthetic_files[:max_files]
    
    # Get real file lists from CSV
    df = pd.read_csv(real_csv)
    df = df.loc[df["HC_vs_LTLE_vs_RTLE_string"].isin(["HC", "left", "right"])]
    print(f"Found {len(df)} real images in CSV")
    df = df.sample(max_files, random_state=42)
    real_nii_paths = df['file'].tolist()
    
    print(f"Processing {len(synthetic_files)} synthetic and {len(real_nii_paths)} real images")
    
    # Create shared feature extractor
    print("Initializing shared Inception feature extractor...")
    shared_feat_extractor = InceptionModel(return_features=True, device=device)
    
    # Initialize FID metrics for each position
    # Determine dimensions based on the data format
    # Synthetic: D,H,W and Real: (112, 136, 112) after processing
    D_dim, H_dim, W_dim = 112, 136, 112  # Based on your preprocessing
    
    fid_metrics = {}
    for axis in ['axis0', 'axis1', 'axis2']:
        fid_metrics[axis] = {}
        if axis == 'axis0':
            dim_size = D_dim
        elif axis == 'axis1':
            dim_size = H_dim
        else:
            dim_size = W_dim
            
        for pos in range(dim_size):
            fid_metrics[axis][pos] = FID(num_features=2048, feature_extractor=shared_feat_extractor, device=device)
            fid_metrics[axis][pos].reset()
    
    # Process data in batches
    total_files = max(len(synthetic_files), len(real_nii_paths))
    num_batches = (total_files + volume_batch_size - 1) // volume_batch_size
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * volume_batch_size
        
        print(f"Processing batch {batch_idx + 1}/{num_batches}")
        
        # Load synthetic volumes for this batch
        synthetic_volumes = load_synthetic_volumes_batch(synthetic_files, batch_start, volume_batch_size)
        
        # Load real volumes for this batch
        real_volumes = load_real_volumes_batch(real_nii_paths, batch_start, volume_batch_size)
        
        # Process each axis and position
        for axis_idx, axis_name in enumerate(['axis0', 'axis1', 'axis2']):
            if axis_name == 'axis0':
                dim_size = D_dim
                target_height, target_width = 136, 112
            elif axis_name == 'axis1':
                dim_size = H_dim
                target_height, target_width = 112, 112
            else:
                dim_size = W_dim
                target_height, target_width = 112, 136
            
            for pos in range(dim_size):
                real_slices = []
                synthetic_slices = []
                
                # Extract slices for this position from real volumes
                for volume in real_volumes:
                    if axis_idx == 0:  # axis0 (sagittal)
                        slice_2d = volume[pos, :, :]
                    elif axis_idx == 1:  # axis1 (coronal)  
                        slice_2d = volume[:, pos, :]
                    else:  # axis2 (axial)
                        slice_2d = volume[:, :, pos]
                    
                    img_tensor = preprocess_slice_for_inception(slice_2d, target_height, target_width, minmax_for_synthetic=False)
                    real_slices.append(img_tensor)
                
                # Extract slices for this position from synthetic volumes
                for volume in synthetic_volumes:
                    if axis_idx == 0:  # axis0 (sagittal)
                        slice_2d = volume[pos, :, :] #if pos < volume.shape[0] else np.zeros((volume.shape[1], volume.shape[2]))
                    elif axis_idx == 1:  # axis1 (coronal)  
                        slice_2d = volume[:, pos, :] #if pos < volume.shape[1] else np.zeros((volume.shape[0], volume.shape[2]))
                    else:  # axis2 (axial)
                        slice_2d = volume[:, :, pos] #if pos < volume.shape[2] else np.zeros((volume.shape[0], volume.shape[1]))
                    
                    img_tensor = preprocess_slice_for_inception(slice_2d, target_height, target_width, minmax_for_synthetic=False)
                    synthetic_slices.append(img_tensor)
                
                # Process in sub-batches and update FID metric
                min_slices = min(len(real_slices), len(synthetic_slices))
                if min_slices > 0:
                    for sub_batch_start in range(0, min_slices, batch_size):
                        sub_batch_end = min(sub_batch_start + batch_size, min_slices)
                        
                        real_batch = torch.stack(real_slices[sub_batch_start:sub_batch_end], dim=0).to(device)
                        synthetic_batch = torch.stack(synthetic_slices[sub_batch_start:sub_batch_end], dim=0).to(device)
                        
                        fid_metrics[axis_name][pos].update((real_batch, synthetic_batch))
                        
                        del real_batch, synthetic_batch
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Clean up memory
        del synthetic_volumes, real_volumes
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Compute final FID scores
    axis0_fids = {}
    axis1_fids = {}
    axis2_fids = {}
    
    print("Computing final FID scores...")
    for axis_name, axis_fids in [('axis0', axis0_fids), ('axis1', axis1_fids), ('axis2', axis2_fids)]:
        print(f"Computing FIDs for {axis_name}...")
        if axis_name == 'axis0':
            dim_size = D_dim
        elif axis_name == 'axis1':
            dim_size = H_dim
        else:
            dim_size = W_dim
            
        for pos in range(dim_size):
            try:
                fid_score = fid_metrics[axis_name][pos].compute()
                # Convert to float if it's a tensor, otherwise use as-is
                if isinstance(fid_score, torch.Tensor):
                    fid_score_val = float(fid_score.item())
                else:
                    fid_score_val = float(fid_score)
                
                # Check for valid values
                if not (math.isnan(fid_score_val) or math.isinf(fid_score_val)):
                    axis_fids[pos] = fid_score_val
                    if pos % 20 == 0:  # Print every 20th position
                        print(f"  {axis_name}, Position {pos}: FID = {fid_score_val:.4f}")
                else:
                    print(f"  Skipping {axis_name} position {pos}: Invalid FID score (nan/inf)")
            except Exception as e:
                print(f"  Error computing FID for {axis_name} position {pos}: {e}")
    
    # Convert to arrays and calculate statistics
    axis0_fid_array = np.array([axis0_fids[k] for k in sorted(axis0_fids.keys())])
    axis1_fid_array = np.array([axis1_fids[k] for k in sorted(axis1_fids.keys())])
    axis2_fid_array = np.array([axis2_fids[k] for k in sorted(axis2_fids.keys())])
    
    avg_fid_axis0 = np.mean(axis0_fid_array) if len(axis0_fid_array) > 0 else float('nan')
    avg_fid_axis1 = np.mean(axis1_fid_array) if len(axis1_fid_array) > 0 else float('nan')
    avg_fid_axis2 = np.mean(axis2_fid_array) if len(axis2_fid_array) > 0 else float('nan')
    
    all_fids = np.concatenate([axis0_fid_array, axis1_fid_array, axis2_fid_array])
    overall_avg_fid = np.mean(all_fids) if len(all_fids) > 0 else float('nan')
    
    print(f"\n=== FID Results Summary (PyTorch Ignite - Diffusion) ===")
    print(f"Average FID for axis 0 (sagittal): {avg_fid_axis0:.4f}")
    print(f"Average FID for axis 1 (coronal): {avg_fid_axis1:.4f}")
    print(f"Average FID for axis 2 (axial): {avg_fid_axis2:.4f}")
    print(f"Overall average FID: {overall_avg_fid:.4f}")
    
    return {
        'axis0_fids': axis0_fid_array,
        'axis1_fids': axis1_fid_array,
        'axis2_fids': axis2_fid_array,
        'axis0_fids_by_position': axis0_fids,
        'axis1_fids_by_position': axis1_fids,
        'axis2_fids_by_position': axis2_fids,
        'avg_fid_axis0': avg_fid_axis0,
        'avg_fid_axis1': avg_fid_axis1,
        'avg_fid_axis2': avg_fid_axis2,
        'overall_avg_fid': overall_avg_fid
    }


def main():
    parser = argparse.ArgumentParser(description='Calculate per-slice FID using PyTorch Ignite for Diffusion paper')
    parser.add_argument('--synthetic_folder', type=str, 
                        default='/space/mcdonald-syn01/1/projects/jsawant/Diffusion_paper/synthetic_data_pkls_HC',
                        help='Path to folder containing synthetic .pkl files')
    parser.add_argument('--synthetic_folder_2', type=str,
                        default='/space/mcdonald-syn01/1/projects/jsawant/Diffusion_paper/synthetic_data_pkls_TLE',
                        help='Path to second folder containing synthetic .pkl files')
    parser.add_argument('--real_csv', type=str,
                        default='/space/mcdonald-syn01/1/projects/jsawant/Diffusion_paper/data_csvs/val.csv',
                        help='CSV file containing paths to real brain images')
    parser.add_argument('--output_dir', type=str,
                        default='./fid_results_ignite',
                        help='Directory to save FID results')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda, cuda:0, cuda:1, cpu, or auto)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for Inception inference')
    parser.add_argument('--max_files', type=int, default=477,
                        help='Maximum number of files to process')
    parser.add_argument('--volume_batch_size', type=int, default=16,
                        help='Number of volumes to load at once to manage memory')
    
    args = parser.parse_args()
    
    if args.device == 'auto':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    config = {
        'synthetic_folder': args.synthetic_folder,
        'synthetic_folder_2': args.synthetic_folder_2,
        'real_csv': args.real_csv,
        'device': device,
        'batch_size': args.batch_size,
        'max_files': args.max_files,
        'volume_batch_size': args.volume_batch_size
    }
    
    results = calculate_per_slice_fid_ignite(config)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save results
    np.save(os.path.join(args.output_dir, 'axis0_fids.npy'), results['axis0_fids'])
    np.save(os.path.join(args.output_dir, 'axis1_fids.npy'), results['axis1_fids'])
    np.save(os.path.join(args.output_dir, 'axis2_fids.npy'), results['axis2_fids'])
    
    for axis in ['axis0', 'axis1', 'axis2']:
        positions = np.array(sorted(results[f'{axis}_fids_by_position'].keys()))
        values = np.array([results[f'{axis}_fids_by_position'][k] for k in positions])
        np.savez(os.path.join(args.output_dir, f'{axis}_fids_by_position.npz'), 
                 positions=positions, values=values)
    
    with open(os.path.join(args.output_dir, 'fid_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    with open(os.path.join(args.output_dir, 'fid_summary.txt'), 'w') as f:
        f.write("=== Diffusion Paper FID Results Summary (PyTorch Ignite) ===\n")
        f.write(f"Synthetic images folder: {args.synthetic_folder}\n")
        f.write(f"Synthetic images folder 2: {args.synthetic_folder_2}\n")
        f.write(f"Real images CSV: {args.real_csv}\n")
        f.write(f"Max files processed: {args.max_files}\n")
        f.write(f"Device used: {device}\n\n")
        f.write(f"Average FID for axis 0 (sagittal): {results['avg_fid_axis0']:.4f}\n")
        f.write(f"Average FID for axis 1 (coronal): {results['avg_fid_axis1']:.4f}\n")
        f.write(f"Average FID for axis 2 (axial): {results['avg_fid_axis2']:.4f}\n")
        f.write(f"Overall average FID: {results['overall_avg_fid']:.4f}\n")
    
    print(f"\nResults saved to {args.output_dir}")
    
    # Generate plots
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(15, 5))
        
        axis_names = ['axis0', 'axis1', 'axis2']
        axis_titles = ['Axis 0 (Sagittal)', 'Axis 1 (Coronal)', 'Axis 2 (Axial)']
        colors = ['b', 'g', 'r']
        
        for i, (axis, title, color) in enumerate(zip(axis_names, axis_titles, colors)):
            plt.subplot(1, 3, i+1)
            positions = sorted(results[f'{axis}_fids_by_position'].keys())
            values = [results[f'{axis}_fids_by_position'][k] for k in positions]
            plt.plot(positions, values, f'{color}-o', linewidth=2, markersize=4)
            plt.title(f'{title} FIDs (Ignite)')
            plt.xlabel('Slice Position')
            plt.ylabel('FID Score')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(args.output_dir, 'per_slice_fids_diffusion_ignite.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {plot_path}")
        
    except ImportError:
        print("Matplotlib not available, skipping plot generation")


if __name__ == "__main__":
    main() 