"""
Dataset for the cross-validation experiments (supports 3D and 2D backends).

The data-loading / preprocessing logic is kept identical to
Diffusion_paper/classifier/dataset.py (z-normalization, slice range 10:125,
resize 112x112 -> pad to 128x128, synthetic .pkl mixing, stratified
real subsampling via num_samples, synthetic top-up via num_synth_samples,
and the BalancedBatchSampler).

The output shape is selected with `volume_3d`:
  * volume_3d=True  -> [1, D, 128, 128]  single-channel volume for the 3D CNN
                        (cnn_ben_3d.MRIClassifier).
  * volume_3d=False -> [D, 128, 128]     slices-as-channels for the 2D
                        EfficientNetV2 (efficientNetV2.MRIClassifier), matching
                        the original Diffusion classifier dataset exactly.
"""

import torch
from torch.utils.data import Dataset, Sampler
import os
from PIL import Image
import numpy as np
from torchvision import transforms
import pandas as pd
import nibabel as nib
import glob
import pickle
import random


class MRIDataset(Dataset):
    def __init__(self, df, hparams, train=True, num_samples=None, num_synth_samples=None,
                 volume_3d=True):
        # Keep only the three classes we care about
        self.data = df.loc[df["HC_vs_LTLE_vs_RTLE_string"].isin(["right", "left", "HC"])]
        self.train = train
        # Output shape: True -> [1, D, H, W] (3D CNN); False -> [D, H, W] (2D CNN)
        self.volume_3d = volume_3d

        # Whether to include REAL volumes in the sample list:
        #   - val/test (train=False): always use ALL provided real rows
        #   - train (train=True): include real ONLY when num_samples is given, so
        #     synthetic-only runs (num_samples=None) contain NO real data.
        include_real = (not train) or (num_samples is not None)

        # Stratified subsample of the *real* data to `num_samples` rows
        if num_samples is not None:
            class_column = "HC_vs_LTLE_vs_RTLE_string"
            class_counts = self.data[class_column].value_counts()
            class_proportions = class_counts / len(self.data)

            stratified_samples = {}
            for class_name, proportion in class_proportions.items():
                class_sample_size = int(np.round(num_samples * proportion))
                class_data = self.data[self.data[class_column] == class_name]

                # Sample with replacement only if we need more than available
                replace = class_sample_size > len(class_data)
                stratified_samples[class_name] = class_data.sample(
                    n=class_sample_size, replace=replace, random_state=hparams['sample_seed']
                )

            self.data = pd.concat(stratified_samples.values()).reset_index(drop=True)

        # Base transforms applied to every slice (no random augmentation)
        self.base_transform = transforms.Compose([
            transforms.Resize((112, 112)),                 # resize in-plane to 112x112
            transforms.Pad(padding=(8, 8, 8, 8), fill=0),  # pad to 128x128
            transforms.ToTensor(),
        ])

        # Load synthetic latents (pkl) if requested
        self.synth_samples = []
        if num_synth_samples is not None:
            synth_files = glob.glob(os.path.join(hparams['synth_hc_folder_path'], "*.pkl"))
            synth_tle_files = glob.glob(os.path.join(hparams['synth_tle_folder_path'], "*.pkl"))

            synth_files = random.sample(synth_files, int(0.423 * num_synth_samples))
            synth_files.extend(random.sample(synth_tle_files, int(0.577 * num_synth_samples)))

            for file in synth_files:
                with open(file, 'rb') as f:
                    latent = pickle.load(f)
                self.synth_samples.append((file, latent['label']))

        # Build the (path, label) sample list
        def diag_to_label(diag):
            return 0 if diag == "HC" else 1

        if include_real:
            self.samples = [
                (row["file"], diag_to_label(row["HC_vs_LTLE_vs_RTLE_string"]))
                for _, row in self.data.iterrows()
            ]
        else:
            # synthetic-only run: do NOT load any real volumes
            self.samples = []

        if self.train and num_synth_samples is not None:
            self.samples.extend(self.synth_samples)
            np.random.shuffle(self.samples)

        self.SLICE_START = 10
        self.SLICE_END = 125

    def samples_dataframe(self):
        """DataFrame describing the exact samples used: columns [file, label, source]."""
        rows = [
            {'file': path, 'label': int(label),
             'source': 'synthetic' if '.pkl' in path else 'real'}
            for path, label in self.samples
        ]
        return pd.DataFrame(rows, columns=['file', 'label', 'source'])

    def __len__(self):
        return len(self.samples)

    def preprocess(self, image):
        # Crop synthetic latent images to the matching 115-slice depth
        image = image[:, 9:124, :]
        return image

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        if '.nii' in img_path:
            image = nib.load(img_path)
            image = image.get_fdata()
            image = (image - np.mean(image)) / np.std(image)    # z-norm
            image = image[:, self.SLICE_START:self.SLICE_END, :]  # [H, 115, W]
        elif '.pkl' in img_path:
            with open(img_path, 'rb') as f:
                latent = pickle.load(f)
            image = self.preprocess(latent['image'])

        slices = []
        for i in range(image.shape[1]):
            slice_img = Image.fromarray(image[:, i, :])
            slice_tensor = self.base_transform(slice_img)
            slices.append(slice_tensor)

        image = torch.stack(slices, dim=0)  # [D, 1, 128, 128]
        image = image.squeeze(1)            # [D, 128, 128]  (slices-as-channels, 2D CNN)
        if self.volume_3d:
            image = image.unsqueeze(0)      # [1, D, 128, 128]  single-channel volume (3D CNN)

        return image, label, img_path


class BalancedBatchSampler(Sampler):
    """Yields class-balanced batches (half positive, half negative)."""

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

        self.positive_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == 1]
        self.negative_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == 0]

        assert batch_size % 2 == 0, "Batch size must be even for balanced sampling"

        self.num_batches = max(len(self.positive_indices), len(self.negative_indices)) // (batch_size // 2)

    def __iter__(self):
        if len(self.positive_indices) > len(self.negative_indices):
            pos_indices = self.positive_indices.copy()
            np.random.shuffle(pos_indices)

            for i in range(self.num_batches):
                batch_indices = []
                start_idx = i * (self.batch_size // 2)
                end_idx = start_idx + (self.batch_size // 2)
                batch_pos_indices = pos_indices[start_idx:end_idx]

                batch_neg_indices = np.random.choice(
                    self.negative_indices, self.batch_size // 2, replace=True
                )
                batch_indices.extend(batch_pos_indices)
                batch_indices.extend(batch_neg_indices)
                np.random.shuffle(batch_indices)
                yield batch_indices
        else:
            neg_indices = self.negative_indices.copy()
            np.random.shuffle(neg_indices)

            for i in range(self.num_batches):
                batch_indices = []
                start_idx = i * (self.batch_size // 2)
                end_idx = start_idx + (self.batch_size // 2)
                batch_neg_indices = neg_indices[start_idx:end_idx]

                batch_pos_indices = np.random.choice(
                    self.positive_indices, self.batch_size // 2, replace=True
                )
                batch_indices.extend(batch_pos_indices)
                batch_indices.extend(batch_neg_indices)
                np.random.shuffle(batch_indices)
                yield batch_indices

    def __len__(self):
        return self.num_batches
