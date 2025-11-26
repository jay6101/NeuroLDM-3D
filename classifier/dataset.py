import torch
from torch.utils.data import Dataset, DataLoader, Sampler
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
    def __init__(self, df, hparams, train=True, num_samples=None, num_synth_samples=None):
        # Read the CSV file using pandas
        #df = pd.read_csv(csv_file)
        self.data = df.loc[df["HC_vs_LTLE_vs_RTLE_string"].isin(["right","left","HC"])]
        
        # If num_samples is specified, create a stratified sample of the data
        if num_samples is not None:
            # Calculate the proportion of each class
            class_column = "HC_vs_LTLE_vs_RTLE_string"
            class_counts = self.data[class_column].value_counts()
            class_proportions = class_counts / len(self.data)
            
            # Create a stratified sample with the same proportions
            stratified_samples = {}
            for class_name, proportion in class_proportions.items():
                class_sample_size = int(np.round(num_samples * proportion))
                class_data = self.data[self.data[class_column] == class_name]
                
                # If we need more samples than available, use sampling with replacement
                replace = class_sample_size > len(class_data)
                stratified_samples[class_name] = class_data.sample(n=class_sample_size, replace=replace, random_state=hparams['sample_seed'])
            
            # Combine all stratified samples
            self.data = pd.concat(stratified_samples.values()).reset_index(drop=True)
        
        self.train = train
        
        # Define base transforms that are always applied
        self.base_transform = transforms.Compose([
            transforms.Resize((112, 112)),  # First resize to 113x113
            transforms.Pad(padding=(8, 8, 8, 8), fill=0),  # Add asymmetric padding to reach 128x128
            transforms.ToTensor(),
            #transforms.Lambda(lambda x: x.mul(2).sub(1))
        ])
        if num_synth_samples is not None:
            synth_files = glob.glob(os.path.join(hparams['synth_hc_folder_path'],"*.pkl"))
            synth_tle_files = glob.glob(os.path.join(hparams['synth_tle_folder_path'],"*.pkl"))
            
            synth_files = random.sample(synth_files, int(0.423*num_synth_samples))
            synth_files.extend(random.sample(synth_tle_files, int(0.577*num_synth_samples)))
            
            self.synth_samples = []
            for file in synth_files:
                with open(file, 'rb') as f:
                    latent = pickle.load(f)
                self.synth_samples.append((file, latent['label']))
        
        # Define additional transforms for training
        # Note: Removed from Compose to apply with same random state
        self.random_horizontal_flip = transforms.RandomHorizontalFlip(p=0.5)
        self.random_vertical_flip = transforms.RandomVerticalFlip(p=0.5)
        self.random_rotation = transforms.RandomRotation(degrees=15)
        
        # Create samples list from CSV data
        def diag_to_label(diag):
            return 0 if diag == "HC" else 1

        self.samples = [
            (row["file"], diag_to_label(row["HC_vs_LTLE_vs_RTLE_string"]))
            for _, row in self.data.iterrows()
        ]
        if self.train and num_synth_samples is not None:
            #self.samples = []
            self.samples.extend(self.synth_samples)
            np.random.shuffle(self.samples)
        
        self.SLICE_START = 10
        self.SLICE_END = 125
    
    def __len__(self):
        return len(self.samples)
    
    def preprocess(self,image):
        image = image[:,9:124,:]
        return image
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        if '.nii' in img_path:
            image = nib.load(img_path)
            image = image.get_fdata()
            image = (image - np.mean(image)) / np.std(image)    # z-norm
            # Get slices from 10 to 125 
            image = image[:, self.SLICE_START:self.SLICE_END, :]  # Shape: [H, 115, W]
        elif '.pkl' in img_path:
            with open(img_path, 'rb') as f:
                latent = pickle.load(f)
            image = self.preprocess(latent['image'])
        # Generate random states for transforms if in training mode
        if self.train:
            flip_h = torch.rand(1) < 0.5
            flip_v = torch.rand(1) < 0.5
            angle = torch.rand(1) * 30 - 15  # Random angle between -15 and 15 degrees
            channel_flip = torch.rand(1) < 0.5
            # if channel_flip:
            #     image = image[:,::-1,:]
        
        # Process all slices with the same random state
        slices = []
        for i in range(image.shape[1]):
            slice_img = Image.fromarray(image[:, i, :])
            # Apply base transforms
            slice_tensor = self.base_transform(slice_img)
            
            # Apply training transforms with same random state
            if self.train:
                if flip_h:
                    slice_tensor = transforms.functional.hflip(slice_tensor)
                if flip_v:
                    slice_tensor = transforms.functional.vflip(slice_tensor)
                slice_tensor = transforms.functional.rotate(slice_tensor, float(angle))
            slices.append(slice_tensor)
        
        # Stack all slices
        image = torch.stack(slices, dim=0)  # Shape: [115, 1, 128, 128]
        
        return image.squeeze(1), label, img_path

class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        
        # Get indices for positive and negative samples
        self.positive_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == 1]
        self.negative_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == 0]
        
        # Ensure batch_size is even
        assert batch_size % 2 == 0, "Batch size must be even for balanced sampling"
        
        # Calculate number of batches per epoch based on negative samples
        # Each negative sample should appear exactly once
        self.num_batches = max(len(self.positive_indices),len(self.negative_indices)) // (batch_size//2)
        
    def __iter__(self):
        # Shuffle negative indices once per epoch
        
        if len(self.positive_indices) > len(self.negative_indices):
            pos_indices = self.positive_indices.copy()
            np.random.shuffle(pos_indices)
            
            # Create batches
            for i in range(self.num_batches):
                batch_indices = []
                
                # Get negative indices for this batch (without replacement)
                start_idx = i * (self.batch_size // 2)
                end_idx = start_idx + (self.batch_size // 2)
                batch_pos_indices = pos_indices[start_idx:end_idx]

                batch_neg_indices = np.random.choice(
                    self.negative_indices, 
                    self.batch_size//2, 
                    replace=True
                )
                batch_indices.extend(batch_pos_indices)
                batch_indices.extend(batch_neg_indices)
                np.random.shuffle(batch_indices)
                
                yield batch_indices
        else:
            neg_indices = self.negative_indices.copy()
            np.random.shuffle(neg_indices)
            
            # Create batches
            for i in range(self.num_batches):
                batch_indices = []
                
                # Get negative indices for this batch (without replacement)
                start_idx = i * (self.batch_size // 2)
                end_idx = start_idx + (self.batch_size // 2)
                batch_neg_indices = neg_indices[start_idx:end_idx]

                batch_pos_indices = np.random.choice(
                    self.positive_indices, 
                    self.batch_size//2, 
                    replace=True)   
                batch_indices.extend(batch_pos_indices)
                batch_indices.extend(batch_neg_indices)
                np.random.shuffle(batch_indices)
            
                yield batch_indices
            
    def __len__(self):
        return self.num_batches
