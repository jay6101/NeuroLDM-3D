import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import os
from PIL import Image
import numpy as np
from torchvision import transforms
import pandas as pd
import glob
import pickle

class MRIDataset(Dataset):
    def __init__(self, root_dir, split='train', train=True):
        # Read the CSV file using pandas
        paths = os.path.join(root_dir,"train/*.pkl")
        self.train_files = glob.glob(paths)
        self.train = train
        
        self.samples = []
        
        for file in self.train_files:
            with open(file, 'rb') as f:
                latent = pickle.load(f)# HC
            self.samples.append((file, latent['label']))
        
    def __len__(self):
        return len(self.samples)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def __getitem__(self, idx):
        latent_path, _ = self.samples[idx]
        
        with open(latent_path, 'rb') as f:
            latent = pickle.load(f)
        
        latent_z = self.reparameterize(torch.from_numpy(latent['mu']).float(),torch.from_numpy(latent['logvar']).float())
        label = torch.tensor(latent['label'])
        
        return latent_z, label

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
