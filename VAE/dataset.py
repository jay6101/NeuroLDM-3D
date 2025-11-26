import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import os
from PIL import Image
import numpy as np
from torchvision import transforms
import pandas as pd
import nibabel as nib

class MRIDataset(Dataset):
    def __init__(self, csv_file, train=True):
        # Read the CSV file using pandas
        df = pd.read_csv(csv_file)
        self.data = df.loc[df["HC_vs_LTLE_vs_RTLE_string"].isin(["right","left","HC"])]
        self.data = self.data[~self.data['file'].str.contains("/space/mcdonald-syn01/1/BIDS//enigma_conglom//derivatives//cat12_copy/sub-upenn", na=False)]
        self.train = train
        
        # Define base transforms that are always applied
        self.base_transform = transforms.Compose([
            transforms.Resize((112, 112)),  # First resize to 113x113
            #transforms.Pad(padding=(7, 7, 8, 8), fill=0),  # Add asymmetric padding to reach 128x128
            transforms.ToTensor(),
            #transforms.Lambda(lambda x: x.mul(2).sub(1))
        ])
        
        # Define additional transforms for training
        # Note: Removed from Compose to apply with same random state
        # self.random_horizontal_flip = transforms.RandomHorizontalFlip(p=0.5)
        # self.random_vertical_flip = transforms.RandomVerticalFlip(p=0.5)
        # self.random_rotation = transforms.RandomRotation(degrees=15)
        
        # Create samples list from CSV data
        def diag_to_label(diag):
            return 0 if diag == "HC" else 1

        self.samples = [
            (row["file"], diag_to_label(row["HC_vs_LTLE_vs_RTLE_string"]))
            for _, row in self.data.iterrows()
        ]
        self.SLICE_START = 1
        self.SLICE_END = 137
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        image = nib.load(img_path)
        image = image.get_fdata()
        #image = (image - np.min(image)) / (np.max(image) - np.min(image))   # z-norm
        image = (image - np.mean(image)) / np.std(image)  
        #image = image*2 - 1
        # Get slices from 10 to 125 
        image = image[:, self.SLICE_START:self.SLICE_END, :]  # Shape: [H, 115, W]
        
        # Generate random states for transforms if in training mode
        if self.train:
            flip_h = torch.rand(1) < 0.5
            flip_v = torch.rand(1) < 0.5
            # angle = torch.rand(1) * 30 - 15  # Random angle between -15 and 15 degrees
            # channel_flip = torch.rand(1) < 0.5
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
                #slice_tensor = transforms.functional.rotate(slice_tensor, float(0))
            slices.append(slice_tensor)
        
        # Stack all slices
        image = torch.stack(slices, dim=0)  # Shape: [115, 1, 128, 128]
        image = image.squeeze(1)
        image = image.permute(1,0,2)
        image = image.unsqueeze(0)
        return image, label, img_path

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
