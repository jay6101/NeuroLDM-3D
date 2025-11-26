import os
import torch
import shutil
import json
from torch.utils.data import DataLoader
from dataset import MRIDataset, BalancedBatchSampler
from model.maisi_vae import VAE_Lite
from model.lpips3D import LPIPSLoss3D
from utils import (
    set_random_seeds,
    train_one_epoch,
    validate_one_epoch
)
from datetime import datetime

def train_vae_gan(hparams):
    # 1. Set random seeds
    #set_random_seeds(hparams['random_seed'])
    
    # 2. Create run directory for logs/checkpoints
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #run_dir = os.path.join(hparams['runs_dir'], f'vae_gan_run_{timestamp}')
    run_dir = "/space/mcdonald-syn01/1/projects/jsawant/DSC250/VAE_GAN/gan"
    #os.makedirs(run_dir, exist_ok=True)
    
    # Save hyperparameters
    hparams_save = hparams.copy()
    hparams_save['device'] = str(hparams['device'])  # Convert torch.device to string
    with open(os.path.join(run_dir, 'hparams.json'), 'w') as f:
        json.dump(hparams_save, f, indent=4)
    
    # Initialize dictionary to store losses
    losses = {
        'train_losses': [],
        'val_losses': [],
        'best_epoch': None,
        'best_losses': {
            'train': None,
            'val': None
        }
    }
    
    # 3. Optionally copy source files (for reproducibility)
    source_files = ['train.py', 'utils.py', 'dataset.py', 'model/vae.py', 'model/enc_dec.py', 'model/resnet_blocks.py']
    for file in source_files:
        src_path = os.path.join(os.path.dirname(__file__), file)
        dst_path = os.path.join(run_dir, os.path.basename(file))
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
    
    print(f"Using device: {hparams['device']}")
    
    # 4. Datasets & Dataloaders (train/val only)
    train_dataset = MRIDataset(hparams["train_csv_file"], train=True)
    val_dataset   = MRIDataset(hparams["val_csv_file"],   train=False)
    
    train_sampler = BalancedBatchSampler(train_dataset, hparams['batch_size'])
    train_loader  = DataLoader(train_dataset, batch_size=hparams['batch_size'],shuffle=True,#batch_sampler=train_sampler,
                               num_workers=hparams['num_workers'], pin_memory=True)
    val_loader    = DataLoader(val_dataset, batch_size=hparams['batch_size'],
                               shuffle=False, num_workers=hparams['num_workers'], pin_memory=True)
    
    # 5. Initialize VAE (generator), Discriminator, LPIPS model
    # vae = VAE(use_reparam=True).to(hparams['device'])
    vae = VAE_Lite(
    spatial_dims=3,           # 3D model
    in_channels=1,            # e.g., single-channel input
    out_channels=1,           # single-channel reconstruction
    channels=(32, 64, 128),   # downsampling channels
    num_res_blocks=(1, 1, 1),    # one ResBlock per level
    attention_levels=(False, False, False),
    latent_channels=4,
    norm_num_groups=16,
    norm_eps=1e-5,
    with_encoder_nonlocal_attn=False,
    with_decoder_nonlocal_attn=False,
    include_fc=False,
    use_combined_linear=False,
    use_flash_attention=False,
    use_convtranspose=False,
    num_splits=8,
    dim_split=0,
    norm_float16=False,
    print_info=False,
    save_mem=True,
    ).to(hparams['device'])
    
    checkpoint = torch.load(hparams['vae_checkpoint'], 
                          map_location=hparams['device'])
    vae.load_state_dict(checkpoint['vae_state_dict'])
    del checkpoint
    
    lpips_model = LPIPSLoss3D(hparams['lpips_model']).to(hparams['device'])
    # Freeze LPIPS model weights (assuming it's already loaded in model.py)
    lpips_model.eval()
    for p in lpips_model.parameters():
        p.requires_grad = False
    
    # 6. Optimizer
    optimizer_vae = torch.optim.Adam(
        vae.parameters(),
        lr=hparams['vae_lr'],
        weight_decay=hparams['weight_decay']
    )
    
    # Set up scheduler
    scheduler_vae = None
    if hparams.get('use_cosine_scheduler', False):
        scheduler_vae = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer_vae,
            T_0=hparams['cosine_t0'],
            T_mult=hparams['cosine_t_mult'],
            eta_min=hparams['cosine_eta_min']
        )
    
    # 7. Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(hparams['num_epochs']):
        # --- Train ---
        vae.train()
        print(f"--------------- Epoch {epoch} -------------------")
        train_loss = train_one_epoch(
            vae, lpips_model,
            train_loader, optimizer_vae,
            hparams, epoch
        )
        
        # --- Validate ---
        vae.eval()
        val_loss = validate_one_epoch(
            vae, lpips_model,
            val_loader, hparams
        )
        
        print(f"[Epoch {epoch+1}] "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Store losses
        losses['train_losses'].append(float(train_loss))
        losses['val_losses'].append(float(val_loss))
        
        # Check if this is the best model so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Update best losses
            losses['best_epoch'] = epoch + 1
            losses['best_losses']['train'] = float(train_loss)
            losses['best_losses']['val'] = float(val_loss)
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'vae_state_dict': vae.state_dict(),
                'val_loss': val_loss
            }, os.path.join(run_dir, 'best_vae_gan.pth'))
            print(f"** Best model saved at epoch {epoch+1}")
        else:
            patience_counter += 1
            # Early stopping if desired
            if hparams.get('use_early_stopping', False) and \
               patience_counter >= hparams['early_stopping_patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
        # Save losses after each epoch
        with open(os.path.join(run_dir, 'losses.json'), 'w') as f:
            json.dump(losses, f, indent=4)
        
        # (Optional) step scheduler
        if scheduler_vae is not None:
            scheduler_vae.step()

    print("Training complete.")


if __name__ == "__main__":
    hparams = {
        'train_csv_file': "/space/mcdonald-syn01/1/projects/jsawant/DSC250/data_csvs/train.csv",
        'val_csv_file':   "/space/mcdonald-syn01/1/projects/jsawant/DSC250/data_csvs/val.csv",
        'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        'batch_size': 2,
        'disc_lr': 1e-6,
        'vae_lr': 1e-4,
        'weight_decay': 0,
        'num_epochs': 100,
        'runs_dir': '/space/mcdonald-syn01/1/projects/jsawant/DSC250/VAE_GAN/runs',
        'random_seed': 42,
        'use_early_stopping': True,
        'early_stopping_patience': 35,
        'lambda_adv': 0.04,
        'lambda_lpips': 0.6,
        'lambda_kl': 1e-6,
        'lambda_recon': 1.0,
        'lambda_disc': 0, #1e-6,
        'num_workers': 4,
        'warmup_epochs': 1,
        'gan_loss_type': 'lsgan',  # bce, lsgan, hinge
        'recon_loss_type': 'l1',    # l1, mse
        'use_cosine_scheduler': True,  # Whether to use cosine annealing scheduler
        'cosine_t0': 5,  # Number of epochs for first restart
        'cosine_t_mult': 2,  # Multiply T_0 by this factor after each restart
        'cosine_eta_min': 5e-6,  # Minimum learning rate
        'use_discriminator': False,  # New parameter to control discriminator usage
        'lpips_model': 'alex',
        'disc_checkpoint': '/space/mcdonald-syn01/1/projects/jsawant/DSC250/VAE_GAN/gan/best_disc.pth',
        'vae_checkpoint': '/space/mcdonald-syn01/1/projects/jsawant/DSC250/VAE_GAN/gan/best_vae_gan.pth'
    }
    train_vae_gan(hparams)
