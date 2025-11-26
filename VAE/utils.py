import torch
import torch.nn.functional as F
import random, os, numpy as np

def set_random_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def kl_divergence(mu, logvar):
    # KL( q(z|x) || p(z)=N(0,I) )
    kld = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1, dim=[1,2,3,4])
    return kld.mean()

def compute_recon_loss(x_recon, images, loss_type):
    """Helper function to compute reconstruction losses based on specified type"""
    if loss_type == 'l1':
        return F.l1_loss(x_recon, images)
    elif loss_type == 'mse':
        return F.mse_loss(x_recon, images)
    else:
        raise ValueError(f"Unsupported reconstruction loss type: {loss_type}")

def train_one_epoch(vae, lpips_model, train_loader, opt_vae, hparams, current_epoch):
    vae.train()
    device = hparams['device']

    epoch_running_loss = 0.0
    epoch_running_recon = 0.0
    epoch_running_kl = 0.0
    epoch_running_lpips = 0.0
    
    running_loss = 0.0
    running_recon = 0.0
    running_kl = 0.0
    running_lpips = 0.0
    
    for batch_idx, (images, labels, paths) in enumerate(train_loader):
        images = images.float().to(device)

        # Update VAE
        opt_vae.zero_grad()
        x_recon, mu, logvar = vae(images)

        # Compute losses
        recon_loss = compute_recon_loss(x_recon, images, hparams['recon_loss_type'])
        kld = 0.0
        if mu is not None and logvar is not None:
            kld = kl_divergence(mu, logvar)

        with torch.no_grad():
            lpips_model.eval()
        lpips_val = lpips_model(x_recon, images)
        if lpips_val.dim() > 0:
            lpips_val = lpips_val.mean()
        
        # Weighted sum
        total_loss = (hparams['lambda_recon'] * recon_loss
                     + hparams['lambda_kl'] * kld
                     + hparams['lambda_lpips'] * lpips_val)
        
        total_loss.backward()
        opt_vae.step()

        # Update running averages for both 10-batch and epoch-level metrics
        epoch_running_loss += total_loss.item()
        epoch_running_recon += recon_loss.item()
        epoch_running_kl += kld
        epoch_running_lpips += lpips_val.item()

        # Update running averages for 10-batch metrics
        running_loss += total_loss.item()
        running_recon += recon_loss.item()
        running_kl += kld
        running_lpips += lpips_val.item()

        # Print and reset running averages every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(f"Train Batch [{batch_idx + 1}/{len(train_loader)}] Average of last 10 - "
                  f"Total: {running_loss/10:.4f}, "
                  f"Recon: {running_recon/10:.4f}, "
                  f"KL: {running_kl/10:.4f}, "
                  f"LPIPS: {running_lpips/10:.4f}")
            
            # Reset running losses
            running_loss = 0.0
            running_recon = 0.0
            running_kl = 0.0
            running_lpips = 0.0

    # Calculate epoch averages using the actual number of batches
    n_batches = len(train_loader)
    avg_recon = epoch_running_recon / n_batches
    avg_kl = epoch_running_kl / n_batches
    avg_lpips = epoch_running_lpips / n_batches
    
    print(f"\nTrain Epoch Averages:")
    print(f"Total Loss: {epoch_running_loss / n_batches:.4f}")
    print(f"Individual Losses - Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}, "
          f"LPIPS: {avg_lpips:.4f}")
    
    return epoch_running_loss / n_batches

def validate_one_epoch(vae, lpips_model, val_loader, hparams):
    vae.eval()
    device = hparams['device']
    
    running_loss = 0.0
    running_recon = 0.0
    running_kl = 0.0
    running_lpips = 0.0
    
    with torch.no_grad():
        for images, labels, paths in val_loader:
            images = images.float().to(device)
            
            # Generate reconstructions
            x_recon, mu, logvar = vae(images)
            
            # Compute losses
            recon_loss = compute_recon_loss(x_recon, images, hparams['recon_loss_type'])
            kld = 0.0
            if mu is not None and logvar is not None:
                kld = kl_divergence(mu, logvar)

            lpips_val = lpips_model(x_recon, images)
            if lpips_val.dim() > 0:
                lpips_val = lpips_val.mean()
            
            total_loss = (hparams['lambda_recon'] * recon_loss
                         + hparams['lambda_kl'] * kld
                         + hparams['lambda_lpips'] * lpips_val)
                
            running_loss += total_loss.item()
            running_recon += recon_loss.item()
            running_kl += kld
            running_lpips += lpips_val.item()
    
    n_batches = len(val_loader)
    avg_recon = running_recon / n_batches
    avg_kl = running_kl / n_batches
    avg_lpips = running_lpips / n_batches
    
    print(f"\nValidation Epoch Averages:")
    print(f"Total Loss: {running_loss / n_batches:.4f}")
    print(f"Individual Losses - Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}, "
          f"LPIPS: {avg_lpips:.4f}")
    
    return running_loss / n_batches
