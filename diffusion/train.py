#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist

import argparse
import os
import numpy as np
from copy import deepcopy
from collections import OrderedDict

# -------------
#  Your custom dataset for MRI latents
# -------------
from dataset import MRIDataset  # e.g. "from data.mri_dataset import MRIDataset"

# -------------
#  The 3D DiT with window-based attention:
# -------------
from model.dit3d_window_attn import DiT3D_models_WindAttn  # e.g. from your local .py
# Or import your custom dictionary if you named it differently

# -------------
#  Logging utilities
# -------------
import random

###############################################################################
#                                Utilities                                    #
###############################################################################

def set_seed(opt):
    """Set random seed for reproducibility."""
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed_all(opt.manualSeed)

def setup_logging(output_dir):
    """Simple logger that prints to screen and writes to file."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'log.txt')
    logger = open(log_file, 'a')

    class LoggerWrapper:
        def info(self, msg):
            print(msg)
            logger.write(msg + '\n')
            logger.flush()

        def close(self):
            logger.close()

    return LoggerWrapper()

def get_output_dir(base_dir, experiment_name):
    """Create the output directory for logs/checkpoints."""
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)
    return outdir

def copy_source(file_path, output_dir):
    """If needed, copy this source file into the output directory."""
    # For reproducibility. You can remove if not needed.
    import shutil
    if os.path.exists(file_path):
        shutil.copy(file_path, os.path.join(output_dir, os.path.basename(file_path)))

def requires_grad(model, flag=True):
    """Set requires_grad flag for all parameters in a model."""
    for p in model.parameters():
        p.requires_grad = flag

###############################################################################
#                  A Minimal Gaussian Diffusion Implementation                 #
###############################################################################

def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + (mean1 - mean2) ** 2 * torch.exp(-logvar2)
    )

class GaussianDiffusion:
    """
    Diffusion schedule and sampling/training logic.
    """

    def __init__(self, betas, loss_type, model_mean_type, model_var_type):
        """
        Args:
          betas: np.ndarray of length T (num_timesteps).
          loss_type: 'mse' or 'kl' ...
          model_mean_type: 'eps' => model predicts noise
          model_var_type: 'fixedsmall', 'fixedlarge', etc.  
        """
        self.loss_type = loss_type
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type

        self.np_betas = betas = betas.astype(np.float64)
        self.num_timesteps = len(betas)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        # Convert all these to torch tensors
        self.betas = torch.from_numpy(betas).float()
        self.alphas_cumprod = torch.from_numpy(alphas_cumprod).float()
        self.alphas_cumprod_prev = torch.from_numpy(alphas_cumprod_prev).float()

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # Posterior q(x_{t-1}|x_t, x_0)
        self.posterior_variance = (
            self.betas
            * (1.0 - self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.max(self.posterior_variance, torch.tensor(1e-20))
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(torch.from_numpy(alphas).float())
            / (1.0 - self.alphas_cumprod)
        )

    @staticmethod
    def _extract(a, t, x_shape):
        # move 'a' (e.g. self.betas) to the same device as 't'
        a = a.to(t.device)
        out = a.gather(dim=0, index=t)
        return out.reshape(len(t), *([1] * (len(x_shape) - 1)))

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse x_start for t steps. x_start: (B, C, D, H, W)
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self._extract(
            self.sqrt_alphas_cumprod, t, x_start.shape
        )
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Posterior mean and variance of q(x_{t-1}|x_t, x_0).
        """
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_var = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_var_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_var, posterior_log_var_clipped

    def p_mean_variance(self, denoise_fn, x_t, t, label, clip_denoised=True, return_pred_xstart=False):
        """
        Compute model_mean, model_variance, model_log_variance.
        denoise_fn is your model forward:  x_{t} -> x_{t-1} (or noise).
        """
        model_out = denoise_fn(x_t, t, label)

        if self.model_var_type in ['fixedsmall', 'fixedlarge']:
            if self.model_var_type == 'fixedlarge':
                model_variance = self.betas
                model_log_variance = torch.log(
                    torch.cat([self.posterior_variance[1:2], self.betas[1:]])
                )
            else:
                model_variance = self.posterior_variance
                model_log_variance = self.posterior_log_variance_clipped

            model_variance = self._extract(model_variance, t, x_t.shape)
            model_log_variance = self._extract(model_log_variance, t, x_t.shape)
        else:
            raise NotImplementedError("Only 'fixedsmall'/'fixedlarge' are supported in this example.")

        if self.model_mean_type == 'eps':
            # model_out is predicted noise
            pred_xstart = self._predict_xstart_from_eps(x_t, t, eps=model_out)
            if clip_denoised:
                pred_xstart.clamp_(-0.5, 0.5)
            model_mean, _, _ = self.q_posterior_mean_variance(pred_xstart, x_t, t)
        else:
            raise NotImplementedError("Only 'eps' is supported in this example.")

        if return_pred_xstart:
            return model_mean, model_variance, model_log_variance, pred_xstart
        else:
            return model_mean, model_variance, model_log_variance

    def _predict_xstart_from_eps(self, x_t, t, eps):
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    @torch.no_grad()
    def p_sample(self, denoise_fn, x_t, t, label):
        """
        Sample x_{t-1} given x_t.
        """
        model_mean, _, model_log_variance, _ = self.p_mean_variance(
            denoise_fn, x_t, t, label, clip_denoised=False, return_pred_xstart=True
        )
        # If t=0, skip noise
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1, *([1]*(len(x_t.shape)-1)))
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise

    @torch.no_grad()
    def p_sample_loop(self, denoise_fn, shape, device, label):
        """
        Produce samples by starting at x_T ~ N(0,I) and iteratively sampling.
        shape: (batch, C, D, H, W)
        """
        evolve = []
        x = torch.randn(shape, device=device)
        evolve.append(x.detach().cpu().numpy())
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample(denoise_fn, x, t, label)
            if i%100==0 or i==999 or i==970 or i==950:
                evolve.append(x.detach().cpu().numpy())
        return x, evolve

    def training_losses(self, denoise_fn, x_start, t, label, noise=None):
        """
        Compute training losses. We do the MSE on noise by default.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise=noise)
        model_out = denoise_fn(x_noisy, t, label)

        if self.loss_type == 'mse':
            # Model predicts noise, so MSE between predicted noise and true noise:
            loss = nn.MSELoss(reduction='none')(model_out, noise)
            return loss.mean(dim=list(range(1, len(loss.shape))))
        elif self.loss_type == 'l1':
            loss = nn.L1Loss(reduction='none')(model_out, noise)
            return loss.mean(dim=list(range(1, len(loss.shape))))
        else:
            raise NotImplementedError("Only 'mse' and 'l1' loss is shown here.")

###############################################################################
#                             The DiT + Diffusion Model                       #
###############################################################################

class Model(nn.Module):
    """
    Wrapper that ties together:
      - The DiT3D (window-attn) model
      - The diffusion schedule
      - The forward pass that predicts noise
    """

    def __init__(self, args, betas, loss_type, model_mean_type, model_var_type):
        super().__init__()
        self.diffusion = GaussianDiffusion(
            betas=betas,
            loss_type=loss_type,
            model_mean_type=model_mean_type,
            model_var_type=model_var_type
        )

        # We pick from the dictionary of window-attn 3D DiT models
        # 'DiT3D_models_WindAttn' is assumed to define multiple variants
        # e.g. "DiT-XL/2" or "DiT-B/4" etc.
        self.model = DiT3D_models_WindAttn[args.model_type](
            in_size=(28, 34, 28),     # Your MRI latent shape (D,H,W)
            in_channels=4,           # 4 channels
            num_classes=2,           # binary label => 2
            window_size=args.window_size,
            window_block_indexes=args.window_block_indexes,
        )

    def forward_diffusion_loss(self, z, label):
        """
        The standard forward pass for training:
          1) sample a random t
          2) compute noisy z_t
          3) compute MSE with predicted noise
        """
        B = z.shape[0]
        t = torch.randint(0, self.diffusion.num_timesteps, (B,), device=z.device)
        losses = self.diffusion.training_losses(self._denoise, z, t, label)
        return losses.mean()

    def _denoise(self, x_t, t, label):
        """
        The 'denoise_fn' passed into the diffusion logic: we apply the model
        to produce noise predictions. The model forward signature is:
           model(x, t, y) => returns predicted noise
        """
        # Our DiT expects input shape (B, C, D, H, W) for x,
        # as well as (B,) for t, (B,) for label
        return self.model(x_t, t, label)

    @torch.no_grad()
    def sample(self, batch_size, device):
        """
        Generate random samples from the model.
        """
        shape = (batch_size, 4, 28, 34, 28)  # (B, C, D, H, W)
        # For demonstration, let's pick random labels:
        rand_labels = torch.randint(0, 2, (batch_size,), device=device)
        samples = self.diffusion.p_sample_loop(
            denoise_fn=self._denoise,
            shape=shape,
            device=device,
            label=rand_labels
        )
        return samples

###############################################################################
#                          Training Loop & Utilities                          #
###############################################################################

def get_betas(schedule_type, b_start, b_end, time_num):
    """
    A simple function to create a beta schedule from b_start -> b_end.
    """
    if schedule_type == 'linear':
        betas = np.linspace(b_start, b_end, time_num, dtype=np.float64)
    elif schedule_type == 'cosine':
        # Cosine schedule parameters
        s = 0.008  # small offset as in Nichol & Dhariwal's paper
        steps = time_num + 1
        x = np.linspace(0, time_num, steps)
        # Compute cumulative alphas using a cosine schedule
        alphas_cumprod = np.cos((x / time_num + s) / (1 + s) * (np.pi / 2)) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        # Derive betas from cumulative alphas
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = np.clip(betas, 0, 0.999)
        betas = betas.astype(np.float64)
    else:
        raise NotImplementedError(f"Only 'linear' is implemented here. Got {schedule_type}.")
    return betas

def setup_dataset_and_loader(opt):
    """
    Create your MRIDataset and a DataLoader for it.
    Assume MRIDataset returns:
      - z: shape (4,28,34,28)
      - label: 0 or 1
    """
    # Example usage
    train_dataset = MRIDataset(opt.dataroot, split='train')
    # (You can define 'root' or any other param your MRIDataset needs.)

    # No test dataset is shown. Add if needed.
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.bs,
        shuffle=True,
        num_workers=opt.workers,
        drop_last=True,
    )
    return train_dataloader

def update_ema(ema_model, current_model, decay=0.9999):
    """
    Exponential moving average for model parameters.
    """
    ema_params = dict(ema_model.named_parameters())
    model_params = dict(current_model.named_parameters())
    for k in model_params.keys():
        ema_params[k].data.mul_(decay).add_(model_params[k].data, alpha=1 - decay)

def train_loop(gpu, opt, output_dir):
    set_seed(opt)
    logger = setup_logging(output_dir)
    logger.info("Starting training...")

    # If distributed is used:
    if opt.distribution_type == 'multi':
        # Setup multi-GPU with torch.distributed
        if opt.dist_url == "env://" and opt.rank == -1:
            opt.rank = int(os.environ["RANK"])
        base_rank = opt.rank * opt.ngpus_per_node
        opt.rank = base_rank + gpu
        dist.init_process_group(
            backend=opt.dist_backend,
            init_method=opt.dist_url,
            world_size=opt.world_size,
            rank=opt.rank,
        )
        torch.cuda.set_device(gpu)

    # Prepare dataset/loader
    train_dataloader = setup_dataset_and_loader(opt)

    # Create betas and build the diffusion+DiT model
    betas = get_betas(opt.schedule_type, opt.beta_start, opt.beta_end, opt.time_num)
    diffusion_model = Model(opt, betas, opt.loss_type, opt.model_mean_type, opt.model_var_type)

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    diffusion_model.to(device)

    # Optionally wrap in DDP
    if opt.distribution_type == 'multi':
        diffusion_model = nn.parallel.DistributedDataParallel(
            diffusion_model, device_ids=[gpu], output_device=gpu
        )
    elif opt.distribution_type == 'single':
        diffusion_model = nn.DataParallel(diffusion_model)

    # Setup optimizer
    optimizer = optim.Adam(diffusion_model.parameters(), lr=opt.lr)

    # Possibly an EMA model
    if opt.use_ema:
        ema_model = deepcopy(diffusion_model).to(device)
        requires_grad(ema_model, False)

    start_epoch = 0
    global_step = 0
    running_loss = 0.0  # Initialize running loss
    running_loss_steps = 0  # Initialize step counter for running loss
    best_epoch_loss = float('inf')

    # Main training loop
    print("trainloader length", len(train_dataloader))
    for epoch in range(start_epoch, opt.niter):
        epoch_loss =  0.0
        diffusion_model.train()
        for i, batch in enumerate(train_dataloader):
            # batch = (z, label)
            z, label = batch  # from MRIDataset
            # shape of z => (B,4,28,34,28), label => (B,)
            #print(z.size())

            z = z.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            # forward diffusion loss
            loss = diffusion_model.module.forward_diffusion_loss(z, label) \
                   if isinstance(diffusion_model, nn.DataParallel) or isinstance(diffusion_model, nn.parallel.DistributedDataParallel) \
                   else diffusion_model.forward_diffusion_loss(z, label)

            optimizer.zero_grad()
            loss.backward()
            if opt.grad_clip is not None:
                nn.utils.clip_grad_norm_(diffusion_model.parameters(), opt.grad_clip)
            optimizer.step()

            # Update EMA if enabled
            if opt.use_ema:
                with torch.no_grad():
                    if isinstance(diffusion_model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
                        update_ema(ema_model.module, diffusion_model.module)
                    else:
                        update_ema(ema_model, diffusion_model)

            # Print running loss every 50 steps
            #running_loss += loss.item()
            epoch_loss += loss.item()
            #running_loss_steps +=1
            # if global_step % 200 == 0:
            #     avg_running_loss = running_loss / running_loss_steps
            #     logger.info(f"Epoch [{epoch}/{opt.niter}] Iter [{i}/{len(train_dataloader)}] "
            #                 f"Global Step: {global_step}  Running Loss: {avg_running_loss:.4f}")
            #     running_loss = 0.0  # Reset running loss
            #     running_loss_steps = 0  # Reset step counter

            global_step += 1
        
        epoch_loss = epoch_loss/len(train_dataloader)
        print(f"Epoch Loss: {epoch_loss}")
        print(f"-----------------------------Epoch: {epoch} complete-----------------------------")
        if epoch_loss < best_epoch_loss:
            best_epoch_loss = epoch_loss
            if isinstance(diffusion_model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
                sd = diffusion_model.module.state_dict()
            else:
                sd = diffusion_model.state_dict()

            save_dict = {
                'epoch': epoch,
                'model_state': sd,
                'optimizer_state': optimizer.state_dict(),
            }
            if opt.use_ema:
                save_dict['ema_state'] = ema_model.state_dict()

            ckpt_path = os.path.join(output_dir, f"epoch_best.pth")
            torch.save(save_dict, ckpt_path)
            logger.info(f"Saved checkpoint to {ckpt_path}")    

        # End of epoch: Save checkpoint periodically
        if ((epoch+1) % opt.saveIter) == 0:
            if isinstance(diffusion_model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
                sd = diffusion_model.module.state_dict()
            else:
                sd = diffusion_model.state_dict()

            save_dict = {
                'epoch': epoch,
                'model_state': sd,
                'optimizer_state': optimizer.state_dict(),
            }
            if opt.use_ema:
                save_dict['ema_state'] = ema_model.state_dict()

            ckpt_path = os.path.join(output_dir, f"epoch_{epoch}.pth")
            torch.save(save_dict, ckpt_path)
            logger.info(f"Saved checkpoint to {ckpt_path}")

        # Example: generate some sample latents for monitoring
        # if ((epoch+1) % opt.sampleIter) == 0:
        #     diffusion_model.eval()
        #     sample_batch_size = 4
        #     with torch.no_grad():
        #         if isinstance(diffusion_model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        #             x_gen = diffusion_model.module.sample(sample_batch_size, device)
        #         else:
        #             x_gen = diffusion_model.sample(sample_batch_size, device)
        #     logger.info(f"Generated sample latents shape: {x_gen.shape}")

    logger.info("Training complete.")
    logger.close()

    if opt.distribution_type == 'multi':
        dist.destroy_process_group()

###############################################################################
#                                 Main & Argparse                              #
###############################################################################

def main():
    opt = parse_args()
    output_dir = get_output_dir(opt.model_dir, opt.experiment_name)
    copy_source(__file__, output_dir)

    # If using distributed
    if opt.distribution_type == 'multi':
        opt.ngpus_per_node = torch.cuda.device_count()
        opt.world_size = opt.ngpus_per_node * opt.world_size
        mp.spawn(train_loop, nprocs=opt.ngpus_per_node, args=(opt, output_dir))
    else:
        # single or no distributed
        train_loop(opt.gpu, opt, output_dir)

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='./checkpoints_2', help='path to save trained model weights')
    parser.add_argument('--experiment_name', type=str, default='dit3d', help='experiment name (used for checkpointing and logging)')

    # Data params
    parser.add_argument('--dataroot', default='/space/mcdonald-syn01/1/projects/jsawant/Diffusion_paper/diffusion/latent_data')
    parser.add_argument('--category', default='chair')
    parser.add_argument('--num_classes', type=int, default=2)

    parser.add_argument('--bs', type=int, default=4, help='input batch size')
    parser.add_argument('--workers', type=int, default=2, help='workers')
    parser.add_argument('--niter', type=int, default=10000, help='number of epochs to train for')

    # parser.add_argument('--nc', default=3)
    # parser.add_argument('--npoints', default=2048)
    
    '''model'''
    parser.add_argument("--model_type", type=str, choices=["DiT-S/4","DiT-B/4"], default="DiT-S/4")
    parser.add_argument('--beta_start', default=0.0001)
    parser.add_argument('--beta_end', default=0.02)
    parser.add_argument('--schedule_type', default='linear')
    parser.add_argument('--time_num', type=int, default=1000)

    #params
    parser.add_argument('--window_size', type=int, default=0)
    parser.add_argument('--window_block_indexes', type=tuple, default='')
    parser.add_argument('--attention', default=True)
    parser.add_argument('--dropout', default=0.1)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--loss_type', default='mse')
    parser.add_argument('--model_mean_type', default='eps')
    parser.add_argument('--model_var_type', default='fixedsmall')

    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate for E, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--decay', type=float, default=0, help='weight decay for EBM')
    parser.add_argument('--grad_clip', type=float, default=None, help='weight decay for EBM')
    parser.add_argument('--lr_gamma', type=float, default=0.998, help='lr decay for EBM')

    parser.add_argument('--model', default='', help="path to model (to continue training)")


    '''distributed'''
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed nodes.')
    parser.add_argument('--node', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=12345)
    parser.add_argument('--dist_url', type=str, default='tcp://localhost:12345')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--distribution_type', default='single', choices=['multi', 'single', None],
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use. None means using all available GPUs.')

    '''eval'''
    parser.add_argument('--saveIter', default=100, type=int, help='unit: epoch')
    parser.add_argument('--diagIter', default=50000, type=int, help='unit: epoch')
    parser.add_argument('--vizIter', default=50000, type=int, help='unit: epoch')
    parser.add_argument('--print_freq', default=10, type=int, help='unit: iter')

    parser.add_argument('--manualSeed', default=42, type=int, help='random seed')

    parser.add_argument('--debug', action='store_true', default=False, help = 'debug mode')
    parser.add_argument('--use_tb', action='store_true', default=False, help = 'use tensorboard')
    parser.add_argument('--use_pretrained', action='store_true', default=False, help = 'use pretrained 2d DiT weights')
    parser.add_argument('--use_ema', action='store_true', default=True, help = 'use ema')
    
    parser.add_argument('--sampleIter', type=int, default=10, help='Generate sample latents every X epochs')

    opt = parser.parse_args()

    return opt

if __name__ == '__main__':
    main()
