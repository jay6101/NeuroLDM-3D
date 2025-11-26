import os
import torch
import json
import numpy as np
from dataset import MRIDataset
from model.maisi_vae import VAE_Lite
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
import argparse
from model.efficientNetV2 import MRIClassifier
import random
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from train import Model, get_betas
import pickle

def load_models(run_folder, device):
    """Load the trained VAE model."""
    # Load hyperparameters
    with open(os.path.join(run_folder, 'hparams.json'), 'r') as f:
        hparams = json.load(f)
    hparams['device'] = device
    
    # Initialize models
    #vae = VAE(use_reparam=True).to(device)
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
    #disc = Discriminator().to(device)
    
    # Load checkpoint
    checkpoint = torch.load(os.path.join(run_folder, 'best_vae_gan.pth'), 
                          map_location=device)
    
    vae.load_state_dict(checkpoint['vae_state_dict'])
    #disc.load_state_dict(checkpoint['disc_state_dict'])
    
    return vae, hparams

def load_diffusion(ckpt_path, opt, device):
    betas = get_betas('linear', 0.0001, 0.02, 1000)
    diffusion_model = Model(opt, betas, opt.loss_type, opt.model_mean_type, opt.model_var_type)
    # diffusion_model = nn.DataParallel(diffusion_model)
    diffusion_model.to(device)
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["model_state"]
    
    new_state_dict = {}
    for k, v in state_dict.items():
        # Remove "module." so that it matches a non-DataParallel model
        if k.startswith("module."):
            new_key = k.replace("module.", "")
        else:
            new_key = k
        new_state_dict[new_key] = v

    diffusion_model.load_state_dict(new_state_dict, strict=False)

    diffusion_model.eval()
    
    print(f"Epoch loaded: {checkpoint["epoch"]}")
    
    return diffusion_model

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='./checkpoints_2', help='path to save trained model weights')
    parser.add_argument('--experiment_name', type=str, default='dit3d', help='experiment name (used for checkpointing and logging)')

    # Data params
    parser.add_argument('--dataroot', default='/space/mcdonald-syn01/1/projects/jsawant/DSC250/diffusion/latent_data')
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
    parser.add_argument('--schedule_type', default='cosine')
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

    opt = parser.parse_args([])

    return opt


def generate_and_save(diffusion_model, vae, num_samples, label, device):
    for i in tqdm(range(num_samples)):
        custom_label = torch.tensor([label], device=device)  # shape (batch_size,)
        samples, _ = diffusion_model.diffusion.p_sample_loop(
            denoise_fn=diffusion_model._denoise,
            shape=(8, 4, 28, 34, 28),
            device=device,
            label=custom_label
        )
        with torch.no_grad():
            out = vae.decode(samples)

        out = out.cpu().numpy()
        for j in range(len(out)):
            pick = {'image' : out[j][0]}
            pick['label'] = custom_label.item()

            with open(f"/space/mcdonald-syn01/1/projects/jsawant/Diffusion_paper/synthetic_data_pkls_HC/{i}_{pick['label']}.pkl", 'wb') as file:
                pickle.dump(pick, file)
    return

vae_run_folder = "/space/mcdonald-syn01/1/projects/jsawant/Diffusion_paper/VAE/best_runs/vae_run_20250226_161525"
diffusion_model_path = "/space/mcdonald-syn01/1/projects/jsawant/Diffusion_paper/diffusion/checkpoints_2/dit3d/epoch_9999.pth"
device = 'cuda:0'
opt = get_args()

vae, hparams = load_models(vae_run_folder, device)
vae.eval()
diffusion_model = load_diffusion(diffusion_model_path, opt, device)

label = 0
num_samples = 2000
generate_and_save(diffusion_model, vae, num_samples, label, device)