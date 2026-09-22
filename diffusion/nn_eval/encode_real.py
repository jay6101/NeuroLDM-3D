"""Stage 2: VAE-encode real images from a split CSV into latents.

Uses the SAME recipe as VAE/save_latent.py:
  - VAE_Lite + best_vae.pth
  - VAE/dataset.py:MRIDataset(csv, train=False)  (deterministic, no flips)
  - posterior mean `mu` is the latent we store (plus logvar for diagnostics)

Output: <out_dir>/latents.npz with keys
  latents (N,4,28,34,28) float32  -> mu
  logvars (N,4,28,34,28) float16
  labels  (N,) int64              -> 0=HC, 1=epilepsy
  stems   (N,) str

Run (encode the reference DB and the test set):
  PY=/home/jsawant/.conda/envs/jay2/bin/python
  $PY .../nn_eval/encode_real.py --split train --device cuda:0
  $PY .../nn_eval/encode_real.py --split test  --device cuda:0
"""
import os
import json
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import paths

import sys
sys.path.insert(0, paths.VAE_DIR)  # so `dataset` and `model.maisi_vae` resolve to the VAE pkg
from dataset import MRIDataset            # noqa: E402  (VAE/dataset.py)
from model.maisi_vae import VAE_Lite      # noqa: E402


def load_vae(device):
    vae = VAE_Lite(
        spatial_dims=3, in_channels=1, out_channels=1,
        channels=(32, 64, 128), num_res_blocks=(1, 1, 1),
        attention_levels=(False, False, False), latent_channels=4,
        norm_num_groups=16, norm_eps=1e-5,
        with_encoder_nonlocal_attn=False, with_decoder_nonlocal_attn=False,
        include_fc=False, use_combined_linear=False, use_flash_attention=False,
        use_convtranspose=False, num_splits=8, dim_split=0,
        norm_float16=False, print_info=False, save_mem=True,
    ).to(device)
    ckpt = torch.load(paths.VAE_CKPT, map_location=device)
    vae.load_state_dict(ckpt["vae_state_dict"])
    vae.eval()
    print(f"Loaded VAE from {paths.VAE_CKPT}")
    return vae


@torch.no_grad()
def encode_csv(vae, csv_file, device, bs, workers):
    ds = MRIDataset(csv_file, train=False)
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers, pin_memory=True)
    print(f"Encoding {len(ds)} images from {csv_file}")

    mus, logvars, labels, stems = [], [], [], []
    for images, label, path in tqdm(loader, desc="encoding"):
        images = images.float().to(device)
        mu, logvar = vae.encode(images)  # deterministic posterior mean
        mus.append(mu.cpu().numpy().astype(np.float32))
        logvars.append(logvar.cpu().numpy().astype(np.float16))
        labels.append(label.cpu().numpy().astype(np.int64))
        stems.extend(os.path.basename(p).split(".")[0] for p in path)
    return (
        np.concatenate(mus, 0),
        np.concatenate(logvars, 0),
        np.concatenate(labels, 0),
        np.asarray(stems),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(paths.SPLITS.keys()), required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    csv_file, out_dir = paths.SPLITS[args.split]
    os.makedirs(out_dir, exist_ok=True)
    print(f"Device: {device} | split={args.split} | out={out_dir}")

    vae = load_vae(device)
    mus, logvars, labels, stems = encode_csv(vae, csv_file, device, args.bs, args.workers)

    out = os.path.join(out_dir, paths.LATENTS_FILE)
    np.savez_compressed(out, latents=mus, logvars=logvars, labels=labels, stems=stems)
    dist = {int(c): int((labels == c).sum()) for c in np.unique(labels)}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"split": args.split, "csv": csv_file, "vae_ckpt": paths.VAE_CKPT,
                   "n": int(len(labels)), "label_counts": dist,
                   "latent_shape": list(paths.LATENT_SHAPE)}, f, indent=2)
    print(f"Saved {mus.shape} latents -> {out}; label counts (0=HC,1=epd): {dist}")


if __name__ == "__main__":
    main()
