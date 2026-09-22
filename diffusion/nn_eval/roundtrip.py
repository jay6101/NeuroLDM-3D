"""Stage 3b: 'roundtrip' latents through the VAE => encode(decode(latent)).

Why: the diffusion outputs z (which carries the VAE posterior-noise floor), so
fakes must be decoded+re-encoded to a clean mu before they are comparable to
the real/train mu. But that decode+encode pass itself nudges a latent toward
the VAE manifold. To keep the comparison fair, we apply the SAME decode+encode
to the reals too. After running this on all sets, every set is represented as
encode(decode(latent)) -> stored under key `latents_rt`.

  fakes:  encode(decode(z))    -- also removes the diffusion/VAE noise
  reals:  encode(decode(mu))   -- same operation, for symmetry

Reuses already-saved latents (no re-sampling / no re-encoding from images).
Writes key `latents_rt` into each set's latents.npz (other keys preserved).

Run:
  /home/jsawant/.conda/envs/jay2/bin/python \
      /space/mcdonald-syn01/1/projects/jsawant/Diffusion_paper/diffusion/nn_eval/roundtrip.py \
      --set all --device cuda:0
"""
import os
import argparse

import numpy as np
import torch
from tqdm import tqdm

import paths

import sys
sys.path.insert(0, paths.DIFF_DIR)  # diffusion's copy of the VAE
from model.maisi_vae import VAE_Lite  # noqa: E402

SETS = {"train": paths.REAL_TRAIN_DIR, "test": paths.REAL_TEST_DIR, "fakes": paths.SYNTH_DIR}


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
def roundtrip(vae, lat, device, bs):
    out = []
    for i in tqdm(range(0, len(lat), bs), desc="decode->encode"):
        x = torch.from_numpy(np.ascontiguousarray(lat[i:i + bs])).to(device)
        img = vae.decode(x)        # latent -> image
        mu, _ = vae.encode(img)    # image -> clean mu
        out.append(mu.cpu().numpy().astype(np.float32))
    return np.concatenate(out, 0)


def process(vae, name, set_dir, device, bs):
    npz = os.path.join(set_dir, paths.LATENTS_FILE)
    d = dict(np.load(npz, allow_pickle=True))
    d.pop("latents_mu", None)  # drop deprecated key if present
    rt = roundtrip(vae, d["latents"].astype(np.float32), device, bs)
    d["latents_rt"] = rt
    np.savez_compressed(npz, **d)
    print(f"[{name}] wrote latents_rt {rt.shape} -> {npz}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=list(SETS) + ["all"], default="all")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--bs", type=int, default=8)
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    vae = load_vae(device)
    todo = SETS if args.set == "all" else {args.set: SETS[args.set]}
    for name, set_dir in todo.items():
        process(vae, name, set_dir, device, args.bs)


if __name__ == "__main__":
    main()
