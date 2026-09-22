"""Stage 3: generate synthetic latents from the trained LDM (DiT-S/4).

We sample directly in VAE-latent space (the memorization analysis is done
entirely in latent space, so no VAE decode is needed). Labels are class-matched
to the classifier test set (HC=0, epilepsy=1) so the real-vs-fake comparison is
fair per class and overall.

Loading/sampling mirrors diffusion/save_synth.py exactly:
  - get_betas('linear', 1e-4, 0.02, 1000)
  - Model(opt, betas, 'mse', 'eps', 'fixedsmall'), DiT-S/4, window_size=0
  - checkpoint['model_state'], strip 'module.', load_state_dict(strict=False)
  - diffusion.p_sample_loop(shape=(B,4,28,34,28), label=...)

Output: latents/synthetic/latents.npz with
  latents (N,4,28,34,28) float32  -> sampled z
  labels  (N,) int64

Run:
  /home/jsawant/.conda/envs/jay2/bin/python \
      /space/mcdonald-syn01/1/projects/jsawant/Diffusion_paper/diffusion/nn_eval/generate_fake_latents.py \
      --device cuda:0 --bs 16
"""
import os
import json
import argparse
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm

import paths  # type: ignore

import sys
sys.path.insert(0, paths.DIFF_DIR)  # so `train` + `model.*` resolve to the diffusion pkg
from train import Model, get_betas   # noqa: E402


def load_diffusion(ckpt_path, device):
    # checkpoint used NO windowed attention: empty tuple so `i in ()` is False
    # for every block (an empty string '' breaks `i in window_block_indexes`).
    opt = SimpleNamespace(model_type="DiT-S/4", window_size=0, window_block_indexes=())
    betas = get_betas("linear", 0.0001, 0.02, 1000)
    model = Model(opt, betas, "mse", "eps", "fixedsmall").to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt["model_state"]
    sd = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    print(f"Loaded LDM (epoch {ckpt.get('epoch')}) from {ckpt_path}")
    return model


@torch.no_grad()
def generate(model, n_per_class, device, bs, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    all_lat, all_lab = [], []
    for cls, n in n_per_class.items():
        done = 0
        pbar = tqdm(total=n, desc=f"sampling class {cls} ({paths.CLASS_NAMES[cls]})")
        while done < n:
            b = min(bs, n - done)
            label = torch.full((b,), cls, device=device, dtype=torch.long)
            x, _ = model.diffusion.p_sample_loop(
                denoise_fn=model._denoise,
                shape=(b, *paths.LATENT_SHAPE),
                device=device,
                label=label,
            )
            all_lat.append(x.detach().cpu().numpy().astype(np.float32))
            all_lab.append(np.full((b,), cls, dtype=np.int64))
            done += b
            pbar.update(b)
        pbar.close()
    return np.concatenate(all_lat, 0), np.concatenate(all_lab, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--bs", type=int, default=16, help="sampling batch size")
    ap.add_argument("--n_hc", type=int, default=paths.N_TEST_HC, help="num HC (0) fakes")
    ap.add_argument("--n_epd", type=int, default=paths.N_TEST_EPD, help="num epilepsy (1) fakes")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    os.makedirs(paths.SYNTH_DIR, exist_ok=True)
    print(f"Device: {device}")

    model = load_diffusion(paths.LDM_CKPT, device)
    n_per_class = {0: args.n_hc, 1: args.n_epd}
    latents, labels = generate(model, n_per_class, device, args.bs, args.seed)

    out = os.path.join(paths.SYNTH_DIR, paths.LATENTS_FILE)
    np.savez_compressed(out, latents=latents, labels=labels)
    dist = {int(c): int((labels == c).sum()) for c in np.unique(labels)}
    with open(os.path.join(paths.SYNTH_DIR, "meta.json"), "w") as f:
        json.dump({"ldm_ckpt": paths.LDM_CKPT, "n": int(len(labels)),
                   "label_counts": dist, "seed": args.seed,
                   "latent_shape": list(paths.LATENT_SHAPE)}, f, indent=2)
    print(f"Saved {latents.shape} latents -> {out}; label counts (0=HC,1=epd): {dist}")


if __name__ == "__main__":
    main()
