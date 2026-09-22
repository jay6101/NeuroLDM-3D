"""Shared pieces for VAE reconstruction evaluation: model loading, data, metrics."""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

VAE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if VAE_DIR not in sys.path:
    sys.path.insert(0, VAE_DIR)

from dataset import MRIDataset  # noqa: E402
from model.maisi_vae import VAE_Lite  # noqa: E402

RUN_FOLDER = os.path.join(VAE_DIR, "best_runs", "vae_run_20250226_161525")
CSV_DIR = os.path.join(os.path.dirname(VAE_DIR), "data_csvs")


class MultiCSVMRIDataset(MRIDataset):
    """MRIDataset over the union of several CSVs, de-duplicated on the image path.

    val.csv is a strict superset of test.csv in this project, so a naive
    concatenation would count 303 subjects twice.
    """

    def __init__(self, csv_files, train=False):
        if isinstance(csv_files, str):
            csv_files = [csv_files]
        df = pd.concat([pd.read_csv(c) for c in csv_files], ignore_index=True)
        df = df.drop_duplicates(subset="file", keep="first").reset_index(drop=True)
        fd, tmp = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        try:
            df.to_csv(tmp, index=False)
            super().__init__(tmp, train=train)
        finally:
            os.remove(tmp)


def load_vae(run_folder=RUN_FOLDER, device="cuda:0"):
    """Instantiate VAE_Lite with the architecture used at training time and load weights."""
    vae = VAE_Lite(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(32, 64, 128),
        num_res_blocks=(1, 1, 1),
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
    ).to(device)
    ckpt = torch.load(os.path.join(run_folder, "best_vae.pth"), map_location=device, weights_only=False)
    vae.load_state_dict(ckpt["vae_state_dict"])
    vae.eval()
    return vae


@torch.no_grad()
def reconstruct(vae, images, stochastic=False):
    """Reconstruct a batch. Deterministic mode decodes the posterior mean instead of a sample."""
    mean, logvar = vae.encode(images)
    z = vae.reparameterize(mean, logvar) if stochastic else mean
    return vae.decode(z), mean, logvar


class LPIPS3D(nn.Module):
    """Slice-wise LPIPS averaged over all slices of all three axes.

    Numerically equivalent to VAE/model/lpips3D.py but evaluates each axis as a
    batch of slices instead of one slice at a time.
    """

    def __init__(self, net="alex", chunk=64):
        super().__init__()
        import lpips

        self.lpips_fn = lpips.LPIPS(net=net, verbose=False)
        self.chunk = chunk

    def _sum_over_slices(self, s1, s2, n_slices, batch):
        """s1/s2: [batch*n_slices, C, h, w] -> per-sample sum of slice LPIPS, shape [batch]."""
        out = []
        for i in range(0, s1.shape[0], self.chunk):
            out.append(self.lpips_fn(s1[i : i + self.chunk], s2[i : i + self.chunk]).flatten())
        return torch.cat(out).view(batch, n_slices).sum(dim=1)

    def forward(self, v1, v2):
        """v1/v2: [B, C, D, H, W] -> [B] LPIPS values."""
        b, c, d, h, w = v1.shape
        total = self._sum_over_slices(
            v1.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w),
            v2.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w),
            d, b,
        )
        total = total + self._sum_over_slices(
            v1.permute(0, 4, 1, 2, 3).reshape(b * w, c, d, h),
            v2.permute(0, 4, 1, 2, 3).reshape(b * w, c, d, h),
            w, b,
        )
        total = total + self._sum_over_slices(
            v1.permute(0, 3, 1, 2, 4).reshape(b * h, c, d, w),
            v2.permute(0, 3, 1, 2, 4).reshape(b * h, c, d, w),
            h, b,
        )
        return total / (d + w + h)


def to_unit_range(real, recon):
    """Rescale both volumes to [-1, 1] using the real volume's range (LPIPS input convention)."""
    flat = real.flatten(1)
    lo = flat.min(dim=1).values.view(-1, 1, 1, 1, 1)
    hi = flat.max(dim=1).values.view(-1, 1, 1, 1, 1)
    scale = (hi - lo).clamp_min(1e-8)
    return (real - lo) / scale * 2 - 1, ((recon - lo) / scale * 2 - 1).clamp(-1, 1)


def voxel_metrics(real, recon):
    """Per-sample MSE, MAE and PSNR over the model-input (z-scored) intensity scale."""
    diff = (recon - real).flatten(1)
    mse = diff.pow(2).mean(dim=1)
    mae = diff.abs().mean(dim=1)
    flat = real.flatten(1)
    data_range = flat.max(dim=1).values - flat.min(dim=1).values
    psnr = 10 * torch.log10(data_range.pow(2) / mse.clamp_min(1e-12))
    return mse, mae, psnr


def subject_id_from_path(path):
    base = os.path.basename(path)
    for token in base.split("_"):
        if token.startswith("sub-"):
            return token
        if token.startswith("bsub-"):
            return token[1:]
    return base.split(".")[0]


def preprocessed_affine(orig_affine, orig_shape, out_shape, slice_start):
    """Affine mapping preprocessed voxel indices back to the original scanner space.

    The preprocessing resizes axis 0 and axis 2 to `out_shape` and crops axis 1 at
    `slice_start`. Resampling uses the half-pixel convention of torchvision Resize.
    """
    sx = orig_shape[0] / out_shape[0]
    sz = orig_shape[2] / out_shape[2]
    m = np.array(
        [
            [sx, 0.0, 0.0, 0.5 * sx - 0.5],
            [0.0, 1.0, 0.0, float(slice_start)],
            [0.0, 0.0, sz, 0.5 * sz - 0.5],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    return orig_affine @ m
