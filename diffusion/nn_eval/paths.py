"""Central configuration for the nearest-neighbor memorization pipeline.

All scripts import from here so paths/constants live in one place.

Pipeline (run in order):
  1. build_csvs.py            -> data_csvs/{train,val,test}.csv
  2. encode_real.py --split train   -> latents/real_train/latents.npz
     encode_real.py --split test    -> latents/real_test/latents.npz
  3. generate_fake_latents.py       -> latents/synthetic/latents.npz
  4. nn_memorization.py             -> results/
"""
import os

# --- project roots ---
PROJECT = "/space/mcdonald-syn01/1/projects/jsawant/Diffusion_paper"
DIFF_DIR = os.path.join(PROJECT, "diffusion")
VAE_DIR = os.path.join(PROJECT, "VAE")

# --- checkpoints ---
VAE_RUN = os.path.join(VAE_DIR, "best_runs", "vae_run_20250226_161525")
VAE_CKPT = os.path.join(VAE_RUN, "best_vae.pth")          # the VAE actually in use
VAE_HPARAMS = os.path.join(VAE_RUN, "hparams.json")
LDM_CKPT = os.path.join(DIFF_DIR, "checkpoints_2", "dit3d", "epoch_9999.pth")

# --- source CSVs ---
DATA_CSV_DIR = os.path.join(PROJECT, "data_csvs")
PICKLE_PREP = os.path.join(DATA_CSV_DIR, "pickle_prep_2025_01_17no_duplicates.csv")
TEST_PERF_CSV = os.path.join(
    PROJECT, "classifier", "new_runs", "runs_500", "real_500_syn_50",
    "fold_1", "individual_test_performance.csv",
)

# Existing latents are used ONLY to recover the train/val split (filenames),
# never their stored values. We re-encode everything fresh.
LATENT_DATA = os.path.join(DIFF_DIR, "latent_data")
LATENT_TRAIN_DIR = os.path.join(LATENT_DATA, "train")
LATENT_VAL_DIR = os.path.join(LATENT_DATA, "val")

# --- sub-CSVs we build ---
TRAIN_CSV = os.path.join(DATA_CSV_DIR, "train.csv")
VAL_CSV = os.path.join(DATA_CSV_DIR, "val.csv")
TEST_CSV = os.path.join(DATA_CSV_DIR, "test.csv")

# --- outputs ---
NN_DIR = os.path.join(DIFF_DIR, "nn_eval")
LATENTS_DIR = os.path.join(NN_DIR, "latents")
REAL_TRAIN_DIR = os.path.join(LATENTS_DIR, "real_train")
REAL_TEST_DIR = os.path.join(LATENTS_DIR, "real_test")
SYNTH_DIR = os.path.join(LATENTS_DIR, "synthetic")
RESULTS_DIR = os.path.join(NN_DIR, "results")
LATENTS_FILE = "latents.npz"  # filename inside each latents/<set>/ folder

# Map a split name -> (source csv, output latent dir)
SPLITS = {
    "train": (TRAIN_CSV, REAL_TRAIN_DIR),
    "val": (VAL_CSV, REAL_TEST_DIR.replace("real_test", "real_val")),
    "test": (TEST_CSV, REAL_TEST_DIR),
}

# --- model / latent geometry (fixed by the trained models) ---
LATENT_SHAPE = (4, 28, 34, 28)        # C, D, H, W  (latent dim = 106,624)
VAE_INPUT_SHAPE = (1, 112, 136, 112)  # C, D, H, W
NUM_CLASSES = 2
CLASS_NAMES = {0: "HC", 1: "epilepsy"}

# Test-set class counts (used as defaults for class-matched fake generation)
N_TEST_HC = 128
N_TEST_EPD = 175
