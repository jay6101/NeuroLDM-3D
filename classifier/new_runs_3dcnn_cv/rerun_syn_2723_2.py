"""Re-run runs_syn/syn_2723 with 3D CNN, new init -> runs/runs_syn/syn_2723_2/"""
import argparse
import os

import torch

import config
from train import train_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--random-seed", type=int, default=123,
                   help="model init seed (original syn_2723 used 42)")
    args = p.parse_args()

    hp = config.build_base_hyperparams("3d")
    hp.update({
        "device": torch.device(args.device if torch.cuda.is_available() else "cpu"),
        "num_samples": None,
        "num_synth_samples": 2723,
        "variant_name": "syn_2723_2",
        "group": "runs_syn",
        "run_dir": os.path.join(config.default_runs_dir("3d"), "runs_syn", "syn_2723_2"),
        "random_seed": args.random_seed,
    })
    train_model(hp, hp["model_name"])


if __name__ == "__main__":
    main()
