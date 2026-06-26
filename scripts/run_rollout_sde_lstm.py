#!/usr/bin/env python
"""Generate and visualise stochastic rollouts from the SDE-LSTM."""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adsb_sde.config import ensure_dir, load_config
from adsb_sde.inference import load_checkpoint
from adsb_sde.rollout import rollout_batch
from adsb_sde.utils import get_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SDE-LSTM rollouts")
    parser.add_argument("--config", default="configs/sde_lstm_default.yaml")
    parser.add_argument("--checkpoint", default="outputs/sde_lstm/sde_lstm.pt")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=30)
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()

    print(f"Loading checkpoint: {args.checkpoint}")
    model, ckpt_config = load_checkpoint(args.checkpoint, device)

    data_dir = Path(config["data_dir"])
    X_test = np.load(data_dir / config["test_file"], mmap_mode="r")

    n = min(args.num_samples, len(X_test))
    initial_windows = np.array(X_test[:n], dtype=np.float32)

    output_dir = ensure_dir(config["output_dir"])

    print(f"Generating deterministic rollouts ({n} samples, {args.rollout_steps} steps)...")
    det_rollouts = rollout_batch(model, initial_windows, args.rollout_steps, device, deterministic=True)

    print(f"Generating stochastic rollouts ({n} samples, {args.rollout_steps} steps)...")
    sto_rollouts = rollout_batch(model, initial_windows, args.rollout_steps, device, deterministic=False)

    np.save(output_dir / "rollout_deterministic.npy", det_rollouts)
    np.save(output_dir / "rollout_stochastic.npy", sto_rollouts)
    print(f"Rollout arrays saved to {output_dir}")

    # Plot EN trajectories
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for i in range(n):
        ax = axes[i]
        context_E = initial_windows[i, :, 0]
        context_N = initial_windows[i, :, 1]
        det_E = det_rollouts[i, :, 0]
        det_N = det_rollouts[i, :, 1]
        sto_E = sto_rollouts[i, :, 0]
        sto_N = sto_rollouts[i, :, 1]

        ax.plot(context_E, context_N, "k-o", markersize=3, label="context", linewidth=1)
        ax.plot(det_E, det_N, "b--", label="deterministic", linewidth=1)
        ax.plot(sto_E, sto_N, "r:", alpha=0.7, label="stochastic", linewidth=1)
        ax.set_title(f"Seq {i}", fontsize=9)
        ax.set_xlabel("E (norm)", fontsize=8)
        ax.set_ylabel("N (norm)", fontsize=8)
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("SDE-LSTM Rollout Examples", fontsize=11)
    plt.tight_layout()
    plot_path = output_dir / "rollout_examples.png"
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
