#!/usr/bin/env python
"""Train the SDE-style probabilistic LSTM transition model."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adsb_sde.config import load_config
from adsb_sde.training import fit_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SDE-LSTM")
    parser.add_argument(
        "--config",
        default="configs/sde_lstm_default.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    fit_model(config)


if __name__ == "__main__":
    main()
