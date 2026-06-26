#!/usr/bin/env python
"""Calibrate and apply the stationary-clutter kinematic rule."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adsb_sde.config import ensure_dir, load_config
from adsb_sde.kinematic_rules import (
    apply_stationary_rule,
    calibrate_stationary_thresholds,
    save_thresholds,
)
from adsb_sde.reporting import print_summary_table, save_dataframe


def flag_rate(df: pd.DataFrame, col: str) -> float:
    return float(df[col].mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate and apply stationary-clutter rule")
    parser.add_argument("--config", default="configs/sde_lstm_default.yaml")
    args = parser.parse_args()

    from adsb_sde.config import load_config
    config = load_config(args.config)

    data_dir = Path(config["data_dir"])
    output_dir = ensure_dir(config["output_dir"])

    print("Loading data...")
    X_train = np.load(data_dir / config["train_file"], mmap_mode="r")
    X_test  = np.load(data_dir / config["test_file"],  mmap_mode="r")

    print(f"Calibrating stationary thresholds on {len(X_train)} train sequences...")
    thresholds = calibrate_stationary_thresholds(np.asarray(X_train, dtype=np.float32))
    save_thresholds(thresholds, output_dir / "stationary_thresholds.json")
    print("Thresholds:")
    for k, v in thresholds.items():
        print(f"  {k}: {v}")

    print("\nApplying to train...")
    train_feats = apply_stationary_rule(np.asarray(X_train, dtype=np.float32), thresholds)
    save_dataframe(train_feats, output_dir / "train_stationary_features.csv")

    print("Applying to test...")
    test_feats = apply_stationary_rule(np.asarray(X_test, dtype=np.float32), thresholds)
    save_dataframe(test_feats, output_dir / "test_stationary_features.csv")

    bool_cols = ["low_mean_speed", "low_displacement", "low_path_length",
                 "low_step_displacement", "stationary_flag"]

    summary = pd.DataFrame([
        {
            "split": "train",
            "stationary_flag_rate":        flag_rate(train_feats, "stationary_flag"),
            "low_mean_speed_rate":         flag_rate(train_feats, "low_mean_speed"),
            "low_displacement_rate":       flag_rate(train_feats, "low_displacement"),
            "low_path_length_rate":        flag_rate(train_feats, "low_path_length"),
            "low_step_displacement_rate":  flag_rate(train_feats, "low_step_displacement"),
        },
        {
            "split": "test",
            "stationary_flag_rate":        flag_rate(test_feats, "stationary_flag"),
            "low_mean_speed_rate":         flag_rate(test_feats, "low_mean_speed"),
            "low_displacement_rate":       flag_rate(test_feats, "low_displacement"),
            "low_path_length_rate":        flag_rate(test_feats, "low_path_length"),
            "low_step_displacement_rate":  flag_rate(test_feats, "low_step_displacement"),
        },
    ])
    save_dataframe(summary, output_dir / "stationary_rule_summary.csv")

    print("\nStationary rule summary:")
    print_summary_table(summary)
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
