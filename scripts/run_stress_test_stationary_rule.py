#!/usr/bin/env python
"""Stress-test the stationary-clutter kinematic rule across all corruptions."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adsb_sde.config import ensure_dir, load_config
from adsb_sde.corruption import (
    position_jump,
    random_walk_velocity,
    speed_scale,
    stationary_clutter,
    sudden_turn_90,
)
from adsb_sde.kinematic_rules import (
    apply_stationary_rule,
    calibrate_stationary_thresholds,
    load_thresholds,
    save_thresholds,
)
from adsb_sde.reporting import print_summary_table, save_dataframe

CORRUPTIONS = {
    "speed_scaled_1.5":   lambda x: speed_scale(x, factor=1.5),
    "speed_scaled_2.0":   lambda x: speed_scale(x, factor=2.0),
    "random_walk_velocity": lambda x: random_walk_velocity(x),
    "sudden_turn_90":     lambda x: sudden_turn_90(x),
    "position_jump":      lambda x: position_jump(x),
    "stationary_clutter": lambda x: stationary_clutter(x),
}


def evaluate_case(X: np.ndarray, thresholds: dict, name: str) -> dict:
    feats = apply_stationary_rule(X, thresholds)
    return {
        "case":                       name,
        "stationary_detection_rate":  float(feats["stationary_flag"].mean()),
        "low_mean_speed_rate":        float(feats["low_mean_speed"].mean()),
        "low_displacement_rate":      float(feats["low_displacement"].mean()),
        "low_path_length_rate":       float(feats["low_path_length"].mean()),
        "low_step_displacement_rate": float(feats["low_step_displacement"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stress-test stationary-clutter rule")
    parser.add_argument("--config", default="configs/sde_lstm_default.yaml")
    parser.add_argument("--max-samples", type=int, default=50000)
    args = parser.parse_args()

    config = load_config(args.config)
    data_dir = Path(config["data_dir"])
    output_dir = ensure_dir(config["output_dir"])

    X_train = np.load(data_dir / config["train_file"], mmap_mode="r")
    X_test  = np.load(data_dir / config["test_file"],  mmap_mode="r")

    thresholds_path = output_dir / "stationary_thresholds.json"
    if thresholds_path.exists():
        print(f"Loading thresholds from {thresholds_path}")
        thresholds = load_thresholds(thresholds_path)
    else:
        print(f"Calibrating thresholds on {len(X_train)} train sequences...")
        thresholds = calibrate_stationary_thresholds(
            np.asarray(X_train, dtype=np.float32)
        )
        save_thresholds(thresholds, thresholds_path)
        print(f"Thresholds saved to {thresholds_path}")

    n = min(args.max_samples, len(X_test))
    X_sub = np.asarray(X_test[:n], dtype=np.float32)

    print(f"\nEvaluating on {n} test sequences...")
    results = [evaluate_case(X_sub, thresholds, "clean")]

    for name, fn in CORRUPTIONS.items():
        print(f"  {name}...")
        X_corrupt = fn(X_sub)
        results.append(evaluate_case(X_corrupt, thresholds, name))

    summary = pd.DataFrame(results)
    save_dataframe(summary, output_dir / "stationary_rule_stress_summary.csv")

    print("\nStationary rule stress-test summary:")
    print_summary_table(summary)
    print(f"\nSaved to {output_dir / 'stationary_rule_stress_summary.csv'}")


if __name__ == "__main__":
    main()
