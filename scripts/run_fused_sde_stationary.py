#!/usr/bin/env python
"""Fused SDE + stationary-clutter rule evaluation across all corruptions."""
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
from adsb_sde.inference import load_checkpoint, score_sequences
from adsb_sde.kinematic_rules import apply_stationary_rule, load_thresholds
from adsb_sde.reporting import print_summary_table, save_dataframe
from adsb_sde.utils import get_device

CORRUPTIONS = {
    "speed_scaled_1.5":     lambda x: speed_scale(x, factor=1.5),
    "speed_scaled_2.0":     lambda x: speed_scale(x, factor=2.0),
    "random_walk_velocity": lambda x: random_walk_velocity(x),
    "sudden_turn_90":       lambda x: sudden_turn_90(x),
    "position_jump":        lambda x: position_jump(x),
    "stationary_clutter":   lambda x: stationary_clutter(x),
}

CASES_ORDER = [
    "clean",
    "speed_scaled_1.5",
    "speed_scaled_2.0",
    "random_walk_velocity",
    "sudden_turn_90",
    "position_jump",
    "stationary_clutter",
]


def evaluate_case(
    X: np.ndarray,
    model,
    sde_threshold: float,
    score_name: str,
    stat_thresholds: dict,
    batch_size: int,
    device,
    case_name: str,
    quantile: float,
) -> dict:
    sde_scores = score_sequences(model, X, batch_size, device)
    sde_flag = sde_scores[score_name].values > sde_threshold

    stat_feats = apply_stationary_rule(X, stat_thresholds)
    stat_flag = stat_feats["stationary_flag"].values

    fused_flag = sde_flag | stat_flag

    sde_rate   = float(sde_flag.mean())
    stat_rate  = float(stat_flag.mean())
    fused_rate = float(fused_flag.mean())

    return {
        "case":                    case_name,
        "score_name":              score_name,
        "quantile":                quantile,
        "sde_threshold":           sde_threshold,
        "sde_detection_rate":      sde_rate,
        "stationary_detection_rate": stat_rate,
        "fused_detection_rate":    fused_rate,
        "stationary_adds_pp":      fused_rate - sde_rate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fused SDE + stationary rule evaluation")
    parser.add_argument("--config",      default="configs/sde_lstm_default.yaml")
    parser.add_argument("--checkpoint",  default="outputs/sde_lstm/sde_lstm.pt")
    parser.add_argument("--score-name",  default="total_nll")
    parser.add_argument("--quantile",    type=float, default=0.99)
    parser.add_argument("--max-samples", type=int,   default=50000)
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()
    output_dir = ensure_dir(config["output_dir"])

    print(f"Loading SDE checkpoint: {args.checkpoint}")
    model, _ = load_checkpoint(args.checkpoint, device)

    thresholds_path = output_dir / "sde_thresholds.csv"
    if not thresholds_path.exists():
        print(f"SDE thresholds not found at {thresholds_path}. Run run_score_sde_lstm.py first.")
        sys.exit(1)
    thresh_df = pd.read_csv(thresholds_path)
    row = thresh_df[
        (thresh_df["score_name"] == args.score_name)
        & (thresh_df["quantile"] == args.quantile)
    ]
    if row.empty:
        print(f"No threshold for score_name={args.score_name!r} quantile={args.quantile}")
        sys.exit(1)
    sde_threshold = float(row["threshold"].iloc[0])
    print(f"SDE threshold ({args.score_name} p{int(args.quantile*100)}): {sde_threshold:.4f}")

    stat_thresholds_path = output_dir / "stationary_thresholds.json"
    if not stat_thresholds_path.exists():
        print(f"Stationary thresholds not found at {stat_thresholds_path}. Run run_stationary_rule.py first.")
        sys.exit(1)
    stat_thresholds = load_thresholds(stat_thresholds_path)

    data_dir = Path(config["data_dir"])
    X_test = np.load(data_dir / config["test_file"], mmap_mode="r")
    n = min(args.max_samples, len(X_test))
    X_sub = np.asarray(X_test[:n], dtype=np.float32)
    batch_size = config["batch_size"]

    results = []
    for case_name in CASES_ORDER:
        print(f"Evaluating {case_name}...")
        if case_name == "clean":
            X_case = X_sub
        else:
            X_case = CORRUPTIONS[case_name](X_sub)
        results.append(
            evaluate_case(
                X_case, model, sde_threshold, args.score_name,
                stat_thresholds, batch_size, device, case_name, args.quantile,
            )
        )

    summary = pd.DataFrame(results)
    out_path = output_dir / "fused_sde_stationary_summary.csv"
    save_dataframe(summary, out_path)

    print(f"\n{'='*72}")
    print(f"FUSED SDE + STATIONARY RULE  |  {args.score_name}  p{int(args.quantile*100)}")
    print(f"{'='*72}")
    display = summary[[
        "case", "sde_detection_rate", "stationary_detection_rate",
        "fused_detection_rate", "stationary_adds_pp",
    ]].copy()
    display.columns = ["case", "SDE p99", "stationary", "fused", "+pp"]
    print_summary_table(display)
    print(f"{'='*72}")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
