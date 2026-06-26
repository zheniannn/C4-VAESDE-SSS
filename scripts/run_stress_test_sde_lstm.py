#!/usr/bin/env python
"""Stress-test the SDE-LSTM with synthetic corruptions."""
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
from adsb_sde.reporting import print_summary_table, save_dataframe
from adsb_sde.utils import get_device


CORRUPTIONS = {
    "speed_scaled_1.5": lambda x: speed_scale(x, factor=1.5),
    "speed_scaled_2.0": lambda x: speed_scale(x, factor=2.0),
    "random_walk_velocity": lambda x: random_walk_velocity(x),
    "sudden_turn_90": lambda x: sudden_turn_90(x),
    "position_jump": lambda x: position_jump(x),
    "stationary_clutter": lambda x: stationary_clutter(x),
}


def score_case(
    model,
    X: np.ndarray,
    batch_size: int,
    device,
    score_name: str,
    threshold: float,
    case_name: str,
    quantile: float,
) -> dict:
    scores = score_sequences(model, X, batch_size, device)
    vals = scores[score_name].values
    return {
        "case": case_name,
        "score_name": score_name,
        "quantile": quantile,
        "threshold": threshold,
        "mean_score": float(vals.mean()),
        "p95_score": float(np.percentile(vals, 95)),
        "detection_rate_at_threshold": float((vals > threshold).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stress-test SDE-LSTM")
    parser.add_argument("--config", default="configs/sde_lstm_default.yaml")
    parser.add_argument("--checkpoint", default="outputs/sde_lstm/sde_lstm.pt")
    parser.add_argument("--max-samples", type=int, default=50000)
    parser.add_argument("--score-name", default="total_nll")
    parser.add_argument("--quantile", type=float, default=0.99)
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()

    print(f"Loading checkpoint: {args.checkpoint}")
    model, ckpt_config = load_checkpoint(args.checkpoint, device)

    thresholds_path = Path(config["output_dir"]) / "sde_thresholds.csv"
    if not thresholds_path.exists():
        print(
            f"Thresholds file not found at {thresholds_path}.\n"
            "Please run run_score_sde_lstm.py first."
        )
        sys.exit(1)

    thresholds_df = pd.read_csv(thresholds_path)
    row = thresholds_df[
        (thresholds_df["score_name"] == args.score_name)
        & (thresholds_df["quantile"] == args.quantile)
    ]
    if row.empty:
        print(
            f"No threshold found for score_name={args.score_name!r} "
            f"quantile={args.quantile}. Check sde_thresholds.csv."
        )
        sys.exit(1)
    threshold = float(row["threshold"].iloc[0])
    print(
        f"Using threshold {threshold:.4f} "
        f"({args.score_name} p{int(args.quantile * 100)} from train)"
    )

    data_dir = Path(config["data_dir"])
    X_test = np.load(data_dir / config["test_file"], mmap_mode="r")
    n = min(args.max_samples, len(X_test))
    X_test_sub = np.array(X_test[:n], dtype=np.float32)

    batch_size = config["batch_size"]
    results: list[dict] = []

    print(f"\nScoring clean ({n} sequences)...")
    results.append(
        score_case(model, X_test_sub, batch_size, device, args.score_name, threshold, "clean", args.quantile)
    )

    for name, fn in CORRUPTIONS.items():
        print(f"Scoring {name}...")
        X_corrupt = fn(X_test_sub)
        results.append(
            score_case(model, X_corrupt, batch_size, device, args.score_name, threshold, name, args.quantile)
        )

    summary = pd.DataFrame(results)
    output_dir = ensure_dir(config["output_dir"])
    out_path = output_dir / "sde_stress_summary.csv"

    # Merge with any existing results for other (score_name, quantile) pairs
    # so that run_compare_thresholds.py can pivot across quantiles.
    if out_path.exists():
        existing = pd.read_csv(out_path)
        existing = existing[
            ~(
                (existing["score_name"] == args.score_name)
                & (existing["quantile"] == args.quantile)
            )
        ]
        summary = pd.concat([existing, summary], ignore_index=True)

    save_dataframe(summary, out_path)

    print("\nStress-test summary (this run):")
    print_summary_table(pd.DataFrame(results))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
