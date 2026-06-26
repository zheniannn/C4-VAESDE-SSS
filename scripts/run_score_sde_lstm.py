#!/usr/bin/env python
"""Score train/test sequences and compute detection thresholds."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adsb_sde.config import ensure_dir, load_config
from adsb_sde.inference import load_checkpoint, score_sequences
from adsb_sde.reporting import print_summary_table, save_dataframe
from adsb_sde.utils import get_device


THRESHOLD_SCORE_COLUMNS = ["total_nll", "mahalanobis", "max_step_nll", "final_step_nll"]
QUANTILES = [0.90, 0.95, 0.99]


def compute_thresholds(
    train_scores: pd.DataFrame,
    test_scores: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for col in THRESHOLD_SCORE_COLUMNS:
        if col not in train_scores.columns:
            continue
        for q in QUANTILES:
            threshold = train_scores[col].quantile(q)
            test_flag_rate = (test_scores[col] > threshold).mean()
            rows.append({
                "score_name": col,
                "quantile": q,
                "threshold": threshold,
                "test_flag_rate": test_flag_rate,
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score SDE-LSTM")
    parser.add_argument("--config", default="configs/sde_lstm_default.yaml")
    parser.add_argument("--checkpoint", default="outputs/sde_lstm/sde_lstm.pt")
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()

    print(f"Loading checkpoint: {args.checkpoint}")
    model, ckpt_config = load_checkpoint(args.checkpoint, device)

    data_dir = Path(config["data_dir"])
    X_train = np.load(data_dir / config["train_file"], mmap_mode="r")
    X_test = np.load(data_dir / config["test_file"], mmap_mode="r")

    output_dir = ensure_dir(config["output_dir"])
    batch_size = config["batch_size"]

    print(f"Scoring train sequences ({len(X_train)})...")
    train_scores = score_sequences(model, X_train, batch_size, device)

    print(f"Scoring test sequences ({len(X_test)})...")
    test_scores = score_sequences(model, X_test, batch_size, device)

    save_dataframe(train_scores, output_dir / "train_sde_scores.csv")
    save_dataframe(test_scores, output_dir / "test_sde_scores.csv")
    print(f"Scores saved to {output_dir}")

    thresholds = compute_thresholds(train_scores, test_scores)
    save_dataframe(thresholds, output_dir / "sde_thresholds.csv")

    print("\nThresholds:")
    print_summary_table(thresholds)


if __name__ == "__main__":
    main()
