#!/usr/bin/env python
"""Compare p95 vs p99 stress-test detection performance."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adsb_sde.reporting import print_summary_table

STRESS_SUMMARY_PATH = Path("outputs/sde_lstm/sde_stress_summary.csv")

CASES_ORDER = [
    "clean",
    "speed_scaled_1.5",
    "speed_scaled_2.0",
    "random_walk_velocity",
    "sudden_turn_90",
    "position_jump",
    "stationary_clutter",
]


def main() -> None:
    if not STRESS_SUMMARY_PATH.exists():
        print(
            f"Stress summary not found at {STRESS_SUMMARY_PATH}.\n"
            "Run run_stress_test_sde_lstm.py first (optionally at both p95 and p99)."
        )
        sys.exit(1)

    df = pd.read_csv(STRESS_SUMMARY_PATH)

    quantiles = sorted(df["quantile"].unique())
    score_names = df["score_name"].unique()

    for score_name in score_names:
        print(f"\n=== Score: {score_name} ===")
        sub = df[df["score_name"] == score_name].copy()

        pivot = sub.pivot_table(
            index="case",
            columns="quantile",
            values="detection_rate_at_threshold",
        )
        pivot = pivot.reindex([c for c in CASES_ORDER if c in pivot.index])
        pivot.columns = [f"detect@p{int(q * 100)}" for q in pivot.columns]
        pivot = pivot.reset_index()
        print_summary_table(pivot)


if __name__ == "__main__":
    main()
