from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)


def save_json(obj: object, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2)


def print_summary_table(df: pd.DataFrame) -> None:
    with pd.option_context("display.max_columns", None, "display.width", 120):
        print(df.to_string(index=False))
