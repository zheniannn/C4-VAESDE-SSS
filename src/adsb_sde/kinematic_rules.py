from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def compute_kinematic_features(X: np.ndarray) -> pd.DataFrame:
    # X: (N, 30, 4) — [E, N, vE, vN] normalised
    N = X.shape[0]

    vE = X[:, :, 2]  # (N, 30)
    vN = X[:, :, 3]  # (N, 30)
    E  = X[:, :, 0]  # (N, 30)
    Npos = X[:, :, 1]

    speed = np.sqrt(vE ** 2 + vN ** 2)  # (N, 30)
    mean_speed = speed.mean(axis=1)
    max_speed  = speed.max(axis=1)
    min_speed  = speed.min(axis=1)

    dE = np.diff(E, axis=1)      # (N, 29)
    dN = np.diff(Npos, axis=1)
    step_disp = np.sqrt(dE ** 2 + dN ** 2)   # (N, 29)
    path_length = step_disp.sum(axis=1)
    mean_step_disp = step_disp.mean(axis=1)
    max_step_disp  = step_disp.max(axis=1)

    start_end_disp = np.sqrt(
        (E[:, -1] - E[:, 0]) ** 2 + (Npos[:, -1] - Npos[:, 0]) ** 2
    )
    disp_to_path = start_end_disp / (path_length + 1e-8)

    return pd.DataFrame({
        "sequence_index":               np.arange(N),
        "mean_speed_norm":              mean_speed,
        "max_speed_norm":               max_speed,
        "min_speed_norm":               min_speed,
        "start_end_displacement_norm":  start_end_disp,
        "path_length_norm":             path_length,
        "displacement_to_path_ratio":   disp_to_path,
        "mean_step_displacement_norm":  mean_step_disp,
        "max_step_displacement_norm":   max_step_disp,
    })


def calibrate_stationary_thresholds(
    train_X: np.ndarray,
    quantile: float = 0.01,
) -> dict:
    feats = compute_kinematic_features(train_X)
    return {
        "mean_speed_p01":              float(feats["mean_speed_norm"].quantile(quantile)),
        "start_end_displacement_p01":  float(feats["start_end_displacement_norm"].quantile(quantile)),
        "path_length_p01":             float(feats["path_length_norm"].quantile(quantile)),
        "mean_step_displacement_p01":  float(feats["mean_step_displacement_norm"].quantile(quantile)),
        "quantile":                    quantile,
    }


def apply_stationary_rule(
    X: np.ndarray,
    thresholds: dict,
) -> pd.DataFrame:
    feats = compute_kinematic_features(X)

    feats["low_mean_speed"] = (
        feats["mean_speed_norm"] < thresholds["mean_speed_p01"]
    )
    feats["low_displacement"] = (
        feats["start_end_displacement_norm"] < thresholds["start_end_displacement_p01"]
    )
    feats["low_path_length"] = (
        feats["path_length_norm"] < thresholds["path_length_p01"]
    )
    feats["low_step_displacement"] = (
        feats["mean_step_displacement_norm"] < thresholds["mean_step_displacement_p01"]
    )
    feats["stationary_flag"] = (
        feats["low_mean_speed"]
        | feats["low_displacement"]
        | feats["low_path_length"]
        | feats["low_step_displacement"]
    )
    return feats


def save_thresholds(thresholds: dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(thresholds, f, indent=2)


def load_thresholds(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)
