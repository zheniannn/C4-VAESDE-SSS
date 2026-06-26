from __future__ import annotations

import numpy as np


def speed_scale(x: np.ndarray, factor: float = 1.5) -> np.ndarray:
    # x: (N, 30, 4) - [E, N, vE, vN] normalised
    out = x.copy()
    out[:, :, 2] = x[:, :, 2] * factor   # vE
    out[:, :, 3] = x[:, :, 3] * factor   # vN
    # Recompute E, N from starting position using scaled velocities
    out[:, 0, 0] = x[:, 0, 0]
    out[:, 0, 1] = x[:, 0, 1]
    for t in range(1, 30):
        out[:, t, 0] = out[:, t - 1, 0] + out[:, t - 1, 2]
        out[:, t, 1] = out[:, t - 1, 1] + out[:, t - 1, 3]
    return out


def random_walk_velocity(x: np.ndarray, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = x.copy()
    N, T, _ = x.shape
    noise_vE = rng.standard_normal((N, T)).astype(np.float32)
    noise_vN = rng.standard_normal((N, T)).astype(np.float32)
    out[:, :, 2] = noise_vE
    out[:, :, 3] = noise_vN
    out[:, 0, 0] = x[:, 0, 0]
    out[:, 0, 1] = x[:, 0, 1]
    for t in range(1, T):
        out[:, t, 0] = out[:, t - 1, 0] + out[:, t - 1, 2]
        out[:, t, 1] = out[:, t - 1, 1] + out[:, t - 1, 3]
    return out


def sudden_turn_90(x: np.ndarray) -> np.ndarray:
    out = x.copy()
    N, T, _ = x.shape
    mid = T // 2
    # Rotate velocity by 90 degrees from mid onward
    vE_orig = x[:, mid:, 2].copy()
    vN_orig = x[:, mid:, 3].copy()
    out[:, mid:, 2] = -vN_orig
    out[:, mid:, 3] = vE_orig
    # Recompute positions from mid onward
    for t in range(mid, T):
        if t == mid:
            out[:, t, 0] = x[:, t, 0]
            out[:, t, 1] = x[:, t, 1]
        else:
            out[:, t, 0] = out[:, t - 1, 0] + out[:, t - 1, 2]
            out[:, t, 1] = out[:, t - 1, 1] + out[:, t - 1, 3]
    return out


def position_jump(x: np.ndarray, jump_normalised: float = 2.0) -> np.ndarray:
    out = x.copy()
    mid = x.shape[1] // 2
    out[:, mid:, 0] = x[:, mid:, 0] + jump_normalised
    out[:, mid:, 1] = x[:, mid:, 1] + jump_normalised
    return out


def stationary_clutter(x: np.ndarray) -> np.ndarray:
    out = x.copy()
    # Fix position to starting value, near-zero velocity
    out[:, :, 0] = x[:, 0:1, 0]
    out[:, :, 1] = x[:, 0:1, 1]
    out[:, :, 2] = 0.0
    out[:, :, 3] = 0.0
    return out
