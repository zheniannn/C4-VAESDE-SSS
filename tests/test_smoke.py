"""Smoke tests — no real data required."""
import math

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from adsb_sde.corruption import (
    position_jump,
    random_walk_velocity,
    speed_scale,
    stationary_clutter,
    sudden_turn_90,
)
from adsb_sde.dataset import SequenceDataset
from adsb_sde.inference import compute_sde_scores
from adsb_sde.kinematic_rules import (
    apply_stationary_rule,
    calibrate_stationary_thresholds,
    compute_kinematic_features,
)
from adsb_sde.loss import gaussian_nll_loss, mse_for_monitoring
from adsb_sde.model import ProbabilisticMotionLSTM
from adsb_sde.rollout import sample_rollout

N, T, F = 128, 30, 4
rng = np.random.default_rng(0)
SYNTHETIC = rng.standard_normal((N, T, F)).astype(np.float32)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

def test_dataset_shapes():
    ds = SequenceDataset(SYNTHETIC)
    x, y = ds[0]
    assert x.shape == (29, 4)
    assert y.shape == (29, 4)
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32


def test_dataset_len():
    ds = SequenceDataset(SYNTHETIC, max_samples=64)
    assert len(ds) == 64


def test_dataset_bad_shape():
    with pytest.raises(ValueError):
        SequenceDataset(np.zeros((10, 20, 4)))


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

def _make_model():
    return ProbabilisticMotionLSTM(
        input_dim=4, hidden_dim=32, num_layers=2, dropout=0.0,
        min_logvar=-8.0, max_logvar=4.0,
    )


def test_model_output_shapes():
    model = _make_model()
    x = torch.randn(4, 29, 4)
    mu, logvar, hidden = model(x)
    assert mu.shape == (4, 29, 4)
    assert logvar.shape == (4, 29, 4)
    assert len(hidden) == 2  # (h_n, c_n)


def test_logvar_clamped():
    model = _make_model()
    x = torch.randn(4, 29, 4) * 100  # extreme input
    _, logvar, _ = model(x)
    assert logvar.min().item() >= -8.0 - 1e-5
    assert logvar.max().item() <= 4.0 + 1e-5


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #

def test_gaussian_nll_finite():
    mu = torch.zeros(8, 29, 4)
    logvar = torch.zeros(8, 29, 4)
    target = torch.randn(8, 29, 4)
    loss = gaussian_nll_loss(mu, logvar, target)
    assert loss.ndim == 0
    assert math.isfinite(loss.item())


def test_gaussian_nll_none_reduction():
    mu = torch.zeros(8, 29, 4)
    logvar = torch.zeros(8, 29, 4)
    target = torch.randn(8, 29, 4)
    nll = gaussian_nll_loss(mu, logvar, target, reduction="none")
    assert nll.shape == (8, 29, 4)


def test_mse_for_monitoring():
    mu = torch.ones(4, 29, 4)
    target = torch.zeros(4, 29, 4)
    mse = mse_for_monitoring(mu, target)
    assert abs(mse.item() - 1.0) < 1e-5


# --------------------------------------------------------------------------- #
# Train/eval loop
# --------------------------------------------------------------------------- #

def test_train_eval_loop():
    model = _make_model()
    ds = SequenceDataset(SYNTHETIC, max_samples=32)
    loader = DataLoader(ds, batch_size=16, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for x_input, y_target in loader:
        mu, logvar, _ = model(x_input)
        loss = gaussian_nll_loss(mu, logvar, y_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        assert math.isfinite(loss.item())

    model.eval()
    with torch.no_grad():
        for x_input, y_target in loader:
            mu, logvar, _ = model(x_input)
            loss = gaussian_nll_loss(mu, logvar, y_target)
            assert math.isfinite(loss.item())


# --------------------------------------------------------------------------- #
# SDE scores
# --------------------------------------------------------------------------- #

SCORE_COLUMNS = [
    "sequence_index", "total_nll", "pos_nll", "vel_nll",
    "mahalanobis", "final_step_nll", "max_step_nll",
    "mean_std", "pos_std", "vel_std",
    "mean_abs_error", "mean_drift_norm", "mean_diffusion_norm",
]


def test_compute_sde_scores_columns():
    model = _make_model()
    model.eval()
    ds = SequenceDataset(SYNTHETIC, max_samples=16)
    loader = DataLoader(ds, batch_size=8, shuffle=False)
    device = torch.device("cpu")
    df = compute_sde_scores(model, loader, device)
    for col in SCORE_COLUMNS:
        assert col in df.columns, f"Missing column: {col}"
    assert len(df) == 16


# --------------------------------------------------------------------------- #
# Corruptions
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("fn", [
    lambda x: speed_scale(x, 1.5),
    lambda x: random_walk_velocity(x),
    lambda x: sudden_turn_90(x),
    lambda x: position_jump(x),
    lambda x: stationary_clutter(x),
])
def test_corruption_shape(fn):
    out = fn(SYNTHETIC)
    assert out.shape == SYNTHETIC.shape
    # Original must not be mutated
    assert not np.shares_memory(out, SYNTHETIC) or not np.array_equal(out, SYNTHETIC)


# --------------------------------------------------------------------------- #
# Rollout
# --------------------------------------------------------------------------- #

def test_sample_rollout_shape():
    model = _make_model()
    model.eval()
    context = np.random.randn(30, 4).astype(np.float32)
    device = torch.device("cpu")
    result = sample_rollout(model, context, steps=10, device=device, deterministic=True)
    # context (30) + steps (10) rows
    assert result.shape == (40, 4)


def test_sample_rollout_stochastic_shape():
    model = _make_model()
    model.eval()
    context = np.random.randn(30, 4).astype(np.float32)
    device = torch.device("cpu")
    result = sample_rollout(model, context, steps=5, device=device, deterministic=False)
    assert result.shape == (35, 4)


# --------------------------------------------------------------------------- #
# Kinematic rules
# --------------------------------------------------------------------------- #

KINEMATIC_COLUMNS = [
    "sequence_index",
    "mean_speed_norm",
    "max_speed_norm",
    "min_speed_norm",
    "start_end_displacement_norm",
    "path_length_norm",
    "displacement_to_path_ratio",
    "mean_step_displacement_norm",
    "max_step_displacement_norm",
]

THRESHOLD_KEYS = [
    "mean_speed_p01",
    "start_end_displacement_p01",
    "path_length_p01",
    "mean_step_displacement_p01",
    "quantile",
]


def test_compute_kinematic_features_columns():
    df = compute_kinematic_features(SYNTHETIC)
    assert len(df) == N
    for col in KINEMATIC_COLUMNS:
        assert col in df.columns, f"Missing column: {col}"


def test_calibrate_stationary_thresholds_keys():
    thresholds = calibrate_stationary_thresholds(SYNTHETIC, quantile=0.01)
    for key in THRESHOLD_KEYS:
        assert key in thresholds, f"Missing key: {key}"
    assert thresholds["quantile"] == 0.01


def test_apply_stationary_rule_has_flag():
    thresholds = calibrate_stationary_thresholds(SYNTHETIC, quantile=0.01)
    df = apply_stationary_rule(SYNTHETIC, thresholds)
    assert "stationary_flag" in df.columns
    assert df["stationary_flag"].dtype == bool or df["stationary_flag"].isin([True, False]).all()


def test_stationary_clutter_higher_flag_rate_than_moving():
    # Build synthetic moving data: non-trivial velocities
    rng2 = np.random.default_rng(1)
    moving = rng2.standard_normal((256, 30, 4)).astype(np.float32)
    moving[:, :, 2] += 1.0   # positive vE baseline
    moving[:, :, 3] += 1.0   # positive vN baseline
    # Recompute positions
    for t in range(1, 30):
        moving[:, t, 0] = moving[:, t - 1, 0] + moving[:, t - 1, 2]
        moving[:, t, 1] = moving[:, t - 1, 1] + moving[:, t - 1, 3]

    clutter = stationary_clutter(moving)

    thresholds = calibrate_stationary_thresholds(moving, quantile=0.05)
    moving_feats  = apply_stationary_rule(moving,  thresholds)
    clutter_feats = apply_stationary_rule(clutter, thresholds)

    moving_rate  = moving_feats["stationary_flag"].mean()
    clutter_rate = clutter_feats["stationary_flag"].mean()
    assert clutter_rate > moving_rate, (
        f"Expected stationary clutter flag rate ({clutter_rate:.3f}) "
        f"> moving flag rate ({moving_rate:.3f})"
    )


def test_kinematic_module_imports():
    # Verify all public symbols are importable
    from adsb_sde.kinematic_rules import (  # noqa: F401
        apply_stationary_rule,
        calibrate_stationary_thresholds,
        compute_kinematic_features,
        load_thresholds,
        save_thresholds,
    )
