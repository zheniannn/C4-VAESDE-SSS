# C4-VAESDE-ADSB-SDE

Probabilistic SDE-style LSTM transition model for ADS-B trajectory anomaly detection.

## Purpose

C4 extends the ADS-B motion-prior pipeline with a **probabilistic transition model**. Where C3 learns a deterministic next-state prediction, C4 learns the full distribution over next states — enabling calibrated uncertainty estimates and likelihood-based anomaly scoring.

## Relationship to C1, C2, and C3

| Repo | Role |
|------|------|
| C1-VAESDE-ADSB-PREPROCESSING | Produces normalised ADS-B EN motion windows |
| C2-VAESDE-ADSB-TRAINING | Trains VAE + kinematic motion prior |
| C3-VAESDE-ADSB-LSTM | Trains deterministic LSTM transition model |
| **C4-VAESDE-ADSB-SDE** | **Trains probabilistic SDE-style LSTM transition model** |

C4 is standalone. It may optionally load a trained C3 checkpoint to initialise the mean network.

## Why SDE-LSTM

The deterministic LSTM (C3) predicts:

```
x_{t+1} = f_theta(history)
```

The SDE-LSTM (C4) predicts:

```
x_{t+1} ~ Normal(mu_theta(history), sigma_theta(history)^2)
```

This gives:
- **Calibrated uncertainty** — the model knows when it is uncertain.
- **Likelihood-based anomaly scores** — unusual transitions are flagged by low probability, not just high error.
- **Probabilistic rollouts** — trajectories can be sampled, not just predicted.

## SDE Interpretation

The model can be read as a discrete-time SDE:

```
x_{t+1} = x_t + drift_theta(history) * dt + diffusion_theta(history) * sqrt(dt) * epsilon

where:
  dt     = 1 normalised timestep
  epsilon ~ Normal(0, I)
  drift  = (mu - x_t) / dt
  diffusion = exp(0.5 * logvar) / sqrt(dt)
```

The predicted mean represents the expected transition (drift).
The predicted variance represents uncertainty (diffusion).

## Input Data

Place in `data/`:

| File | Required | Shape |
|------|----------|-------|
| `X_train.npy` | Yes | (N, 30, 4) |
| `X_test.npy` | Yes | (N, 30, 4) |
| `normalisation_mean.csv` | No | — |
| `normalisation_std.csv` | No | — |
| `train_sequence_metadata.csv` | No | — |
| `test_sequence_metadata.csv` | No | — |

Feature order: `[E_m, N_m, vE_mps, vN_mps]` (normalised).

## Model Objective

Gaussian negative log-likelihood:

```
NLL = 0.5 * [(target - mu)^2 / var + logvar]
```

The model outputs `mu` and `logvar` for each timestep and feature.

## Repository Structure

```
C4-VAESDE-ADSB-SDE/
├── configs/
│   └── sde_lstm_default.yaml      # Default training config
├── scripts/
│   ├── run_train_sde_lstm.py      # Train the model
│   ├── run_score_sde_lstm.py      # Score sequences, compute thresholds
│   ├── run_stress_test_sde_lstm.py # Stress tests with synthetic corruptions
│   ├── run_rollout_sde_lstm.py    # Generate stochastic rollouts
│   └── run_compare_thresholds.py  # Compare p95/p99 detection rates
├── src/adsb_sde/
│   ├── config.py      # load_config, set_seed, ensure_dir
│   ├── dataset.py     # SequenceDataset
│   ├── model.py       # ProbabilisticMotionLSTM
│   ├── loss.py        # gaussian_nll_loss, decompose_nll
│   ├── training.py    # fit_model
│   ├── inference.py   # load_checkpoint, score_sequences
│   ├── corruption.py  # Stress-test corruptions
│   ├── rollout.py     # sample_rollout, rollout_batch
│   ├── reporting.py   # save_dataframe, print_summary_table
│   └── utils.py       # get_device, count_parameters, describe_array
├── tests/
│   └── test_smoke.py  # Smoke tests (no real data required)
├── data/              # Place X_train.npy and X_test.npy here
├── checkpoints/
└── outputs/
```

## Quick Start

```bash
pip install -e .
pytest
```

## Copy Data from C1

```bash
cp ../C1-VAESDE-ADSB-PREPROCESSING/data/X_train.npy data/
cp ../C1-VAESDE-ADSB-PREPROCESSING/data/X_test.npy data/
```

Optional: initialise from C3 checkpoint (set in config):

```
c3_checkpoint_path: ../C3-VAESDE-ADSB-LSTM/outputs/lstm/motion_lstm.pt
```

## Training

```bash
python scripts/run_train_sde_lstm.py --config configs/sde_lstm_default.yaml
```

Outputs:
- `outputs/sde_lstm/sde_lstm.pt` — model checkpoint
- `outputs/sde_lstm/history.csv` — per-epoch loss history

## Scoring

```bash
python scripts/run_score_sde_lstm.py --config configs/sde_lstm_default.yaml
```

Outputs:
- `outputs/sde_lstm/train_sde_scores.csv`
- `outputs/sde_lstm/test_sde_scores.csv`
- `outputs/sde_lstm/sde_thresholds.csv`

## Stress Testing

```bash
python scripts/run_stress_test_sde_lstm.py \
  --config configs/sde_lstm_default.yaml \
  --score-name total_nll \
  --quantile 0.99
```

Outputs:
- `outputs/sde_lstm/sde_stress_summary.csv`

To compare p95 vs p99 (run stress test twice, then):

```bash
python scripts/run_compare_thresholds.py
```

## Rollout Generation

```bash
python scripts/run_rollout_sde_lstm.py --config configs/sde_lstm_default.yaml
```

Outputs:
- `outputs/sde_lstm/rollout_deterministic.npy`
- `outputs/sde_lstm/rollout_stochastic.npy`
- `outputs/sde_lstm/rollout_examples.png`

## Stationary-Clutter Rule

The SDE-LSTM detects **dynamic** abnormalities via transition NLL — trajectories
where the actual next state is unlikely under the learned distribution. However,
stationary clutter (a nearly-fixed aircraft or ghost return) is a *highly
predictable* sequence: constant position, near-zero velocity. The model learns
this pattern well, so NLL stays low and the SDE score does not flag it.

To close this gap, C4 adds a simple kinematic **lower-tail rule** calibrated
from clean train data:

> **Flag stationary clutter if any of:**
> - `mean_speed_norm` < train p1
> - `start_end_displacement_norm` < train p1
> - `path_length_norm` < train p1
> - `mean_step_displacement_norm` < train p1

These thresholds are data-driven — calibrated at the 1st percentile of the
clean training set in normalised units — so no physical m/s cutoff is hardcoded.

**Final C4 detector:**

```
flag_abnormal = (SDE_total_nll > train_p99) OR (stationary_rule == True)
```

### Stationary-rule commands

Calibrate and apply the rule on train/test:

```bash
python scripts/run_stationary_rule.py --config configs/sde_lstm_default.yaml
```

Stress-test the rule across all corruptions:

```bash
python scripts/run_stress_test_stationary_rule.py \
  --config configs/sde_lstm_default.yaml \
  --max-samples 50000
```

Fused evaluation (SDE + stationary rule):

```bash
python scripts/run_fused_sde_stationary.py \
  --config configs/sde_lstm_default.yaml \
  --score-name total_nll \
  --quantile 0.99 \
  --max-samples 50000
```

Outputs:
- `outputs/sde_lstm/stationary_thresholds.json`
- `outputs/sde_lstm/train_stationary_features.csv`
- `outputs/sde_lstm/test_stationary_features.csv`
- `outputs/sde_lstm/stationary_rule_summary.csv`
- `outputs/sde_lstm/stationary_rule_stress_summary.csv`
- `outputs/sde_lstm/fused_sde_stationary_summary.csv`

## Interpretation of SDE Scores

| Score | Meaning |
|-------|---------|
| `total_nll` | Mean NLL over all timesteps and features — main anomaly score |
| `pos_nll` | NLL for E, N position only |
| `vel_nll` | NLL for vE, vN velocity only |
| `mahalanobis` | Mean squared Mahalanobis term (error / variance) |
| `final_step_nll` | NLL at the last predicted timestep |
| `max_step_nll` | Worst-case NLL across timesteps |
| `mean_std` | Mean predicted uncertainty — high means the model is unsure |
| `mean_drift_norm` | Mean magnitude of predicted drift |
| `mean_diffusion_norm` | Mean magnitude of predicted diffusion |

A trajectory is suspicious if the actual next state is unlikely under the learned
transition distribution — i.e., the NLL is high.

## How to Compare with C3

Run C3 scoring first to get MSE-based anomaly scores, then run C4 scoring for
NLL-based scores. Compare detection rates on the same stress-test cases.

C3 deterministic LSTM:
- Anomaly score = reconstruction MSE
- Cannot separate large error from genuinely uncertain regions

C4 SDE-LSTM:
- Anomaly score = NLL = error + predicted variance
- High NLL when error is large *relative to* predicted uncertainty

## How to Combine with C2 and C3

The pipeline layers four complementary signals:

| Layer | Model | What it detects |
|-------|-------|-----------------|
| Kinematic rules | Hard thresholds | Speed jumps, impossible states |
| VAE (C2) | Whole-window reconstruction | Globally implausible trajectories |
| LSTM (C3) | Step-to-step MSE | Deterministic transition errors |
| SDE-LSTM (C4) | Probabilistic NLL | Unlikely transitions, calibrated uncertainty |

Each layer flags different failure modes. Combining all four gives the most robust
anomaly detection.
