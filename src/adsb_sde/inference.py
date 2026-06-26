from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dataset import SequenceDataset
from .loss import decompose_nll
from .model import ProbabilisticMotionLSTM, build_model, predict_diffusion, predict_drift
from .utils import get_device


def load_checkpoint(
    path: str | Path,
    device: Optional[torch.device] = None,
) -> tuple[ProbabilisticMotionLSTM, dict]:
    if device is None:
        device = get_device()
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config


@torch.no_grad()
def compute_sde_scores(
    model: ProbabilisticMotionLSTM,
    loader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    records: list[dict] = []
    seq_idx = 0
    dt = 1.0

    for x_input, y_target in loader:
        x_input = x_input.to(device)
        y_target = y_target.to(device)

        mu, logvar = model(x_input)
        decomp = decompose_nll(mu, logvar, y_target)

        nll_feat = decomp["nll_per_feature"]          # (B, T, 4)
        mah_feat = decomp["squared_mahalanobis_per_feature"]  # (B, T, 4)
        std_feat = decomp["std"]                       # (B, T, 4)

        drift = predict_drift(mu, x_input, dt=dt)             # (B, T, 4)
        diffusion = predict_diffusion(logvar, dt=dt)          # (B, T, 4)

        B = x_input.size(0)
        for i in range(B):
            nll_i = nll_feat[i]       # (T, 4)
            mah_i = mah_feat[i]       # (T, 4)
            std_i = std_feat[i]       # (T, 4)
            mu_i = mu[i]              # (T, 4)
            tgt_i = y_target[i]       # (T, 4)
            drift_i = drift[i]        # (T, 4)
            diff_i = diffusion[i]     # (T, 4)

            records.append({
                "sequence_index": seq_idx,
                "total_nll": nll_i.mean().item(),
                "pos_nll": nll_i[:, :2].mean().item(),
                "vel_nll": nll_i[:, 2:].mean().item(),
                "mahalanobis": mah_i.mean().item(),
                "final_step_nll": nll_i[-1, :].mean().item(),
                "max_step_nll": nll_i.mean(dim=1).max().item(),
                "mean_std": std_i.mean().item(),
                "pos_std": std_i[:, :2].mean().item(),
                "vel_std": std_i[:, 2:].mean().item(),
                "mean_abs_error": (mu_i - tgt_i).abs().mean().item(),
                "mean_drift_norm": drift_i.norm(dim=-1).mean().item(),
                "mean_diffusion_norm": diff_i.norm(dim=-1).mean().item(),
            })
            seq_idx += 1

    return pd.DataFrame(records)


def score_sequences(
    model: ProbabilisticMotionLSTM,
    X: np.ndarray,
    batch_size: int,
    device: torch.device,
    max_samples: Optional[int] = None,
) -> pd.DataFrame:
    ds = SequenceDataset(X, max_samples=max_samples)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return compute_sde_scores(model, loader, device)
