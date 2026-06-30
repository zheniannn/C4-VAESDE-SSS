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
    # Vectorised: reduce each batch to per-sequence column tensors and move to
    # CPU once per batch, instead of ~12 .item() GPU->CPU syncs per sequence.
    columns = [
        "total_nll", "pos_nll", "vel_nll", "mahalanobis",
        "final_step_nll", "max_step_nll", "mean_std", "pos_std", "vel_std",
        "mean_abs_error", "mean_drift_norm", "mean_diffusion_norm",
    ]
    chunks: dict[str, list[np.ndarray]] = {c: [] for c in columns}
    dt = 1.0

    for x_input, y_target in loader:
        x_input = x_input.to(device)
        y_target = y_target.to(device)

        mu, logvar, _ = model(x_input)
        decomp = decompose_nll(mu, logvar, y_target)

        nll_feat = decomp["nll_per_feature"]          # (B, T, 4)
        mah_feat = decomp["squared_mahalanobis_per_feature"]  # (B, T, 4)
        std_feat = decomp["std"]                       # (B, T, 4)

        drift = predict_drift(mu, x_input, dt=dt)             # (B, T, 4)
        diffusion = predict_diffusion(logvar, dt=dt)          # (B, T, 4)

        batch = {
            "total_nll": nll_feat.mean(dim=(1, 2)),
            "pos_nll": nll_feat[:, :, :2].mean(dim=(1, 2)),
            "vel_nll": nll_feat[:, :, 2:].mean(dim=(1, 2)),
            "mahalanobis": mah_feat.mean(dim=(1, 2)),
            "final_step_nll": nll_feat[:, -1, :].mean(dim=1),
            "max_step_nll": nll_feat.mean(dim=2).max(dim=1).values,
            "mean_std": std_feat.mean(dim=(1, 2)),
            "pos_std": std_feat[:, :, :2].mean(dim=(1, 2)),
            "vel_std": std_feat[:, :, 2:].mean(dim=(1, 2)),
            "mean_abs_error": (mu - y_target).abs().mean(dim=(1, 2)),
            "mean_drift_norm": drift.norm(dim=-1).mean(dim=1),
            "mean_diffusion_norm": diffusion.norm(dim=-1).mean(dim=1),
        }
        for c in columns:
            chunks[c].append(batch[c].detach().cpu().numpy())

    data = {c: np.concatenate(chunks[c]) if chunks[c] else np.array([]) for c in columns}
    n = len(data["total_nll"])
    return pd.DataFrame({"sequence_index": np.arange(n), **data})


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
