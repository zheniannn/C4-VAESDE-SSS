from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


class ProbabilisticMotionLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        min_logvar: float = -8.0,
        max_logvar: float = 4.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.mean_head = nn.Linear(hidden_dim, input_dim)
        self.logvar_head = nn.Linear(hidden_dim, input_dim)

    def forward(
        self,
        x: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # x: (B, T, 4)
        out, hidden_out = self.lstm(x, hidden)   # (B, T, hidden_dim)
        mu = self.mean_head(out)                  # (B, T, 4)
        logvar = self.logvar_head(out)            # (B, T, 4)
        logvar = torch.clamp(logvar, self.min_logvar, self.max_logvar)
        return mu, logvar, hidden_out


def predict_drift(mu: torch.Tensor, x_input: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
    return (mu - x_input) / dt


def predict_diffusion(logvar: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    return std / math.sqrt(dt)


def build_model(config: dict) -> ProbabilisticMotionLSTM:
    return ProbabilisticMotionLSTM(
        input_dim=config.get("n_features", 4),
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        min_logvar=config.get("min_logvar", -8.0),
        max_logvar=config.get("max_logvar", 4.0),
    )


def initialise_from_c3_checkpoint(
    model: ProbabilisticMotionLSTM,
    c3_checkpoint_path: str | Path,
    strict: bool = False,
) -> dict:
    path = Path(c3_checkpoint_path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    # Unwrap state dict from checkpoint formats
    if "model_state_dict" in checkpoint:
        c3_state = checkpoint["model_state_dict"]
    elif "model_state" in checkpoint:
        c3_state = checkpoint["model_state"]
    elif isinstance(checkpoint, dict) and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        c3_state = checkpoint
    else:
        print("[C3 init] Unrecognised checkpoint format; trying top-level dict.")
        c3_state = checkpoint

    loaded: list[str] = []
    skipped: list[str] = []

    c4_state = model.state_dict()

    # Map C3 lstm.* -> C4 lstm.*
    for key, val in c3_state.items():
        if key.startswith("lstm."):
            c4_key = key
        elif key.startswith("head."):
            # C3 deterministic head -> C4 mean_head
            c4_key = "mean_head." + key[len("head."):]
        else:
            skipped.append(key)
            continue

        if c4_key in c4_state and c4_state[c4_key].shape == val.shape:
            c4_state[c4_key] = val
            loaded.append(f"{key} -> {c4_key}")
        else:
            msg = f"[C3 init] Shape mismatch or missing key: {key} -> {c4_key}"
            if strict:
                raise RuntimeError(msg)
            print(msg)
            skipped.append(key)

    model.load_state_dict(c4_state)

    report = {"loaded": loaded, "skipped": skipped}
    print(f"[C3 init] Loaded {len(loaded)} weight tensors, skipped {len(skipped)}.")
    return report
