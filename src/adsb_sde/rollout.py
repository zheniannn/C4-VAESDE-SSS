from __future__ import annotations

import numpy as np
import torch

from .model import ProbabilisticMotionLSTM


@torch.no_grad()
def sample_rollout(
    model: ProbabilisticMotionLSTM,
    initial_context: np.ndarray,
    steps: int,
    device: torch.device,
    deterministic: bool = False,
) -> np.ndarray:
    # initial_context: (context_len, 4)
    # Pass the full context once to warm up hidden state, then step one token at a time.
    model.eval()
    context = torch.tensor(initial_context, dtype=torch.float32, device=device)
    generated: list[np.ndarray] = [context.cpu().numpy()]

    # Warm up: run the full context through the LSTM to get hidden state.
    # Shape: (1, context_len, 4)
    ctx = context.unsqueeze(0)
    mu, logvar, hidden = model(ctx)  # hidden carries (h_n, c_n) after full context

    # The prediction for the next step after the last context token:
    mu_last     = mu[0, -1, :]      # (4,)
    logvar_last = logvar[0, -1, :]  # (4,)

    for _ in range(steps):
        if deterministic:
            next_state = mu_last
        else:
            std = torch.exp(0.5 * logvar_last)
            next_state = mu_last + std * torch.randn_like(std)

        generated.append(next_state.cpu().numpy()[np.newaxis, :])

        # Feed just the single new token, reusing the LSTM hidden state.
        token = next_state.unsqueeze(0).unsqueeze(0)  # (1, 1, 4)
        mu, logvar, hidden = model(token, hidden)
        mu_last     = mu[0, -1, :]
        logvar_last = logvar[0, -1, :]

    return np.concatenate(generated, axis=0)  # (context_len + steps, 4)


@torch.no_grad()
def rollout_batch(
    model: ProbabilisticMotionLSTM,
    initial_windows: np.ndarray,
    rollout_steps: int,
    device: torch.device,
    deterministic: bool = False,
) -> np.ndarray:
    # initial_windows: (B, 30, 4)
    model.eval()
    B = initial_windows.shape[0]
    results = np.zeros((B, rollout_steps, 4), dtype=np.float32)

    for i in range(B):
        traj = sample_rollout(
            model,
            initial_windows[i],
            rollout_steps,
            device,
            deterministic=deterministic,
        )
        results[i] = traj[-rollout_steps:]

    return results
