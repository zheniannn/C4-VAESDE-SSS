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
    model.eval()
    context = torch.tensor(initial_context, dtype=torch.float32, device=device)
    generated = [context.cpu().numpy()]

    history = context.unsqueeze(0)  # (1, context_len, 4)

    for _ in range(steps):
        mu, logvar = model(history)   # (1, T, 4)
        # Take prediction at the last timestep
        mu_last = mu[0, -1, :]       # (4,)
        logvar_last = logvar[0, -1, :]  # (4,)

        if deterministic:
            next_state = mu_last
        else:
            std = torch.exp(0.5 * logvar_last)
            eps = torch.randn_like(std)
            next_state = mu_last + std * eps

        generated.append(next_state.cpu().numpy()[np.newaxis, :])
        history = torch.cat([history, next_state.unsqueeze(0).unsqueeze(0)], dim=1)

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
