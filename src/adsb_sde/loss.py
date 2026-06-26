from __future__ import annotations

import torch


def gaussian_nll_loss(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    # nll = 0.5 * [(target - mu)^2 / var + logvar]  (no 2pi constant)
    var = torch.exp(logvar)
    nll = 0.5 * ((target - mu) ** 2 / var + logvar)
    if reduction == "mean":
        return nll.mean()
    elif reduction == "none":
        return nll
    else:
        raise ValueError(f"Unknown reduction: {reduction!r}")


def mse_for_monitoring(mu: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.mse_loss(mu, target)


def decompose_nll(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    var = torch.exp(logvar)
    std = torch.exp(0.5 * logvar)
    squared_mahalanobis = (target - mu) ** 2 / var   # (B, T, 4)
    nll_per_feature = 0.5 * (squared_mahalanobis + logvar)  # (B, T, 4)
    return {
        "nll_per_feature": nll_per_feature,
        "squared_mahalanobis_per_feature": squared_mahalanobis,
        "std": std,
    }
