from __future__ import annotations

import numpy as np
import torch


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def describe_array(name: str, X: np.ndarray, sample_size: int = 10_000) -> None:
    total = X.shape[0]
    n = min(sample_size, total)
    sample = np.array(X[:n], dtype=np.float32)
    print(
        f"{name}: shape={X.shape}  dtype={X.dtype}  "
        f"mean={sample.mean():.4f}  std={sample.std():.4f}  "
        f"min={sample.min():.4f}  max={sample.max():.4f}  "
        f"(stats from first {n}/{total} rows)"
    )
