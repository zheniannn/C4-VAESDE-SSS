from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import ensure_dir, load_config, set_seed
from .dataset import SequenceDataset
from .loss import gaussian_nll_loss, mse_for_monitoring
from .model import build_model, initialise_from_c3_checkpoint
from .utils import get_device


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip: float,
) -> tuple[float, float]:
    model.train()
    total_nll = 0.0
    total_mse = 0.0
    n_batches = 0

    for x_input, y_target in tqdm(loader, desc="  train", leave=False):
        x_input = x_input.to(device)
        y_target = y_target.to(device)

        mu, logvar = model(x_input)
        loss = gaussian_nll_loss(mu, logvar, y_target)
        mse = mse_for_monitoring(mu, y_target)

        optimizer.zero_grad()
        loss.backward()
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        total_nll += loss.item()
        total_mse += mse.item()
        n_batches += 1

    return total_nll / n_batches, total_mse / n_batches


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_nll = 0.0
    total_mse = 0.0
    n_batches = 0

    for x_input, y_target in loader:
        x_input = x_input.to(device)
        y_target = y_target.to(device)

        mu, logvar = model(x_input)
        loss = gaussian_nll_loss(mu, logvar, y_target)
        mse = mse_for_monitoring(mu, y_target)

        total_nll += loss.item()
        total_mse += mse.item()
        n_batches += 1

    return {"nll": total_nll / n_batches, "mse": total_mse / n_batches}


def fit_model(config: dict) -> None:
    set_seed(config["seed"])

    data_dir = Path(config["data_dir"])
    output_dir = ensure_dir(config["output_dir"])
    train_path = data_dir / config["train_file"]
    test_path = data_dir / config["test_file"]

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Missing X_train.npy or X_test.npy. "
            "Copy them from C1-VAESDE-ADSB-PREPROCESSING into data/."
        )

    X_train = np.load(train_path, mmap_mode="r")
    X_test = np.load(test_path, mmap_mode="r")

    debug_mode = config.get("debug_mode", False)
    max_train = config.get("debug_train_size") if debug_mode else None
    max_test = config.get("debug_test_size") if debug_mode else None

    train_ds = SequenceDataset(X_train, max_samples=max_train)
    test_ds = SequenceDataset(X_test, max_samples=max_test)

    num_workers = config.get("num_workers", 0)
    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    device = get_device()
    model = build_model(config).to(device)

    c3_report = {"loaded": [], "skipped": []}
    if config.get("initialise_from_c3", False):
        c3_path = Path(config["c3_checkpoint_path"])
        if c3_path.exists():
            c3_report = initialise_from_c3_checkpoint(
                model, c3_path, strict=config.get("strict_c3_load", False)
            )
            print(f"[C3 init] Loaded from {c3_path}")
            for entry in c3_report["loaded"]:
                print(f"  loaded: {entry}")
        else:
            print(f"[C3 init] Checkpoint not found at {c3_path}, training from scratch.")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 0.0),
    )

    gradient_clip = config.get("gradient_clip", 1.0)
    epochs = config["epochs"]
    history: list[dict] = []

    print(f"\nTraining SDE-LSTM for {epochs} epochs on {device}")
    print(f"  train sequences: {len(train_ds)}, test sequences: {len(test_ds)}\n")

    for epoch in range(1, epochs + 1):
        train_nll, train_mse = train_one_epoch(
            model, train_loader, optimizer, device, gradient_clip
        )
        test_metrics = evaluate(model, test_loader, device)

        row = {
            "epoch": epoch,
            "train_nll": train_nll,
            "train_mse": train_mse,
            "test_nll": test_metrics["nll"],
            "test_mse": test_metrics["mse"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train NLL {train_nll:.4f}  MSE {train_mse:.6f} | "
            f"test NLL {test_metrics['nll']:.4f}  MSE {test_metrics['mse']:.6f}"
        )

    checkpoint_path = output_dir / "sde_lstm.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "final_train_nll": history[-1]["train_nll"],
            "final_train_mse": history[-1]["train_mse"],
            "final_test_nll": history[-1]["test_nll"],
            "final_test_mse": history[-1]["test_mse"],
            "history": history,
            "c3_initialisation_report": c3_report,
        },
        checkpoint_path,
    )
    print(f"\nCheckpoint saved to {checkpoint_path}")

    history_path = output_dir / "history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    print(f"Training history saved to {history_path}")
