#!/usr/bin/env python3
"""Post-training FP16 quantization for BEED (Bangalore EEG Epilepsy Dataset).

Requirements:
- PyTorch
- pandas
- scikit-learn

This script forces CUDA usage and does not allow CPU fallback.
"""

import argparse
import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FP16 post-training quantization for BEED CNN model")
    parser.add_argument("--csv", default="BEED_Data.csv", help="Path to BEED CSV file")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for DataLoaders")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--calibration-fraction", type=float, default=0.2, help="Fraction of train data for calibration")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


class CNNModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5)
        self.relu = nn.ReLU(inplace=True)
        # Input length 16 -> conv1: 12 -> conv2: 8, channels=64 => 64 * 8 = 512
        self.fc = nn.Linear(64 * 8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.flatten(1)
        x = self.fc(x)
        return x


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def load_and_preprocess(csv_path: str, seed: int) -> Tuple[TensorDataset, TensorDataset]:
    df = pd.read_csv(csv_path)
    # Features: X1-X16, Label: column 17 (0,1,2,3)
    X = df.iloc[:, 0:16].values.astype(np.float32)
    y = df.iloc[:, 16].values.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to tensors and reshape to (batch, 1, 16)
    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    return train_dataset, test_dataset


def build_dataloaders(
    train_dataset: TensorDataset,
    test_dataset: TensorDataset,
    batch_size: int,
    calibration_fraction: float,
    seed: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # Optional calibration split (not used for FP16)
    train_size = len(train_dataset)
    cal_size = int(train_size * calibration_fraction)
    main_size = train_size - cal_size
    generator = torch.Generator().manual_seed(seed)
    main_train, calibration = torch.utils.data.random_split(
        train_dataset, [main_size, cal_size], generator=generator
    )

    train_loader = DataLoader(
        main_train,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
        drop_last=False,
    )
    calibration_loader = DataLoader(
        calibration,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
        drop_last=False,
    )
    return train_loader, calibration_loader, test_loader


def apply_fp16(model: nn.Module) -> nn.Module:
    # FP16 post-training quantization on GPU via half-precision weights
    model = model.half()
    return model


def evaluate_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_fp16: bool,
) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if use_fp16:
                x = x.half()
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    model.train()
    running_loss = 0.0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        total += x.size(0)
    return running_loss / max(total, 1)


def train_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> None:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        _ = train_one_epoch(model, loader, device, optimizer, criterion)


def measure_latency_ms(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_fp16: bool,
) -> Tuple[float, float]:
    model.eval()
    total_ms = 0.0
    total_samples = 0

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    # Warmup
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            if use_fp16:
                x = x.half()
            _ = model(x)
            break
    torch.cuda.synchronize()

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            if use_fp16:
                x = x.half()

            torch.cuda.synchronize()
            starter.record()
            _ = model(x)
            ender.record()
            torch.cuda.synchronize()

            batch_ms = starter.elapsed_time(ender)
            total_ms += batch_ms
            total_samples += x.size(0)

    avg_ms_per_batch = total_ms / max(len(loader), 1)
    avg_ms_per_sample = total_ms / max(total_samples, 1)
    return avg_ms_per_batch, avg_ms_per_sample


def measure_peak_memory_bytes(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_fp16: bool,
) -> int:
    model.eval()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            if use_fp16:
                x = x.half()
            _ = model(x)
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated(device))


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = 0
    nonzero = 0
    for p in model.parameters():
        total += p.numel()
        nonzero += int(torch.count_nonzero(p).item())
    return total, nonzero


def main() -> None:
    args = parse_args()

    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required and CPU fallback is disallowed.")

    set_seed(args.seed)

    train_dataset, test_dataset = load_and_preprocess(args.csv, args.seed)
    train_loader, calibration_loader, test_loader = build_dataloaders(
        train_dataset,
        test_dataset,
        args.batch_size,
        args.calibration_fraction,
        args.seed,
    )

    model = CNNModel().to(device)
    train_model(model, train_loader, device, args.epochs, args.lr)

    fp32_accuracy = evaluate_accuracy(model, test_loader, device, use_fp16=False)

    model_fp16 = apply_fp16(model)
    accuracy = evaluate_accuracy(model_fp16, test_loader, device, use_fp16=True)
    avg_ms_per_batch, avg_ms_per_sample = measure_latency_ms(model_fp16, test_loader, device, use_fp16=True)
    peak_mem_bytes = measure_peak_memory_bytes(model_fp16, test_loader, device, use_fp16=True)
    params_total, params_nonzero = count_parameters(model_fp16)
    drop_pct = (fp32_accuracy - accuracy) * 100.0

    results: Dict[str, float] = {
        "test_accuracy": accuracy,
        "fp32_test_accuracy": fp32_accuracy,
        "accuracy_drop_pct": drop_pct,
        "latency_ms_per_batch": avg_ms_per_batch,
        "latency_ms_per_sample": avg_ms_per_sample,
        "peak_mem_bytes": peak_mem_bytes,
        "peak_mem_mb": peak_mem_bytes / (1024 ** 2),
        "params_total": params_total,
        "params_nonzero": params_nonzero,
    }

    print("Results:")
    for k, v in results.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
