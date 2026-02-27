#!/usr/bin/env python3
import argparse
import random
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class EEGCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5)
        # 16 -> 12 -> 8, flatten size = 64 * 8 = 512
        self.fc1 = nn.Linear(64 * 8, 128)
        self.fc_out = nn.Linear(128, 4)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc_out(x)
        return x


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    X = df.iloc[:, :16].values.astype(np.float32)
    y = df.iloc[:, 16].values.astype(np.int64)
    return X, y


def prepare_dataloaders(X, y, batch_size=64):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).long()
    X_test_tensor = torch.from_numpy(X_test).float()
    y_test_tensor = torch.from_numpy(y_test).long()

    train_ds = TensorDataset(X_train_tensor, y_train_tensor)
    test_ds = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        X_batch = X_batch.view(-1, 1, 16)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(outputs, dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    return correct / total if total > 0 else 0.0


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    total_latency_ms = 0.0
    latency_batches = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            X_batch = X_batch.view(-1, 1, 16)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            outputs = model(X_batch)
            end_event.record()
            torch.cuda.synchronize()
            total_latency_ms += start_event.elapsed_time(end_event)
            latency_batches += 1
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    avg_latency_ms = total_latency_ms / latency_batches if latency_batches > 0 else 0.0
    return (correct / total if total > 0 else 0.0), avg_latency_ms


def main():
    parser = argparse.ArgumentParser(description="PyTorch CNN for BEED dataset (CUDA only)")
    parser.add_argument("--data", type=str, required=True, help="Path to BEED CSV file")
    args = parser.parse_args()

    set_seed(42)

    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available. Device forced to 'cuda'.")

    X, y = load_data(args.data)
    train_loader, test_loader = prepare_dataloaders(X, y, batch_size=64)

    model = EEGCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_time = time.time()

    final_train_acc = 0.0
    final_test_acc = 0.0
    avg_gpu_latency_ms = 0.0

    for _ in range(20):
        final_train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        final_test_acc, avg_gpu_latency_ms = evaluate(model, test_loader, device)

    torch.cuda.synchronize()
    total_time = time.time() - start_time

    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    total_params = sum(p.numel() for p in model.parameters())

    results = {
        "final_train_accuracy": final_train_acc,
        "final_test_accuracy": final_test_acc,
        "training_time_seconds": total_time,
        "peak_gpu_memory_mb": peak_mem_mb,
        "avg_gpu_latency_ms": avg_gpu_latency_ms,
        "total_params": total_params,
    }

    print(f"Final train accuracy: {final_train_acc:.4f}")
    print(f"Final test accuracy: {final_test_acc:.4f}")
    print(f"Total training time (s): {total_time:.2f}")
    print(f"Peak GPU memory usage (MB): {peak_mem_mb:.2f}")
    print(f"Average GPU latency (ms): {avg_gpu_latency_ms:.3f}")
    print(f"Total parameters: {total_params}")


if __name__ == "__main__":
    main()
