import argparse
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EEGCNN(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        # After Conv1d(5) -> length 12, then Conv1d(5) -> length 8
        # Flatten size = 64 * 8 = 512
        self.fc1 = nn.Linear(64 * 8, 128)
        self.fc_out = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape: (batch, 1, 16)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc_out(x)
        return x


def apply_unstructured_pruning(model: nn.Module, amount: float = 0.2):
    prune.l1_unstructured(model.conv1, name="weight", amount=amount)
    prune.l1_unstructured(model.conv2, name="weight", amount=amount)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    torch.cuda.synchronize()
    start = time.time()

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)

    torch.cuda.synchronize()
    end = time.time()

    accuracy = correct / total
    avg_latency_ms = ((end - start) / total) * 1000.0  # per-sample latency in ms
    return accuracy, avg_latency_ms


def count_nonzero_params(model: nn.Module):
    nonzero = 0
    total = 0
    for name, module in model.named_modules():
        if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
            w = module.weight.detach()
            nonzero += torch.count_nonzero(w).item()
            total += w.numel()
        if hasattr(module, "bias") and isinstance(module.bias, torch.Tensor) and module.bias is not None:
            b = module.bias.detach()
            nonzero += torch.count_nonzero(b).item()
            total += b.numel()
    return int(nonzero), int(total)


def main(csv_path: str):
    set_seeds(42)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available. Aborting per instructions.")

    device = torch.device("cuda")

    # Load data
    df = pd.read_csv(csv_path)
    X = df.iloc[:, 0:16].astype(np.float32).values
    y = df.iloc[:, 16].astype(np.int64).values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardize (fit on train only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to tensors (keep on CPU; move to GPU inside loop)
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    # Reshape for Conv1d: (batch, 1, 16)
    X_train_t = X_train_t.view(-1, 1, 16)
    X_test_t = X_test_t.view(-1, 1, 16)

    train_ds = TensorDataset(X_train_t, y_train_t)
    test_ds = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, pin_memory=True)

    model = EEGCNN(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Baseline training
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    train_start = time.time()

    for epoch in range(20):
        train_one_epoch(model, train_loader, criterion, optimizer, device)

    # One-shot pruning after baseline training
    apply_unstructured_pruning(model, amount=0.2)

    # Retraining after pruning
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(10):
        train_one_epoch(model, train_loader, criterion, optimizer, device)

    torch.cuda.synchronize()
    train_end = time.time()
    total_training_time_sec = train_end - train_start

    # Evaluation
    test_accuracy, avg_latency_ms = evaluate(model, test_loader, device)
    peak_mem_bytes = torch.cuda.max_memory_allocated()
    nonzero_params, total_params = count_nonzero_params(model)

    results = {
        "test_accuracy": test_accuracy,
        "avg_inference_latency_ms": avg_latency_ms,
        "peak_gpu_memory_bytes": peak_mem_bytes,
        "nonzero_params": nonzero_params,
        "total_params": total_params,
        "total_training_time_sec": total_training_time_sec,
    }

    print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unstructured magnitude pruning on BEED dataset")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to BEED CSV file")
    args = parser.parse_args()
    main(args.csv_path)
