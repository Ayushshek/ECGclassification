import os
import time
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def load_data(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    features = df.iloc[:, 0:16].values.astype(np.float32)
    labels = df.iloc[:, 16].values.astype(np.int64)
    return features, labels


def prepare_dataloaders(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 64,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


class EEGCNN(nn.Module):
    def __init__(self, num_features: int = 16, num_classes: int = 4) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5)
        self.relu = nn.ReLU()

        # Input length = 16
        # After conv1 (kernel=5, stride=1, padding=0): 16 - 5 + 1 = 12
        # After conv2 (kernel=5, stride=1, padding=0): 12 - 5 + 1 = 8
        conv_output_len = 8
        flattened_size = 64 * conv_output_len

        self.fc1 = nn.Linear(flattened_size, 128)
        self.out = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # (batch, 1, 16)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.out(x)
        return x


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct / total


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    process: psutil.Process,
    peak_memory: int,
) -> Tuple[float, int]:
    model.train()
    running_correct = 0
    running_total = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1)
        running_correct += (preds == batch_y).sum().item()
        running_total += batch_y.size(0)

        current_mem = process.memory_info().rss
        if current_mem > peak_memory:
            peak_memory = current_mem

    train_acc = running_correct / running_total
    return train_acc, peak_memory


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    running_correct = 0
    running_total = 0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            preds = torch.argmax(logits, dim=1)
            running_correct += (preds == batch_y).sum().item()
            running_total += batch_y.size(0)

    return running_correct / running_total


def main() -> None:
    device = torch.device("cpu")
    torch.manual_seed(42)
    np.random.seed(42)

    csv_path = os.path.join(os.path.dirname(__file__), "BEED_Data.csv")
    features, labels = load_data(csv_path)
    train_loader, test_loader = prepare_dataloaders(features, labels, batch_size=64)

    model = EEGCNN(num_features=16, num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    process = psutil.Process(os.getpid())
    peak_memory = process.memory_info().rss

    start_time = time.time()
    final_train_acc = 0.0

    for _ in range(50):
        final_train_acc, peak_memory = train_one_epoch(
            model, train_loader, criterion, optimizer, device, process, peak_memory
        )

    total_time = time.time() - start_time
    final_test_acc = evaluate(model, test_loader, device)

    peak_memory_mb = peak_memory / (1024 * 1024)

    results: Dict[str, float] = {
        "train_accuracy": final_train_acc,
        "test_accuracy": final_test_acc,
        "training_time_seconds": total_time,
        "peak_memory_mb": peak_memory_mb,
    }

    print(f"Final train accuracy: {final_train_acc:.4f}")
    print(f"Final test accuracy: {final_test_acc:.4f}")
    print(f"Total training time (s): {total_time:.2f}")
    print(f"Peak memory usage (MB): {peak_memory_mb:.2f}")
    print("Results dict:", results)


if __name__ == "__main__":
    main()
