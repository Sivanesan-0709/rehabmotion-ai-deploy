"""
AI Exercise Form & Posture Intelligence Platform
Temporal Deep Learning Sequence Classifier (PyTorch LSTM)
"""

import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from app.config import MODELS_DIR


class LSTMSequenceNet(nn.Module):
    """Deep Recurrent Neural Network for temporal kinematics sequence modeling."""

    def __init__(
        self,
        input_dim: int = 68,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 5,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, seq_len, input_dim]
        lstm_out, _ = self.lstm(x)
        # Use last timestep representation
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden)


class PyTorchTemporalModel:
    """Wrapper for training, evaluating, and benchmarking LSTM sequence models."""

    def __init__(
        self,
        input_dim: int = 68,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 5,
        device: Optional[str] = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = LSTMSequenceNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
        ).to(self.device)
        self.is_trained = False
        self.num_classes = num_classes

    def train(
        self,
        X_train_seq: np.ndarray,
        y_train_seq: np.ndarray,
        epochs: int = 25,
        batch_size: int = 32,
        lr: float = 0.002,
    ) -> float:
        """
        Trains temporal LSTM network on 3D sequence tensors [N, Seq_Len, Features].
        Returns training duration in seconds.
        """
        t0 = time.perf_counter()
        self.model.train()

        tensor_x = torch.tensor(X_train_seq, dtype=torch.float32)
        tensor_y = torch.tensor(y_train_seq, dtype=torch.long)

        dataset = TensorDataset(tensor_x, tensor_y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        duration = time.perf_counter() - t0
        self.is_trained = True
        return duration

    def predict(self, X_seq: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            tensor_x = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
            outputs = self.model(tensor_x)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
        return preds

    def predict_proba(self, X_seq: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            tensor_x = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
            outputs = self.model(tensor_x)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
        return probs

    def benchmark_inference(self, sample_seq: np.ndarray, num_iterations: int = 100) -> Tuple[float, float]:
        self.model.eval()
        if len(sample_seq.shape) == 2:
            sample_seq = np.expand_dims(sample_seq, axis=0)

        tensor_x = torch.tensor(sample_seq, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            # Warm up
            for _ in range(10):
                _ = self.model(tensor_x)

            t0 = time.perf_counter()
            for _ in range(num_iterations):
                _ = self.model(tensor_x)
            total_time = time.perf_counter() - t0

        latency_ms = (total_time / num_iterations) * 1000.0
        fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
        return round(latency_ms, 2), round(fps, 1)

    def save(self, filepath: Optional[str] = None):
        target = filepath or str(MODELS_DIR / "lstm_temporal_model.pt")
        torch.save(self.model.state_dict(), target)

    def load(self, filepath: Optional[str] = None):
        target = filepath or str(MODELS_DIR / "lstm_temporal_model.pt")
        self.model.load_state_dict(torch.load(target, map_location=self.device))
        self.model.eval()
        self.is_trained = True
