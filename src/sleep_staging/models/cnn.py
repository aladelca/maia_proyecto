"""CNN pequeña por época; normalización ajustada solo en entrenamiento."""

import copy
from typing import Any, Self

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class CNNClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self, epochs: int = 3, batch_size: int = 32, device: str = "cpu", seed: int = 42
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.seed = seed

    def fit(self, X: Any, y: np.ndarray, groups: np.ndarray | None = None) -> Self:
        torch.set_num_threads(2)
        torch.manual_seed(self.seed)
        self.classes_, encoded = np.unique(y, return_inverse=True)
        x = np.asarray(X, dtype="float32")
        self.n_features_in_ = x.shape[1]
        training = np.arange(len(x))
        validation = np.array([], dtype=int)
        if groups is not None:
            training, validation = next(
                GroupShuffleSplit(
                    n_splits=1, test_size=0.2, random_state=self.seed
                ).split(x, y, groups)
            )
        self.mean_ = float(x[training].mean())
        self.std_ = max(float(x[training].std()), 1e-6)
        values = torch.from_numpy(((x - self.mean_) / self.std_)[:, None, :])
        target = torch.from_numpy(encoded.astype("int64"))
        if self.device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS no disponible; selecciona device: cpu.")
        self.device_ = (
            "mps"
            if self.device in {"auto", "mps"} and torch.backends.mps.is_available()
            else "cpu"
        )
        self.network_ = nn.Sequential(
            nn.Conv1d(1, 16, 25, stride=5, padding=12),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(16, 32, 9, padding=4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(6),
            nn.Flatten(),
            nn.Linear(192, len(self.classes_)),
        ).to(self.device_)
        optimizer = torch.optim.Adam(self.network_.parameters(), lr=0.001)
        counts = np.bincount(encoded[training], minlength=len(self.classes_))
        weights = len(training) / (len(self.classes_) * np.maximum(counts, 1))
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32, device=self.device_)
        )
        loader = DataLoader(
            TensorDataset(values[training], target[training]),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            generator=torch.Generator().manual_seed(self.seed),
        )
        best_loss = float("inf")
        best_state = None
        self.best_epoch_ = self.epochs
        self.history_ = []
        for epoch in range(self.epochs):
            self.network_.train()
            losses = []
            for batch, label in loader:
                optimizer.zero_grad()
                loss = criterion(
                    self.network_(batch.to(self.device_)), label.to(self.device_)
                )
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            validation_loss = float(np.mean(losses))
            if len(validation):
                self.network_.eval()
                with torch.no_grad():
                    chunks = []
                    for start in range(0, len(validation), self.batch_size):
                        indices = validation[start : start + self.batch_size]
                        chunks.append(
                            float(
                                criterion(
                                    self.network_(values[indices].to(self.device_)),
                                    target[indices].to(self.device_),
                                ).cpu()
                            )
                        )
                    validation_loss = float(np.mean(chunks))
                if validation_loss < best_loss:
                    best_loss = validation_loss
                    best_state = copy.deepcopy(self.network_.state_dict())
                    self.best_epoch_ = epoch + 1
            self.history_.append(
                {
                    "epoch": epoch + 1,
                    "training_loss": float(np.mean(losses)),
                    "validation_loss": validation_loss,
                }
            )
            if len(validation) and epoch + 1 - self.best_epoch_ >= 3:
                break
        if best_state is not None:
            self.network_.load_state_dict(best_state)
        self.network_.to("cpu").eval()
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        x = np.asarray(X, dtype="float32")
        if x.ndim != 2 or x.shape[1] != self.n_features_in_:
            raise ValueError("Número de muestras incompatible con el modelo.")
        values = torch.from_numpy(((x - self.mean_) / self.std_)[:, None, :])
        chunks = []
        with torch.no_grad():
            for batch in values.split(self.batch_size):
                chunks.append(torch.softmax(self.network_(batch), dim=1).numpy())
        return np.concatenate(chunks)

    def predict(self, X: Any) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
