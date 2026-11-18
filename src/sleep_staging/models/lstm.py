"""Compact feature-sequence LSTM with an explicit MPS execution contract."""

import os

import torch
from torch import nn


def require_mps() -> torch.device:
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        raise ValueError(
            "CPU fallback is forbidden; set PYTORCH_ENABLE_MPS_FALLBACK=0 before importing torch."
        )
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise ValueError("This experiment requires an available MPS device.")
    return torch.device("mps")


class SleepLSTM(nn.Module):
    def __init__(
        self,
        inputs: int,
        hidden_size: int,
        dropout: float,
        bidirectional: bool,
        target_position: int,
    ) -> None:
        super().__init__()
        self.target_position = target_position
        self.projection = nn.Sequential(nn.Linear(inputs, hidden_size), nn.ReLU())
        self.recurrent = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0,
            proj_size=0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden_size * (2 if bidirectional else 1), 5)
        )

    def target_representation(self, output: torch.Tensor) -> torch.Tensor:
        return output[:, self.target_position, :]

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        output, _ = self.recurrent(self.projection(values))
        return self.head(self.target_representation(output))
