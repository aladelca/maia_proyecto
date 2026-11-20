"""Masked channel encoders and temporal classifiers for five stages and movement."""

import torch
from torch import nn
from torch.nn import functional as F

from sleep_staging.models.tcn import TemporalConv


class FullNightNet(nn.Module):
    def __init__(self, view: str, head: str, width: int = 24):
        super().__init__()
        if view not in {"wave", "spectrum"} or head not in {"five", "six", "movement"}:
            raise ValueError("Unknown signal representation or output head.")
        self.view, self.head, self.width = view, head, width
        if view == "wave":
            self.encoder = nn.Sequential(
                nn.Conv1d(1, 8, 50, stride=25),
                nn.GELU(),
                nn.Conv1d(8, 8, 5, stride=3),
                nn.GELU(),
                nn.AvgPool1d(6),
                nn.Flatten(),
                nn.Linear(24, width),
            )
        else:
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 8, (3, 7), stride=(2, 3)),
                nn.GELU(),
                nn.Conv2d(8, 8, 3, stride=2),
                nn.GELU(),
                nn.AvgPool2d((3, 2)),
                nn.Flatten(),
                nn.Linear(24, width),
            )
        self.envelope = nn.Sequential(
            nn.Linear(30, width), nn.GELU(), nn.Linear(width, width)
        )
        self.fuse = nn.Sequential(
            nn.Linear(4 * width + 4, width), nn.LayerNorm(width), nn.GELU()
        )
        self.temporal = TemporalConv(width, 0.1)
        self.classifier = nn.Linear(width, 5 if head == "five" else 6)

    def encode(self, x: torch.Tensor, emg: torch.Tensor) -> torch.Tensor:
        batch = len(x)
        eeg_eog = self.encoder(x.reshape(batch * 3, 1, *x.shape[2:])).reshape(
            batch, 3, self.width
        )
        return torch.cat((eeg_eog, self.envelope(emg).unsqueeze(1)), dim=1)

    def forward(self, encoded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        per_epoch = torch.cat(
            ((encoded * mask.unsqueeze(-1)).flatten(1), mask.to(encoded.dtype)), dim=1
        )
        h = self.temporal.sequence(self.fuse(per_epoch).unsqueeze(0)).squeeze(0)
        return output_log_probs(self.classifier(h), self.head)


def output_log_probs(logits: torch.Tensor, head: str) -> torch.Tensor:
    if head == "movement":
        m = logits[:, 5:6]
        return torch.cat(
            (F.log_softmax(logits[:, :5], dim=1) + F.logsigmoid(-m), F.logsigmoid(m)),
            dim=1,
        )
    return F.log_softmax(logits, dim=1)


def predict_sequence(
    net: FullNightNet, enc: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Score one contiguous run of eligible EEG; callers split at gaps and night edges."""
    return net(enc, mask)
