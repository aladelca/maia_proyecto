"""Epoch encoders and three matched sequence heads for offline sleep staging."""

import torch
from torch import nn

from sleep_staging.models.lstm import SleepLSTM


def research_device(requested: str) -> torch.device:
    if requested == "auto":
        for name, available in (
            ("mps", torch.backends.mps.is_available),
            ("cuda", torch.cuda.is_available),
        ):
            if available():
                return torch.device(name)
        raise ValueError("No MPS/CUDA device detected. Use --device cpu for a CPU run.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable in this PyTorch process.")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in this PyTorch process.")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported device: {requested}")
    return torch.device(requested)


class WaveEncoder(nn.Module):
    """One embedding per 30-second epoch, learned with single-epoch supervision."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.filters = nn.Sequential(
            nn.Conv1d(3, 16, 25, stride=5, padding=12),
            nn.GELU(),
            nn.MaxPool1d(4),
            nn.Conv1d(16, 32, 9, stride=2, padding=4),
            nn.GELU(),
            nn.MaxPool1d(3),
            nn.Conv1d(32, width, 5, padding=2),
            nn.GELU(),
        )
        self.classifier = nn.Linear(width, 5)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.filters(x).mean(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.embed(x))


class TemporalConv(nn.Module):
    """Three symmetric dilations cover exactly 15 epochs at the center."""

    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.dilations = (1, 2, 4)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.Linear(3 * width, width), nn.GELU(), nn.Dropout(dropout))
            for _ in self.dilations
        )

    def sequence(self, x: torch.Tensor) -> torch.Tensor:
        # A shared linear map over three offset taps is a dilated convolution.
        # Keeping [batch, time, channels] also supports single-epoch inputs on Metal.
        length = x.shape[1]
        for dilation, block in zip(self.dilations, self.blocks, strict=True):
            padded = nn.functional.pad(x, (0, 0, dilation, dilation))
            taps = torch.cat(
                [
                    padded[:, shift : shift + length]
                    for shift in (0, dilation, 2 * dilation)
                ],
                dim=-1,
            )
            x = x + block(taps)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sequence(x)[:, x.shape[1] // 2]


class SequenceHead(nn.Module):
    def __init__(
        self, kind: str, inputs: int, width: int, length: int, dropout: float
    ) -> None:
        super().__init__()
        self.kind = kind
        self.length = length
        if kind == "bilstm":
            self.recurrent = SleepLSTM(inputs, width, dropout, True, length // 2)
        elif kind in {"transformer", "tcn"}:
            self.project = nn.Linear(inputs, width)
            if kind == "tcn":
                self.temporal = TemporalConv(width, dropout)
            else:
                self.position = nn.Parameter(torch.zeros(1, length, width))
                nn.init.normal_(self.position, std=0.02)
                self.attention = nn.ModuleList(
                    AttentionBlock(width, dropout) for _ in range(2)
                )
            self.stage = nn.Sequential(nn.Dropout(dropout), nn.Linear(width, 5))
            if kind == "transformer":
                self.boundary = nn.Linear(width, 1)
        else:
            raise ValueError(f"Unknown temporal architecture: {kind}")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.kind == "bilstm":
            return self.recurrent(x), None
        h = self.project(x)
        if self.kind == "tcn":
            return self.stage(self.temporal(h)), None
        h = h + self.position
        for block in self.attention:
            h = block(h)
        center = h[:, self.length // 2]
        return self.stage(center), self.boundary(center).squeeze(-1)


class AttentionBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(
            width, num_heads=4, dropout=dropout, batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, 2 * width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * width, width),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        attended, _ = self.attention(h, h, h, need_weights=True)
        x = x + self.dropout(attended)
        return x + self.dropout(self.feed_forward(x))
