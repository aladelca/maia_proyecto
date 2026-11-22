"""Matched EEG + EOG reference using the published Braindecode U-Sleep module."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from sleep_staging.fullnight.data import RATE, Inputs, valid_blocks
from sleep_staging.fullnight.training import Night

PROFILE = "eeg_fpz_cz+eog"


@dataclass(frozen=True)
class Budget:
    passes: int = 2
    windows_per_night: int = 4
    window: int = 35
    batch: int = 4
    inference_chunk: int = 96
    halo: int = 32
    seed: int = 42

    def __post_init__(self) -> None:
        if (
            min(
                self.passes,
                self.windows_per_night,
                self.window,
                self.batch,
                self.inference_chunk,
            )
            < 1
            or self.halo < 0
        ):
            raise ValueError("Presupuesto U-Sleep inválido.")


def build() -> nn.Module:
    from braindecode.models import USleep

    return USleep(
        n_chans=2,
        n_outputs=5,
        sfreq=RATE,
        input_window_seconds=30,
        ensure_odd_conv_size=True,
        depth=12,
        n_time_filters=5,
        complexity_factor=1.67,
    )


def windows(night: Night, length: int) -> list[np.ndarray]:
    # A sampled epoch is the centre of a window entirely inside a valid EEG block.
    choices = np.zeros(len(night.y), dtype=bool)
    half = length // 2
    for a, b in valid_blocks(night.inp.available[:, 0]):
        if b - a >= length:
            choices[a + half : b - length + half + 1] = True
    return [np.flatnonzero(choices & (night.y == c)) - half for c in range(5)]


def pair(x: Inputs, start: int, stop: int) -> np.ndarray:
    return np.asarray(x.wave[start:stop, [0, 2], :], dtype=np.float32)


def fit(nights: list[Night], cfg: Budget, seed: int, device: torch.device) -> nn.Module:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = build().to(device).train()
    opt = torch.optim.Adam(net.parameters(), lr=0.001)
    by_night = [
        [starts for starts in windows(n, cfg.window) if starts.size] for n in nights
    ]
    for _ in range(cfg.passes):
        batch_starts: list[tuple[int, int]] = []
        for i, groups in enumerate(by_night):
            if groups:
                for _ in range(cfg.windows_per_night):
                    starts = groups[rng.integers(len(groups))]
                    batch_starts.append((i, int(rng.choice(starts))))
        if not batch_starts:
            raise ValueError("Sin ventanas de entrenamiento para U-Sleep.")
        rng.shuffle(batch_starts)
        for j in range(0, len(batch_starts), cfg.batch):
            xs, ys = [], []
            for i, start in batch_starts[j : j + cfg.batch]:
                n = nights[i]
                xs.append(pair(n.inp, start, start + cfg.window))
                y = n.y[start : start + cfg.window].astype("int64")
                ys.append(np.where((y >= 0) & (y < 5), y, -100))
            x = torch.from_numpy(np.stack(xs)).to(device)
            labels = torch.from_numpy(np.stack(ys)).to(device)
            logits = net(x).reshape(len(xs), 5, cfg.window)
            loss = nn.functional.cross_entropy(logits, labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return net.eval()


@torch.inference_mode()
def predict(net: nn.Module, x: Inputs, cfg: Budget, device: torch.device) -> np.ndarray:
    """Discard overlapping context at chunk edges; stop at every invalid EEG epoch."""
    net.eval()
    p = np.full((len(x.wave), 5), np.nan, dtype=np.float32)
    for a, b in valid_blocks(x.available[:, 0]):
        for start in range(a, b, cfg.inference_chunk):
            end = min(start + cfg.inference_chunk, b)
            left, right = max(a, start - cfg.halo), min(b, end + cfg.halo)
            signal = torch.from_numpy(pair(x, left, right)).unsqueeze(0).to(device)
            logits = net(signal).reshape(5, right - left)
            p[start:end] = (
                logits.softmax(0)[:, start - left : end - left].T.cpu().numpy()
            )
    return p
