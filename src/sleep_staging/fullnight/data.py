"""Label-free, complete 30-second inputs; labels are joined after preparation."""

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

from sleep_staging.config import CLASSES, channel_profiles
from sleep_staging.data.edf import Signal
from sleep_staging.data.labels import epoch_annotations

CHANNELS = ("eeg_fpz_cz", "eeg_pz_oz", "eog", "emg")
PROFILES = channel_profiles()
LABELS = CLASSES["task_6"]
RATE = 50
FEATURES = ("mean", "std", "range", "step", "delta", "theta", "alpha", "sigma", "beta")


def profile_bits(name: str) -> np.ndarray:
    if name not in PROFILES:
        raise ValueError("Perfil inválido. Se requiere al menos un EEG compatible.")
    return np.array([ch in PROFILES[name] for ch in CHANNELS], dtype=bool)


def valid_blocks(valid: np.ndarray) -> list[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(np.r_[False, valid, False]))
    return list(zip(edges[::2], edges[1::2], strict=True))


@dataclass
class Inputs:
    wave: np.ndarray
    spectrum: np.ndarray
    emg: np.ndarray
    available: np.ndarray
    features: np.ndarray

    def eligible(self, bits: np.ndarray) -> np.ndarray:
        return (self.available[:, :2] & bits[:2]).any(axis=1)

    def save(self, directory: Path) -> None:
        for f in fields(self):
            np.save(directory / f.name, getattr(self, f.name), allow_pickle=False)

    @classmethod
    def load(cls, directory: Path) -> "Inputs":
        return cls(
            **{
                f.name: np.load(directory / (f.name + ".npy"), mmap_mode="r")
                for f in fields(cls)
            }
        )


def make_inputs(signals: dict[str, Signal], duration: float) -> Inputs:
    n = int(duration // 30)
    if not n or not set(CHANNELS[:2]).intersection(signals):
        raise ValueError("Se requiere al menos una época completa y un EEG.")
    wave = np.zeros((n, 3, 1500), dtype=np.float16)
    spectra = np.zeros((n, 3, 15, 50), dtype=np.float16)
    emg = np.zeros((n, 30), dtype=np.float16)
    available = np.zeros((n, 4), dtype=bool)
    features = np.full((n, 4, len(FEATURES)), np.nan, dtype=np.float32)
    window = np.hanning(100).astype(np.float32)
    hz = np.fft.rfftfreq(100, 1 / RATE)[1:]
    for c, name in enumerate(CHANNELS):
        if name not in signals:
            continue
        signal = signals[name]
        rate = signal.sfreq
        if rate not in ((1, 100) if c == 3 else (100,)):
            raise ValueError(f"Frecuencia no admitida para {name}: {rate}")
        count = int(rate * 30)
        if len(signal.values) < n * count:
            raise ValueError(f"Duración incompatible para {name}")
        src = signal.values[: n * count].reshape(n, count)
        good = np.isfinite(src).all(axis=1)
        if c < 3:
            good &= np.ptp(src, axis=1) > 1e-8
        values = np.where(good[:, None], src, 0).astype(np.float32)
        available[:, c] = good
        if c == 3 and rate == 100:
            filtered = sosfiltfilt(
                butter(4, [10, 45], fs=100, btype="bandpass", output="sos"),
                values,
                axis=1,
            )
            values = np.sqrt(np.mean(filtered.reshape(n, 30, 100) ** 2, axis=2)).astype(
                np.float32
            )
        elif c < 3:
            # Resample each epoch independently so a damaged epoch cannot affect a neighbour.
            values = resample_poly(values, 1, 2, axis=1, padtype="line").astype(
                np.float32
            )
        features[:, c, :4] = np.stack(
            (
                values.mean(1),
                values.std(1),
                np.ptp(values, axis=1),
                np.abs(np.diff(values, axis=1)).mean(1),
            ),
            axis=1,
        )
        if good.any():
            sample = values[good, :: max(1, values.shape[1] // 30)]
            lo, med, hi = np.percentile(sample, [5, 50, 95])
            scale = max(float((hi - lo) / 3.29), 1.0)
            z = np.clip((values - med) / scale, -20, 20)
            z[~good] = 0
            if c == 3:
                emg[:] = z
            else:
                wave[:, c] = z
                blocks = values.reshape(n, 15, 100) * window
                power = np.abs(np.fft.rfft(blocks, axis=-1)[:, :, 1:]) ** 2 / np.sum(
                    window**2
                )
                spectra[:, c] = np.log1p(power / scale**2)
                for j, (low, high) in enumerate(
                    ((0.5, 4), (4, 8), (8, 12), (12, 16), (16, 25.1)), 4
                ):
                    features[:, c, j] = np.log1p(
                        power[:, :, (hz >= low) & (hz < high)].mean(axis=(1, 2))
                    )
        features[~good, c] = np.nan
    return Inputs(wave, spectra, emg, available, features)


def targets(
    annotations: list[tuple[float, float, str]], duration: float
) -> tuple[np.ndarray, np.ndarray]:
    frame = epoch_annotations(annotations, duration)
    frame = frame[frame.duration_s.eq(30)]
    y = (
        frame.task_6.map({s: i for i, s in enumerate(LABELS)})
        .fillna(-1)
        .to_numpy(dtype=np.int8)
    )
    sleep = frame.onset_s[(y > 0) & (y < 5)]
    crop = frame.onset_s.between(sleep.min() - 1800, sleep.max() + 1800).to_numpy()
    return y, crop
