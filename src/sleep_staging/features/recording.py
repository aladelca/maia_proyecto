"""Native-rate features shared by training and inference of the tabular champion."""

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from sleep_staging.data.edf import Signal
from sleep_staging.features.research import batch_signal_features

SIGNAL_CHANNELS = ["eeg_fpz_cz", "eeg_pz_oz", "eog", "emg"]
FPZ_FLAT = "eeg_fpz_cz__eeg__flat"


def recording_features(
    signals: dict[str, Signal], onset_seconds: np.ndarray
) -> pd.DataFrame:
    """Filter whole channels before slicing the requested 30-second epochs."""
    features: dict[str, np.ndarray] = {}
    for channel in SIGNAL_CHANNELS:
        signal = signals[channel]
        values = np.asarray(signal.values, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Señal no finita en {channel}.")
        if signal.kind != "emg_envelope":
            low, high = (
                (0.3, 35)
                if signal.kind == "eeg"
                else (0.3, 15)
                if signal.kind == "eog"
                else (10, 45)
            )
            values = sosfiltfilt(
                butter(4, [low, high], fs=signal.sfreq, btype="bandpass", output="sos"),
                values,
            )
        batches: dict[str, list[np.ndarray]] = {}
        samples = int(30 * signal.sfreq)
        for start in range(0, len(onset_seconds), 128):
            onsets = onset_seconds[start : start + 128]
            array = np.stack(
                [
                    values[
                        int(round(onset * signal.sfreq)) : int(
                            round(onset * signal.sfreq)
                        )
                        + samples
                    ].astype(np.float32)
                    for onset in onsets
                ]
            )
            if array.shape[1] != samples:
                raise ValueError("Incomplete signal epoch.")
            for name, column in batch_signal_features(
                array, signal.sfreq, signal.kind
            ).items():
                batches.setdefault(name, []).append(column.astype(np.float32))
        for name, parts in batches.items():
            features[f"{channel}__{signal.kind}__{name}"] = np.concatenate(parts)
        del values, signal
    return pd.DataFrame(features)
