"""Retrospective five-stage inference from an unannotated Sleep Cassette EDF."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sleep_staging.data.edf import ALIASES, read_psg, validated_edf_channels
from sleep_staging.features.recording import (
    FPZ_FLAT,
    SIGNAL_CHANNELS,
    recording_features,
)
from sleep_staging.serving.contract import KEYS, LABELS, SCHEMA, predictions


class IncompatibleEDF(ValueError):
    """Input error that can be shown to the uploader."""


def prepare_edf(path: Path, recording_id: str) -> pd.DataFrame:
    infos = validated_edf_channels(path)
    for label, canonical in ALIASES.items():
        matches = [info for info in infos if info["name"] == label]
        rate = 1 if canonical == "emg" else 100
        if not matches:
            raise IncompatibleEDF(f"LightGBM requiere el canal {label}.")
        if len(matches) != 1:
            raise IncompatibleEDF(f"Canal duplicado: {label}.")
        if matches[0]["sfreq"] != rate:
            raise IncompatibleEDF(f"LightGBM requiere {label} a {rate} Hz.")
        if matches[0]["unit"] not in {"uV", "µV", "μV", "mV", "V"}:
            raise IncompatibleEDF(f"Unidad incompatible para {label}.")
    signals, _, duration = read_psg(path, channels=SIGNAL_CHANNELS)
    onsets = np.arange(int(duration // 30), dtype=np.int64) * 30
    features = recording_features(signals, onsets)
    if not np.isfinite(features.to_numpy()).all():
        raise IncompatibleEDF("El EDF contiene características no finitas.")
    keys = pd.DataFrame(
        {
            "recording_id": recording_id,
            "epoch_id": [f"{recording_id}:{t * 100}" for t in onsets],
            "onset_s": onsets,
            "eligible": ~features[FPZ_FLAT].astype(bool),
        }
    )
    return pd.concat([keys, features], axis=1)


def score_night(model: Any, manifest: dict, frame: pd.DataFrame) -> pd.DataFrame:
    if (
        manifest["preprocessing_id"] != "sc-w30-context435-v1"
        or manifest["input_schema"] != SCHEMA
        or manifest["classes"] != LABELS
        or set(frame) != set(KEYS + manifest["raw_columns"])
    ):
        raise RuntimeError("La release LightGBM no corresponde al protocolo EDF.")
    # Flat Fpz epochs were absent from training, including temporal normalization.
    valid = predictions(model, manifest, frame.loc[frame.eligible])
    return frame[KEYS].join(valid.drop(columns=KEYS))
