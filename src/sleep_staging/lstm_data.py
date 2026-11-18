"""Signal-only feature windows with explicit recording and time boundaries."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sleep_staging.config import CLASSES
from sleep_staging.research_data import SIGNAL_CHANNELS


def sequence_index(
    frame: pd.DataFrame, length: int, centered: bool
) -> tuple[np.ndarray, np.ndarray]:
    if length < 1 or centered and length % 2 != 1:
        raise ValueError("Window length must be positive and centered windows odd.")
    if frame.duplicated(["recording_id", "onset_s"]).any():
        raise ValueError("Duplicate recording/onset.")
    positions = pd.Series(np.arange(len(frame)), index=frame.index)
    indices = np.empty((len(frame), length), dtype=np.int32)
    padding = np.empty((len(frame), length), dtype=bool)
    offsets = np.arange(length) - (length // 2 if centered else length - 1)
    for _, night in frame.groupby("recording_id", sort=True):
        night = night.sort_values("onset_s")
        for _, block in night.groupby(night.onset_s.diff().ne(30).cumsum(), sort=False):
            rows = positions.loc[block.index].to_numpy()
            neighbors = np.arange(len(rows))[:, None] + offsets
            padding[rows] = (neighbors < 0) | (neighbors >= len(rows))
            indices[rows] = rows[np.clip(neighbors, 0, len(rows) - 1)]
    return indices, padding


@dataclass
class FeatureScaler:
    median: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "FeatureScaler":
        if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
            raise ValueError("Scaler requires nonempty finite training features.")
        quantiles = np.quantile(values, [0.05, 0.5, 0.95], axis=0)
        return cls(
            quantiles[1].astype(np.float32),
            np.maximum((quantiles[2] - quantiles[0]) / 3.29, 1e-6).astype(np.float32),
        )

    def transform(self, values: np.ndarray) -> np.ndarray:
        if values.ndim != 2 or values.shape[1] != len(self.median):
            raise ValueError("Incompatible feature columns for scaler.")
        return np.clip((values - self.median) / self.scale, -10, 10).astype(np.float32)

    def to_dict(self) -> dict:
        return {
            "median": self.median.tolist(),
            "scale": self.scale.tolist(),
            "clip": 10,
        }

    @classmethod
    def from_dict(cls, value: dict) -> "FeatureScaler":
        return cls(
            np.asarray(value["median"], dtype=np.float32),
            np.asarray(value["scale"], dtype=np.float32),
        )


@dataclass
class SequenceData:
    frame: pd.DataFrame
    values: np.ndarray
    columns: list[str]
    targets: np.ndarray
    labels: np.ndarray

    @classmethod
    def from_frame(
        cls, frame: pd.DataFrame, columns: list[str] | None = None
    ) -> "SequenceData":
        frame = frame.sort_values(["recording_id", "onset_s"]).reset_index(drop=True)
        if (
            frame.epoch_id.duplicated().any()
            or frame.duplicated(["recording_id", "onset_s"]).any()
        ):
            raise ValueError("Duplicate epoch identities.")
        if frame.groupby("recording_id").subject_id.nunique().max() != 1:
            raise ValueError("Each recording must belong to one subject.")
        if frame.groupby("subject_id").fold.nunique().max() != 1:
            raise ValueError("Each subject must belong to one fold.")
        eligible = [
            c for c in frame if any(c.startswith(ch + "__") for ch in SIGNAL_CHANNELS)
        ]
        columns = columns if columns is not None else eligible
        if (
            not columns
            or len(set(columns)) != len(columns)
            or set(columns) - set(eligible)
        ):
            raise ValueError("Only unique signal feature columns are allowed.")
        values = frame[columns].to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Expected finite signal features.")
        classes = CLASSES["task_5"]
        if set(frame.task_5.dropna()) - set(classes):
            raise ValueError("Unexpected supervised classes.")
        labels = (
            frame.task_5.map({name: i for i, name in enumerate(classes)})
            .fillna(-1)
            .to_numpy(dtype=np.int64)
        )
        return cls(frame, values, columns, np.flatnonzero(labels >= 0), labels)
