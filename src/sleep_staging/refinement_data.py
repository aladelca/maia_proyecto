"""Label-independent retrospective feature views and tuning metrics."""

import numpy as np
import pandas as pd

from sleep_staging.features.research import temporal_features
from sleep_staging.lstm_data import FeatureScaler, SequenceData


def class_weights(labels: np.ndarray, power: float) -> np.ndarray:
    counts = np.bincount(labels, minlength=5)
    if (
        len(counts) != 5
        or not len(labels)
        or not (counts > 0).all()
        or not 0 <= power <= 1
    ):
        raise ValueError("Five nonempty classes and weight power in [0, 1] required.")
    return (len(labels) / (5 * counts)) ** power


def macro_f1_eval(
    y_true: np.ndarray | None, probabilities: np.ndarray
) -> tuple[str, float, bool]:
    if y_true is None:
        raise ValueError("Validation labels are required.")
    if probabilities.ndim != 2 or probabilities.shape != (len(y_true), 5):
        raise ValueError("Expected five-class prediction probabilities.")
    confusion = np.bincount(
        5 * y_true.astype(int) + probabilities.argmax(axis=1), minlength=25
    ).reshape(5, 5)
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    scores = np.divide(
        2 * np.diag(confusion), denominator, out=np.zeros(5), where=denominator > 0
    )
    return "macro_f1", float(scores.mean()), True


def normalize_recordings(data: SequenceData) -> SequenceData:
    """Whole-record signal-feature adaptation, using no labels or other records."""
    values = np.empty_like(data.values)
    for positions in data.frame.groupby("recording_id", sort=False).indices.values():
        values[positions] = FeatureScaler.fit(data.values[positions]).transform(
            data.values[positions]
        )
    frame = pd.concat(
        [
            data.frame.drop(columns=data.columns),
            pd.DataFrame(values, columns=data.columns, index=data.frame.index),
        ],
        axis=1,
    )
    return SequenceData.from_frame(frame, data.columns)


def refinement_views(data: SequenceData) -> dict[str, SequenceData]:
    context = temporal_features(data.frame, data.columns)
    full = SequenceData.from_frame(
        pd.concat([data.frame, context], axis=1), data.columns + list(context)
    )
    return {
        "raw145": data,
        "night145": normalize_recordings(data),
        "context435": full,
        "night435": normalize_recordings(full),
    }
