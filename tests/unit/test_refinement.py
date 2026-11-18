import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from sleep_staging.lstm_data import SequenceData
from sleep_staging.lstm_research import LSTMConfig
from sleep_staging.refinement_data import (
    class_weights,
    macro_f1_eval,
    normalize_recordings,
    refinement_views,
)


def sample_frame():
    return pd.DataFrame(
        {
            "epoch_id": [f"e{i}" for i in range(12)],
            "recording_id": ["a"] * 6 + ["b"] * 6,
            "subject_id": ["a"] * 6 + ["b"] * 6,
            "fold": [0] * 6 + [1] * 6,
            "onset_s": list(range(0, 180, 30)) * 2,
            "task_5": ["W", "N1", None, "N2", "N3", "REM"] * 2,
            "eeg_fpz_cz__eeg__std": np.r_[np.arange(6), 100 + np.arange(6)].astype(
                np.float32
            ),
        }
    )


def test_recording_normalization_is_independent_of_targets_and_other_nights():
    data = SequenceData.from_frame(sample_frame())
    expected = normalize_recordings(data)
    changed = data.frame.copy()
    changed["task_5"] = "REM"
    changed.loc[changed.recording_id.eq("b"), data.columns] *= 1e5
    actual = normalize_recordings(SequenceData.from_frame(changed))
    np.testing.assert_allclose(expected.values[:6], actual.values[:6])
    np.testing.assert_allclose(np.median(expected.values[:6], axis=0), 0, atol=1e-6)


def test_all_feature_views_preserve_targets_and_the_unscored_context_row():
    data = SequenceData.from_frame(sample_frame())
    views = refinement_views(data)
    assert set(views) == {"raw145", "night145", "context435", "night435"}
    for view in views.values():
        assert view.frame.epoch_id.tolist() == data.frame.epoch_id.tolist()
        np.testing.assert_array_equal(view.targets, data.targets)
        assert len(view.values) == 12 and np.isfinite(view.values).all()
    assert views["context435"].values.shape[1] == 3


def test_macro_callback_matches_unweighted_fixed_class_f1():
    y = np.array([0, 0, 0, 1, 2, 3, 4])
    predicted = np.array([0, 1, 0, 1, 2, 2, 4])
    p = np.eye(5)[predicted]
    name, score, higher = macro_f1_eval(y, p)
    assert name == "macro_f1" and higher
    assert score == pytest.approx(
        f1_score(y, predicted, labels=range(5), average="macro")
    )


def test_weight_power_zero_half_one_and_legacy_default_are_well_defined():
    labels = np.array([0] * 10 + [1] * 5 + [2] * 3 + [3] * 2 + [4])
    old = len(labels) / (5 * np.bincount(labels, minlength=5))
    np.testing.assert_array_equal(class_weights(labels, 0), np.ones(5))
    np.testing.assert_allclose(class_weights(labels, 0.5), np.sqrt(old))
    np.testing.assert_array_equal(class_weights(labels, 1), old)
    assert LSTMConfig().weight_power == 1
    with pytest.raises(ValueError):
        LSTMConfig(weight_power=-1)
