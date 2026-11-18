import numpy as np
import pandas as pd
import pytest

from sleep_staging.lstm_data import FeatureScaler, SequenceData, sequence_index


def chronology():
    return pd.DataFrame(
        {
            "epoch_id": list("abcdefg"),
            "recording_id": ["night-a"] * 5 + ["night-b"] * 2,
            "subject_id": ["subject-a"] * 5 + ["subject-b"] * 2,
            "fold": [0] * 5 + [1] * 2,
            "onset_s": [0, 30, 60, 150, 180, 0, 30],
            "task_5": ["W", None, "N1", "N2", "REM", "N3", "W"],
            "eeg_fpz_cz__eeg__std": np.arange(7, dtype=np.float32),
            "emg__emg_envelope__median": np.arange(7, dtype=np.float32) + 10,
        }
    )


def test_windows_preserve_m_context_without_bridging_gaps_or_nights():
    frame = chronology()
    neighbors, padding = sequence_index(frame, 3, centered=False)
    np.testing.assert_array_equal(neighbors[2], [0, 1, 2])
    np.testing.assert_array_equal(neighbors[3], [3, 3, 3])
    np.testing.assert_array_equal(neighbors[5], [5, 5, 5])
    np.testing.assert_array_equal(padding[0], [True, True, False])
    data = SequenceData.from_frame(frame)
    assert data.targets.tolist() == [0, 2, 3, 4, 5, 6]
    assert 1 in neighbors[2]  # M contributes signal features, never its label.


def test_centered_windows_keep_target_position_and_are_order_invariant():
    original = chronology()
    shuffled = original.sample(frac=1, random_state=17).reset_index(drop=True)
    expected, _ = sequence_index(original, 3, centered=True)
    actual, _ = sequence_index(shuffled, 3, centered=True)
    for pos, row in shuffled.iterrows():
        source = original.index[original.epoch_id.eq(row.epoch_id)][0]
        assert (
            shuffled.iloc[actual[pos]].epoch_id.tolist()
            == original.iloc[expected[source]].epoch_id.tolist()
        )
        assert actual[pos, 1] == pos


def test_features_are_independent_of_labels_and_forbid_metadata():
    frame = chronology()
    changed = frame.assign(task_5="REM")
    a, b = SequenceData.from_frame(frame), SequenceData.from_frame(changed)
    np.testing.assert_array_equal(a.values, b.values)
    assert a.columns == ["eeg_fpz_cz__eeg__std", "emg__emg_envelope__median"]
    with pytest.raises(ValueError, match="signal feature"):
        SequenceData.from_frame(frame, columns=["fold"])


def test_duplicate_epochs_and_recordings_with_mixed_subjects_are_rejected():
    frame = chronology()
    with pytest.raises(ValueError, match="Duplicate"):
        SequenceData.from_frame(pd.concat([frame, frame.iloc[:1]], ignore_index=True))
    frame.loc[0, "subject_id"] = "other-subject"
    with pytest.raises(ValueError, match="recording"):
        SequenceData.from_frame(frame)


def test_scaler_uses_training_only_and_retains_constant_columns():
    values = np.array([[1, 4], [2, 4], [3, 4], [1e8, 9]], dtype=np.float32)
    scaler = FeatureScaler.fit(values[:3])
    before = scaler.transform(values[:3])
    scaler.transform(values[3:])
    np.testing.assert_array_equal(before, scaler.transform(values[:3]))
    np.testing.assert_array_equal(scaler.median, [2, 4])
    assert np.isfinite(scaler.transform(values)).all()
    assert np.abs(scaler.transform(values)).max() <= 10
    assert scaler.transform(values).dtype == np.float32


def test_window_parameters_and_nonfinite_features_are_rejected():
    with pytest.raises(ValueError):
        sequence_index(chronology(), 0, centered=False)
    with pytest.raises(ValueError):
        sequence_index(chronology(), 4, centered=True)
    frame = chronology()
    frame.loc[0, "eeg_fpz_cz__eeg__std"] = np.inf
    with pytest.raises(ValueError, match="finite"):
        SequenceData.from_frame(frame)
