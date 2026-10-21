import pandas as pd
import pytest

from sleep_staging.data.splits import assign_subjects


def sample_epochs():
    return pd.DataFrame(
        [
            {
                "subject_id": f"sc:{s:02d}",
                "recording_id": f"sc:{s}:{night}",
                "task_6": label,
            }
            for s in range(10)
            for night in [1, 2]
            for label in ["W", "N1", "N2", "N3", "REM", "M"]
        ]
    )


def test_subject_and_both_nights_stay_together_and_splits_are_deterministic():
    epochs = sample_epochs()
    first = assign_subjects(epochs, n_splits=5, seed=42)
    pd.testing.assert_frame_equal(
        first, assign_subjects(epochs.sample(frac=1), n_splits=5, seed=42)
    )
    merged = epochs.merge(first, on="subject_id", validate="many_to_one")
    assert merged.groupby("subject_id").fold.nunique().max() == 1
    assert set(first[first.partition == "development"].fold) == set(range(5))
    assert set(first[first.partition == "test"].fold) == {-1}


def test_insufficient_subjects_fail_instead_of_splitting_epochs():
    with pytest.raises(ValueError, match="sujetos"):
        assign_subjects(sample_epochs().query('subject_id == "sc:00"'), n_splits=5)
