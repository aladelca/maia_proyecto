import numpy as np

from sleep_staging.data.labels import epoch_annotations, map_label


def test_label_mapping_retains_movement_only_for_six_classes():
    assert map_label("Sleep stage 3", "task_6") == "N3"
    assert map_label("Sleep stage 4", "task_5") == "N3"
    assert map_label("Sleep stage R", "task_5") == "REM"
    assert map_label("Movement time", "task_6") == "M"
    assert map_label("Movement time", "task_5") is None
    assert map_label("Sleep stage ?", "task_6") is None


def test_long_annotations_gaps_and_partial_tail():
    epochs = epoch_annotations(
        [(0, 60, "Sleep stage 2"), (90, 30, "Movement time")], 125
    )
    assert epochs.onset_s.tolist() == [0, 30, 60, 90, 120]
    assert epochs.reason.tolist() == ["", "", "annotation_gap", "", "incomplete"]
    assert epochs.iloc[3].task_6 == "M"
    assert epochs.iloc[0].onset_sample_eeg == 0
    assert epochs.iloc[1].onset_sample_eeg == 3000


def test_overlaps_and_unaligned_annotations_are_not_silently_labeled():
    epochs = epoch_annotations(
        [(0, 60, "Sleep stage 2"), (29, 60, "Sleep stage R")], 90
    )
    assert not np.any(epochs.reason == "")
