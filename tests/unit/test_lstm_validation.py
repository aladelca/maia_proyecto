import numpy as np
import pandas as pd
import pytest

from sleep_staging.lstm_research import (
    LSTMConfig,
    choose_trial,
    read_checkpoint,
    split_subjects,
    write_checkpoint,
)


def test_config_requires_mps_and_odd_centered_windows():
    with pytest.raises(ValueError):
        LSTMConfig(device="cpu")
    with pytest.raises(ValueError):
        LSTMConfig(variants=[{"name": "bad", "length": 4, "centered": True}])


def test_inner_split_does_not_contain_external_subjects_or_labels():
    frame = pd.DataFrame(
        {
            "subject_id": np.repeat(np.arange(20), 3),
            "fold": np.repeat(np.arange(20) % 5, 3),
        }
    )
    training, validation, external = split_subjects(frame, fold=2, seed=44)
    assert not set(training) & set(validation)
    assert not (set(training) | set(validation)) & set(external)
    assert set(external) == set(frame.loc[frame.fold.eq(2), "subject_id"])
    assert set(training) | set(validation) | set(external) == set(frame.subject_id)


def test_selection_uses_internal_f1_then_epoch_count_then_candidate_order():
    trials = [
        {"candidate": 1, "best_f1_macro": 0.7, "best_epoch": 4},
        {"candidate": 0, "best_f1_macro": 0.7, "best_epoch": 4},
        {"candidate": 2, "best_f1_macro": 0.69, "best_epoch": 1},
    ]
    assert choose_trial(trials)["candidate"] == 0
    trials[1]["best_epoch"] = 5
    assert choose_trial(trials)["candidate"] == 1


def test_checkpoint_integrity_and_identity_are_verified_before_loading(tmp_path):
    path = tmp_path / "last.pt"
    write_checkpoint(path, {"identity": "a", "epoch": 3})
    assert read_checkpoint(path, "a")["epoch"] == 3
    with pytest.raises(ValueError, match="identity"):
        read_checkpoint(path, "b")
    path.write_bytes(b"modified")
    with pytest.raises(ValueError, match="checksum"):
        read_checkpoint(path, "a")
