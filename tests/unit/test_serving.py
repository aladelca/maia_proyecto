import numpy as np
import pandas as pd
import pytest

from sleep_staging.serving.contract import context_matrix, validate_night


def test_night_keeps_excluded_context_and_sorts_epochs():
    data = pd.DataFrame(
        {
            "recording_id": ["public"] * 3,
            "epoch_id": ["c", "a", "b"],
            "onset_s": [60, 0, 30],
            "eligible": [True, True, False],
            "ch__feature": [3.0, 1.0, 2.0],
        }
    )
    ordered = validate_night(data, ["ch__feature"])
    assert ordered.epoch_id.tolist() == ["a", "b", "c"]
    assert context_matrix(ordered, ["ch__feature"]).shape == (3, 3)
    for change in (
        {"epoch_id": ["a"] * 3},
        {"epoch_id": ["=1+1", "a", "b"]},
        {"onset_s": [0, 30, 31]},
        {"onset_s": [0, 30, 30]},
        {"ch__feature": [0, 1, np.inf]},
        {"task_5": ["W"] * 3},
        {"eligible": [1, 0, 1]},
        {"recording_id": ["a", "b", "a"]},
    ):
        with pytest.raises(ValueError):
            validate_night(data.assign(**change), ["ch__feature"])
