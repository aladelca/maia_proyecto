"""Protocol regressions specific to pairing historical LightGBM and FullNight scores."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_stage_only_features_remove_movement_mass(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    from compare_fullnight_lgbm import stacked

    p = np.array([[0.4, 0.2, 0.1, 0.1, 0, 0.2], [0.2, 0.1, 0.05, 0.05, 0, 0.6]])
    x = pd.DataFrame({"feature": [9, 11]}, index=[7, 3])
    stages = stacked(x, p, movement=False)
    with_motion = stacked(x, p, movement=True)
    np.testing.assert_allclose(stages.iloc[0, 1:], stages.iloc[1, 1:])
    pd.testing.assert_frame_equal(stages, with_motion.drop(columns="fullnight_M"))
    assert with_motion.index.tolist() == [7, 3]
    assert with_motion.fullnight_M.tolist() == [0.2, 0.6]


def test_missing_epoch_or_patient_cannot_pass_pairing(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    from compare_fullnight_lgbm import SIX, aligned

    meta = pd.DataFrame(
        dict(
            recording_id=["a", "b"],
            subject_id=["p", "q"],
            onset_s=[30, 30],
            fold=[0, 1],
            task_5=["N2", "REM"],
        )
    )
    pred = meta.drop(columns="task_5").assign(target=[2, 4])
    pred[SIX] = np.eye(6)[[2, 4]]
    np.testing.assert_array_equal(aligned(meta, pred.iloc[::-1]), np.eye(6)[[2, 4]])
    with pytest.raises(ValueError):
        aligned(meta, pred.assign(subject_id=["p", "p"]))
    with pytest.raises(ValueError):
        aligned(meta, pred.iloc[:1])
    with pytest.raises(ValueError):
        aligned(meta, pred.assign(target=[2, 3]))


def test_native_movement_on_true_stage_is_counted_as_an_error(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    from report_fullnight_lgbm import from_counts

    m = np.eye(5, 6, dtype=int)
    m[1, 5] = 2
    result = from_counts(m)
    assert result["epochs"] == 7
    assert result["movement_calls"] == 2
    assert result["f1_N1"] == 0.5
    assert result["accuracy"] == 5 / 7
    assert result["macro_f1"] == 0.9
