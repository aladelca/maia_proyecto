import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sleep_staging.serving.contract import LABELS, context_matrix
from sleep_staging.stack_training import SIX

pytest.importorskip("boto3")
pytest.importorskip("lightgbm")

spec = importlib.util.spec_from_file_location(
    "maia_train_cloud", Path(__file__).resolve().parents[2] / "scripts/train_cloud.py"
)
assert spec and spec.loader
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)


def cohort() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    raw = pd.DataFrame(
        {
            "subject_id": list("abcde"),
            "epoch_id": list("abcde"),
            "recording_id": list("abcde"),
            "onset_s": [0] * 5,
            "fold": range(5),
            "task_5": ["W", "N1", "N2", "N3", "REM"],
            "eog__feature": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )
    split = pd.concat(
        [
            raw[["subject_id", "fold"]].assign(partition="development"),
            pd.DataFrame([dict(subject_id="held", fold=-1, partition="test")]),
        ],
        ignore_index=True,
    )
    cols = ["eog__feature"]
    manifest = dict(raw_columns=cols, model_columns=list(context_matrix(raw, cols)))
    return raw, split, manifest


def test_only_frozen_development_enters_refit():
    raw, split, manifest = cohort()
    data = training.development(raw, split, manifest)
    assert len(data.targets) == 5 and data.columns == manifest["model_columns"]
    with pytest.raises(ValueError):
        training.development(
            raw.assign(subject_id=list("abcd") + ["held"]), split, manifest
        )
    with pytest.raises(ValueError):
        training.development(
            raw, pd.concat([split, split.iloc[[0]].assign(partition="test")]), manifest
        )


def test_stack_appends_out_of_fold_fullnight_probabilities():
    raw, split, manifest = cohort()
    data = training.development(raw, split, manifest)
    six = np.full((5, 6), 0.1)
    six[np.arange(5), np.arange(5)] = 0.5
    oof = raw[["recording_id", "subject_id", "onset_s", "fold"]].assign(target=range(5))
    oof[SIX] = six
    x, y = training.stack_inputs(data, oof)
    neural = ["fullnight_" + s for s in LABELS]
    assert list(x) == manifest["model_columns"] + neural
    assert y.tolist() == LABELS
    # Movement mass is removed and the five stages are renormalised.
    np.testing.assert_allclose(x[neural].sum(1), 1, rtol=1e-6)
    np.testing.assert_allclose(np.diag(x[neural]), 0.5 / 0.9, rtol=1e-6)
    with pytest.raises(ValueError):
        training.stack_inputs(data, oof.iloc[1:])
    with pytest.raises(ValueError):
        training.stack_inputs(data, oof.assign(target=[1, 0, 2, 3, 4]))
