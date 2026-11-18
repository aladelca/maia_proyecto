import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from sleep_staging.serving.contract import context_matrix

pytest.importorskip("boto3")
pytest.importorskip("lightgbm")

spec = importlib.util.spec_from_file_location(
    "maia_train_cloud", Path(__file__).resolve().parents[2] / "scripts/train_cloud.py"
)
assert spec and spec.loader
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)


def test_only_frozen_development_enters_refit():
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
