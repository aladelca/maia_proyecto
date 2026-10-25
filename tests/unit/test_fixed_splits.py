import pandas as pd
import pytest

from sleep_staging.config import ExperimentConfig
from sleep_staging.experiments import create_splits


def test_fixed_split_respects_csv(tmp_path):
    subjects = [f"sleep-cassette:{i:02d}" for i in range(8)]

    expected = pd.DataFrame({
        "subject_id": subjects,
        "partition": ["development"] * 5 + ["test"] * 3,
        "fold": [0, 1, 2, 3, 4, -1, -1, -1],
    })

    csv = tmp_path / "configs/splits/fixed.csv"
    csv.parent.mkdir(parents=True)
    expected.to_csv(csv, index=False)

    prepared = tmp_path / "prepared/synthetic"
    prepared.mkdir(parents=True)

    epochs = pd.DataFrame({
        "subject_id": subjects * 2,
    })
    epochs.to_parquet(prepared / "epochs.parquet", index=False)

    cfg = ExperimentConfig(
        split_file="configs/splits/fixed.csv",
        n_splits=5,
    )

    result = create_splits(prepared, cfg, tmp_path)
    actual = pd.read_parquet(result / "subjects.parquet")

    pd.testing.assert_frame_equal(
        actual.sort_values("subject_id").reset_index(drop=True),
        expected.sort_values("subject_id").reset_index(drop=True),
    )

    assert set(actual[actual.partition == "test"].fold) == {-1}


def test_fixed_split_rejects_missing_subject(tmp_path):
    subjects = [f"sleep-cassette:{i:02d}" for i in range(8)]

    csv = tmp_path / "configs/splits/fixed.csv"
    csv.parent.mkdir(parents=True)

    pd.DataFrame({
        "subject_id": subjects[:-1],
        "partition": ["development"] * 5 + ["test"] * 2,
        "fold": [0, 1, 2, 3, 4, -1, -1],
    }).to_csv(csv, index=False)

    prepared = tmp_path / "prepared/synthetic"
    prepared.mkdir(parents=True)

    pd.DataFrame({
        "subject_id": subjects,
    }).to_parquet(prepared / "epochs.parquet", index=False)

    cfg = ExperimentConfig(
        split_file="configs/splits/fixed.csv",
        n_splits=5,
    )

    with pytest.raises(ValueError, match="Partición incompatible"):
        create_splits(prepared, cfg, tmp_path)
