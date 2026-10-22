import numpy as np
import pandas as pd
import pytest

from sleep_staging.config import ExperimentConfig
from sleep_staging.data.edf import read_recording
from sleep_staging.data.preprocess import extract_dataset_features, prepare_dataset

pytestmark = [pytest.mark.integration, pytest.mark.real_data]


def test_real_pipeline_cache_features_and_epoch_keys(real_dataset):
    data, root = real_dataset
    cfg = ExperimentConfig(task="task_5", max_epochs_per_recording=240)
    prepared = prepare_dataset(data, cfg, root)
    epochs = pd.read_parquet(prepared / "epochs.parquet")
    assert epochs.epoch_id.is_unique
    assert epochs.subject_id.nunique() == 8
    assert set(epochs.task_5.dropna()) == {"W", "N1", "N2", "N3", "REM"}
    before = (prepared / "epochs.parquet").stat().st_mtime_ns
    assert prepare_dataset(data, cfg, root) == prepared
    assert (prepared / "epochs.parquet").stat().st_mtime_ns == before
    features = extract_dataset_features(prepared, cfg, root)
    frame = pd.read_parquet(features)
    assert frame.attrs["preprocessing_id"] == prepared.name
    assert set(frame.epoch_id) == set(epochs.query('reason == ""').epoch_id)
    assert np.isfinite(frame.filter(like="__").to_numpy()).all()


def test_real_edf_preserves_native_emg_and_alignment(real_dataset):
    data, root = real_dataset
    record = pd.read_parquet(
        root / "data/manifests" / data.name / "recordings.parquet"
    ).iloc[0]
    signals, annotations, duration = read_recording(
        root / record.psg_file, root / record.hyp_file
    )
    assert signals["eeg_fpz_cz"].sfreq == 100
    assert signals["emg"].sfreq == 1
    assert signals["emg"].kind == "emg_envelope"
    assert len(signals["emg"].values) == duration
    assert annotations[0][0] == 0
    assert annotations[0][2] == "Sleep stage W"
