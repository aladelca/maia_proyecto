import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from sleep_staging.config import ExperimentConfig
from sleep_staging.experiments import run_experiment
from sleep_staging.inference import predict_features
from sleep_staging.tracking import require_tracking


def test_offline_tracking_fails_without_creating_filestore():
    with pytest.raises(ValueError, match="MLflow"):
        require_tracking("http://127.0.0.1:1")


@pytest.mark.integration
@pytest.mark.real_data
def test_real_experiment_logs_model_and_reload(real_dataset, tracking_server):
    data, root = real_dataset
    cfg = ExperimentConfig(
        name="integration-real", task="task_5", rf_trees=8, max_epochs_per_recording=240
    )
    result = run_experiment(data, cfg, root, tracking_server)
    assert set(result["leaderboard"].model) == {"dummy", "rf"}
    assert result["oof"].groupby(["model", "epoch_id"]).size().max() == 1
    assert not set(result["oof"].subject_id) & set(result["test_subjects"])
    client = MlflowClient(tracking_uri=tracking_server)
    assert client.get_run(result["run_id"]).info.status == "FINISHED"
    assert client.list_artifacts(result["run_id"])
    frame = pd.read_parquet(result["features_path"])
    prediction = predict_features(result["model_uri"], frame, tracking_server)
    assert len(prediction) == len(frame)
    np.testing.assert_allclose(prediction.filter(like="prob_").sum(axis=1), 1)
    with pytest.raises(ValueError, match="características"):
        predict_features(result["model_uri"], frame[["epoch_id"]], tracking_server)
    incompatible = frame.copy()
    incompatible.attrs["preprocessing_id"] = "different"
    with pytest.raises(ValueError, match="preprocesamiento"):
        predict_features(result["model_uri"], incompatible, tracking_server)
    model = mlflow.sklearn.load_model(result["model_uri"])
    assert set(model.classes_) == {"W", "N1", "N2", "N3", "REM"}
    epochs = pd.read_parquet(
        root / "data/interim" / result["preprocessing_id"] / "epochs.parquet"
    )
    development_ids = epochs[
        ~epochs.subject_id.isin(result["test_subjects"])
        & epochs.task_5.notna()
        & epochs.reason.eq("")
        & ~epochs.flat_eeg_fpz_cz
    ].epoch_id
    columns = list(model.feature_names_in_)
    expected = (
        model.named_steps["impute"]
        .transform(frame[frame.epoch_id.isin(development_ids)][columns])
        .mean(axis=0)
    )
    np.testing.assert_allclose(model.named_steps["scale"].mean_, expected)


@pytest.mark.integration
@pytest.mark.real_data
def test_six_class_pilot_fails_explicitly_when_movement_has_no_training_support(
    real_dataset, tracking_server
):
    data, root = real_dataset
    cfg = ExperimentConfig(
        name="six-class-support", task="task_6", max_epochs_per_recording=240
    )
    with pytest.raises(ValueError, match="faltan clases"):
        run_experiment(data, cfg, root, tracking_server)
