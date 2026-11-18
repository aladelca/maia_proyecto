import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
import torch
from mlflow.tracking import MlflowClient

from sleep_staging.lstm_data import SequenceData
from sleep_staging.lstm_research import fit_sequence
from sleep_staging.refinement import (
    BoosterCandidate,
    NeuralCandidate,
    RefinementConfig,
    fit_booster,
    neural_config,
    predict_fit,
)
from sleep_staging.refinement_data import refinement_views


def real_views():
    root = Path(__file__).resolve().parents[2]
    reference = root / "var/research/sc-w30-v2/campaign.json"
    if not reference.exists():
        pytest.skip("Requires real research feature cache")
    raw = pd.read_parquet(json.loads(reference.read_text())["dataset"])
    subjects = []
    selected = []
    for subject, group in raw.groupby("subject_id", sort=True):
        if group.task_5.nunique() == 5:
            subjects.append(subject)
            selected.append(group[group.task_5.notna()].groupby("task_5").head(8))
        if len(subjects) == 2:
            break
    return refinement_views(
        SequenceData.from_frame(pd.concat(selected, ignore_index=True))
    ), subjects


@pytest.mark.integration
def test_lightgbm_inner_metric_refit_and_checkpoint_use_real_data(tmp_path):
    views, subjects = real_views()
    old = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment("real-refinement-test")
    candidate = BoosterCandidate(
        name="small",
        n_estimators=5,
        num_leaves=3,
        min_child_samples=2,
        weight_power=0.5,
    )
    cfg = RefinementConfig(
        lgbm=[candidate],
        bilstm=[NeuralCandidate(name="unused", view="raw145")],
        lgbm_patience=2,
    )
    try:
        fit = fit_booster(
            views["context435"],
            cfg,
            candidate,
            subjects[:1],
            subjects[1:],
            tmp_path / "inner",
            "inner",
            42,
        )
        history = json.loads((tmp_path / "inner/history.json").read_text())
        assert set(history["valid_0"]) == {"macro_f1"}
        assert 1 <= fit["best_epoch"] <= 5
        refit = fit_booster(
            views["context435"],
            cfg,
            candidate,
            subjects,
            [],
            tmp_path / "refit",
            "refit",
            42,
            fit["best_epoch"],
        )
        assert refit["estimator_n_estimators"] == fit["best_epoch"]
        assert refit["torch_loaded"] is False
        assert refit["worker_pid"] != __import__("os").getpid()
        probability = predict_fit(
            "lgbm", views, cfg, candidate, refit, views["raw145"].targets[:5]
        )
        np.testing.assert_allclose(probability.sum(axis=1), 1)
        assert (
            fit_booster(
                views["context435"],
                cfg,
                candidate,
                subjects[:1],
                subjects[1:],
                tmp_path / "inner",
                "inner",
                42,
            )
            == fit
        )
        assert MlflowClient().get_run(fit["mlflow_run_id"]).data.tags["device"] == "cpu"
        with pytest.raises(ValueError, match="overlap"):
            fit_booster(
                views["context435"],
                cfg,
                candidate,
                subjects,
                subjects,
                tmp_path / "bad",
                "bad",
                42,
            )
    finally:
        mlflow.set_tracking_uri(old)


@pytest.mark.parametrize("hidden,power", [(64, 0), (128, 0.5)])
@pytest.mark.integration
def test_improved_bilstm_trains_with_new_features_and_loss_on_mps(
    tmp_path, monkeypatch, hidden, power
):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    views, subjects = real_views()
    candidate = NeuralCandidate(
        name="real_bilstm", view="night435", hidden_size=hidden, weight_power=power
    )
    cfg = RefinementConfig(
        lgbm=[BoosterCandidate(name="unused")],
        bilstm=[candidate],
        nn_max_epochs=1,
        batch_size=16,
    )
    local, spec = neural_config(cfg, candidate)
    old = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment("real-bilstm-refinement")
    try:
        fit = fit_sequence(
            views[candidate.view],
            local,
            spec,
            subjects[:1],
            subjects[1:],
            candidate.lr,
            tmp_path / "fit",
            "real",
            42,
        )
        probability = predict_fit(
            "bilstm", views, cfg, candidate, fit, views["raw145"].targets[:5]
        )
        np.testing.assert_allclose(probability.sum(axis=1), 1, atol=1e-6)
        audit = json.loads((tmp_path / "fit/audit.json").read_text())
        assert audit["params"]["weight_power"] == power
        assert len(audit["columns"]) == 435
        assert audit["params"]["hidden_size"] == hidden
        assert MlflowClient().get_run(fit["mlflow_run_id"]).data.tags["device"] == "mps"
    finally:
        mlflow.set_tracking_uri(old)
