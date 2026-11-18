import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from sleep_staging.lstm_data import FeatureScaler, SequenceData, sequence_index
from sleep_staging.models.lstm import SleepLSTM, require_mps


@pytest.mark.parametrize("centered", [False, True])
@pytest.mark.integration
def test_real_feature_batch_trains_and_reloads_on_mps(tmp_path, centered, monkeypatch):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable on this host")
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    root = Path(__file__).resolve().parents[2]
    reference = root / "var/research/sc-w30-v2/campaign.json"
    if not reference.exists():
        pytest.skip("Requires the real research feature cache")
    raw = pd.read_parquet(json.loads(reference.read_text())["dataset"])
    data = SequenceData.from_frame(raw.iloc[:300])
    indices, _ = sequence_index(data.frame, 15, centered)
    scaler = FeatureScaler.fit(data.values)
    rows = data.targets[:32]
    device = require_mps()
    x = torch.from_numpy(scaler.transform(data.values)[indices[rows]]).to(device)
    y = torch.from_numpy(data.labels[rows]).to(device)
    model = SleepLSTM(len(data.columns), 64, 0.3, centered, 7 if centered else 14).to(
        device
    )
    observed = {}
    model.recurrent.register_forward_hook(
        lambda module, args, result: observed.update(recurrent=result[0].detach())
    )
    model.head.register_forward_pre_hook(
        lambda module, args: observed.update(head_input=args[0].detach())
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    loss = torch.nn.functional.cross_entropy(model(x), y)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    optimizer.step()
    assert all(p.device.type == "mps" for p in model.parameters())
    assert any(not torch.equal(before[k], v) for k, v in model.state_dict().items())
    model.eval()
    with torch.no_grad():
        prediction = model(x).cpu().numpy()
        torch.testing.assert_close(
            observed["head_input"], observed["recurrent"][:, 7 if centered else 14]
        )
        reversed_prediction = model(x.flip(0)).flip(0).cpu().numpy()
    np.testing.assert_allclose(prediction, reversed_prediction, atol=1e-5)
    path = tmp_path / "weights.pt"
    torch.save(model.state_dict(), path)
    other = SleepLSTM(len(data.columns), 64, 0.3, centered, 7 if centered else 14).to(
        device
    )
    other.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    other.eval()
    with torch.no_grad():
        actual = other(x).cpu().numpy()
        probabilities = other(x).softmax(dim=1).cpu().numpy()
    np.testing.assert_allclose(prediction, actual, atol=1e-6)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)


def test_cpu_fallback_and_unavailable_mps_are_rejected(monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    with pytest.raises(ValueError, match="fallback"):
        require_mps()
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(ValueError, match="MPS"):
        require_mps()


@pytest.mark.integration
def test_interrupted_real_training_resumes_on_mps_and_is_idempotent(
    tmp_path, monkeypatch
):
    import mlflow
    from mlflow.tracking import MlflowClient

    from sleep_staging.lstm_research import LSTMConfig, WindowSpec, fit_sequence

    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable on this host")
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    root = Path(__file__).resolve().parents[2]
    reference = root / "var/research/sc-w30-v2/campaign.json"
    if not reference.exists():
        pytest.skip("Requires the real research feature cache")
    raw = pd.read_parquet(json.loads(reference.read_text())["dataset"])
    selected = []
    subjects = []
    for subject, group in raw.groupby("subject_id", sort=True):
        if group.task_5.nunique() == 5:
            subjects.append(subject)
            selected.append(group[group.task_5.notna()].groupby("task_5").head(4))
        if len(subjects) == 2:
            break
    data = SequenceData.from_frame(pd.concat(selected, ignore_index=True))
    previous_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment("real-lstm-resume-test")
    cfg = LSTMConfig(max_epochs=2, batch_size=16, hidden_size=8)
    spec = WindowSpec(name="real-test", length=3, centered=False)
    args = (
        data,
        cfg,
        spec,
        subjects[:1],
        subjects[1:],
        1e-3,
        tmp_path / "fit",
        "resume-test",
        42,
    )
    original = mlflow.log_metrics

    def interruption(*args, **kwargs):
        raise RuntimeError("deliberate interruption after checkpoint")

    try:
        monkeypatch.setattr(mlflow, "log_metrics", interruption)
        with pytest.raises(RuntimeError, match="deliberate interruption"):
            fit_sequence(*args)
        assert (tmp_path / "fit/last.pt").exists()
        monkeypatch.setattr(mlflow, "log_metrics", original)
        resumed = fit_sequence(*args)
        assert resumed["epochs_completed"] == 2
        assert (
            MlflowClient().get_run(resumed["mlflow_run_id"]).info.status == "FINISHED"
        )
        assert fit_sequence(*args) == resumed
        experiment_id = (
            MlflowClient().get_run(resumed["mlflow_run_id"]).info.experiment_id
        )
        assert len(MlflowClient().search_runs([experiment_id])) == 1
    finally:
        mlflow.set_tracking_uri(previous_uri)
