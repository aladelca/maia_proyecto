import numpy as np
import pandas as pd
import pytest

from sleep_staging.config import ExperimentConfig
from sleep_staging.data.preprocess import prepare_dataset
from sleep_staging.experiments import waveform_frame

torch = pytest.importorskip("torch")


@pytest.mark.integration
@pytest.mark.real_data
@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_cnn_real_epochs_normalizer_and_probabilities(real_dataset, device):
    from sleep_staging.models.cnn import CNNClassifier

    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS no disponible")
    data, root = real_dataset
    cfg = ExperimentConfig(task="task_5", max_epochs_per_recording=240)
    prepared = prepare_dataset(data, cfg, root)
    epochs = pd.read_parquet(prepared / "epochs.parquet")
    eligible = epochs[epochs.reason.eq("") & epochs.task_5.notna()]
    sample = eligible.groupby("task_5", group_keys=False).head(4)
    X = waveform_frame(sample, prepared, "eeg_fpz_cz")
    assert X.attrs["preprocessing_id"] == prepared.name
    model = CNNClassifier(epochs=1, batch_size=10, device=device)
    model.fit(X, sample.task_5.to_numpy())
    np.testing.assert_allclose(model.mean_, X.to_numpy().mean(), rtol=1e-6)
    before = model.mean_
    probability = model.predict_proba(X.iloc[-2:])
    assert model.mean_ == before
    assert probability.shape == (2, 5)
    np.testing.assert_allclose(probability.sum(axis=1), 1, atol=1e-6)
