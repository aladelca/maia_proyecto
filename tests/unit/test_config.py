import pytest
from pydantic import ValidationError

from sleep_staging.config import ExperimentConfig, channel_profiles


def test_all_twelve_profiles_require_eeg():
    profiles = channel_profiles()
    assert len(profiles) == 12
    assert all(set(p) & {"eeg_fpz_cz", "eeg_pz_oz"} for p in profiles.values())
    with pytest.raises(ValidationError, match="EEG"):
        ExperimentConfig(channels=["eog"])


@pytest.mark.parametrize(
    "kwargs", [{"task": "other"}, {"n_splits": 1}, {"eeg_high": 50}, {"workers": 0}]
)
def test_invalid_experiment_is_rejected(kwargs):
    with pytest.raises(ValidationError):
        ExperimentConfig(**kwargs)
