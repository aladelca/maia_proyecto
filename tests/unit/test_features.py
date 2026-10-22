import numpy as np

from sleep_staging.features.extract import signal_features


def test_known_frequency_is_in_expected_band():
    t = np.arange(3000) / 100
    f = signal_features(np.sin(2 * np.pi * 10 * t), 100, "eeg")
    assert f["relative_alpha"] > 0.95
    assert f["relative_delta"] < 0.01


def test_zero_signal_is_finite_and_envelope_has_no_waveform_bands():
    f = signal_features(np.zeros(3000), 100, "eeg")
    assert all(np.isfinite(list(f.values())))
    envelope = signal_features(np.arange(30), 1, "emg_envelope")
    assert "relative_alpha" not in envelope
    assert envelope["mean"] == 14.5
