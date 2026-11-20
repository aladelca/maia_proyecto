import numpy as np
import pytest
import torch

from sleep_staging.config import channel_profiles
from sleep_staging.data.edf import Signal
from sleep_staging.fullnight.data import CHANNELS, make_inputs, profile_bits, targets
from sleep_staging.fullnight.model import FullNightNet, output_log_probs


def signals(epochs=5):
    seconds = np.arange(epochs * 3000) / 100
    eeg = (15 * np.sin(2 * np.pi * 10 * seconds)).astype("float32")
    return {
        CHANNELS[0]: Signal(np.zeros_like(eeg), 100, "eeg", "EEG Fpz-Cz"),
        CHANNELS[1]: Signal(eeg, 100, "eeg", "EEG Pz-Oz"),
        CHANNELS[3]: Signal(
            np.ones(epochs * 30, dtype="float32"), 1, "emg_envelope", "EMG submental"
        ),
    }


def test_inputs_use_available_eeg_and_complete_signal_epochs():
    src = signals()
    src[CHANNELS[1]].values[6000:9000] = 0
    x = make_inputs(src, duration=150)
    assert x.wave.shape == (5, 3, 1500)
    assert x.spectrum.shape[:2] == (5, 3)
    np.testing.assert_array_equal(
        x.eligible(profile_bits(CHANNELS[1])), [1, 1, 0, 1, 1]
    )
    assert not x.eligible(profile_bits(CHANNELS[0])).any()
    assert len(channel_profiles()) == 12
    assert np.isfinite(x.wave).all()
    with pytest.raises(ValueError):
        profile_bits("emg")
    y, crop = targets(
        [
            (0, 60, "Sleep stage W"),
            (60, 30, "Movement time"),
            (90, 30, "Sleep stage 2"),
            (120, 30, "Sleep stage ?"),
        ],
        150,
    )
    np.testing.assert_array_equal(y, [0, 0, 5, 2, -1])
    assert crop.tolist() == [True] * 5


def test_native_and_derived_emg_share_one_hz_representation():
    src = signals()
    src[CHANNELS[3]] = Signal(
        np.sin(np.arange(15000) * 1.7).astype("float32"),
        100,
        "emg_raw",
        "EMG submental",
    )
    x = make_inputs(src, duration=150)
    assert x.emg.shape == (5, 30)
    assert np.isfinite(x.emg).all()


@pytest.mark.parametrize(
    "view,head",
    [("wave", "five"), ("spectrum", "five"), ("wave", "six"), ("wave", "movement")],
)
def test_absent_signals_are_invisible_and_probabilities_normalize(view, head):
    torch.set_num_threads(1)
    net = FullNightNet(view, head, width=8).eval()
    inp = make_inputs(signals(), duration=150)
    x = torch.from_numpy(
        (inp.wave if view == "wave" else inp.spectrum).astype("float32")
    )
    emg = torch.from_numpy(inp.emg.astype("float32"))
    mask = torch.tensor([[False, True, False, False]]).expand(5, -1)
    a = net.encode(x, emg)
    b = net.encode(x.clone().index_fill(1, torch.tensor([0, 2]), 99), emg + 99)
    from sleep_staging.fullnight.model import predict_sequence

    first = predict_sequence(net, a, mask)
    second = predict_sequence(net, b, mask)
    assert first.shape == (5, 5 if head == "five" else 6)
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first.exp().sum(-1), torch.ones(5))
    first.mean().backward()


def test_two_stage_movement_probability_has_correct_class_order():
    logits = torch.tensor([[1.0, 2, 3, 4, 5, 0]])
    p = output_log_probs(logits, "movement").exp()
    torch.testing.assert_close(p[:, :5], logits[:, :5].softmax(1) * 0.5)
    torch.testing.assert_close(p[:, 5], torch.tensor([0.5]))
