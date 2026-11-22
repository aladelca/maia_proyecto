from multiprocessing import get_context

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from sleep_staging.lstm_data import SequenceData
from sleep_staging.models.tcn import TemporalConv, research_device
from sleep_staging.tcn_data import KEYS, boundary_targets
from sleep_staging.tcn_research import PROBS, aligned


def cohort() -> SequenceData:
    frame = pd.DataFrame(
        {
            "epoch_id": list("abcdefghi"),
            "recording_id": ["r"] * 7 + ["s"] * 2,
            "subject_id": ["p"] * 7 + ["q"] * 2,
            "fold": [0] * 7 + [1] * 2,
            "onset_s": [0, 30, 60, 90, 120, 180, 210, 0, 30],
            "task_5": ["W", "W", "W", "N1", None, "N1", "N2", "N1", "N1"],
            "eog__example": np.zeros(9),
        }
    )
    return SequenceData.from_frame(frame)


def test_transition_targets_mask_unscored_gaps_and_night_edges() -> None:
    np.testing.assert_array_equal(
        boundary_targets(cohort()), [-1, 0, 1, -1, -1, -1, -1, -1, -1]
    )


def test_oof_alignment_requires_same_targets_and_fold() -> None:
    data = cohort()
    table = (
        data.frame.iloc[data.targets][KEYS + ["task_5"]]
        .rename(columns={"task_5": "y_true"})
        .copy()
    )
    for col in PROBS:
        table[col] = 0.2
    assert aligned(data, table.iloc[::-1]).epoch_id.tolist() == table.epoch_id.tolist()
    bad = table.copy()
    bad.loc[0, "fold"] = 2
    with pytest.raises(AssertionError):
        aligned(data, bad)
    with pytest.raises(ValueError):
        aligned(data, table.iloc[:-1])


def check_dilated_convolution(length: int) -> None:
    width = 4
    model = TemporalConv(width, 0)
    values = torch.randn(3, length, width, requires_grad=True)
    expected = values.transpose(1, 2)
    for dilation, block in zip(model.dilations, model.blocks, strict=True):
        linear = block[0]
        weights = linear.weight.reshape(width, 3, width).transpose(1, 2)
        expected = expected + nn.functional.gelu(
            nn.functional.conv1d(
                expected, weights, linear.bias, dilation=dilation, padding=dilation
            )
        )
    torch.testing.assert_close(model(values), expected[:, :, length // 2])
    model(values).sum().backward()
    assert values.grad is not None and torch.isfinite(values.grad).all()


@pytest.mark.parametrize("length", [1, 15])
def test_tcn_matches_dilated_convolutions(length: int) -> None:
    # Isolate Torch from native OpenMP runtimes loaded by the booster tests on macOS.
    process = get_context("spawn").Process(
        target=check_dilated_convolution, args=(length,)
    )
    process.start()
    process.join(60)
    if process.is_alive():
        process.terminate()
        process.join()
        pytest.fail("Torch convolution equivalence check timed out")
    assert process.exitcode == 0


def test_device_selection_without_host_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert research_device("auto").type == research_device("cuda").type == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA"):
        research_device("cuda")
    with pytest.raises(ValueError, match="MPS/CUDA"):
        research_device("auto")
    assert research_device("cpu").type == "cpu"
