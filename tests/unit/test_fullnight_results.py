import numpy as np
import pandas as pd
import pytest
import torch

from sleep_staging.fullnight.data import Inputs
from sleep_staging.fullnight.metrics import state_slices, summarize_counts
from sleep_staging.fullnight.trees import tree_features


def test_state_slices_keep_unscored_and_movement_outside_five_stage_boundaries():
    y = np.array([0, 0, 0, 1, 5, 1, 2, 2, -1, 2])
    transition, stable = state_slices(y)
    assert np.flatnonzero(transition).tolist() == [2, 6]
    assert np.flatnonzero(stable).tolist() == [1]


def test_movement_false_alarms_use_nights_not_number_of_subjects():
    matrix = np.diag([9, 3, 5, 8, 4, 2])
    matrix[2, 5] = 4
    metrics = summarize_counts(matrix, 2)
    assert metrics["m_false_positives_per_night"] == 2
    assert metrics["m_recall"] == 1
    assert metrics["m_precision"] == 1 / 3


def test_tree_context_never_crosses_invalid_eeg_or_uses_unselected_eeg():
    valid = np.ones((7, 4), dtype=bool)
    valid[3, 1] = False
    f = np.arange(7 * 4 * 9, dtype=np.float32).reshape(7, 4, 9)
    x = Inputs(np.empty(0), np.empty(0), np.empty(0), valid, f.copy())
    a = tree_features(x, "eeg_pz_oz")
    x.features[4:] += 1000
    x.features[:, 0] = np.nan
    b = tree_features(x, "eeg_pz_oz")
    np.testing.assert_equal(a[:3], b[:3])


def test_usleep_training_windows_and_predictions_respect_gaps_and_eeg_selection(
    tmp_path,
):
    pytest.importorskip("braindecode")
    from sleep_staging.fullnight.training import Night
    from sleep_staging.models import usleep_reference as us

    torch.set_num_threads(1)
    rng = np.random.default_rng(4)
    available = np.ones((8, 4), dtype=bool)
    available[3, 0] = False
    x = Inputs(
        rng.normal(size=(8, 3, 1500)).astype("float16"),
        np.empty(0),
        np.empty(0),
        available,
        np.empty(0),
    )
    y = np.array([0, 1, 2, 0, 0, 1, 2, 1])
    night = Night("r", "s", 0, tmp_path, x, y, np.ones(8, bool))
    assert us.windows(night, 3)[1].tolist() == [0, 4]
    net = us.build().eval()
    budget = us.Budget(inference_chunk=3, halo=1)
    probabilities = us.predict(net, x, budget, torch.device("cpu"))
    x.wave[4:] += 1
    x.wave[:, 1] = 0
    changed = us.predict(net, x, budget, torch.device("cpu"))
    assert np.isnan(probabilities[3]).all()
    np.testing.assert_array_equal(probabilities[:3], changed[:3])
    np.testing.assert_allclose(probabilities[available[:, 0]].sum(1), 1, atol=1e-6)


def test_closed_holdout_reads_saved_results_without_reopening_psgs_and_detects_damage(
    tmp_path,
):
    from sleep_staging.fullnight.experiment import evaluate_frozen, result_hashes
    from sleep_staging.provenance import fingerprint, hash_file, write_json

    folder = tmp_path / "campaign"
    splits = tmp_path / "configs/splits"
    splits.mkdir(parents=True)
    split = splits / "sc_subjects_v1.csv"
    pd.DataFrame(
        {"subject_id": ["a", "b"], "partition": ["development", "test"]}
    ).to_csv(split, index=False)
    conditions = splits / "st_conditions_v1.csv"
    conditions.write_text("recording_id,condition\n")
    protocol = {"split_sha256": hash_file(split), "code": {"archived": "earlier"}}
    run_id = fingerprint(protocol)
    write_json(folder / "protocol.json", protocol)
    write_json(
        folder / "freeze.json",
        {"run": run_id, "smoke": False, "subjects": ["a"], "weights": {}},
    )
    dest = folder / "reserved"
    dest.mkdir()
    for f in [
        "metrics.csv",
        "predictions.parquet",
        "counts.json",
        "confusion.json",
        "paired-subject-bootstrap.json",
    ]:
        (dest / f).write_bytes(b"evaluated")
    write_json(
        dest / "completed.json",
        {
            "run": run_id,
            "conditions_sha256": hash_file(conditions),
            "artifacts": result_hashes(dest),
        },
    )
    args = (tmp_path, tmp_path / "absent-raw", tmp_path, folder, torch.device("cpu"))
    assert evaluate_frozen(*args) == dest
    (dest / "metrics.csv").write_text("changed")
    with pytest.raises(ValueError, match="resultado"):
        evaluate_frozen(*args)
