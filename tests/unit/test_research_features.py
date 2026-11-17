import numpy as np
import pandas as pd
import pytest

from sleep_staging.features.extract import signal_features
from sleep_staging.features.research import batch_signal_features, temporal_features
from sleep_staging.provenance import hash_file
from sleep_staging.research_data import build_research_dataset, development_recordings


@pytest.mark.parametrize(
    "kind,sfreq,samples",
    [("eeg", 100, 3000), ("eog", 100, 3000), ("emg_envelope", 1, 30)],
)
def test_vectorized_original_features_match_the_existing_protocol(kind, sfreq, samples):
    x = np.random.default_rng(42).normal(size=(3, samples)).astype(np.float32)
    batch = batch_signal_features(x, sfreq, kind)
    for row in range(len(x)):
        original = signal_features(x[row], sfreq, kind)
        for name, expected in original.items():
            np.testing.assert_allclose(
                batch[name][row], expected, rtol=1e-10, atol=1e-10
            )


def test_nonlinear_features_remain_finite_on_constant_signals():
    features = batch_signal_features(np.zeros((2, 3000)), 100, "eeg")
    assert all(np.isfinite(values).all() for values in features.values())
    np.testing.assert_allclose(features["permutation_entropy"], 0)
    np.testing.assert_allclose(features["petrosian_fd"], 1)
    np.testing.assert_allclose(features["flat"], 1)


def test_entropy_distinguishes_monotone_and_noise():
    monotone = np.arange(3000, dtype=float)
    noise = np.random.default_rng(13).normal(size=3000)
    features = batch_signal_features(np.stack([monotone, noise]), 100, "eeg")
    assert features["permutation_entropy"][0] == 0
    assert features["permutation_entropy"][1] > 0.99


def test_context_is_row_order_invariant_and_independent_of_labels_and_other_records():
    frame = pd.DataFrame(
        {
            "recording_id": ["a"] * 20 + ["b"] * 20,
            "onset_s": list(range(0, 600, 30)) * 2,
            "task_5": ["W"] * 40,
            "eeg_fpz_cz__eeg__power_delta": np.arange(40, dtype=float),
        }
    )
    columns = ["eeg_fpz_cz__eeg__power_delta"]
    expected = temporal_features(frame, columns)
    changed = frame.sample(frac=1, random_state=2)
    changed.loc[:, "task_5"] = "REM"
    changed.loc[changed.recording_id.eq("b"), columns] = 1e8
    actual = temporal_features(changed, columns).sort_index()
    pd.testing.assert_frame_equal(expected.iloc[:20], actual.iloc[:20])
    pd.testing.assert_frame_equal(
        expected, temporal_features(frame.drop(columns="task_5"), columns)
    )


def test_context_resets_at_a_missing_epoch():
    # Each segment is constant. Bridging the gap would create mixed rolling means.
    frame = pd.DataFrame(
        {
            "recording_id": ["a"] * 12,
            "onset_s": list(range(0, 180, 30)) + list(range(300, 480, 30)),
            "eeg_fpz_cz__eeg__std": [1.0] * 6 + [100.0] * 6,
        }
    )
    context = temporal_features(frame, ["eeg_fpz_cz__eeg__std"])
    np.testing.assert_allclose(
        context.iloc[:6].to_numpy(),
        context.iloc[0].to_numpy()[None, :].repeat(6, axis=0),
    )
    np.testing.assert_allclose(
        context.iloc[6:].to_numpy(),
        context.iloc[6].to_numpy()[None, :].repeat(6, axis=0),
    )


def test_context_rejects_target_columns_and_duplicate_epochs():
    frame = pd.DataFrame(
        {
            "recording_id": ["a", "a"],
            "onset_s": [0, 0],
            "task_5": [1, 2],
            "eeg_fpz_cz__eeg__std": [1, 2],
        }
    )
    with pytest.raises(ValueError, match="signal features"):
        temporal_features(frame, ["task_5"])
    with pytest.raises(ValueError, match="Duplicate"):
        temporal_features(frame, ["eeg_fpz_cz__eeg__std"])


def test_dataset_selection_excludes_reserved_edf_paths_before_reading_signals(tmp_path):
    subjects = pd.DataFrame(
        {
            "subject_id": [f"sc:{i}" for i in range(7)],
            "partition": ["development"] * 5 + ["test"] * 2,
            "fold": [0, 1, 2, 3, 4, -1, -1],
        }
    )
    splits = tmp_path / "splits.csv"
    subjects.to_csv(splits, index=False)
    records = pd.DataFrame(
        {
            "recording_id": [f"record{i}" for i in range(7)],
            "subject_id": subjects.subject_id,
            "psg_file": [f"nonexistent/{i}-psg.edf" for i in range(7)],
            "hyp_file": [f"nonexistent/{i}-hyp.edf" for i in range(7)],
        }
    )
    path = tmp_path / "data/manifests/sleep-cassette/recordings.parquet"
    path.parent.mkdir(parents=True)
    records.to_parquet(path, index=False)
    selected, frozen = development_recordings(tmp_path, splits)
    assert selected.subject_id.tolist() == subjects.subject_id.iloc[:5].tolist()
    assert not set(selected.psg_file) & set(records.psg_file.iloc[5:])
    assert len(frozen) == 7


def test_research_cache_rejects_modified_features(tmp_path, monkeypatch):
    subjects = pd.DataFrame(
        {
            "subject_id": [f"sc:{i}" for i in range(7)],
            "partition": ["development"] * 5 + ["test"] * 2,
            "fold": [0, 1, 2, 3, 4, -1, -1],
        }
    )
    splits = tmp_path / "splits.csv"
    subjects.to_csv(splits, index=False)
    manifest = tmp_path / "data/manifests/sleep-cassette"
    manifest.mkdir(parents=True)
    records, files = [], []
    for i in range(7):
        record = {"recording_id": f"record{i}", "subject_id": f"sc:{i}"}
        for key in ["psg_file", "hyp_file"]:
            relative = f"{i}-{key}.edf"
            (tmp_path / relative).write_bytes(b"unit-test-checksum-only")
            record[key] = relative
            files.append({"path": relative, "sha256": hash_file(tmp_path / relative)})
        records.append(record)
    pd.DataFrame(records).to_parquet(manifest / "recordings.parquet")
    pd.DataFrame(files).to_parquet(manifest / "files.parquet")

    def fake_record(record, source_root, margin):
        return pd.DataFrame(
            {
                "epoch_id": [record["recording_id"] + ":0"],
                "subject_id": [record["subject_id"]],
                "task_5": ["W"],
            }
        )

    monkeypatch.setattr("sleep_staging.research_data._record_features", fake_record)
    output = tmp_path / "output"
    dataset = build_research_dataset(tmp_path, splits, output)
    assert build_research_dataset(tmp_path, splits, output) == dataset
    frame = pd.read_parquet(dataset)
    frame.loc[0, "task_5"] = "REM"
    frame.to_parquet(dataset)
    with pytest.raises(ValueError, match="cache checksum"):
        build_research_dataset(tmp_path, splits, output)
