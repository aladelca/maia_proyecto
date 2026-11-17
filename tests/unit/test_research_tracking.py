import joblib
import pytest

from sleep_staging.provenance import hash_file
from sleep_staging.research_tracking import ensemble_identity, validated_candidate


def test_no_refit_cannot_export_a_previous_candidate(tmp_path):
    (tmp_path / "candidate.joblib").write_bytes(b"previous-campaign")
    with pytest.raises(ValueError, match="no verified candidate refit"):
        validated_candidate({"candidate_refit": False}, tmp_path)


def test_candidate_identity_and_checksum_must_match_the_campaign(tmp_path):
    dataset = tmp_path / "dataset.parquet"
    dataset.write_bytes(b"fixed-data")
    candidate = tmp_path / "candidate.joblib"
    joblib.dump(
        {"spec": {"name": "winner"}, "dataset_sha256": hash_file(dataset)}, candidate
    )
    campaign = {
        "candidate_refit": True,
        "candidate_sha256": hash_file(candidate),
        "dataset": str(dataset),
        "winner": {"name": "winner"},
    }
    assert validated_candidate(campaign, tmp_path)["spec"]["name"] == "winner"
    with pytest.raises(ValueError, match="does not belong"):
        validated_candidate({**campaign, "winner": {"name": "another-model"}}, tmp_path)
    dataset.write_bytes(b"other-data")
    with pytest.raises(ValueError, match="does not belong"):
        validated_candidate(campaign, tmp_path)
    candidate.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        validated_candidate(campaign, tmp_path)


def test_ensemble_tracking_identity_follows_component_runs_and_predictions():
    components = [{"name": "lgbm", "run_id": "run-1", "oof_sha256": "hash-1"}]
    original = ensemble_identity(components, [1.0])
    assert original == ensemble_identity(components, [1.0])
    assert original != ensemble_identity([{**components[0], "run_id": "run-2"}], [1.0])
    assert original != ensemble_identity(
        [{**components[0], "oof_sha256": "hash-2"}], [1.0]
    )
