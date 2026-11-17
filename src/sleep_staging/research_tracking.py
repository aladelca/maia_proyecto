"""Identity and candidate integrity checks for research tracking exports."""

from pathlib import Path

import joblib

from sleep_staging.provenance import fingerprint, hash_file


def ensemble_identity(components: list[dict], weights: list[float]) -> str:
    if len(components) != len(weights) or not components:
        raise ValueError("Each ensemble component requires a weight.")
    return fingerprint({"components": components, "weights": weights})


def validated_candidate(campaign: dict, root: Path) -> dict:
    """Reject stale weights left by a previous campaign or --no-refit run."""
    if campaign.get("candidate_refit") is not True:
        raise ValueError(
            "This campaign has no verified candidate refit; rerun without --no-refit."
        )
    candidate = root / "candidate.joblib"
    if (
        not candidate.is_file()
        or not campaign.get("candidate_sha256")
        or hash_file(candidate) != campaign["candidate_sha256"]
    ):
        raise ValueError("Candidate checksum mismatch.")
    bundle = joblib.load(candidate)
    if bundle["spec"]["name"] != campaign["winner"]["name"] or bundle[
        "dataset_sha256"
    ] != hash_file(Path(campaign["dataset"])):
        raise ValueError("Candidate does not belong to the selected model and dataset.")
    return bundle
