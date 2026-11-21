"""Predict five sleep stages from a Cassette EDF with a trusted serving bundle."""

import argparse
import json
from pathlib import Path

import joblib

from sleep_staging.provenance import atomic_path, hash_file
from sleep_staging.serving.edf_lgbm import prepare_edf, score_night


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edf", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.bundle / "manifest.json").read_text())
    checkpoint = args.bundle / "model.joblib"
    if hash_file(checkpoint) != manifest["model_sha256"]:
        raise ValueError("Checksum del modelo LightGBM incorrecto.")
    frame = prepare_edf(args.edf, "night")
    result = score_night(joblib.load(checkpoint), manifest, frame)
    with atomic_path(args.output) as path:
        result.to_csv(path, index=False)
    print(
        f"{manifest['release_id']} · {result.eligible.sum()}/{len(result)} épocas elegibles · {args.output}"
    )


if __name__ == "__main__":
    main()
