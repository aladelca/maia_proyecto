"""Publish a Batch candidate after validating its frozen example. Activation is a separate Terraform apply."""

import argparse
import json
import tempfile
from pathlib import Path
from uuid import UUID

import boto3
import joblib
import numpy as np
import pandas as pd
from publish_cloud_inputs import publish_bundle

from sleep_staging.provenance import hash_file, write_json
from sleep_staging.serving.contract import LABELS, predictions, read_night


def promote(job: str, baseline: Path, output: Path) -> None:
    s3 = boto3.client("s3")
    inputs = json.loads(baseline.read_text())
    b = inputs["buckets"]
    prefix = "candidates/" + str(UUID(job)) + "/"
    progress = json.loads(
        s3.get_object(Bucket=b["research"], Key=prefix + "progress.json")["Body"].read()
    )
    candidate = progress["candidate"]
    if "stack_five.joblib" in candidate:
        raise ValueError("Candidato híbrido: usar scripts/promote_stack_candidate.py.")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name, ref in candidate.items():
            example = name == "example.parquet"
            if (example and ref != inputs["objects"][name]) or (
                not example and not ref["key"].startswith(prefix)
            ):
                raise ValueError("Unexpected candidate object")
            path = root / ("manifest.json" if name == "manifest" else name)
            s3.download_file(
                b["releases"] if example else b["research"],
                ref["key"],
                str(path),
                ExtraArgs={"VersionId": ref["version"]},
            )
            if hash_file(path) != ref["sha256"]:
                raise ValueError("Candidate checksum mismatch")
        manifest = json.loads((root / "manifest.json").read_text())
        if (
            manifest["training_job"] != job
            or manifest["dataset_sha256"] != inputs["objects"]["dataset"]["sha256"]
            or manifest["split_sha256"] != inputs["objects"]["splits"]["sha256"]
            or manifest["model_sha256"] != candidate["model.joblib"]["sha256"]
        ):
            raise ValueError("Candidate provenance mismatch")
        result = predictions(
            joblib.load(root / "model.joblib"),
            manifest,
            read_night(root / "example.parquet", manifest),
        )
        reference = pd.read_parquet(root / "expected.parquet")
        pd.testing.assert_frame_equal(
            result.drop(columns=["prob_" + c for c in LABELS]),
            reference.drop(columns=["prob_" + c for c in LABELS]),
        )
        cols = ["prob_" + c for c in LABELS]
        np.testing.assert_allclose(
            result[cols], reference[cols], rtol=0, atol=1e-6, equal_nan=True
        )
        inputs["objects"].update(publish_bundle(s3, b["releases"], root, manifest))
        inputs["release"] = manifest["release_id"]
        write_json(output, inputs)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("job")
    p.add_argument("--baseline", type=Path, default=Path("var/cloud/inputs.json"))
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    promote(a.job, a.baseline, a.output)
