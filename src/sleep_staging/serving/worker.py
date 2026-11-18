"""Version-pinned S3 inference for the Step Functions task."""

import hashlib
import json
import os
import tempfile
import time
from contextlib import closing
from functools import lru_cache
from pathlib import Path
from typing import Any

import boto3
import joblib

from sleep_staging.provenance import hash_file
from sleep_staging.serving.contract import MAX_BYTES, predictions, read_night


@lru_cache(maxsize=1)
def load_release() -> tuple[dict, Any]:
    s3 = boto3.client("s3")
    bucket = os.environ["RELEASE_BUCKET"]
    obj = s3.get_object(
        Bucket=bucket,
        Key=os.environ["MANIFEST_KEY"],
        VersionId=os.environ["MANIFEST_VERSION"],
    )
    with closing(obj["Body"]) as stream:
        data = stream.read()
    if hashlib.sha256(data).hexdigest() != os.environ["MANIFEST_SHA256"]:
        raise RuntimeError("Manifest integrity failure")
    manifest = json.loads(data)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "model.joblib"
        s3.download_file(
            bucket,
            manifest["model_key"],
            str(path),
            ExtraArgs={"VersionId": manifest["model_version"]},
        )
        if hash_file(path) != manifest["model_sha256"]:
            raise RuntimeError("Model integrity failure")
        model = joblib.load(path)
    return manifest, model


def handler(event: dict, context: Any) -> dict:
    import pyarrow as pa

    start = time.monotonic()
    s3 = boto3.client("s3")
    manifest, model = load_release()
    jobs = os.environ["JOBS_BUCKET"]
    prefix = f"users/{event['owner']}/results/{event['job']}"
    bucket = os.environ["RELEASE_BUCKET"] if event["example"] else jobs
    obj = s3.get_object(
        Bucket=bucket, Key=event["input_key"], VersionId=event["input_version"]
    )
    with closing(obj["Body"]) as stream, tempfile.TemporaryDirectory() as folder:
        if obj["ContentLength"] > MAX_BYTES:
            return {"state": "failed", "reason": "Archivo demasiado grande."}
        path = Path(folder) / "input.parquet"
        with path.open("wb") as f:
            for chunk in stream.iter_chunks(65536):
                f.write(chunk)
        if hash_file(path) != event["sha256"]:
            return {
                "state": "failed",
                "reason": "El archivo no coincide con su checksum.",
            }
        try:
            frame = read_night(path, manifest)
        except (ValueError, pa.ArrowException, OverflowError):
            return {
                "state": "failed",
                "reason": "El archivo no cumple el contrato de una noche preparada.",
            }
        result = predictions(model, manifest, frame)
        summary = {
            "job_id": event["job"],
            "inference_seconds": round(time.monotonic() - start, 3),
            "model_version": manifest["release_id"],
            "preprocessing_id": manifest["preprocessing_id"],
            "rows": len(frame),
            "eligible": int(frame.eligible.sum()),
            "input_sha256": event["sha256"],
        }
        payload = {
            **summary,
            "epochs": json.loads(result.to_json(orient="records", double_precision=15)),
        }
        for ext, body, mime in [
            ("json", json.dumps(payload, allow_nan=False), "application/json"),
            ("csv", result.to_csv(index=False), "text/csv"),
        ]:
            s3.put_object(
                Bucket=jobs,
                Key=f"{prefix}.{ext}",
                Body=body.encode(),
                ContentType=mime,
                ContentDisposition=(
                    f'attachment; filename="{event["job"]}.csv"'
                    if ext == "csv"
                    else "inline"
                ),
            )
        return {"state": "succeeded", **summary}
