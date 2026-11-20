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
from sleep_staging.sensor_profiles import MAX_EDF_BYTES, validate_lgbm_profile
from sleep_staging.serving.contract import LABELS, MAX_BYTES, predictions, read_night


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


@lru_cache(maxsize=1)
def load_edf_release() -> tuple[Any, dict]:
    import torch

    from sleep_staging.fullnight.training import load_model

    s3 = boto3.client("s3")
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "model.pt"
        s3.download_file(
            os.environ["RELEASE_BUCKET"],
            os.environ["EDF_MODEL_KEY"],
            str(path),
            ExtraArgs={"VersionId": os.environ["EDF_MODEL_VERSION"]},
        )
        if hash_file(path) != os.environ["EDF_MODEL_SHA256"]:
            raise RuntimeError("EDF model integrity failure")
        torch.set_num_threads(2)
        return load_model(path, torch.device("cpu"))


def handler(event: dict, context: Any) -> dict:
    import pyarrow as pa

    start = time.monotonic()
    s3 = boto3.client("s3")
    edf = event.get("format") == "edf"
    jobs = os.environ["JOBS_BUCKET"]
    prefix = f"users/{event['owner']}/results/{event['job']}"
    bucket = os.environ["RELEASE_BUCKET"] if event["example"] else jobs
    obj = s3.get_object(
        Bucket=bucket, Key=event["input_key"], VersionId=event["input_version"]
    )
    with closing(obj["Body"]) as stream, tempfile.TemporaryDirectory() as folder:
        if obj["ContentLength"] > (MAX_EDF_BYTES if edf else MAX_BYTES):
            return {"state": "failed", "reason": "Archivo demasiado grande."}
        path = Path(folder) / ("input.edf" if edf else "input.parquet")
        with path.open("wb") as f:
            for chunk in stream.iter_chunks(65536):
                f.write(chunk)
        if hash_file(path) != event["sha256"]:
            return {
                "state": "failed",
                "reason": "El archivo no coincide con su checksum.",
            }
        if edf and event.get("model") == "lightgbm":
            from sleep_staging.serving.edf_lgbm import (
                IncompatibleEDF,
                prepare_edf,
                score_night,
            )

            manifest, model = load_release()
            try:
                profile = validate_lgbm_profile(event.get("profile", "auto"))
                frame = prepare_edf(path, event["job"])
            except IncompatibleEDF as exc:
                return {"state": "failed", "reason": str(exc)}
            except (ValueError, EOFError, OverflowError, OSError):
                return {"state": "failed", "reason": "EDF incompatible con LightGBM."}
            result = score_night(model, manifest, frame)
            details = {
                "model_version": manifest["release_id"],
                "preprocessing_id": manifest["preprocessing_id"],
                "classes": LABELS,
                "profile": profile,
                "window": "full",
                "rows": len(frame),
                "eligible": int(frame.eligible.sum()),
            }
        elif edf:
            if event.get("model", "fullnight") != "fullnight":
                return {"state": "failed", "reason": "Modelo EDF desconocido."}
            import torch

            from sleep_staging.serving.edf import edf_summary, predict_edf

            net, meta = load_edf_release()
            try:
                result, profile = predict_edf(
                    path,
                    net,
                    event.get("profile", "auto"),
                    event["job"],
                    torch.device("cpu"),
                )
            except (ValueError, EOFError, OverflowError, OSError):
                return {
                    "state": "failed",
                    "reason": "EDF o perfil de canales incompatible.",
                }
            details = edf_summary(result, profile, meta)
        else:
            manifest, model = load_release()
            try:
                frame = read_night(path, manifest)
            except (ValueError, pa.ArrowException, OverflowError):
                return {
                    "state": "failed",
                    "reason": "El archivo no cumple el contrato de una noche preparada.",
                }
            result = predictions(model, manifest, frame)
            details = {
                "model_version": manifest["release_id"],
                "preprocessing_id": manifest["preprocessing_id"],
                "rows": len(frame),
                "eligible": int(frame.eligible.sum()),
            }
        summary = {
            **details,
            "job_id": event["job"],
            "inference_seconds": round(time.monotonic() - start, 3),
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
