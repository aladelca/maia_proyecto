"""Exercise the deployed API and check all probabilities against the frozen night."""

import argparse
import hashlib
import json
import os
import time
import uuid
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
import requests

from sleep_staging.serving.contract import LABELS


def verify(config: dict, bundle: Path) -> dict:
    auth = boto3.client("cognito-idp").initiate_auth(
        ClientId=config["clientId"],
        AuthFlow="USER_PASSWORD_AUTH",
        AuthParameters={
            "USERNAME": os.environ["DEMO_USERNAME"],
            "PASSWORD": os.environ["DEMO_PASSWORD"],
        },
    )["AuthenticationResult"]
    url = config["api"] + "/v1/"
    client = requests.Session()
    if client.get(url + "examples/default", timeout=20).status_code != 401:
        raise RuntimeError("Anonymous request accepted")
    denied = client.get(
        url + "examples/default",
        headers={"Authorization": "Bearer " + auth["IdToken"]},
        timeout=20,
    )
    if denied.status_code not in (401, 403):
        raise RuntimeError("ID token accepted")
    client.headers.update({"Authorization": "Bearer " + auth["AccessToken"]})
    client.get(url + "examples/default", timeout=20).raise_for_status()
    example = (bundle / "example.parquet").read_bytes()
    sign = client.post(
        url + "uploads",
        json={"size": len(example), "sha256": hashlib.sha256(example).hexdigest()},
        timeout=30,
    )
    sign.raise_for_status()
    form = sign.json()
    upload = requests.post(
        form["url"],
        data=form["fields"],
        files={"file": ("night.parquet", example)},
        timeout=60,
        headers={"Origin": config["origin"]},
    )
    if upload.status_code != 201:
        raise RuntimeError(f"Upload failed: HTTP {upload.status_code}")
    expected = pd.read_parquet(bundle / "expected.parquet").set_index("epoch_id")
    report = {}
    for name, body in [
        ("example", {"example": "default"}),
        (
            "upload",
            {
                "upload_id": form["upload_id"],
                "version_id": upload.headers["x-amz-version-id"],
            },
        ),
    ]:
        started = time.monotonic()
        headers = {"Idempotency-Key": str(uuid.uuid4())}
        accepted = client.post(
            url + "predictions", headers=headers, json=body, timeout=30
        )
        accepted.raise_for_status()
        job = accepted.json()["job_id"]
        elapsed = time.monotonic() - started
        for _ in range(80):
            status = client.get(url + "predictions/" + job, timeout=20)
            status.raise_for_status()
            data = status.json()
            if data["state"] != "processing":
                break
            time.sleep(2)
        if data["state"] != "succeeded":
            raise RuntimeError(
                f"Prediction {job}: {data['state']} {data.get('reason', '')}"
            )
        reply = requests.get(data["files"]["json"], timeout=25)
        if not reply.ok:
            raise RuntimeError(f"Result download: HTTP {reply.status_code}")
        result = pd.DataFrame(reply.json()["epochs"]).set_index("epoch_id")
        cols = ["prob_" + c for c in LABELS]
        assert result.index.equals(expected.index)
        pd.testing.assert_series_equal(
            result.prediction, expected.prediction, check_dtype=False
        )
        np.testing.assert_allclose(
            result[cols], expected[cols], rtol=0, atol=1e-6, equal_nan=True
        )
        replay = client.post(
            url + "predictions", headers=headers, json=body, timeout=20
        )
        assert replay.status_code == 202 and replay.json()["job_id"] == job
        if name == "upload":
            assert (
                client.post(
                    url + "predictions",
                    headers=headers,
                    json={"example": "default"},
                    timeout=20,
                ).status_code
                == 409
            )
        report[name] = {
            "job": job,
            "rows": len(result),
            "eligible": data["eligible"],
            "release": data["model_version"],
            "max_difference": float(np.nanmax(abs(result[cols] - expected[cols]))),
            "accept_seconds": round(elapsed, 3),
            "total_seconds": round(time.monotonic() - started, 3),
        }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("frontend/dist/config.json")
    )
    parser.add_argument("--bundle", type=Path, default=Path("var/cloud/bundle"))
    args = parser.parse_args()
    print(
        json.dumps(verify(json.loads(args.config.read_text()), args.bundle), indent=2)
    )
