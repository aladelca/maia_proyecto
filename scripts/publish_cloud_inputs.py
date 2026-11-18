"""Publish immutable seed inputs. Each pointer includes an S3 VersionId and checksum."""

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError

from sleep_staging.provenance import hash_file, write_json


def platform_outputs() -> dict:
    data = json.loads(
        subprocess.check_output(
            ["terraform", "-chdir=infra/platform", "output", "-json"]
        )
    )
    return {k: v["value"] for k, v in data.items()}


def put_verified(s3: Any, bucket: str, key: str, path: Path) -> dict:
    digest = hash_file(path)
    try:
        head = s3.head_object(Bucket=bucket, Key=key)
        if head["Metadata"].get("sha256") != digest:
            raise ValueError("Immutable object differs: " + key)
        version = head["VersionId"]
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in ("404", "NoSuchKey"):
            raise
        with path.open("rb") as f:
            version = s3.put_object(
                Bucket=bucket,
                Key=key,
                Body=f,
                IfNoneMatch="*",
                Metadata={"sha256": digest},
            )["VersionId"]
    return {"key": key, "version": version, "sha256": digest}


def publish_bundle(s3: Any, bucket: str, bundle: Path, m: dict) -> dict:
    prefix = "releases/" + m["release_id"] + "/"
    pointers = {}
    for name in ("model.joblib", "example.parquet", "expected.parquet"):
        pointers[name] = put_verified(s3, bucket, prefix + name, bundle / name)
    model = pointers["model.joblib"]
    m.update(model_key=model["key"], model_version=model["version"])
    write_json(bundle / "cloud-manifest.json", m)
    manifest_key = (
        prefix + "manifests/" + hash_file(bundle / "cloud-manifest.json") + ".json"
    )
    pointers["manifest"] = put_verified(
        s3, bucket, manifest_key, bundle / "cloud-manifest.json"
    )
    return pointers


def publish(bundle: Path, output: Path) -> None:
    s3 = boto3.client("s3")
    buckets = platform_outputs()["buckets"]
    m = json.loads((bundle / "manifest.json").read_text())
    pointers = publish_bundle(s3, buckets["releases"], bundle, m)
    research = json.loads(Path("var/research/sc-w30-v2/campaign.json").read_text())
    dataset = Path(research["dataset"])
    if hash_file(dataset) != m["dataset_sha256"]:
        raise ValueError("Dataset fingerprint mismatch")
    for label, path in [
        ("dataset", dataset),
        ("splits", Path("configs/splits/sc_subjects_v1.csv")),
        ("grid", Path("configs/research/refinement.yaml")),
    ]:
        pointers[label] = put_verified(
            s3,
            buckets["research"],
            "datasets/" + hash_file(path) + "/" + path.name,
            path,
        )
    release = {"buckets": buckets, "objects": pointers, "release": m["release_id"]}
    write_json(output, release)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, default=Path("var/cloud/bundle"))
    p.add_argument("--output", type=Path, default=Path("var/cloud/inputs.json"))
    a = p.parse_args()
    publish(a.bundle, a.output)
