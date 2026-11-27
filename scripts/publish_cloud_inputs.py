"""Publish immutable seed inputs. Each pointer includes an S3 VersionId and checksum."""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import boto3
import pandas as pd
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


def stack_inputs(campaign: Path, folder: Path) -> list[tuple[str, Path, dict]]:
    """Out-of-fold FullNightNet probabilities and the hybrid inputs of the bundled EDF."""
    from sleep_staging.serving.stacked_edf import EXAMPLE_PATH, classifier, inputs
    from sleep_staging.stack_release import CHECKPOINTS
    from sleep_staging.stack_training import fullnight_oof

    freeze = json.loads((campaign / "freeze.json").read_text())
    weights = hash_file(campaign / "refit" / "six_direct.pt")
    if (
        freeze["smoke"]
        or freeze["edf_model"] != "six_direct.pt"
        or weights != freeze["weights"]["six_direct.pt"]
        or weights != CHECKPOINTS["six_direct.pt"]
    ):
        raise ValueError("La campaña FullNightNet no corresponde a la red publicada.")
    oof = folder / "fullnight-oof.parquet"
    fullnight_oof(campaign).to_parquet(oof, index=False)
    result, x = inputs(EXAMPLE_PATH, "night", list(classifier().feature_names_in_))
    example = folder / "stack-example.parquet"
    pd.concat(
        [
            result.loc[result.eligible, ["onset_s"]].reset_index(drop=True),
            x.reset_index(drop=True),
        ],
        axis=1,
    ).to_parquet(example, index=False)
    fullnight = {"run": freeze["run"], "checkpoint_sha256": weights}
    return [("fullnight_oof", oof, fullnight), ("stack_example", example, {})]


def publish(
    bundle: Path,
    output: Path,
    edf_run: Path | None = None,
    stack_run: Path | None = None,
) -> None:
    s3 = boto3.client("s3")
    buckets = platform_outputs()["buckets"]
    m = json.loads((bundle / "manifest.json").read_text())
    edf_model = None
    if edf_run is not None:
        freeze = json.loads((edf_run / "freeze.json").read_text())
        name = freeze["edf_model"]
        if freeze["smoke"] or name not in {"six_direct.pt", "six_movement.pt"}:
            raise ValueError("Se requiere una campaña EDF completa y congelada.")
        edf_model = edf_run / "refit" / name
        if hash_file(edf_model) != freeze["weights"][name]:
            raise ValueError("Checksum del modelo EDF incompatible.")
    pointers = publish_bundle(s3, buckets["releases"], bundle, m)
    if edf_model is not None:
        pointers["edf_model"] = put_verified(
            s3,
            buckets["releases"],
            "releases/ibio-edf/" + hash_file(edf_model) + "/model.pt",
            edf_model,
        )
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
    with tempfile.TemporaryDirectory() as tmp:
        stacked = stack_inputs(stack_run, Path(tmp)) if stack_run else []
        for label, path, extra in stacked:
            pointers[label] = {
                **put_verified(
                    s3,
                    buckets["research"],
                    "datasets/" + hash_file(path) + "/" + path.name,
                    path,
                ),
                **extra,
            }
    release = {"buckets": buckets, "objects": pointers, "release": m["release_id"]}
    write_json(output, release)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, default=Path("var/cloud/bundle"))
    p.add_argument("--output", type=Path, default=Path("var/cloud/inputs.json"))
    p.add_argument(
        "--edf-run",
        type=Path,
        help="Campaña completa con freeze.json para habilitar cargas EDF.",
    )
    p.add_argument(
        "--stack-run",
        type=Path,
        help="Campaña FullNightNet del híbrido: publica las entradas de train_cloud.py --mode stack.",
    )
    a = p.parse_args()
    publish(a.bundle, a.output, a.edf_run, a.stack_run)
