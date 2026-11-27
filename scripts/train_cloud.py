"""Train on the frozen development cohort on EC2. Each fit has its own MLflow run.

`stack` refits the served hybrid (`stack_five`); `refit` and `tune` train its tabular LightGBM.
"""

import argparse
import json
import os
import sqlite3
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import boto3
import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from botocore.exceptions import ClientError
from mlflow.tracking import MlflowClient
from sklearn.base import clone

from sleep_staging.booster_selection import (
    BoosterCandidate,
    choose_trial,
    split_subjects,
)
from sleep_staging.booster_worker import Parameters, fit_booster, tracked_run
from sleep_staging.evaluation import evaluate, scalar_metrics
from sleep_staging.lstm_data import SequenceData
from sleep_staging.provenance import fingerprint, hash_file, write_json
from sleep_staging.serving.contract import (
    LABELS,
    context_matrix,
    predictions,
    read_night,
)
from sleep_staging.stack_release import MODEL as STACK
from sleep_staging.stack_training import KEYS, PROBS, aligned, stacked

RESEARCH_INPUTS = ("dataset", "splits", "grid", "fullnight_oof", "stack_example")
STACK_INPUTS = ("model.joblib", "fullnight_oof", "stack_example")


def download(s3: Any, bucket: str, pointer: dict, path: Path) -> None:
    s3.download_file(
        bucket, pointer["key"], str(path), ExtraArgs={"VersionId": pointer["version"]}
    )
    if hash_file(path) != pointer["sha256"]:
        raise ValueError("Input checksum mismatch: " + path.name)


def development(
    raw: pd.DataFrame, splits: pd.DataFrame, manifest: dict
) -> SequenceData:
    if splits.subject_id.duplicated().any() or set(splits.partition) != {
        "development",
        "test",
    }:
        raise ValueError("Invalid or overlapping subject partitions.")
    frozen = splits.loc[splits.partition.eq("development")].set_index(
        "subject_id", verify_integrity=True
    )
    if (
        set(raw.subject_id) != set(frozen.index)
        or not raw.fold.eq(raw.subject_id.map(frozen.fold)).all()
        or set(raw.fold) != set(range(5))
    ):
        raise ValueError("Development subjects or folds differ from the frozen split.")
    raw = raw.sort_values(["recording_id", "onset_s"]).reset_index(drop=True)
    matrix = context_matrix(raw, manifest["raw_columns"])
    if list(matrix) != manifest["model_columns"]:
        raise ValueError("Training columns differ from serving.")
    meta = raw[["epoch_id", "recording_id", "subject_id", "onset_s", "fold", "task_5"]]
    return SequenceData.from_frame(pd.concat([meta, matrix], axis=1), list(matrix))


def stack_inputs(
    data: SequenceData, oof: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series]:
    """Append the out-of-fold FullNightNet stage probabilities to every labelled epoch."""
    rows = data.frame.iloc[data.targets].reset_index(drop=True)
    p = aligned(rows[["epoch_id", *KEYS, "task_5"]], oof)
    return stacked(rows[data.columns], p), rows.task_5


def stack_expected(model: Any, example: pd.DataFrame) -> pd.DataFrame:
    """Candidate predictions on the frozen inputs of the bundled EDF example."""
    p = model.predict_proba(example[list(model.feature_names_in_)])
    p = p[:, [list(model.classes_).index(s) for s in LABELS]]
    expected = example[["onset_s"]].assign(prediction=np.asarray(LABELS)[p.argmax(1)])
    expected[PROBS] = p
    return expected


def train(mode: str, root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3")
    inputs = json.loads(os.environ["CLOUD_INPUTS"])
    objects = inputs["objects"]
    b = inputs["buckets"]
    campaign = os.environ["AWS_BATCH_JOB_ID"]
    prefix = "candidates/" + campaign + "/"
    state_key = prefix + "progress.json"
    try:
        state = json.loads(
            s3.get_object(Bucket=b["research"], Key=state_key)["Body"].read()
        )
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in ("404", "NoSuchKey"):
            raise
        state = {"fits": {}}
    identity = fingerprint(
        {"inputs": inputs, "mode": mode, "image": os.environ.get("TRAIN_IMAGE", "")}
    )
    if state.get("identity", identity) != identity:
        raise ValueError("Checkpoint belongs to a different campaign.")
    state["identity"] = identity
    names = ["dataset", "splits", "grid", "manifest", "example.parquet"]
    if mode == "stack":
        if missing := [n for n in STACK_INPUTS if n not in objects]:
            raise ValueError(
                "Publish the hybrid inputs with publish_cloud_inputs.py --stack-run: "
                + ", ".join(missing)
            )
        names += STACK_INPUTS
    for name in names:
        bucket = b["research"] if name in RESEARCH_INPUTS else b["releases"]
        download(s3, bucket, objects[name], root / name)
    manifest = json.loads((root / "manifest").read_text())
    if (
        objects["dataset"]["sha256"] != manifest["dataset_sha256"]
        or objects["splits"]["sha256"] != manifest["split_sha256"]
    ):
        raise ValueError("Release and training cohort differ")
    frame = development(
        pd.read_parquet(root / "dataset"), pd.read_csv(root / "splits"), manifest
    )
    config = yaml.safe_load((root / "grid").read_text())
    cfg = Parameters(**config)
    db = root / "mlflow.db"
    managed = os.environ.get("MLFLOW_TRACKING_URI")
    if not managed and "db" in state:
        download(s3, b["research"], state["db"], db)
    mlflow.set_tracking_uri(managed or f"sqlite:///{db.resolve()}")
    name = "sleep-cassette-cloud"
    experiment = mlflow.get_experiment_by_name(name)
    eid = (
        experiment.experiment_id
        if experiment
        else mlflow.create_experiment(
            name,
            artifact_location=f"s3://{b['research']}/tracking/artifacts/{campaign}"
            if not managed
            else None,
        )
    )
    mlflow.set_experiment(experiment_id=eid)
    client = MlflowClient()

    def put(path: Path, key: str) -> dict:
        sha = hash_file(path)
        with path.open("rb") as stream:
            v = s3.put_object(
                Bucket=b["research"],
                Key=prefix + key,
                Body=stream,
                Metadata={"sha256": sha},
            )["VersionId"]
        return {"key": prefix + key, "version": v, "sha256": sha}

    def save() -> None:
        if not managed:
            backup = root / "snapshot.db"
            with sqlite3.connect(db) as src, sqlite3.connect(backup) as dest:
                src.backup(dest)
            state["db"] = put(backup, "tracking.db")
        s3.put_object(
            Bucket=b["research"], Key=state_key, Body=json.dumps(state).encode()
        )

    if "parent" not in state:
        state["parent"] = client.create_run(
            eid,
            tags={
                "mlflow.runName": mode + "-" + campaign,
                "cloud_job": campaign,
                "source_revision": os.environ.get("SOURCE_REVISION", "unversioned"),
                "image_digest": os.environ.get("TRAIN_IMAGE", ""),
                "purpose": mode,
                "test_evaluated": "false",
            },
        ).info.run_id
        save()
    parent = state["parent"]
    os.environ["SLEEP_BOOSTER_PARENT_RUN"] = parent

    def fit(
        key: str,
        candidate: BoosterCandidate,
        train_ids: list,
        valid_ids: list,
        seed: int,
        iterations: int | None = None,
    ) -> dict:
        if key in state["fits"]:
            fit_info = state["fits"][key]
            if client.get_run(fit_info["mlflow_run_id"]).info.status != "FINISHED":
                raise ValueError("Incomplete MLflow checkpoint")
            return fit_info
        folder = root / "fits" / key
        summary = fit_booster(
            frame,
            cfg,
            candidate,
            train_ids,
            valid_ids,
            folder,
            fingerprint([identity, key]),
            seed,
            iterations,
        )
        summary["checkpoint"] = put(
            folder / "model.joblib", "fits/" + key + "/model.joblib"
        )
        state["fits"][key] = summary
        save()
        return summary

    def model_from(info: dict) -> Any:
        path = root / "selected.joblib"
        download(s3, b["research"], info["checkpoint"], path)
        return joblib.load(path)

    def fit_stack() -> dict:
        """Clone the approved LightGBM and refit it on 435 columns plus FullNightNet."""
        if "stack" in state["fits"]:
            fit_info = state["fits"]["stack"]
            if client.get_run(fit_info["mlflow_run_id"]).info.status != "FINISHED":
                raise ValueError("Incomplete MLflow checkpoint")
            return fit_info
        if hash_file(root / "model.joblib") != manifest["model_sha256"]:
            raise ValueError("Base LightGBM differs from the release manifest")
        x, y = stack_inputs(frame, pd.read_parquet(root / "fullnight_oof"))
        folder = root / "fits" / "stack"
        model = clone(joblib.load(root / "model.joblib"))
        start = time.monotonic()
        with tracked_run(
            folder, STACK, "stack_fixed_refit", fingerprint([identity, "stack"])
        ) as run_id:
            model.fit(x, y)
            path = folder / "stack_five.joblib"
            joblib.dump(model, path, compress=3)
            params = model.named_steps["classifier"].get_params()
            mlflow.log_params(
                {
                    **{
                        k: v
                        for k, v in params.items()
                        if isinstance(v, int | float | str)
                    },
                    "base_release": manifest["release_id"],
                    "columns": x.shape[1],
                    "fullnight_run": objects["fullnight_oof"].get("run", ""),
                }
            )
            mlflow.log_metrics(
                {"training_epochs": len(x), "fit_seconds": time.monotonic() - start}
            )
            mlflow.log_artifact(str(path))
        summary = {
            "mlflow_run_id": run_id,
            "model_sha256": hash_file(path),
            "columns": list(x.columns),
            "training_epochs": len(x),
            "training_subjects": int(frame.frame.subject_id.nunique()),
            "elapsed_seconds": time.monotonic() - start,
            "checkpoint": put(path, "fits/stack/stack_five.joblib"),
        }
        state["fits"]["stack"] = summary
        save()
        return summary

    def finish(candidate: dict, files: dict) -> dict:
        write_json(root / "candidate.json", candidate)
        client.log_artifact(parent, str(root / "candidate.json"))
        state["candidate"] = {
            "manifest": put(root / "candidate.json", "manifest.json"),
            "expected.parquet": put(root / "expected.parquet", "expected.parquet"),
            **files,
        }
        client.set_terminated(parent, "FINISHED")
        save()
        print(
            json.dumps(
                {
                    "campaign": campaign,
                    "parent": parent,
                    "candidate": candidate["release_id"],
                    "fits": len(state["fits"]),
                }
            ),
            flush=True,
        )
        return state

    def provenance() -> dict:
        return {
            "training_job": campaign,
            "source_revision": os.environ.get("SOURCE_REVISION", "unversioned"),
            "training_image": os.environ.get("TRAIN_IMAGE"),
            "versions": {
                p: version(p)
                for p in [
                    "numpy",
                    "pandas",
                    "lightgbm",
                    "scikit-learn",
                    "mlflow",
                    "joblib",
                    "pyarrow",
                ]
            },
        }

    seed = cfg.seed + 500
    outer = []
    try:
        if mode == "stack":
            final = fit_stack()
            expected = stack_expected(
                model_from(final), pd.read_parquet(root / "stack_example")
            )
            expected.to_parquet(root / "expected.parquet", index=False)
            fullnight = objects["fullnight_oof"]
            return finish(
                {
                    "model": STACK,
                    "release_id": "stack-five-" + final["mlflow_run_id"][:12],
                    "run_id": final["mlflow_run_id"],
                    "parent_run_id": parent,
                    "model_sha256": final["model_sha256"],
                    "columns": final["columns"],
                    "training_epochs": final["training_epochs"],
                    "training_subjects": final["training_subjects"],
                    "base_release_id": manifest["release_id"],
                    "base_model_sha256": manifest["model_sha256"],
                    "dataset_sha256": manifest["dataset_sha256"],
                    "split_sha256": manifest["split_sha256"],
                    "fullnight_oof_sha256": fullnight["sha256"],
                    "fullnight_run": fullnight.get("run"),
                    "fullnight_checkpoint_sha256": fullnight.get("checkpoint_sha256"),
                    "stack_example_sha256": objects["stack_example"]["sha256"],
                    **provenance(),
                },
                {
                    "stack_five.joblib": final["checkpoint"],
                    "stack_example": objects["stack_example"],
                },
            )
        if mode == "tune":
            grid = [BoosterCandidate(**c) for c in config["lgbm"]]
            for fold in [0, 1, 2, 3, 4, -1]:
                split_seed = cfg.seed + (500 if fold < 0 else fold)
                train_ids, valid_ids, external = split_subjects(
                    frame.frame, fold, split_seed
                )
                trials = [
                    {
                        **fit(
                            f"fold-{fold}/{c.name}", c, train_ids, valid_ids, split_seed
                        ),
                        "candidate": i,
                    }
                    for i, c in enumerate(grid)
                ]
                best = choose_trial(trials)
                final = fit(
                    f"fold-{fold}/refit",
                    grid[best["candidate"]],
                    sorted(train_ids + valid_ids),
                    [],
                    split_seed,
                    best["best_epoch"],
                )
                if external:
                    held = frame.frame.loc[
                        frame.frame.subject_id.isin(external)
                        & frame.frame.task_5.notna()
                    ]
                    labels = model_from(final).predict(held[frame.columns])
                    metrics = scalar_metrics(evaluate(held.task_5, labels, LABELS))
                    for k, v in metrics.items():
                        client.log_metric(final["mlflow_run_id"], "outer_" + k, v)
                    outer.append(
                        held[["epoch_id", "subject_id", "fold", "task_5"]].assign(
                            prediction=labels
                        )
                    )
            oof = pd.concat(outer, ignore_index=True)
            if oof.epoch_id.duplicated().any() or len(oof) != len(frame.targets):
                raise ValueError("Incomplete OOF coverage")
            for k, v in scalar_metrics(
                evaluate(oof.task_5, oof.prediction, LABELS)
            ).items():
                client.log_metric(parent, "oof_" + k, v)
            oof.to_parquet(root / "oof.parquet", index=False)
            client.log_artifact(parent, str(root / "oof.parquet"))
        else:
            final = fit(
                "final",
                BoosterCandidate(**manifest["training_config"]),
                sorted(set(frame.frame.subject_id)),
                [],
                seed,
                manifest["iterations"],
            )
        model = model_from(final)
        night = read_night(root / "example.parquet", manifest)
        expected = predictions(model, manifest, night)
        expected.to_parquet(root / "expected.parquet", index=False)
        candidate = {
            **manifest,
            "release_id": "lgbm-" + final["model_sha256"][:16],
            "model_sha256": final["model_sha256"],
            "model_key": final["checkpoint"]["key"],
            "model_version": final["checkpoint"]["version"],
            "run_id": final["mlflow_run_id"],
            "parent_run_id": parent,
            "iterations": final["best_epoch"],
            "training_config": final["configuration"],
            **provenance(),
        }
        return finish(
            candidate,
            {
                "model.joblib": final["checkpoint"],
                "example.parquet": objects["example.parquet"],
            },
        )
    except Exception:
        client.set_terminated(parent, "FAILED")
        save()
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["stack", "refit", "tune"], default="stack")
    p.add_argument("--root", type=Path, default=Path("/tmp/training"))
    a = p.parse_args()
    train(a.mode, a.root)
