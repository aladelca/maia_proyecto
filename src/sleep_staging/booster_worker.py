"""Isolated CPU LightGBM worker: never imports PyTorch/OpenMP from Torch."""

import json
import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from mlflow.entities import Metric
from mlflow.tracking import MlflowClient
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sleep_staging.config import CLASSES
from sleep_staging.evaluation import evaluate, scalar_metrics
from sleep_staging.lstm_data import SequenceData
from sleep_staging.provenance import atomic_path, hash_file, write_json
from sleep_staging.refinement_data import class_weights, macro_f1_eval

CLASSES_5 = CLASSES["task_5"]


class Parameters(SimpleNamespace):
    def model_dump(self, exclude: set | None = None) -> dict:
        return {k: v for k, v in vars(self).items() if k not in (exclude or set())}


@contextmanager
def tracked_run(
    directory: Path, name: str, purpose: str, identity: str, device: str = "cpu"
) -> Iterator[str]:
    directory.mkdir(parents=True, exist_ok=True)
    journal = directory / "mlflow-run.json"
    run_id = json.loads(journal.read_text())["run_id"] if journal.exists() else None
    tags = {
        "purpose": purpose,
        "identity": identity,
        "test_evaluated": "false",
        "device": device,
        "torch_loaded": "false",
    }
    if parent := os.environ.get("SLEEP_BOOSTER_PARENT_RUN"):
        tags["mlflow.parentRunId"] = parent
    with mlflow.start_run(run_id=run_id, run_name=name, tags=tags) as run:
        write_json(journal, {"run_id": run.info.run_id, "identity": identity})
        yield run.info.run_id


def fit_booster(
    data: SequenceData,
    cfg: Any,
    candidate: Any,
    training_subjects: list,
    validation_subjects: list,
    directory: Path,
    identity: str,
    seed: int,
    fixed_iterations: int | None = None,
) -> dict:
    path = directory / "summary.json"
    if path.exists():
        result = json.loads(path.read_text())
        if (
            result["identity"] != identity
            or hash_file(directory / "model.joblib") != result["model_sha256"]
        ):
            raise ValueError("Booster checkpoint identity/checksum mismatch.")
        if MlflowClient().get_run(result["mlflow_run_id"]).info.status != "FINISHED":
            raise ValueError("Unfinished booster cannot be reused.")
        return result
    if set(training_subjects) & set(validation_subjects):
        raise ValueError("Training and validation subjects overlap.")
    mask = data.frame.subject_id.isin(training_subjects).to_numpy()
    training = data.targets[mask[data.targets]]
    mask = data.frame.subject_id.isin(validation_subjects).to_numpy()
    validation = data.targets[mask[data.targets]]
    if not len(training) or fixed_iterations is None and not len(validation):
        raise ValueError("Nonempty training and validation required.")
    weights = class_weights(data.labels[training], candidate.weight_power)
    if len(validation) and set(data.labels[validation]) != set(range(5)):
        raise ValueError("Missing inner validation classes.")
    features = data.frame[data.columns]
    preprocess = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]
    )
    x_train = preprocess.fit_transform(features.iloc[training])
    params = candidate.model_dump(exclude={"name", "weight_power"})
    params["n_estimators"] = fixed_iterations or candidate.n_estimators
    model = lgb.LGBMClassifier(
        **params,
        class_weight=dict(zip(CLASSES_5, weights, strict=True)),
        objective="multiclass",
        metric="None",
        random_state=seed,
        n_jobs=cfg.lgbm_threads,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
    )
    history: dict = {}
    start = time.monotonic()
    purpose = "lgbm_inner_tuning" if fixed_iterations is None else "lgbm_fixed_refit"
    with tracked_run(
        directory, candidate.name, purpose, identity, device="cpu"
    ) as run_id:
        mlflow.log_params(
            {
                **candidate.model_dump(),
                "seed": seed,
                "fixed_iterations": str(fixed_iterations),
                "threads": cfg.lgbm_threads,
                "selection_metric": "unweighted_macro_f1",
            }
        )
        audit = {
            "training_subjects": training_subjects,
            "validation_subjects": validation_subjects,
            "training_targets": len(training),
            "validation_targets": len(validation),
            "columns": data.columns,
            "class_weights": weights.tolist(),
            "preprocess_mean": preprocess.named_steps["scale"].mean_.tolist(),
            "preprocess_scale": preprocess.named_steps["scale"].scale_.tolist(),
            "candidate": candidate.model_dump(),
        }
        write_json(directory / "audit.json", audit)
        if fixed_iterations is None:
            x_valid = preprocess.transform(features.iloc[validation])
            model.fit(
                x_train,
                np.asarray(CLASSES_5)[data.labels[training]],
                eval_set=[(x_valid, np.asarray(CLASSES_5)[data.labels[validation]])],
                eval_metric=macro_f1_eval,
                callbacks=[
                    lgb.early_stopping(
                        cfg.lgbm_patience,
                        first_metric_only=True,
                        verbose=False,
                        min_delta=cfg.min_delta,
                    ),
                    lgb.record_evaluation(history),
                ],
            )
        else:
            model.fit(x_train, np.asarray(CLASSES_5)[data.labels[training]])
        iterations = int(model.best_iteration_ or model.n_iter_)
        pipeline = Pipeline([*preprocess.steps, ("classifier", model)])
        metrics = (
            evaluate(
                np.asarray(CLASSES_5)[data.labels[validation]],
                pipeline.predict(features.iloc[validation]),
                CLASSES_5,
            )
            if len(validation)
            else None
        )
        with atomic_path(directory / "model.joblib") as temporary:
            joblib.dump(pipeline, temporary)
        write_json(directory / "history.json", history)
        result = {
            "identity": identity,
            "directory": str(directory.resolve()),
            "best_epoch": iterations,
            "best_f1_macro": metrics["f1_macro"] if metrics else None,
            "inner_metrics": metrics,
            "elapsed_seconds": time.monotonic() - start,
            "model_sha256": hash_file(directory / "model.joblib"),
            "mlflow_run_id": run_id,
            "configuration": candidate.model_dump(),
            "device": "cpu",
            "torch_loaded": "torch" in sys.modules,
            "worker_pid": os.getpid(),
            "estimator_n_estimators": model.n_estimators,
        }
        if metrics:
            mlflow.log_metrics(
                {"inner_" + k: v for k, v in scalar_metrics(metrics).items()}
            )
            curve = history["valid_0"]["macro_f1"]
            timestamp = int(time.time() * 1000)
            MlflowClient().log_batch(
                run_id,
                metrics=[
                    Metric("inner_macro_f1_curve", float(value), timestamp, i + 1)
                    for i, value in enumerate(curve)
                ],
            )
        mlflow.log_metrics(
            {
                "selected_iterations": iterations,
                "elapsed_seconds": result["elapsed_seconds"],
            }
        )
        mlflow.log_artifacts(str(directory), "training")
    write_json(path, result)
    MlflowClient().log_artifact(run_id, str(path), artifact_path="training")
    print(
        f"FIT_LGBM {candidate.name} f1={result['best_f1_macro']} trees={iterations} seconds={result['elapsed_seconds']:.1f}",
        flush=True,
    )
    return result


def main() -> None:
    request = json.loads(Path(sys.argv[1]).read_text())
    if "torch" in sys.modules:
        raise RuntimeError("CPU LightGBM worker must not import PyTorch.")
    mlflow.set_tracking_uri(request["tracking_uri"])
    mlflow.set_experiment(experiment_id=request["experiment_id"])
    cache = Path(request["input_root"])
    schema = json.loads((cache / "schema.json").read_text())
    for name in ["features.npy", "metadata.parquet"]:
        if hash_file(cache / name) != schema["hashes"][name]:
            raise ValueError("Booster input cache checksum mismatch.")
    matrix = np.load(cache / "features.npy", mmap_mode="r")
    frame = pd.concat(
        [
            pd.read_parquet(cache / "metadata.parquet"),
            pd.DataFrame(matrix, columns=schema["columns"]),
        ],
        axis=1,
    )
    data = SequenceData.from_frame(frame, schema["columns"])
    if request["mode"] == "fit":
        fit_booster(
            data,
            Parameters(**request["config"]),
            Parameters(**request["candidate"]),
            request["training_subjects"],
            request["validation_subjects"],
            Path(request["directory"]),
            request["identity"],
            request["seed"],
            request["fixed_iterations"],
        )
    else:
        model = joblib.load(request["model_path"])
        p = model.predict_proba(data.frame.iloc[request["rows"]][data.columns])
        p = p[:, [list(model.classes_).index(c) for c in CLASSES_5]]
        with atomic_path(Path(request["probability_path"])) as temporary:
            with temporary.open("wb") as stream:
                np.save(stream, p)


if __name__ == "__main__":
    main()
