"""Bounded nested tuning of LightGBM and feature-sequence BiLSTM."""

import hashlib
import json
import os
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import mlflow
import numpy as np
import pandas as pd
from filelock import FileLock
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, ConfigDict, Field, model_validator

from sleep_staging.config import CLASSES, read_yaml
from sleep_staging.evaluation import evaluate, plot_confusion, scalar_metrics
from sleep_staging.lstm_data import SequenceData
from sleep_staging.lstm_research import (
    LSTMConfig,
    WindowSpec,
    choose_trial,
    fit_sequence,
    load_reference,
    predict_checkpoint,
    split_subjects,
    tracked_run,
)
from sleep_staging.models.lstm import require_mps
from sleep_staging.provenance import atomic_path, fingerprint, hash_file, write_json
from sleep_staging.refinement_data import refinement_views
from sleep_staging.research import paired_subject_bootstrap

CLASSES_5 = CLASSES["task_5"]


class BoosterCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    n_estimators: int = Field(default=450, ge=1)
    learning_rate: float = Field(default=0.05, gt=0)
    num_leaves: int = Field(default=31, ge=2)
    min_child_samples: int = Field(default=100, ge=1)
    max_bin: int = Field(default=63, ge=2)
    reg_alpha: float = Field(default=0, ge=0)
    reg_lambda: float = Field(default=3, ge=0)
    colsample_bytree: float = Field(default=0.8, gt=0, le=1)
    subsample: float = Field(default=0.8, gt=0, le=1)
    subsample_freq: int = Field(default=1, ge=0)
    weight_power: float = Field(default=1, ge=0, le=1)


class NeuralCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    view: Literal["raw145", "night145", "context435", "night435"]
    hidden_size: int = Field(default=64, ge=1)
    dropout: float = Field(default=0.4, ge=0, lt=1)
    weight_power: float = Field(default=0.5, ge=0, le=1)
    lr: float = Field(default=0.0003, gt=0)


class RefinementConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: int = 42
    lgbm_patience: int = Field(default=40, ge=1)
    lgbm_threads: int = Field(default=4, ge=1, le=8)
    min_delta: float = Field(default=0.0001, ge=0)
    nn_max_epochs: int = Field(default=50, ge=1)
    nn_patience: int = Field(default=7, ge=1)
    batch_size: int = Field(default=256, ge=1)
    lgbm: list[BoosterCandidate]
    bilstm: list[NeuralCandidate]

    @model_validator(mode="after")
    def valid_candidates(self) -> "RefinementConfig":
        for names in [[c.name for c in self.lgbm], [c.name for c in self.bilstm]]:
            if not names or len(set(names)) != len(names):
                raise ValueError("Each family requires unique nonempty candidates.")
        return self


def neural_config(
    cfg: RefinementConfig, candidate: NeuralCandidate
) -> tuple[LSTMConfig, WindowSpec]:
    spec = WindowSpec(name=candidate.name, length=15, centered=True)
    local = LSTMConfig(
        seed=cfg.seed,
        hidden_size=candidate.hidden_size,
        dropout=candidate.dropout,
        batch_size=cfg.batch_size,
        max_epochs=cfg.nn_max_epochs,
        patience=cfg.nn_patience,
        min_delta=cfg.min_delta,
        weight_power=candidate.weight_power,
        learning_rates=[candidate.lr],
        variants=[spec],
    )
    return local, spec


def _booster_inputs(data: SequenceData, directory: Path) -> Path:
    root = next((p for p in directory.parents if p.name == "lgbm"), directory.parent)
    metadata = data.frame[
        ["epoch_id", "recording_id", "subject_id", "onset_s", "fold", "task_5"]
    ]
    key = fingerprint(
        {
            "values": hashlib.sha256(
                np.ascontiguousarray(data.values).view(np.uint8)
            ).hexdigest(),
            "metadata": hashlib.sha256(
                pd.util.hash_pandas_object(metadata, index=False).to_numpy().tobytes()
            ).hexdigest(),
            "columns": data.columns,
        }
    )
    cache = root / "booster-inputs" / key
    cache.mkdir(parents=True, exist_ok=True)
    if not (cache / "schema.json").exists():
        with atomic_path(cache / "features.npy") as temporary:
            with temporary.open("wb") as stream:
                np.save(stream, data.values)
        with atomic_path(cache / "metadata.parquet") as temporary:
            metadata.to_parquet(temporary, index=False)
        write_json(
            cache / "schema.json",
            {
                "columns": data.columns,
                "hashes": {
                    name: hash_file(cache / name)
                    for name in ["features.npy", "metadata.parquet"]
                },
            },
        )
    return cache


def _experiment_id() -> str:
    active = mlflow.active_run()
    value = (
        active.info.experiment_id if active else os.environ.get("MLFLOW_EXPERIMENT_ID")
    )
    if value is None:
        raise ValueError("Initialize an MLflow experiment before training.")
    return value


def _booster_process(request: dict, request_path: Path) -> None:
    write_json(request_path, request)
    environment = os.environ.copy()
    active = mlflow.active_run()
    if active is not None:
        environment["SLEEP_BOOSTER_PARENT_RUN"] = active.info.run_id
    subprocess.run(
        [sys.executable, "-m", "sleep_staging.booster_worker", str(request_path)],
        env=environment,
        check=True,
    )


def fit_booster(
    data: SequenceData,
    cfg: RefinementConfig,
    candidate: BoosterCandidate,
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
    directory.mkdir(parents=True, exist_ok=True)
    request = {
        "mode": "fit",
        "input_root": str(_booster_inputs(data, directory).resolve()),
        "config": cfg.model_dump(),
        "candidate": candidate.model_dump(),
        "training_subjects": training_subjects,
        "validation_subjects": validation_subjects,
        "directory": str(directory.resolve()),
        "identity": identity,
        "seed": seed,
        "fixed_iterations": fixed_iterations,
        "tracking_uri": mlflow.get_tracking_uri(),
        "experiment_id": _experiment_id(),
    }
    _booster_process(request, directory / "worker-request.json")
    return json.loads(path.read_text())


def fit_candidate(
    family: str,
    views: dict[str, SequenceData],
    cfg: RefinementConfig,
    candidate: BoosterCandidate | NeuralCandidate,
    training: list,
    validation: list,
    directory: Path,
    identity: str,
    seed: int,
    fixed: int | None = None,
) -> dict:
    if family == "lgbm":
        assert isinstance(candidate, BoosterCandidate)
        return fit_booster(
            views["context435"],
            cfg,
            candidate,
            training,
            validation,
            directory,
            identity,
            seed,
            fixed,
        )
    assert isinstance(candidate, NeuralCandidate)
    local, spec = neural_config(cfg, candidate)
    result = fit_sequence(
        views[candidate.view],
        local,
        spec,
        training,
        validation,
        candidate.lr,
        directory,
        identity,
        seed,
        fixed_epochs=fixed,
        purpose="bilstm_inner_tuning" if fixed is None else "bilstm_fixed_refit",
    )
    return {**result, "configuration": candidate.model_dump(), "device": "mps"}


def predict_fit(
    family: str,
    views: dict[str, SequenceData],
    cfg: RefinementConfig,
    candidate: BoosterCandidate | NeuralCandidate,
    fit: dict,
    rows: np.ndarray,
) -> np.ndarray:
    directory = Path(fit["directory"])
    if family == "lgbm":
        directory.mkdir(parents=True, exist_ok=True)
        key = fingerprint({"model": fit["model_sha256"], "rows": rows.tolist()})
        probability_path = directory / f"prediction-{key}.npy"
        request = {
            "mode": "predict",
            "input_root": str(
                _booster_inputs(views["context435"], directory).resolve()
            ),
            "model_path": str((directory / "model.joblib").resolve()),
            "rows": rows.tolist(),
            "probability_path": str(probability_path.resolve()),
            "tracking_uri": mlflow.get_tracking_uri(),
            "experiment_id": _experiment_id(),
        }
        _booster_process(request, directory / f"prediction-{key}.json")
        return np.load(probability_path)
    assert isinstance(candidate, NeuralCandidate)
    local, spec = neural_config(cfg, candidate)
    return predict_checkpoint(
        directory / "model.pt",
        views[candidate.view],
        rows,
        local,
        spec,
        fit["identity"],
    )


def tune_and_refit(
    family: str,
    views: dict[str, SequenceData],
    cfg: RefinementConfig,
    candidates: list,
    fold: int,
    directory: Path,
    identity: str,
) -> dict:
    training, validation, external = split_subjects(
        views["raw145"].frame, fold, cfg.seed + (fold if fold >= 0 else 500)
    )
    trials = []
    seed = cfg.seed + (fold if fold >= 0 else 500)
    for index, candidate in enumerate(candidates):
        key = fingerprint(
            {"parent": identity, "phase": "inner", "candidate": candidate.model_dump()}
        )
        result = fit_candidate(
            family,
            views,
            cfg,
            candidate,
            training,
            validation,
            directory / f"trial-{index:02d}-{candidate.name}",
            key,
            seed,
        )
        trials.append({**result, "candidate": index})
    chosen = choose_trial(trials)
    candidate = candidates[chosen["candidate"]]
    key = fingerprint(
        {
            "parent": identity,
            "phase": "refit",
            "candidate": candidate.model_dump(),
            "iterations": chosen["best_epoch"],
        }
    )
    refit = fit_candidate(
        family,
        views,
        cfg,
        candidate,
        sorted(training + validation),
        [],
        directory / "refit",
        key,
        seed,
        fixed=chosen["best_epoch"],
    )
    return {
        "trials": trials,
        "selected": chosen,
        "refit": refit,
        "training_subjects": training,
        "validation_subjects": validation,
        "external_subjects": external,
    }


def initialize(reference: Path, root: Path) -> str:
    mlflow.set_tracking_uri(f"sqlite:///{reference.resolve().parent / 'mlflow.db'}")
    name = "sleep-cassette-lgbm-bilstm-refinement-v1"
    exp = mlflow.get_experiment_by_name(name)
    experiment_id = (
        exp.experiment_id
        if exp
        else mlflow.create_experiment(
            name, artifact_location=str((root / "mlflow-artifacts").resolve())
        )
    )
    mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_id


def prepare(
    config: Path, reference: Path
) -> tuple[RefinementConfig, dict[str, SequenceData], dict, dict, pd.DataFrame]:
    cfg = RefinementConfig.model_validate(read_yaml(config))
    package_root = Path(__file__).resolve().parents[2]
    data, baseline, identity, oof = load_reference(
        reference, package_root / "configs/splits/sc_subjects_v1.csv"
    )
    views = refinement_views(data)
    identity.pop("device", None)
    identity.pop("precision", None)
    identity.update(
        {
            "execution_devices": {"lgbm": "isolated_cpu", "bilstm": "mps"},
            "feature_dtype": "float32",
            "bilstm_precision": "float32",
            "config": cfg.model_dump(),
            "lightgbm_version": version("lightgbm"),
            "lockfile_sha256": hash_file(package_root / "uv.lock"),
            "view_hashes": {
                name: hashlib.sha256(
                    np.ascontiguousarray(view.values).view(np.uint8)
                ).hexdigest()
                for name, view in views.items()
            },
        }
    )
    return cfg, views, baseline, identity, oof


def run_family(
    config: Path, reference: Path, root: Path, family: str, mode: str = "run"
) -> dict:
    root = root.resolve()
    if root == reference.resolve().parent:
        raise ValueError("Refinement must not overwrite the reference campaign.")
    if family not in {"lgbm", "bilstm"} or mode not in {"run", "benchmark"}:
        raise ValueError("Invalid family or mode.")
    if family == "bilstm":
        require_mps()
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / f"{family}.lock"), timeout=0):
        cfg, views, baseline, provenance, reference_oof = prepare(config, reference)
        initialize(reference, root)
        identity = fingerprint(
            {"provenance": provenance, "family": family, "mode": mode}
        )
        directory = (
            root / ("runs" if mode == "run" else "technical") / identity / family
        )
        if (directory / "result.json").exists():
            result = json.loads((directory / "result.json").read_text())
            if mode == "run":
                if hash_file(directory / "oof.parquet") != result["oof_sha256"]:
                    raise ValueError("Completed OOF checksum mismatch.")
                path = Path(result["candidate"]["refit"]["directory"]) / (
                    "model.pt" if family == "bilstm" else "model.joblib"
                )
                if hash_file(path) != result["candidate"]["refit"]["model_sha256"]:
                    raise ValueError("Candidate checksum mismatch.")
                write_json(root / f"{family}.json", result)
            return result
        candidates = cfg.lgbm if family == "lgbm" else cfg.bilstm
        device = "cpu" if family == "lgbm" else "mps"
        with tracked_run(
            directory,
            f"{family}-{mode}",
            "hyperparameter_family" if mode == "run" else "technical_benchmark",
            identity,
            nested=False,
            device=device,
        ) as run_id:
            mlflow.log_dict(provenance, "provenance.json")
            mlflow.log_params(
                {
                    "family": family,
                    "candidates_per_fold": len(candidates),
                    "subjects": 62,
                    "outer_folds": 5,
                    "mode": mode,
                    "device": device,
                }
            )
            if mode == "benchmark":
                training, validation, _ = split_subjects(
                    views["raw145"].frame, 0, cfg.seed
                )
                trial = candidates[0]
                if family == "lgbm":
                    trial = trial.model_copy(update={"n_estimators": 30})
                else:
                    trial = candidates[-1]
                    cfg = cfg.model_copy(update={"nn_max_epochs": 1})
                fitted = fit_candidate(
                    family,
                    views,
                    cfg,
                    trial,
                    training,
                    validation,
                    directory / "benchmark",
                    fingerprint(
                        {
                            "parent": identity,
                            "candidate": trial.model_dump(),
                            "mode": "benchmark",
                        }
                    ),
                    cfg.seed,
                )
                MlflowClient().set_tag(
                    fitted["mlflow_run_id"], "purpose", "technical_benchmark_fit"
                )
                result = {
                    "identity": identity,
                    "family": family,
                    "fit": fitted,
                    "mlflow_run_id": run_id,
                    "purpose": "technical_benchmark",
                    "provenance": provenance,
                }
                mlflow.log_dict(result, "result.json")
            else:
                tables, folds = [], []
                for fold in range(5):
                    fold_dir = directory / f"fold-{fold}"
                    key = fingerprint({"parent": identity, "fold": fold})
                    if (fold_dir / "result.json").exists():
                        result = json.loads((fold_dir / "result.json").read_text())
                        if (
                            result["identity"] != key
                            or hash_file(fold_dir / "oof.parquet")
                            != result["oof_sha256"]
                        ):
                            raise ValueError("Fold cache mismatch.")
                        oof = pd.read_parquet(fold_dir / "oof.parquet")
                    else:
                        with tracked_run(
                            fold_dir,
                            f"{family}-fold-{fold}",
                            "outer_fold",
                            key,
                            device=device,
                        ) as fold_run:
                            result = tune_and_refit(
                                family, views, cfg, candidates, fold, fold_dir, key
                            )
                            rows = views["raw145"].targets[
                                views["raw145"]
                                .frame.subject_id.isin(result["external_subjects"])
                                .to_numpy()[views["raw145"].targets]
                            ]
                            selected = candidates[result["selected"]["candidate"]]
                            probability = predict_fit(
                                family, views, cfg, selected, result["refit"], rows
                            )
                            np.testing.assert_allclose(
                                probability.sum(axis=1), 1, atol=1e-5
                            )
                            oof = (
                                views["raw145"]
                                .frame.iloc[rows][
                                    [
                                        "epoch_id",
                                        "subject_id",
                                        "recording_id",
                                        "onset_s",
                                        "fold",
                                    ]
                                ]
                                .copy()
                            )
                            oof["y_true"] = (
                                views["raw145"].frame.iloc[rows].task_5.to_numpy()
                            )
                            oof["y_pred"] = np.asarray(CLASSES_5)[
                                probability.argmax(axis=1)
                            ]
                            oof[["prob_" + c for c in CLASSES_5]] = probability
                            metrics = evaluate(
                                oof.y_true.to_numpy(), oof.y_pred.to_numpy(), CLASSES_5
                            )
                            with atomic_path(fold_dir / "oof.parquet") as tmp:
                                oof.to_parquet(tmp, index=False)
                            result.update(
                                {
                                    "identity": key,
                                    "fold": fold,
                                    "metrics": metrics,
                                    "oof_sha256": hash_file(fold_dir / "oof.parquet"),
                                    "mlflow_run_id": fold_run,
                                }
                            )
                            mlflow.log_metrics(scalar_metrics(metrics))
                            mlflow.log_dict(result, "result.json")
                            mlflow.log_artifact(str(fold_dir / "oof.parquet"))
                        write_json(fold_dir / "result.json", result)
                    folds.append(result)
                    tables.append(oof)
                    print(
                        f"OUTER {family} fold={fold} f1={result['metrics']['f1_macro']:.4f} accuracy={result['metrics']['accuracy']:.4f} kappa={result['metrics']['kappa']:.4f} selected={result['selected']['configuration']['name']}",
                        flush=True,
                    )
                oof = pd.concat(tables, ignore_index=True)
                if oof.epoch_id.duplicated().any() or set(oof.epoch_id) != set(
                    reference_oof.epoch_id
                ):
                    raise ValueError("OOF coverage must equal the historical champion.")
                with atomic_path(directory / "oof.parquet") as tmp:
                    oof.to_parquet(tmp, index=False)
                metrics = evaluate(
                    oof.y_true.to_numpy(), oof.y_pred.to_numpy(), CLASSES_5
                )
                bootstrap = paired_subject_bootstrap(reference_oof, oof)
                bootstrap.update(reference=baseline["winner"]["name"], candidate=family)
                with tracked_run(
                    directory / "candidate",
                    f"{family}-full-development",
                    "final_candidate",
                    identity,
                    device=device,
                ) as candidate_run:
                    candidate = tune_and_refit(
                        family,
                        views,
                        cfg,
                        candidates,
                        -1,
                        directory / "candidate",
                        identity,
                    )
                    selected = candidates[candidate["selected"]["candidate"]]
                    rows = views["raw145"].targets[:20]
                    example = predict_fit(
                        family, views, cfg, selected, candidate["refit"], rows
                    )
                    again = predict_fit(
                        family, views, cfg, selected, candidate["refit"], rows
                    )
                    np.testing.assert_allclose(example, again, atol=1e-6)
                    np.testing.assert_allclose(example.sum(axis=1), 1, atol=1e-5)
                    candidate.update(
                        mlflow_run_id=candidate_run,
                        reload_equal=True,
                        probabilities_sum_to_one=True,
                    )
                    mlflow.log_dict(candidate, "candidate.json")
                    checkpoint = Path(candidate["refit"]["directory"]) / (
                        "model.pt" if family == "bilstm" else "model.joblib"
                    )
                    mlflow.log_artifact(str(checkpoint))
                    np.savez_compressed(
                        directory / "candidate/example-predictions.npz",
                        probabilities=example,
                        epoch_id=views["raw145"]
                        .frame.iloc[rows]
                        .epoch_id.to_numpy(dtype=str),
                    )
                    mlflow.log_artifact(
                        str(directory / "candidate/example-predictions.npz")
                    )
                result = {
                    "identity": identity,
                    "family": family,
                    "provenance": provenance,
                    "config": cfg.model_dump(),
                    "folds": folds,
                    "candidate": candidate,
                    "f1_macro": float(
                        np.mean([f["metrics"]["f1_macro"] for f in folds])
                    ),
                    "f1_std": float(np.std([f["metrics"]["f1_macro"] for f in folds])),
                    "accuracy": float(
                        np.mean([f["metrics"]["accuracy"] for f in folds])
                    ),
                    "kappa": float(np.mean([f["metrics"]["kappa"] for f in folds])),
                    "pooled_metrics": metrics,
                    "paired_bootstrap": bootstrap,
                    "oof_sha256": hash_file(directory / "oof.parquet"),
                    "directory": str(directory.resolve()),
                    "mlflow_run_id": run_id,
                    "test_evaluated": False,
                }
                plot_confusion(metrics, directory / "confusion.png")
                mlflow.log_metrics(
                    {k: result[k] for k in ["f1_macro", "f1_std", "accuracy", "kappa"]}
                )
                mlflow.log_dict(result, "result.json")
                mlflow.log_artifact(str(directory / "oof.parquet"))
                mlflow.log_artifact(str(directory / "confusion.png"))
        write_json(directory / "result.json", result)
        write_json(
            root / f"{family}.json"
            if mode == "run"
            else root / f"{family}-benchmark.json",
            result,
        )
        return result


def compare_families(root: Path, reference: Path, neural_reference: Path) -> dict:
    results = {
        family: json.loads((root / f"{family}.json").read_text())
        for family in ["lgbm", "bilstm"]
    }
    if results["lgbm"]["provenance"] != results["bilstm"]["provenance"]:
        raise ValueError("Tuned families have incompatible provenance.")
    baseline = json.loads(reference.read_text())["winner"]
    previous_nn = json.loads(neural_reference.read_text())
    previous_nn = next(
        r
        for r in previous_nn["neural_results"]
        if r["name"] == previous_nn["best_neural"]
    )
    rows = [
        {
            "name": "lightgbm_original",
            **{k: baseline[k] for k in ["f1_macro", "f1_std", "accuracy", "kappa"]},
        },
        {
            "name": "bilstm_original",
            **{k: previous_nn[k] for k in ["f1_macro", "f1_std", "accuracy", "kappa"]},
        },
    ]
    for family, result in results.items():
        if hash_file(Path(result["directory"]) / "oof.parquet") != result["oof_sha256"]:
            raise ValueError("Comparison OOF checksum mismatch.")
        rows.append(
            {
                "name": family + "_tuned",
                **{k: result[k] for k in ["f1_macro", "f1_std", "accuracy", "kappa"]},
            }
        )
    board = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    board.to_csv(root / "leaderboard.csv", index=False)
    result = {
        "winner": board.iloc[0]["name"],
        "leaderboard": board.to_dict("records"),
        "families": results,
        "test_evaluated": False,
    }
    write_json(root / "comparison.json", result)
    initialize(reference, root)
    key = fingerprint({family: r["identity"] for family, r in results.items()})
    with tracked_run(
        root / "comparison",
        "tuning-comparison",
        "post_campaign_comparison",
        key,
        nested=False,
        device="mixed",
    ):
        mlflow.log_dict(result, "comparison.json")
        mlflow.log_artifact(str(root / "leaderboard.csv"))
    return result
