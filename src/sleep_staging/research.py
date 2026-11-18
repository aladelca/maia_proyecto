"""Five frozen outer folds, ablations and grouped inner CatBoost tuning.

This exploratory campaign never evaluates the reserved test subjects. Cached
datasets and run identities make interrupted local experiments resumable.
"""

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from sleep_staging.config import CLASSES, ExperimentConfig
from sleep_staging.evaluation import evaluate, plot_confusion, scalar_metrics
from sleep_staging.features.extract import signal_features
from sleep_staging.features.research import temporal_features
from sleep_staging.models.classical import classical_model
from sleep_staging.provenance import (
    atomic_path,
    code_fingerprint,
    fingerprint,
    hash_file,
    write_json,
)
from sleep_staging.research_data import SIGNAL_CHANNELS, build_research_dataset

RESEARCH_CLASSES = CLASSES["task_5"]


@dataclass(frozen=True)
class ResearchSpec:
    name: str
    family: str
    feature_view: str
    tuned: bool = False


SPECS = [
    ResearchSpec("rf_eeg_v1", "rf", "eeg_v1"),
    ResearchSpec("catboost_eeg_v1", "catboost", "eeg_v1"),
    ResearchSpec("catboost_eeg_extended", "catboost", "eeg_extended"),
    ResearchSpec("catboost_multichannel_v1", "catboost", "multichannel_v1"),
    ResearchSpec("catboost_multichannel_extended", "catboost", "multichannel_extended"),
    ResearchSpec("catboost_context_offline", "catboost", "context_offline"),
    ResearchSpec("lightgbm_context_offline", "lgbm", "context_offline"),
    ResearchSpec("catboost_tuned_context", "catboost", "context_offline", tuned=True),
]

TUNING_CANDIDATES: list[dict[str, Any]] = [
    {
        "iterations": 600,
        "depth": 5,
        "learning_rate": 0.05,
        "l2_leaf_reg": 3,
        "auto_class_weights": "Balanced",
        "rsm": 0.8,
    },
    {
        "iterations": 700,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 5,
        "auto_class_weights": "SqrtBalanced",
        "rsm": 0.8,
    },
    {
        "iterations": 700,
        "depth": 7,
        "learning_rate": 0.04,
        "l2_leaf_reg": 9,
        "auto_class_weights": None,
        "rsm": 0.7,
    },
    {
        "iterations": 600,
        "depth": 6,
        "learning_rate": 0.08,
        "l2_leaf_reg": 9,
        "auto_class_weights": "Balanced",
        "rsm": 0.7,
    },
]


def research_views(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Build contexts before removing M, without using labels as input variables."""
    columns = [
        c
        for c in raw
        if any(c.startswith(channel + "__") for channel in SIGNAL_CHANNELS)
    ]
    if not np.isfinite(raw[columns].to_numpy()).all():
        raise ValueError("Research features must be finite.")
    context = temporal_features(raw, columns)
    selected = raw.task_5.notna()
    frame = raw[selected].reset_index(drop=True)
    if (
        set(frame.fold) != set(range(5))
        or frame.groupby("subject_id").fold.nunique().max() != 1
    ):
        raise ValueError("Incompatible frozen development folds.")
    base_columns: list[str] = []
    for channel in SIGNAL_CHANNELS:
        kind = (
            "eeg"
            if channel.startswith("eeg")
            else "eog"
            if channel == "eog"
            else "emg_envelope"
        )
        names = signal_features(
            np.zeros(30 if kind == "emg_envelope" else 3000),
            1 if kind == "emg_envelope" else 100,
            kind,
        )
        base_columns.extend(f"{channel}__{kind}__{name}" for name in names)
    all_features = raw.loc[selected, columns].reset_index(drop=True)
    views = {
        "eeg_v1": frame[[c for c in base_columns if c.startswith("eeg_fpz_cz__")]],
        "eeg_extended": frame[[c for c in columns if c.startswith("eeg_fpz_cz__")]],
        "multichannel_v1": frame[base_columns],
        "multichannel_extended": all_features,
        "context_offline": pd.concat(
            [all_features, context.loc[selected].reset_index(drop=True)], axis=1
        ),
    }
    return frame, views


def grouped_inner_split(groups: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if len(np.unique(groups)) < 5:
        raise ValueError("At least five subjects required for grouped inner tuning.")
    training, validation = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(
            groups, groups=groups
        )
    )
    if set(groups[training]) & set(groups[validation]):
        raise ValueError("Subject leakage in inner validation.")
    return training, validation


def tune_catboost(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
    candidates: list[dict[str, Any]] | None = None,
) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    from catboost import CatBoostClassifier

    training, validation = grouped_inner_split(groups, seed)
    if set(RESEARCH_CLASSES) - set(y[training]) or set(RESEARCH_CLASSES) - set(
        y[validation]
    ):
        raise ValueError("Missing classes in grouped inner train/validation.")
    rows: list[dict[str, Any]] = []
    for number, params in enumerate(candidates or TUNING_CANDIDATES):
        start = time.monotonic()
        model = CatBoostClassifier(
            **params,
            loss_function="MultiClass",
            eval_metric="TotalF1:average=Macro"
            + (";use_weights=false" if params.get("auto_class_weights") else ""),
            random_seed=seed,
            thread_count=2,
            allow_writing_files=False,
            verbose=False,
        )
        model.fit(
            X.iloc[training],
            y[training],
            eval_set=(X.iloc[validation], y[validation]),
            early_stopping_rounds=50,
            use_best_model=True,
        )
        prediction = np.asarray(model.predict(X.iloc[validation])).reshape(-1)
        metrics = evaluate(y[validation], prediction, RESEARCH_CLASSES)
        best_params = {**params, "iterations": int(model.tree_count_)}
        row = {
            "candidate": number,
            "params": best_params,
            **scalar_metrics(metrics),
            "elapsed_seconds": time.monotonic() - start,
        }
        rows.append(row)
        print(
            f"INNER candidate={number} f1={row['f1_macro']:.4f} accuracy={row['accuracy']:.4f} kappa={row['kappa']:.4f} trees={model.tree_count_}",
            flush=True,
        )
    chosen = sorted(rows, key=lambda r: (-r["f1_macro"], r["elapsed_seconds"]))[0]
    final = CatBoostClassifier(
        **chosen["params"],
        loss_function="MultiClass",
        random_seed=42,
        thread_count=2,
        allow_writing_files=False,
        verbose=False,
    )
    final.fit(X, y)
    audit = {
        "training_subjects": sorted(np.unique(groups[training]).tolist()),
        "validation_subjects": sorted(np.unique(groups[validation]).tolist()),
        "chosen_params": chosen["params"],
        "selection": "inner_subject_holdout_macro_f1",
    }
    return final, rows, audit


def _fit_model(
    spec: ResearchSpec, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, fold: int
) -> tuple[Any, dict]:
    if spec.tuned:
        model, trials, audit = tune_catboost(X, y, groups, seed=42 + fold)
        return model, {"trials": trials, **audit}
    model = classical_model(
        spec.family, ExperimentConfig(task="task_5", rf_trees=64, workers=2)
    )
    model.fit(X, y)
    return model, {"params": model.named_steps["classifier"].get_params()}


def _aligned_probability(model: Any, X: pd.DataFrame) -> np.ndarray:
    probability = np.asarray(model.predict_proba(X))
    order = [list(model.classes_).index(label) for label in RESEARCH_CLASSES]
    return probability[:, order]


def research_identity(spec: ResearchSpec, X: pd.DataFrame, dataset_hash: str) -> dict:
    """Bind cached scores to package dependencies and the actual feature values."""
    return {
        "spec": asdict(spec),
        "dataset_sha256": dataset_hash,
        "runner_sha256": hash_file(Path(__file__)),
        "source_hash": code_fingerprint(),
        "feature_values_sha256": hashlib.sha256(
            pd.util.hash_pandas_object(X, index=True).to_numpy().tobytes()
        ).hexdigest(),
        "feature_dtypes": X.dtypes.astype(str).tolist(),
        "tuning": TUNING_CANDIDATES if spec.tuned else None,
        "columns": list(X),
        "versions": {p: version(p) for p in ["catboost", "lightgbm", "scikit-learn"]},
    }


def evaluate_research_spec(
    spec: ResearchSpec,
    frame: pd.DataFrame,
    X: pd.DataFrame,
    directory: Path,
    dataset_hash: str,
) -> tuple[dict, pd.DataFrame]:
    identity = research_identity(spec, X, dataset_hash)
    run_dir = directory / "runs" / fingerprint(identity)
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "summary.json").exists():
        summary = json.loads((run_dir / "summary.json").read_text())
        if hash_file(run_dir / "oof.parquet") != summary["oof_sha256"]:
            raise ValueError("OOF cache checksum mismatch.")
        print(f"REUSE {spec.name} f1={summary['f1_macro']:.4f}", flush=True)
        return summary, pd.read_parquet(run_dir / "oof.parquet")
    scores, predictions = [], []
    start = time.monotonic()
    with mlflow.start_run(run_name=spec.name, nested=True) as run:
        mlflow.log_params(
            {
                "family": spec.family,
                "view": spec.feature_view,
                "features": X.shape[1],
                "outer_folds": 5,
                "nested_subject_tuning": spec.tuned,
                "test_evaluated": False,
            }
        )
        for fold in range(5):
            checkpoint = run_dir / f"fold-{fold}.json"
            oof_path = run_dir / f"fold-{fold}.parquet"
            validation = frame.fold.eq(fold).to_numpy()
            training = ~validation
            if set(frame.loc[training, "subject_id"]) & set(
                frame.loc[validation, "subject_id"]
            ):
                raise ValueError("Outer subject leakage.")
            if checkpoint.exists() and oof_path.exists():
                result = json.loads(checkpoint.read_text())
                if hash_file(oof_path) != result["oof_sha256"]:
                    raise ValueError("Fold checkpoint checksum mismatch.")
                fold_oof = pd.read_parquet(oof_path)
            else:
                fold_start = time.monotonic()
                model, training_audit = _fit_model(
                    spec,
                    X.loc[training],
                    frame.loc[training, "task_5"].to_numpy(),
                    frame.loc[training, "subject_id"].to_numpy(),
                    fold,
                )
                predicted = np.asarray(model.predict(X.loc[validation])).reshape(-1)
                metrics = evaluate(
                    frame.loc[validation, "task_5"].to_numpy(),
                    predicted,
                    RESEARCH_CLASSES,
                )
                fold_oof = frame.loc[
                    validation,
                    ["epoch_id", "subject_id", "recording_id", "onset_s", "fold"],
                ].copy()
                fold_oof["y_true"] = frame.loc[validation, "task_5"].to_numpy()
                fold_oof["y_pred"] = predicted
                for label in RESEARCH_CLASSES:
                    # Prediction is batched once below for all classes.
                    fold_oof[f"prob_{label}"] = 0.0
                probabilities = _aligned_probability(model, X.loc[validation])
                fold_oof[[f"prob_{c}" for c in RESEARCH_CLASSES]] = (
                    probabilities.astype(np.float32)
                )
                with atomic_path(oof_path) as temporary:
                    fold_oof.to_parquet(temporary, index=False)
                result = {
                    "fold": fold,
                    "metrics": metrics,
                    "training": training_audit,
                    "elapsed_seconds": time.monotonic() - fold_start,
                    "oof_sha256": hash_file(oof_path),
                }
                write_json(checkpoint, result)
            scores.append(result)
            predictions.append(fold_oof)
            mlflow.log_metrics(
                {
                    f"fold_{fold}_{k}": v
                    for k, v in scalar_metrics(result["metrics"]).items()
                }
            )
            mlflow.log_artifact(str(checkpoint), "folds")
            m = result["metrics"]
            print(
                f"FOLD {spec.name} {fold} f1={m['f1_macro']:.4f} accuracy={m['accuracy']:.4f} kappa={m['kappa']:.4f} seconds={result['elapsed_seconds']:.1f}",
                flush=True,
            )
        oof = pd.concat(predictions, ignore_index=True)
        if set(oof.epoch_id) != set(frame.epoch_id) or oof.epoch_id.duplicated().any():
            raise ValueError("OOF must cover each development epoch exactly once.")
        pooled = evaluate(
            oof.y_true.to_numpy(), oof.y_pred.to_numpy(), RESEARCH_CLASSES
        )
        summary = {
            "name": spec.name,
            "family": spec.family,
            "feature_view": spec.feature_view,
            "features": X.shape[1],
            "f1_macro": float(np.mean([s["metrics"]["f1_macro"] for s in scores])),
            "f1_std": float(np.std([s["metrics"]["f1_macro"] for s in scores])),
            "accuracy": float(np.mean([s["metrics"]["accuracy"] for s in scores])),
            "kappa": float(np.mean([s["metrics"]["kappa"] for s in scores])),
            "n1_f1_oof": pooled["report"]["N1"]["f1-score"],
            "rem_f1_oof": pooled["report"]["REM"]["f1-score"],
            "fold_scores": [s["metrics"]["f1_macro"] for s in scores],
            "elapsed_seconds": time.monotonic() - start,
            "mlflow_run_id": run.info.run_id,
            "run_dir": str(run_dir.resolve()),
            "identity": identity,
        }
        with atomic_path(run_dir / "oof.parquet") as temporary:
            oof.to_parquet(temporary, index=False)
        summary["oof_sha256"] = hash_file(run_dir / "oof.parquet")
        write_json(run_dir / "pooled_metrics.json", pooled)
        plot_confusion(pooled, run_dir / "confusion.png")
        write_json(run_dir / "summary.json", summary)
        mlflow.log_metrics(
            {
                key: summary[key]
                for key in [
                    "f1_macro",
                    "f1_std",
                    "accuracy",
                    "kappa",
                    "n1_f1_oof",
                    "rem_f1_oof",
                ]
            }
        )
        mlflow.log_artifacts(str(run_dir), "results")
    print(
        f"RESULT {spec.name} F1={summary['f1_macro']:.4f}+/-{summary['f1_std']:.4f} accuracy={summary['accuracy']:.4f} kappa={summary['kappa']:.4f}",
        flush=True,
    )
    return summary, oof


def paired_subject_bootstrap(
    reference: pd.DataFrame, candidate: pd.DataFrame, repeats: int = 2000
) -> dict:
    a = reference.sort_values("epoch_id").reset_index(drop=True)
    b = candidate.sort_values("epoch_id").reset_index(drop=True)
    if not a[["epoch_id", "subject_id", "y_true"]].equals(
        b[["epoch_id", "subject_id", "y_true"]]
    ):
        raise ValueError(
            "Paired bootstrap requires identical development epochs and labels."
        )
    from sklearn.metrics import confusion_matrix

    matrices = []
    for table in [a, b]:
        matrices.append(
            np.stack(
                [
                    confusion_matrix(
                        group.y_true, group.y_pred, labels=RESEARCH_CLASSES
                    )
                    for _, group in table.groupby("subject_id", sort=True)
                ]
            )
        )

    def macro(matrix: np.ndarray) -> np.ndarray:
        diagonal = np.diagonal(matrix, axis1=-2, axis2=-1)
        denominator = matrix.sum(axis=-1) + matrix.sum(axis=-2)
        return np.divide(
            2 * diagonal,
            denominator,
            out=np.zeros_like(diagonal, dtype=float),
            where=denominator > 0,
        ).mean(axis=-1)

    random = np.random.default_rng(42)
    indices = random.integers(0, len(matrices[0]), size=(repeats, len(matrices[0])))
    delta = macro(matrices[1][indices].sum(axis=1)) - macro(
        matrices[0][indices].sum(axis=1)
    )
    return {
        "metric": "pooled_oof_macro_f1_difference",
        "reference": "rf_eeg_v1",
        "delta": float(macro(matrices[1].sum(axis=0)) - macro(matrices[0].sum(axis=0))),
        "ci95": np.quantile(delta, [0.025, 0.975]).tolist(),
        "subjects": len(matrices[0]),
        "repeats": repeats,
        "scope": "conditional_on_fitted_oof_models_not_final_test",
    }


def run_research_campaign(
    source_root: Path,
    split_file: Path,
    output_root: Path,
    names: list[str] | None = None,
    refit: bool = True,
) -> dict:
    from filelock import FileLock

    output_root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output_root / "campaign.lock"), timeout=0):
        return _campaign(source_root, split_file, output_root, names, refit)


def _campaign(
    source_root: Path,
    split_file: Path,
    output_root: Path,
    names: list[str] | None,
    refit: bool,
) -> dict:
    dataset = build_research_dataset(source_root, split_file, output_root)
    raw = pd.read_parquet(dataset)
    frame, views = research_views(raw)
    selected = [spec for spec in SPECS if names is None or spec.name in names]
    if not selected or names and set(names) - {s.name for s in selected}:
        raise ValueError("Unknown or empty experiment selection.")
    mlflow.set_tracking_uri(f"sqlite:///{output_root.resolve() / 'mlflow.db'}")
    experiment_name = "sleep-cassette-literature-v2"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = (
        experiment.experiment_id
        if experiment
        else mlflow.create_experiment(
            experiment_name,
            artifact_location=str(output_root.resolve() / "mlflow-artifacts"),
        )
    )
    mlflow.set_experiment(experiment_id=experiment_id)
    results, predictions = [], {}
    with mlflow.start_run(
        experiment_id=experiment_id, run_name="literature-features-and-nested-tuning"
    ) as parent:
        mlflow.log_params(
            {
                "dataset_sha256": hash_file(dataset),
                "split_sha256": hash_file(split_file),
                "subjects": frame.subject_id.nunique(),
                "epochs": len(frame),
                "test_evaluated": False,
                "protocol": "annotation_dependent_wake30_offline",
            }
        )
        mlflow.log_artifact(str(dataset.parent / "metadata.json"), "data")
        for spec in selected:
            summary, oof = evaluate_research_spec(
                spec, frame, views[spec.feature_view], output_root, hash_file(dataset)
            )
            results.append(summary)
            predictions[spec.name] = oof
            board = pd.DataFrame(results).sort_values(
                ["f1_macro", "elapsed_seconds"], ascending=[False, True]
            )
            board.drop(columns="identity").to_csv(
                output_root / "leaderboard.csv", index=False
            )
            print(
                f"CURRENT_WINNER {board.iloc[0]['name']} f1={board.iloc[0]['f1_macro']:.4f}",
                flush=True,
            )
        winner = board.iloc[0].to_dict()
        comparison = (
            paired_subject_bootstrap(
                predictions["rf_eeg_v1"], predictions[winner["name"]]
            )
            if "rf_eeg_v1" in predictions
            else None
        )
        if comparison:
            write_json(output_root / "paired_bootstrap.json", comparison)
        if refit:
            spec = next(s for s in selected if s.name == winner["name"])
            print(f"REFIT {spec.name} on development subjects only", flush=True)
            model, audit = _fit_model(
                spec,
                views[spec.feature_view],
                frame.task_5.to_numpy(),
                frame.subject_id.to_numpy(),
                fold=0,
            )
            with atomic_path(output_root / "candidate.joblib") as temporary:
                joblib.dump(
                    {
                        "model": model,
                        "feature_columns": list(views[spec.feature_view]),
                        "spec": asdict(spec),
                        "training_audit": audit,
                        "dataset_sha256": hash_file(dataset),
                        "test_evaluated": False,
                    },
                    temporary,
                )
            mlflow.log_artifact(str(output_root / "candidate.joblib"), "candidate")
        output = {
            "winner": {k: v for k, v in winner.items() if k != "identity"},
            "dataset": str(dataset.resolve()),
            "output_root": str(output_root.resolve()),
            "subjects": int(frame.subject_id.nunique()),
            "epochs": len(frame),
            "classes": RESEARCH_CLASSES,
            "metrics": ["f1_macro", "accuracy", "kappa"],
            "test_evaluated": False,
            "mlflow_parent_run_id": parent.info.run_id,
            "paired_bootstrap": comparison,
            "specs": [asdict(s) for s in selected],
            "candidate_refit": refit,
            "candidate_sha256": hash_file(output_root / "candidate.joblib")
            if refit
            else None,
        }
        write_json(output_root / "campaign.json", output)
        mlflow.log_artifact(str(output_root / "leaderboard.csv"))
        mlflow.log_dict(output, "campaign.json")
    return output
