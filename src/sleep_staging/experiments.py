import json
import tempfile
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from filelock import FileLock
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from sleep_staging.config import CLASSES, DataConfig, ExperimentConfig
from sleep_staging.data.preprocess import extract_dataset_features, prepare_dataset
from sleep_staging.data.splits import assign_subjects
from sleep_staging.evaluation import evaluate, plot_confusion, scalar_metrics
from sleep_staging.models.classical import classical_model
from sleep_staging.provenance import (
    atomic_path,
    code_fingerprint,
    environment,
    fingerprint,
    hash_file,
    write_json,
)
from sleep_staging.tracking import init_tracking


def create_splits(prepared: Path, cfg: ExperimentConfig, root: Path) -> Path:
    split_path = root / cfg.split_file if cfg.split_file else None

    if split_path is not None and not split_path.is_file():
        raise ValueError(f"No existe el archivo de particiones: {split_path}")

    split_hash = hash_file(split_path) if split_path else None

    inputs = {
        "preprocessing_id": prepared.name,
        "n_splits": cfg.n_splits,
        "seed": cfg.seed,
        "test_fraction": cfg.test_fraction,
    }
    if split_hash is not None:
        inputs["fixed_split_sha256"] = split_hash

    key = fingerprint(inputs)
    directory = root / "data/splits" / key
    directory.parent.mkdir(parents=True, exist_ok=True)

    with FileLock(str(directory) + ".lock"):
        if (directory / "metadata.json").exists():
            return directory

        epochs = pd.read_parquet(prepared / "epochs.parquet")

        if split_path is None:
            subjects = assign_subjects(
                epochs, cfg.n_splits, cfg.seed, cfg.test_fraction
            )
        else:
            subjects = pd.read_csv(split_path, dtype={"subject_id": str})
            required = {"subject_id", "partition", "fold"}

            if not required.issubset(subjects.columns):
                raise ValueError("El CSV requiere subject_id, partition y fold.")

            subjects = subjects[["subject_id", "partition", "fold"]].copy()

            if subjects.isna().any().any():
                raise ValueError("Hay valores nulos en las particiones.")

            if subjects.subject_id.duplicated().any():
                raise ValueError("Existen sujetos duplicados.")

            subjects["fold"] = pd.to_numeric(subjects["fold"], errors="raise")

            if not (subjects["fold"] % 1 == 0).all():
                raise ValueError("Los identificadores de fold deben ser enteros.")

            subjects["fold"] = subjects["fold"].astype(int)

            expected = set(epochs.subject_id.astype(str).unique())
            actual = set(subjects.subject_id)

            if expected != actual:
                raise ValueError(
                    f"Partición incompatible. "
                    f"Faltantes: {sorted(expected - actual)}. "
                    f"Adicionales: {sorted(actual - expected)}."
                )

            dev = subjects[subjects.partition.eq("development")]
            test = subjects[subjects.partition.eq("test")]

            if len(dev) + len(test) != len(subjects):
                raise ValueError("Se encontraron particiones desconocidas.")

            if dev.empty or test.empty:
                raise ValueError("Desarrollo y prueba deben tener sujetos.")

            if set(dev.fold) != set(range(cfg.n_splits)):
                raise ValueError("Los folds de desarrollo son incompatibles.")

            if not test.fold.eq(-1).all():
                raise ValueError("Los sujetos de prueba deben tener fold -1.")

        with atomic_path(directory / "subjects.parquet") as temp:
            subjects.to_parquet(temp, index=False)

        metadata = {
            "schema_version": 1,
            "preprocessing_id": prepared.name,
            "seed": cfg.seed,
            "n_splits": cfg.n_splits,
            "test_fraction": cfg.test_fraction,
            "subjects_hash": hash_file(directory / "subjects.parquet"),
            "fixed_split_file": cfg.split_file,
            "fixed_split_sha256": split_hash,
        }
        write_json(directory / "metadata.json", metadata)

    return directory


def experiment_frame(
    prepared: Path, features: Path, splits: Path, cfg: ExperimentConfig
) -> tuple[pd.DataFrame, list[str]]:
    epochs = pd.read_parquet(prepared / "epochs.parquet")
    eligible = epochs.reason.eq("") & epochs[cfg.task].notna()
    for channel in cfg.channels:
        eligible &= epochs[f"has_{channel}"] & ~epochs[f"flat_{channel}"]
    frame = (
        epochs[eligible]
        .merge(pd.read_parquet(features), on="epoch_id", validate="one_to_one")
        .merge(
            pd.read_parquet(splits / "subjects.parquet"),
            on="subject_id",
            validate="many_to_one",
        )
    )
    columns = [
        column
        for column in frame
        if any(column.startswith(channel + "__") for channel in cfg.channels)
    ]
    if {"eeg_fpz_cz", "eeg_pz_oz"}.issubset(cfg.channels):
        columns += ["cross_eeg__correlation"]
    if not columns or frame.empty:
        raise ValueError("Sin características/épocas elegibles para este perfil.")
    return frame, columns


def waveform_frame(frame: pd.DataFrame, prepared: Path, channel: str) -> pd.DataFrame:
    values = []
    arrays = {}
    for row in frame.itertuples():
        if row.recording_id not in arrays:
            arrays[row.recording_id] = np.load(
                prepared / row.recording_id / f"{channel}.npy", mmap_mode="r"
            )
        values.append(arrays[row.recording_id][int(row.array_row)])
    result = pd.DataFrame(
        np.stack(values),
        index=frame.index,
        columns=[f"sample_{i:04d}" for i in range(3000)],
    )
    result.attrs["preprocessing_id"] = prepared.name
    return result


def run_experiment(
    data: DataConfig, cfg: ExperimentConfig, root: Path, tracking_uri: str
) -> dict[str, Any]:
    init_tracking(tracking_uri, cfg.name)
    lock_path = root / "var/experiment.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock_path), timeout=0):
        return _run(data, cfg, root, tracking_uri)


def _run(
    data: DataConfig, cfg: ExperimentConfig, root: Path, tracking_uri: str
) -> dict[str, Any]:
    prepared = prepare_dataset(data, cfg, root)
    features = extract_dataset_features(prepared, cfg, root)
    splits = create_splits(prepared, cfg, root)
    frame, feature_columns = experiment_frame(prepared, features, splits, cfg)
    development = frame[frame.partition.eq("development")].copy()
    if development.fold.nunique() != cfg.n_splits:
        raise ValueError(
            "El perfil no conserva todos los folds; define una cohorte elegible común."
        )
    classes = CLASSES[cfg.task]
    waveform = (
        waveform_frame(frame, prepared, cfg.channels[0])
        if "cnn" in cfg.models
        else None
    )
    tables: dict[str, pd.DataFrame] = {}
    for family in cfg.models:
        tables[family] = waveform if family == "cnn" else frame[feature_columns]
    with mlflow.start_run(run_name=f"{data.name}-{cfg.task}") as parent:
        report_dir = root / "var/reports" / parent.info.run_id
        report_dir.mkdir(parents=True)
        mlflow.set_tags(
            {
                "dataset": data.name,
                "cohort": data.source,
                "task": cfg.task,
                "channels": "+".join(cfg.channels),
                "source_hash": code_fingerprint(),
                "preprocessing_id": prepared.name,
                "split_id": splits.name,
                "purpose": "pilot" if cfg.max_epochs_per_recording else "full_cohort",
            }
        )
        mlflow.log_params(
            {
                "seed": cfg.seed,
                "n_splits": cfg.n_splits,
                "development_subjects": development.subject_id.nunique(),
                "test_subjects": frame[frame.partition.eq("test")].subject_id.nunique(),
                "eligible_epochs": len(frame),
                "features": len(feature_columns),
            }
        )
        mlflow.log_dict(cfg.model_dump(), "config.json")
        mlflow.log_dict(environment(), "environment.json")
        mlflow.log_artifact(str(prepared / "metadata.json"), "provenance")
        mlflow.log_artifact(str(splits / "subjects.parquet"), "provenance")
        mlflow.log_artifact(str(splits / "metadata.json"), "provenance")
        manifest = root / "data/manifests" / data.name / "files.parquet"
        mlflow.log_artifact(str(manifest), "provenance")
        if (root / "uv.lock").exists():
            mlflow.log_artifact(str(root / "uv.lock"), "provenance")
        rows, predictions, selected_epochs = [], [], []
        for family in cfg.models:
            fold_scores = []
            fold_accuracy = []
            fold_kappa = []
            start = time.monotonic()
            for fold in range(cfg.n_splits):
                training = development[development.fold.ne(fold)]
                validation = development[development.fold.eq(fold)]
                if validation.empty or set(classes) - set(training[cfg.task]):
                    raise ValueError(
                        f"Fold {fold}: faltan clases en entrenamiento o épocas de validación. Revisa soporte; no se cambia el protocolo automáticamente."
                    )
                with mlflow.start_run(run_name=f"{family}-fold-{fold}", nested=True):
                    if family == "cnn":
                        from sleep_staging.models.cnn import CNNClassifier

                        model = CNNClassifier(
                            epochs=cfg.cnn_epochs,
                            batch_size=cfg.batch_size,
                            device=cfg.device,
                            seed=cfg.seed,
                        )
                        model.fit(
                            tables[family].loc[training.index],
                            training[cfg.task].to_numpy(),
                            groups=training.subject_id.to_numpy(),
                        )
                        selected_epochs.append(model.best_epoch_)
                    else:
                        model = classical_model(family, cfg)
                        model.fit(
                            tables[family].loc[training.index], training[cfg.task]
                        )
                    predicted = np.asarray(
                        model.predict(tables[family].loc[validation.index])
                    ).reshape(-1)

                    if len(predicted) != len(validation):
                        raise ValueError(
                            "Número de predicciones incompatible con validación"
                        )
                    metrics = evaluate(
                        validation[cfg.task].to_numpy(), predicted, classes
                    )
                    fold_scores.append(metrics["f1_macro"])
                    fold_accuracy.append(metrics["accuracy"])
                    if metrics["kappa"] is not None:
                        fold_kappa.append(metrics["kappa"])
                    mlflow.log_params(
                        {
                            "model": family,
                            "fold": fold,
                            "training_epochs": len(training),
                            "validation_epochs": len(validation),
                        }
                    )
                    mlflow.log_metrics(scalar_metrics(metrics))
                    mlflow.log_dict(metrics, "metrics.json")
                    mlflow.log_dict(
                        {
                            "training": training[cfg.task].value_counts().to_dict(),
                            "validation": validation[cfg.task].value_counts().to_dict(),
                        },
                        "class_support.json",
                    )
                    if family == "cnn":
                        mlflow.log_dict(
                            {"epochs": model.history_}, "training_history.json"
                        )
                        mlflow.log_params(
                            {
                                "device": model.device_,
                                "best_epoch": model.best_epoch_,
                                "batch_size": cfg.batch_size,
                                "architecture": "conv1d-16-32-adaptive-pool-6",
                                "optimizer": "Adam",
                                "learning_rate": 0.001,
                            }
                        )
                        import torch

                        with tempfile.TemporaryDirectory() as temporary:
                            checkpoint = Path(temporary) / "checkpoint.pt"
                            torch.save(
                                {
                                    "state_dict": model.network_.state_dict(),
                                    "mean": model.mean_,
                                    "std": model.std_,
                                    "classes": model.classes_.tolist(),
                                    "best_epoch": model.best_epoch_,
                                },
                                checkpoint,
                            )
                            mlflow.log_artifact(str(checkpoint), "model")
                    else:
                        mlflow.log_param(
                            "rf_trees", cfg.rf_trees if family == "rf" else 0
                        )
                    out = validation[["epoch_id", "subject_id", "fold"]].copy()
                    out["y_true"], out["y_pred"], out["model"] = (
                        validation[cfg.task].to_numpy(),
                        predicted,
                        family,
                    )
                    predictions.append(out)
            rows.append(
                {
                    "model": family,
                    "f1_macro": float(np.mean(fold_scores)),
                    "f1_std": float(np.std(fold_scores)),
                    "accuracy": float(np.mean(fold_accuracy)),
                    "kappa": float(np.mean(fold_kappa)) if fold_kappa else None,
                    "kappa_std": float(np.std(fold_kappa)) if fold_kappa else None,
                    "fold_scores": fold_scores,
                    "elapsed_seconds": time.monotonic() - start,
                }
            )
        leaderboard = (
            pd.DataFrame(rows)
            .sort_values(["f1_macro", "elapsed_seconds"], ascending=[False, True])
            .reset_index(drop=True)
        )
        winner = str(leaderboard.iloc[0].model)
        if winner == "cnn":
            from sleep_staging.models.cnn import CNNClassifier

            final_model = CNNClassifier(
                epochs=max(1, int(np.median(selected_epochs))),
                batch_size=cfg.batch_size,
                device=cfg.device,
                seed=cfg.seed,
            )
            final_model.fit(
                tables[winner].loc[development.index], development[cfg.task].to_numpy()
            )
        else:
            final_model = classical_model(winner, cfg)
            final_model.fit(
                tables[winner].loc[development.index], development[cfg.task]
            )
        columns = list(tables[winner].columns)
        example = tables[winner].loc[development.index].head(2)
        model_info = mlflow.sklearn.log_model(
            sk_model=final_model,
            name="candidate",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            signature=infer_signature(example, final_model.predict(example)),
            input_example=example,
            code_paths=[str(Path(__file__).parent)],
            pip_requirements=[
                f"{package}=={version(package)}"
                for package in [
                    "mlflow",
                    "scikit-learn",
                    "pandas",
                    "numpy",
                    "cloudpickle",
                    "python-dotenv",
                ]
                + (["torch"] if winner == "cnn" else [])
                + (["lightgbm"] if winner == "lgbm" else [])
                + (["catboost"] if winner == "catboost" else [])
            ],
            metadata={
                "feature_columns": columns,
                "task": cfg.task,
                "channels": cfg.channels,
                "input_kind": "waveform" if winner == "cnn" else "features",
                "preprocessing_id": prepared.name,
                "preprocessing": json.loads((prepared / "metadata.json").read_text())[
                    "preprocessing"
                ],
            },
        )
        name = f"sleep_{data.source}_{cfg.task}_{'_'.join(cfg.channels)}_{winner}"
        registered = mlflow.register_model(model_info.model_uri, name)
        MlflowClient(tracking_uri=tracking_uri).set_registered_model_alias(
            name, "candidate", registered.version
        )
        oof = pd.concat(predictions, ignore_index=True)
        summary = {
            "run_id": parent.info.run_id,
            "winner": winner,
            "model_uri": model_info.model_uri,
            "registry_uri": f"models:/{name}@candidate",
            "preprocessing_id": prepared.name,
            "features_path": str(features),
            "splits_path": str(splits),
            "task": cfg.task,
            "config": cfg.model_dump(),
            "test_subjects": sorted(
                frame[frame.partition.eq("test")].subject_id.unique()
            ),
            "leaderboard": leaderboard.to_dict("records"),
            "coverage": {
                "eligible_epochs": len(frame),
                "total_epochs": len(pd.read_parquet(prepared / "epochs.parquet")),
            },
        }
        write_json(report_dir / "summary.json", summary)
        leaderboard.to_csv(report_dir / "leaderboard.csv", index=False)
        oof.to_parquet(report_dir / "oof.parquet", index=False)
        for family in cfg.models:
            subset = oof[oof.model.eq(family)]
            metrics = evaluate(
                subset.y_true.to_numpy(), subset.y_pred.to_numpy(), classes
            )
            mlflow.log_metrics(
                {
                    f"{family}_{key}": value
                    for key, value in scalar_metrics(metrics).items()
                }
            )
            write_json(report_dir / f"{family}-metrics.json", metrics)
            plot_confusion(metrics, report_dir / f"{family}-confusion.png")
        mlflow.log_artifacts(str(report_dir), "reports")
        mlflow.log_metric("winner_f1_macro_cv", float(leaderboard.iloc[0].f1_macro))
        if leaderboard.iloc[0].kappa is not None:
            mlflow.log_metric("winner_kappa_cv", float(leaderboard.iloc[0].kappa))
        mlflow.set_tag("winner", winner)
        kappa_display = (
            f"; kappa CV = {leaderboard.iloc[0].kappa:.4f} +/- {leaderboard.iloc[0].kappa_std:.4f}"
            if leaderboard.iloc[0].kappa is not None
            else ""
        )
        print(
            f"Winner model: {winner}; F1 macro CV = {leaderboard.iloc[0].f1_macro:.4f} +/- {leaderboard.iloc[0].f1_std:.4f}"
            + kappa_display
        )
    return {**summary, "leaderboard": leaderboard, "oof": oof}
