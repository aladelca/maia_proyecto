"""Nested subject-held-out LSTM experiments, strictly on MPS.

Only development features enter this module. The frozen historical champion is
compared through verified OOF predictions, never through reserved test labels.
"""

import json
import platform
import random
import time
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

import mlflow
import numpy as np
import pandas as pd
import torch
from filelock import FileLock
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sklearn.model_selection import GroupShuffleSplit

from sleep_staging.config import CLASSES, read_yaml
from sleep_staging.evaluation import evaluate, plot_confusion, scalar_metrics
from sleep_staging.lstm_data import FeatureScaler, SequenceData, sequence_index
from sleep_staging.models.lstm import SleepLSTM, require_mps
from sleep_staging.provenance import (
    atomic_path,
    code_fingerprint,
    fingerprint,
    hash_file,
    write_json,
)
from sleep_staging.refinement_data import class_weights
from sleep_staging.research import paired_subject_bootstrap

CLASSES_5 = CLASSES["task_5"]


class WindowSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    length: int = Field(ge=1)
    centered: bool

    @model_validator(mode="after")
    def valid_window(self) -> "WindowSpec":
        if self.centered and self.length % 2 != 1:
            raise ValueError("Centered windows must have odd lengths.")
        return self

    @property
    def target_position(self) -> int:
        return self.length // 2 if self.centered else self.length - 1


class LSTMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    device: Literal["mps"] = "mps"
    seed: int = 42
    hidden_size: int = Field(default=64, ge=1)
    dropout: float = Field(default=0.3, ge=0, lt=1)
    batch_size: int = Field(default=256, ge=1)
    max_epochs: int = Field(default=50, ge=1)
    patience: int = Field(default=7, ge=1)
    min_delta: float = Field(default=1e-4, ge=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    weight_power: float = Field(default=1, ge=0, le=1)
    learning_rates: list[float] = Field(default_factory=lambda: [1e-3, 3e-4])
    variants: list[WindowSpec] = Field(
        default_factory=lambda: [
            WindowSpec(name="lstm_single_epoch", length=1, centered=False),
            WindowSpec(name="lstm_past15", length=15, centered=False),
            WindowSpec(name="bilstm_center15", length=15, centered=True),
        ]
    )

    @model_validator(mode="after")
    def valid_candidates(self) -> "LSTMConfig":
        if not self.learning_rates or any(x <= 0 for x in self.learning_rates):
            raise ValueError("Positive learning rates are required.")
        if not self.variants or len({v.name for v in self.variants}) != len(
            self.variants
        ):
            raise ValueError("Unique nonempty variants are required.")
        return self


def split_subjects(
    frame: pd.DataFrame, fold: int, seed: int
) -> tuple[list, list, list]:
    available = frame.loc[frame.fold.ne(fold)] if fold >= 0 else frame
    groups = available.subject_id.to_numpy()
    a, b = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(
            groups, groups=groups
        )
    )
    training = sorted(set(groups[a]))
    validation = sorted(set(groups[b]))
    external = (
        sorted(set(frame.loc[frame.fold.eq(fold), "subject_id"])) if fold >= 0 else []
    )
    if set(training) & set(validation) or (set(training) | set(validation)) & set(
        external
    ):
        raise ValueError("Subject leakage in nested validation.")
    return training, validation, external


def choose_trial(trials: list[dict]) -> dict:
    return sorted(
        trials, key=lambda t: (-t["best_f1_macro"], t["best_epoch"], t["candidate"])
    )[0]


def write_checkpoint(path: Path, payload: dict) -> None:
    with atomic_path(path) as temporary:
        torch.save(payload, temporary)
    with atomic_path(path.with_suffix(path.suffix + ".sha256")) as temporary:
        temporary.write_text(hash_file(path))


def read_checkpoint(path: Path, identity: str) -> dict:
    checksum = path.with_suffix(path.suffix + ".sha256")
    if not checksum.is_file() or hash_file(path) != checksum.read_text().strip():
        raise ValueError("Checkpoint checksum mismatch.")
    value = torch.load(path, map_location="cpu", weights_only=False)
    if value["identity"] != identity:
        raise ValueError("Checkpoint identity mismatch.")
    return value


def _cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu(v) for v in value)
    return value


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)
    torch.set_num_threads(2)


def _rng() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "mps": torch.mps.get_rng_state(),
    }


def _restore_rng(value: dict) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"])
    torch.mps.set_rng_state(value["mps"])


@contextmanager
def tracked_run(
    directory: Path,
    name: str,
    purpose: str,
    identity: str,
    nested: bool = True,
    device: str = "mps",
) -> Iterator[str]:
    directory.mkdir(parents=True, exist_ok=True)
    journal = directory / "mlflow-run.json"
    run_id = json.loads(journal.read_text())["run_id"] if journal.exists() else None
    with mlflow.start_run(run_id=run_id, run_name=name, nested=nested) as run:
        write_json(journal, {"run_id": run.info.run_id, "identity": identity})
        mlflow.set_tags(
            {
                "purpose": purpose,
                "identity": identity,
                "test_evaluated": "false",
                "device": device,
            }
        )
        yield run.info.run_id


def _network(data: SequenceData, cfg: LSTMConfig, spec: WindowSpec) -> SleepLSTM:
    return SleepLSTM(
        len(data.columns),
        cfg.hidden_size,
        cfg.dropout,
        spec.centered,
        spec.target_position,
    ).to(require_mps())


def _predict_tensor(
    model: SleepLSTM,
    values: torch.Tensor,
    neighbors: np.ndarray,
    rows: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    results = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            index = torch.from_numpy(
                neighbors[rows[start : start + batch_size]].astype(np.int64)
            ).to("mps")
            batch = values[index]
            results.append(model(batch).softmax(dim=1))
    return torch.cat(results).cpu().numpy()


def predict_checkpoint(
    path: Path,
    data: SequenceData,
    rows: np.ndarray,
    cfg: LSTMConfig,
    spec: WindowSpec,
    identity: str,
) -> np.ndarray:
    checkpoint = read_checkpoint(path, identity)
    model = _network(data, cfg, spec)
    model.load_state_dict(checkpoint["model"])
    scaler = FeatureScaler.from_dict(checkpoint["scaler"])
    values = torch.from_numpy(scaler.transform(data.values)).to("mps")
    neighbors, _ = sequence_index(data.frame, spec.length, spec.centered)
    probability = _predict_tensor(model, values, neighbors, rows, cfg.batch_size)
    del values, model
    torch.mps.empty_cache()
    return probability


def fit_sequence(
    data: SequenceData,
    cfg: LSTMConfig,
    spec: WindowSpec,
    training_subjects: list,
    validation_subjects: list,
    lr: float,
    directory: Path,
    identity: str,
    seed: int,
    fixed_epochs: int | None = None,
    purpose: str = "inner_subject_tuning",
) -> dict:
    summary_path = directory / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        read_checkpoint(directory / "model.pt", identity)
        if (
            summary["identity"] != identity
            or MlflowClient().get_run(summary["mlflow_run_id"]).info.status
            != "FINISHED"
        ):
            raise ValueError("Cannot reuse an unfinished or incompatible fit.")
        return summary
    device = require_mps()
    if set(training_subjects) & set(validation_subjects):
        raise ValueError("Training and validation subjects overlap.")
    training_mask = data.frame.subject_id.isin(training_subjects).to_numpy()
    train_rows = data.targets[training_mask[data.targets]]
    validation_mask = data.frame.subject_id.isin(validation_subjects).to_numpy()
    valid_rows = data.targets[validation_mask[data.targets]]
    if not len(train_rows) or fixed_epochs is None and not len(valid_rows):
        raise ValueError("Nonempty training and inner validation are required.")
    for rows in [train_rows, *([valid_rows] if len(valid_rows) else [])]:
        if set(data.labels[rows]) != set(range(5)):
            raise ValueError("Missing classes in training or inner validation.")
    _seed(seed)
    scaler = FeatureScaler.fit(data.values[training_mask])
    values = torch.from_numpy(scaler.transform(data.values)).to(device)
    targets = torch.from_numpy(data.labels).to(device)
    neighbors, padding = sequence_index(data.frame, spec.length, spec.centered)
    weights = class_weights(data.labels[train_rows], cfg.weight_power)
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )
    model = _network(data, cfg, spec)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=cfg.weight_decay
    )
    epochs = fixed_epochs if fixed_epochs is not None else cfg.max_epochs
    last = directory / "last.pt"
    history: list[dict] = []
    best_score, best_epoch, stale = -1.0, 0, 0
    best_state = None
    if last.exists():
        saved = read_checkpoint(last, identity)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        history, best_score, best_epoch, stale = (
            saved["history"],
            saved["best_score"],
            saved["best_epoch"],
            saved["stale"],
        )
        best_state = saved["best_state"]
        _restore_rng(saved["rng"])
    with tracked_run(directory, directory.name, purpose, identity) as run_id:
        params = {
            **cfg.model_dump(exclude={"variants", "learning_rates"}),
            **spec.model_dump(),
            "lr": lr,
            "seed": seed,
            "fixed_epochs": fixed_epochs,
            "scheduler": "none",
            "class_weights": weights.tolist(),
            "parameters": sum(p.numel() for p in model.parameters()),
        }
        mlflow.log_params({k: str(v) for k, v in params.items()})
        audit = {
            "training_subjects": training_subjects,
            "validation_subjects": validation_subjects,
            "scaler_fit_subjects": training_subjects,
            "training_targets": len(train_rows),
            "validation_targets": len(valid_rows),
            "padding_positions": int(padding[data.targets].sum()),
            "classes": CLASSES_5,
            "columns": data.columns,
            "scaler": scaler.to_dict(),
            "params": params,
        }
        write_json(directory / "audit.json", audit)
        mlflow.log_artifact(str(directory / "audit.json"))
        for epoch in range(len(history), epochs):
            if fixed_epochs is None and stale >= cfg.patience:
                break
            model.train()
            torch.mps.synchronize()
            start = time.monotonic()
            total = torch.zeros((), dtype=torch.float32, device=device)
            order = np.random.default_rng(seed + epoch).permutation(train_rows)
            for offset in range(0, len(order), cfg.batch_size):
                rows = order[offset : offset + cfg.batch_size]
                index = torch.from_numpy(neighbors[rows].astype(np.int64)).to(device)
                label_index = torch.from_numpy(rows).to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(values[index])
                loss = criterion(logits, targets[label_index])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=False)
                optimizer.step()
                total += loss.detach() * len(rows)
            torch.mps.synchronize()
            row: dict[str, Any] = {
                "epoch": epoch + 1,
                "train_loss": float(total.cpu()) / len(train_rows),
                "learning_rate": lr,
                "training_seconds": time.monotonic() - start,
            }
            if not np.isfinite(row["train_loss"]):
                raise ValueError("Nonfinite training loss.")
            if len(valid_rows):
                probability = _predict_tensor(
                    model, values, neighbors, valid_rows, cfg.batch_size
                )
                metrics = evaluate(
                    np.asarray(CLASSES_5)[data.labels[valid_rows]],
                    np.asarray(CLASSES_5)[probability.argmax(axis=1)],
                    CLASSES_5,
                )
                row.update(
                    {"inner_" + k: v for k, v in scalar_metrics(metrics).items()}
                )
                selected_probability = probability[
                    np.arange(len(valid_rows)), data.labels[valid_rows]
                ]
                row["inner_loss"] = float(
                    np.average(
                        -np.log(np.maximum(selected_probability, 1e-12)),
                        weights=weights[data.labels[valid_rows]],
                    )
                )
                if metrics["f1_macro"] > best_score + cfg.min_delta:
                    best_score, best_epoch, stale = metrics["f1_macro"], epoch + 1, 0
                    best_state = _cpu(model.state_dict())
                    write_checkpoint(
                        directory / "best.pt",
                        {
                            "identity": identity,
                            "model": best_state,
                            "scaler": scaler.to_dict(),
                        },
                    )
                else:
                    stale += 1
            else:
                best_epoch = epoch + 1
            row["epoch_seconds"] = time.monotonic() - start
            row["mps_allocated_bytes"] = torch.mps.current_allocated_memory()
            history.append(row)
            write_checkpoint(
                last,
                {
                    "identity": identity,
                    "model": _cpu(model.state_dict()),
                    "optimizer": _cpu(optimizer.state_dict()),
                    "history": history,
                    "best_score": best_score,
                    "best_epoch": best_epoch,
                    "stale": stale,
                    "best_state": best_state,
                    "rng": _rng(),
                },
            )
            write_json(directory / "history.json", history)
            mlflow.log_metrics(
                {k: float(v) for k, v in row.items() if k != "epoch"}, step=epoch + 1
            )
            print(
                f"EPOCH {spec.name} {purpose} {directory.name} {epoch + 1}/{epochs} loss={row['train_loss']:.4f} f1={row.get('inner_f1_macro', float('nan')):.4f} accuracy={row.get('inner_accuracy', float('nan')):.4f} kappa={row.get('inner_kappa', float('nan')):.4f} seconds={row['epoch_seconds']:.1f}",
                flush=True,
            )
        state = best_state if fixed_epochs is None else _cpu(model.state_dict())
        write_checkpoint(
            directory / "model.pt",
            {
                "identity": identity,
                "model": state,
                "scaler": scaler.to_dict(),
                "spec": spec.model_dump(),
                "config": cfg.model_dump(),
                "columns": data.columns,
                "classes": CLASSES_5,
            },
        )
        summary = {
            "identity": identity,
            "best_epoch": best_epoch,
            "best_f1_macro": best_score if fixed_epochs is None else None,
            "epochs_completed": len(history),
            "elapsed_seconds": sum(r["epoch_seconds"] for r in history),
            "model_sha256": hash_file(directory / "model.pt"),
            "mlflow_run_id": run_id,
            "directory": str(directory.resolve()),
            "lr": lr,
        }
        mlflow.log_artifacts(str(directory), "training")
    write_json(summary_path, summary)
    MlflowClient().log_artifact(run_id, str(summary_path), artifact_path="training")
    del values, targets, model, optimizer
    torch.mps.empty_cache()
    return summary


def load_reference(
    reference_path: Path, split_path: Path
) -> tuple[SequenceData, dict, dict, pd.DataFrame]:
    campaign = json.loads(reference_path.read_text())
    summary = json.loads(
        (Path(campaign["winner"]["run_dir"]) / "summary.json").read_text()
    )
    dataset = Path(campaign["dataset"])
    if hash_file(dataset) != summary["identity"]["dataset_sha256"]:
        raise ValueError("Reference dataset checksum mismatch.")
    oof_path = Path(summary["run_dir"]) / "oof.parquet"
    if hash_file(oof_path) != summary["oof_sha256"]:
        raise ValueError("Reference OOF checksum mismatch.")
    metadata = json.loads((dataset.parent / "metadata.json").read_text())
    if hash_file(split_path) != metadata["split_sha256"]:
        raise ValueError("Reference split checksum mismatch.")
    data = SequenceData.from_frame(pd.read_parquet(dataset))
    splits = pd.read_csv(split_path, dtype={"subject_id": str})
    if set(data.frame.subject_id) & set(
        splits.loc[splits.partition.eq("test"), "subject_id"]
    ):
        raise ValueError("Reserved test subjects entered research features.")
    frozen = splits.set_index("subject_id")
    if not data.frame.fold.eq(data.frame.subject_id.map(frozen.fold)).all() or set(
        data.frame.fold
    ) != set(range(5)):
        raise ValueError("Incompatible frozen folds.")
    reference = pd.read_parquet(oof_path).sort_values("epoch_id").reset_index(drop=True)
    frame = data.frame.iloc[data.targets].sort_values("epoch_id").reset_index(drop=True)
    if not frame[["epoch_id", "subject_id", "recording_id", "onset_s", "fold"]].equals(
        reference[["epoch_id", "subject_id", "recording_id", "onset_s", "fold"]]
    ) or not frame.task_5.equals(reference.y_true):
        raise ValueError("Reference and neural targets must be identical.")
    identity = {
        "dataset_sha256": hash_file(dataset),
        "split_sha256": hash_file(split_path),
        "source_hash": code_fingerprint(),
        "reference_oof_sha256": summary["oof_sha256"],
        "columns": data.columns,
        "versions": {
            name: version(name)
            for name in ["torch", "numpy", "pandas", "scikit-learn", "mlflow"]
        },
        "platform": platform.platform(),
        "device": "mps",
        "precision": "float32",
        "padding": "replicate_within_contiguous_recording_block",
        "scaler": "training_subjects_median_p5_p95_div329_clip10",
    }
    return data, campaign, identity, reference


def initialize_tracking(reference_path: Path, output_root: Path) -> str:
    uri = f"sqlite:///{reference_path.resolve().parent / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    name = "sleep-cassette-lstm-mps-v1"
    experiment = mlflow.get_experiment_by_name(name)
    experiment_id = (
        experiment.experiment_id
        if experiment
        else mlflow.create_experiment(
            name, artifact_location=str((output_root / "mlflow-artifacts").resolve())
        )
    )
    mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_id


def technical_checks(
    data: SequenceData, cfg: LSTMConfig, identity: dict, root: Path, benchmark: bool
) -> dict:
    key = fingerprint(
        {
            **identity,
            "config": cfg.model_dump(),
            "mode": "benchmark" if benchmark else "preflight",
        }
    )
    directory = root / "technical" / key
    training, validation, _ = split_subjects(data.frame, 0, cfg.seed)
    training_rows = data.targets[
        data.frame.subject_id.isin(training).to_numpy()[data.targets]
    ]
    results = []
    with tracked_run(
        directory,
        "benchmark" if benchmark else "preflight",
        "technical_benchmark" if benchmark else "technical_preflight",
        key,
        nested=False,
    ):
        mlflow.log_dict(identity, "provenance.json")
        for spec in cfg.variants:
            if benchmark:
                local_cfg = cfg.model_copy(update={"max_epochs": 1})
                fit_id = fingerprint({"parent": key, "variant": spec.model_dump()})
                summary = fit_sequence(
                    data,
                    local_cfg,
                    spec,
                    training,
                    validation,
                    cfg.learning_rates[0],
                    directory / spec.name,
                    fit_id,
                    cfg.seed,
                    purpose="technical_benchmark",
                )
                history = json.loads(
                    (Path(summary["directory"]) / "history.json").read_text()
                )
                results.append(
                    {
                        "name": spec.name,
                        "training_rows": len(training_rows),
                        **history[-1],
                        "run_id": summary["mlflow_run_id"],
                    }
                )
            else:
                _seed(cfg.seed)
                scaler = FeatureScaler.fit(
                    data.values[data.frame.subject_id.isin(training)]
                )
                rows = np.concatenate(
                    [
                        training_rows[data.labels[training_rows] == i][:4]
                        for i in range(5)
                    ]
                )
                neighbors, _ = sequence_index(data.frame, spec.length, spec.centered)
                x = torch.from_numpy(scaler.transform(data.values)[neighbors[rows]]).to(
                    require_mps()
                )
                y = torch.from_numpy(data.labels[rows]).to("mps")
                model = _network(data, cfg, spec)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
                before = _cpu(model.state_dict())
                loss = torch.nn.functional.cross_entropy(model(x), y)
                loss.backward()
                finite = bool(torch.isfinite(loss)) and all(
                    p.grad is not None and bool(torch.isfinite(p.grad).all())
                    for p in model.parameters()
                )
                optimizer.step()
                changed = any(
                    not torch.equal(before[k], v.detach().cpu())
                    for k, v in model.state_dict().items()
                )
                if (
                    not finite
                    or not changed
                    or any(p.device.type != "mps" for p in model.parameters())
                ):
                    raise ValueError("Real-data MPS gradient preflight failed.")
                model.eval()
                with torch.no_grad():
                    expected = model(x).softmax(dim=1).cpu().numpy()
                write_checkpoint(
                    directory / f"{spec.name}.pt",
                    {"identity": key, "model": _cpu(model.state_dict())},
                )
                reloaded = _network(data, cfg, spec)
                reloaded.load_state_dict(
                    read_checkpoint(directory / f"{spec.name}.pt", key)["model"]
                )
                reloaded.eval()
                with torch.no_grad():
                    actual = reloaded(x).softmax(dim=1).cpu().numpy()
                np.testing.assert_allclose(expected, actual, atol=1e-6)
                np.testing.assert_allclose(actual.sum(axis=1), 1, atol=1e-6)
                results.append(
                    {
                        "name": spec.name,
                        "device": "mps",
                        "loss": float(loss.detach().cpu()),
                        "gradients_finite": finite,
                        "weights_updated": changed,
                        "reload_equal": True,
                        "batch_rows": len(rows),
                    }
                )
                del model, reloaded, optimizer, x, y
                torch.mps.empty_cache()
        result = {
            "identity": key,
            "config": cfg.model_dump(),
            "results": results,
            "test_evaluated": False,
        }
        write_json(directory / "result.json", result)
        mlflow.log_artifact(str(directory / "result.json"))
    write_json(root / ("benchmark.json" if benchmark else "preflight.json"), result)
    return result


def _fit_fold(
    data: SequenceData,
    cfg: LSTMConfig,
    spec: WindowSpec,
    base_identity: str,
    variant_dir: Path,
    fold: int,
) -> tuple[dict, pd.DataFrame]:
    fold_id = fingerprint(
        {"campaign": base_identity, "spec": spec.model_dump(), "fold": fold}
    )
    directory = variant_dir / f"fold-{fold}"
    if (directory / "summary.json").exists():
        result = json.loads((directory / "summary.json").read_text())
        if (
            result["identity"] != fold_id
            or hash_file(directory / "oof.parquet") != result["oof_sha256"]
        ):
            raise ValueError("Fold OOF cache identity/checksum mismatch.")
        if MlflowClient().get_run(result["mlflow_run_id"]).info.status != "FINISHED":
            raise ValueError("Fold cache has an unfinished MLflow run.")
        return result, pd.read_parquet(directory / "oof.parquet")
    training, validation, external = split_subjects(data.frame, fold, cfg.seed + fold)
    trials = []
    with tracked_run(
        directory, f"{spec.name}-fold-{fold}", "outer_fold", fold_id
    ) as run_id:
        for candidate, lr in enumerate(cfg.learning_rates):
            fit_id = fingerprint(
                {"fold": fold_id, "candidate": candidate, "phase": "inner"}
            )
            summary = fit_sequence(
                data,
                cfg,
                spec,
                training,
                validation,
                lr,
                directory / f"trial-{candidate}",
                fit_id,
                cfg.seed + fold,
            )
            trials.append({**summary, "candidate": candidate})
        chosen = choose_trial(trials)
        refit_id = fingerprint(
            {
                "fold": fold_id,
                "candidate": chosen["candidate"],
                "epochs": chosen["best_epoch"],
                "phase": "refit",
            }
        )
        refit = fit_sequence(
            data,
            cfg,
            spec,
            sorted(training + validation),
            [],
            chosen["lr"],
            directory / "refit",
            refit_id,
            cfg.seed + fold,
            fixed_epochs=chosen["best_epoch"],
            purpose="outer_train_refit",
        )
        external_rows = data.targets[
            data.frame.subject_id.isin(external).to_numpy()[data.targets]
        ]
        probability = predict_checkpoint(
            Path(refit["directory"]) / "model.pt",
            data,
            external_rows,
            cfg,
            spec,
            refit_id,
        )
        np.testing.assert_allclose(probability.sum(axis=1), 1, atol=1e-5)
        oof = data.frame.iloc[external_rows][
            ["epoch_id", "subject_id", "recording_id", "onset_s", "fold"]
        ].copy()
        oof["y_true"] = data.frame.iloc[external_rows].task_5.to_numpy()
        oof["y_pred"] = np.asarray(CLASSES_5)[probability.argmax(axis=1)]
        oof[["prob_" + c for c in CLASSES_5]] = probability
        metrics = evaluate(oof.y_true.to_numpy(), oof.y_pred.to_numpy(), CLASSES_5)
        with atomic_path(directory / "oof.parquet") as temporary:
            oof.to_parquet(temporary, index=False)
        result = {
            "identity": fold_id,
            "fold": fold,
            "metrics": metrics,
            "trials": trials,
            "selected": chosen,
            "refit": refit,
            "training_subjects": training,
            "inner_validation_subjects": validation,
            "external_subjects": external,
            "oof_sha256": hash_file(directory / "oof.parquet"),
            "mlflow_run_id": run_id,
        }
        mlflow.log_metrics(scalar_metrics(metrics))
        mlflow.log_dict(result, "fold-summary.json")
        mlflow.log_artifact(str(directory / "oof.parquet"))
    write_json(directory / "summary.json", result)
    print(
        f"OUTER {spec.name} fold={fold} f1={metrics['f1_macro']:.4f} accuracy={metrics['accuracy']:.4f} kappa={metrics['kappa']:.4f}",
        flush=True,
    )
    return result, oof


def _evaluate_variant(
    data: SequenceData, cfg: LSTMConfig, spec: WindowSpec, identity: str, root: Path
) -> tuple[dict, pd.DataFrame]:
    directory = root / spec.name
    if (directory / "summary.json").exists():
        result = json.loads((directory / "summary.json").read_text())
        if (
            result["identity"] != identity
            or hash_file(directory / "oof.parquet") != result["oof_sha256"]
        ):
            raise ValueError("Variant cache mismatch.")
        return result, pd.read_parquet(directory / "oof.parquet")
    folds, tables = [], []
    with tracked_run(
        directory, spec.name, "five_fold_neural_comparison", identity
    ) as run_id:
        mlflow.log_params(
            {
                **spec.model_dump(),
                "features_per_segment": len(data.columns),
                "outer_folds": 5,
                "architecture": "projection64_lstm64_head5",
                "primary_metric": "f1_macro",
            }
        )
        for fold in range(5):
            result, oof = _fit_fold(data, cfg, spec, identity, directory, fold)
            folds.append(result)
            tables.append(oof)
        oof = pd.concat(tables, ignore_index=True)
        if oof.epoch_id.duplicated().any() or set(oof.epoch_id) != set(
            data.frame.iloc[data.targets].epoch_id
        ):
            raise ValueError("OOF must cover every eligible epoch exactly once.")
        metrics = evaluate(oof.y_true.to_numpy(), oof.y_pred.to_numpy(), CLASSES_5)
        result = {
            "name": spec.name,
            "family": "pytorch_lstm",
            "features": len(data.columns),
            "length": spec.length,
            "centered": spec.centered,
            "identity": identity,
            "f1_macro": float(np.mean([r["metrics"]["f1_macro"] for r in folds])),
            "f1_std": float(np.std([r["metrics"]["f1_macro"] for r in folds])),
            "accuracy": float(np.mean([r["metrics"]["accuracy"] for r in folds])),
            "kappa": float(np.mean([r["metrics"]["kappa"] for r in folds])),
            "n1_f1_oof": metrics["report"]["N1"]["f1-score"],
            "rem_f1_oof": metrics["report"]["REM"]["f1-score"],
            "fold_scores": [r["metrics"]["f1_macro"] for r in folds],
            "folds": folds,
            "directory": str(directory.resolve()),
            "mlflow_run_id": run_id,
        }
        with atomic_path(directory / "oof.parquet") as temporary:
            oof.to_parquet(temporary, index=False)
        result["oof_sha256"] = hash_file(directory / "oof.parquet")
        write_json(directory / "pooled-metrics.json", metrics)
        plot_confusion(metrics, directory / "confusion.png")
        mlflow.log_metrics(
            {
                k: result[k]
                for k in [
                    "f1_macro",
                    "f1_std",
                    "accuracy",
                    "kappa",
                    "n1_f1_oof",
                    "rem_f1_oof",
                ]
            }
        )
        mlflow.log_dict(result, "summary.json")
        for name in ["oof.parquet", "pooled-metrics.json", "confusion.png"]:
            mlflow.log_artifact(str(directory / name))
    write_json(directory / "summary.json", result)
    print(
        f"RESULT {spec.name} F1={result['f1_macro']:.4f} +/-{result['f1_std']:.4f} accuracy={result['accuracy']:.4f} kappa={result['kappa']:.4f}",
        flush=True,
    )
    return result, oof


def _final_candidate(
    data: SequenceData, cfg: LSTMConfig, spec: WindowSpec, identity: str, root: Path
) -> dict:
    directory = root / "candidate"
    training, validation, _ = split_subjects(data.frame, -1, cfg.seed + 500)
    trials = []
    with tracked_run(
        directory, "best_neural_full_development", "neural_candidate_refit", identity
    ) as run_id:
        for candidate, lr in enumerate(cfg.learning_rates):
            fit_id = fingerprint(
                {"campaign": identity, "candidate": candidate, "phase": "final_inner"}
            )
            summary = fit_sequence(
                data,
                cfg,
                spec,
                training,
                validation,
                lr,
                directory / f"trial-{candidate}",
                fit_id,
                cfg.seed + 500,
            )
            trials.append({**summary, "candidate": candidate})
        chosen = choose_trial(trials)
        refit_id = fingerprint(
            {
                "campaign": identity,
                "candidate": chosen["candidate"],
                "epochs": chosen["best_epoch"],
                "phase": "final_refit",
            }
        )
        refit = fit_sequence(
            data,
            cfg,
            spec,
            sorted(training + validation),
            [],
            chosen["lr"],
            directory / "refit",
            refit_id,
            cfg.seed + 500,
            fixed_epochs=chosen["best_epoch"],
            purpose="full_development_refit",
        )
        model_path = Path(refit["directory"]) / "model.pt"
        rows = data.targets[:20]
        a = predict_checkpoint(model_path, data, rows, cfg, spec, refit_id)
        b = predict_checkpoint(model_path, data, rows, cfg, spec, refit_id)
        np.testing.assert_allclose(a, b, atol=1e-6)
        np.testing.assert_allclose(a.sum(axis=1), 1, atol=1e-5)
        neighbors, _ = sequence_index(data.frame, spec.length, spec.centered)
        bundle = read_checkpoint(model_path, refit_id)
        example = FeatureScaler.from_dict(bundle["scaler"]).transform(data.values)[
            neighbors[rows]
        ]
        np.savez_compressed(directory / "example.npz", windows=example, probability=a)
        result = {
            "model_path": str(model_path),
            "model_sha256": hash_file(model_path),
            "checkpoint_identity": refit_id,
            "name": spec.name,
            "spec": spec.model_dump(),
            "selected": chosen,
            "trials": trials,
            "refit": refit,
            "mlflow_run_id": run_id,
            "reload_equal": True,
            "probabilities_sum_to_one": True,
            "input_kind": "scaled_feature_windows",
            "test_evaluated": False,
        }
        write_json(directory / "candidate.json", result)
        mlflow.log_artifact(str(directory / "candidate.json"))
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(model_path.with_suffix(".pt.sha256")))
        mlflow.log_artifact(str(directory / "example.npz"))
    return result


def run_lstm_campaign(
    config_path: Path,
    reference_path: Path,
    output_root: Path,
    mode: str = "run",
    resume: bool = True,
) -> dict:
    require_mps()
    output_root = output_root.resolve()
    reference_path = reference_path.resolve()
    if output_root == reference_path.parent:
        raise ValueError("LSTM outputs must not overwrite the reference campaign.")
    output_root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output_root / "campaign.lock"), timeout=0):
        cfg = LSTMConfig.model_validate(read_yaml(config_path))
        split_path = (
            Path(__file__).resolve().parents[2] / "configs/splits/sc_subjects_v1.csv"
        )
        data, reference_campaign, identity, reference_oof = load_reference(
            reference_path, split_path
        )
        identity["config"] = cfg.model_dump()
        lockfile = Path(__file__).resolve().parents[2] / "uv.lock"
        identity["lockfile_sha256"] = hash_file(lockfile)
        experiment_id = initialize_tracking(reference_path, output_root)
        if mode in {"preflight", "benchmark"}:
            return technical_checks(
                data, cfg, identity, output_root, benchmark=mode == "benchmark"
            )
        if mode != "run":
            raise ValueError("Unknown execution mode.")
        key = fingerprint(identity)
        directory = output_root / "runs" / key
        if (directory / "campaign.json").exists():
            existing = json.loads((directory / "campaign.json").read_text())
            for row in existing["neural_results"]:
                if (
                    hash_file(Path(row["directory"]) / "oof.parquet")
                    != row["oof_sha256"]
                ):
                    raise ValueError("Completed campaign OOF checksum mismatch.")
            if (
                hash_file(Path(existing["candidate"]["model_path"]))
                != existing["candidate"]["model_sha256"]
            ):
                raise ValueError("Completed candidate checksum mismatch.")
            write_json(output_root / "campaign.json", existing)
            return existing
        if directory.exists() and not resume:
            raise ValueError(
                "Campaign exists; use --resume or a different output root."
            )
        with tracked_run(
            directory,
            "lstm_mps_three_variants_five_subject_folds",
            "lstm_research_campaign",
            key,
            nested=False,
        ) as run_id:
            mlflow.log_dict(identity, "identity.json")
            mlflow.log_params(
                {
                    "subjects": int(data.frame.subject_id.nunique()),
                    "recordings": int(data.frame.recording_id.nunique()),
                    "supervised_epochs": len(data.targets),
                    "features": len(data.columns),
                    "device": "mps",
                    "outer_folds": 5,
                    "test_evaluated": False,
                }
            )
            results, bootstraps = [], {}
            champion = reference_campaign["winner"]
            for spec in cfg.variants:
                result, oof = _evaluate_variant(data, cfg, spec, key, directory)
                results.append(result)
                bootstrap = paired_subject_bootstrap(reference_oof, oof)
                bootstrap["reference"] = champion["name"]
                bootstrap["candidate"] = spec.name
                bootstraps[spec.name] = bootstrap
                rows = [
                    {k: v for k, v in r.items() if k not in {"folds"}} for r in results
                ]
                board = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
                board.to_csv(output_root / "leaderboard.csv", index=False)
                print(
                    f"CURRENT_NEURAL_WINNER {board.iloc[0]['name']} f1={board.iloc[0].f1_macro:.4f}; incumbent={champion['name']} f1={champion['f1_macro']:.4f}",
                    flush=True,
                )
            best = max(results, key=lambda r: r["f1_macro"])
            spec = next(v for v in cfg.variants if v.name == best["name"])
            candidate = _final_candidate(data, cfg, spec, key, directory)
            overall = (
                best["name"]
                if best["f1_macro"] > champion["f1_macro"]
                else champion["name"]
            )
            result = {
                "identity": key,
                "provenance": identity,
                "config": cfg.model_dump(),
                "subjects": int(data.frame.subject_id.nunique()),
                "epochs": len(data.targets),
                "neural_results": results,
                "best_neural": best["name"],
                "overall_winner": overall,
                "champion": champion,
                "delta_cv_f1": best["f1_macro"] - champion["f1_macro"],
                "candidate": candidate,
                "paired_bootstrap": bootstraps,
                "mlflow_parent_run_id": run_id,
                "experiment_id": experiment_id,
                "test_evaluated": False,
            }
            write_json(directory / "paired-bootstrap.json", bootstraps)
            write_json(output_root / "paired-bootstrap.json", bootstraps)
            write_json(
                output_root / "tracking-audit.json",
                {
                    "parent": run_id,
                    "experiment_id": experiment_id,
                    "variants": {r["name"]: r["mlflow_run_id"] for r in results},
                    "candidate": candidate["mlflow_run_id"],
                    "overall_winner": overall,
                },
            )
            mlflow.log_metrics(
                {
                    "best_neural_f1_macro": best["f1_macro"],
                    "champion_f1_macro": champion["f1_macro"],
                    "delta_cv_f1": result["delta_cv_f1"],
                }
            )
            mlflow.log_dict(result, "campaign.json")
            for name in [
                "leaderboard.csv",
                "paired-bootstrap.json",
                "tracking-audit.json",
            ]:
                mlflow.log_artifact(str(output_root / name))
        write_json(directory / "campaign.json", result)
        write_json(output_root / "campaign.json", result)
        print(
            f"WINNER_GLOBAL {overall}; BEST_NEURAL {best['name']} F1={best['f1_macro']:.4f} +/-{best['f1_std']:.4f}; DELTA={result['delta_cv_f1']:+.4f}",
            flush=True,
        )
        return result
