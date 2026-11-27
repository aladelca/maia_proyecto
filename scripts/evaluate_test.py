"""Evaluación final, única, del modelo campeón sobre los 16 sujetos de prueba reservados.

Etapas:
1. Construye (o reutiliza) el dataset de desarrollo de 145 variables con el mismo
   código de las campañas (``build_research_dataset``).
2. Reentrena el campeón congelado (LightGBM ``large_regularized``, 261 iteraciones,
   semilla 542) con los 62 sujetos de desarrollo, con el mismo procedimiento del
   refit en la nube (``fit_booster``).
3. Verifica el SHA-256 de los EDF de prueba y extrae sus variables con el mismo
   protocolo (recorte de vigilia de ±30 min, filtros, descriptores).
4. Predice y calcula las métricas una sola vez: F1 macro, exactitud, kappa, F1 por
   estadio, matriz de confusión, métricas por sujeto e intervalos bootstrap del 95 %
   remuestreando sujetos completos.

El script se niega a evaluar de nuevo si ya existen resultados, salvo con
``--allow-reevaluation`` (que queda registrado en el resultado).

Uso (desde la raíz del repositorio):
    uv run --locked --extra boosting python scripts/evaluate_test.py
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from sleep_staging.booster_selection import BoosterCandidate
from sleep_staging.booster_worker import Parameters, fit_booster
from sleep_staging.evaluation import evaluate, scalar_metrics
from sleep_staging.lstm_data import SequenceData
from sleep_staging.provenance import atomic_path, fingerprint, hash_file, write_json
from sleep_staging.research_data import _record_features, build_research_dataset
from sleep_staging.serving.contract import LABELS, context_matrix

META = ["epoch_id", "recording_id", "subject_id", "onset_s", "fold", "task_5"]
CHAMPION = BoosterCandidate(
    name="large_regularized", num_leaves=63, min_child_samples=200, reg_lambda=5
)
ITERATIONS = 261
SEED = 42 + 500  # cfg.seed + 500, igual que el refit en la nube
THREADS = 4


def signal_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if "__" in c]


def with_context(raw: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    raw = raw.sort_values(["recording_id", "onset_s"]).reset_index(drop=True)
    return pd.concat([raw[META], context_matrix(raw, columns)], axis=1)


def test_features(root: Path, splits: Path, cache: Path) -> pd.DataFrame:
    subjects = pd.read_csv(splits, dtype={"subject_id": str})
    test = subjects[subjects.partition.eq("test")]
    records = pd.read_parquet(root / "data/manifests/sleep-cassette/recordings.parquet")
    records = records.merge(test, on="subject_id", validate="many_to_one")
    files = pd.read_parquet(root / "data/manifests/sleep-cassette/files.parquet")
    paths = set(records.psg_file) | set(records.hyp_file)
    files = files[files.path.isin(paths)]
    if set(files.path) != paths:
        raise ValueError("Faltan archivos de prueba en el manifiesto de checksums.")
    for file in files.itertuples():
        if hash_file(root / file.path) != file.sha256:
            raise ValueError(f"Checksum incorrecto: {file.path}")
    frames = []
    for n, record in enumerate(
        records.sort_values("recording_id").to_dict("records"), 1
    ):
        path = cache / f"{record['recording_id']}.parquet"
        if not path.exists():
            frame = _record_features(record, root, margin=30)
            with atomic_path(path) as temporary:
                frame.to_parquet(temporary, index=False)
        frames.append(pd.read_parquet(path))
        print(
            f"TEST {n}/{len(records)} {record['recording_id']} epochs={len(frames[-1])}",
            flush=True,
        )
    return pd.concat(frames, ignore_index=True)


def plot_test_confusion(result: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for ax, key, title, fmt in zip(
        axes,
        ["confusion", "confusion_normalized"],
        ["Conteos (prueba)", "Proporción por estadio real (prueba)"],
        ["{:.0f}", "{:.2f}"],
        strict=True,
    ):
        values = np.array(result[key])
        ax.imshow(values, cmap="Greys")
        ax.set_xticks(range(5), LABELS, fontsize=12)
        ax.set_yticks(range(5), LABELS, fontsize=12)
        ax.set_xlabel("Predicción", fontsize=12)
        ax.set_ylabel("Estadio real", fontsize=12)
        ax.set_title(title, fontsize=13)
        top = values.max()
        for (i, j), v in np.ndenumerate(values):
            ax.text(
                j,
                i,
                fmt.format(v),
                ha="center",
                va="center",
                fontsize=11,
                color="white" if v > top * 0.55 else "black",
            )
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def subject_bootstrap(frame: pd.DataFrame, n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    subjects = frame.subject_id.unique()
    groups = {s: g for s, g in frame.groupby("subject_id")}
    draws: dict[str, list[float]] = {"f1_macro": [], "accuracy": [], "kappa": []}
    for _ in range(n):
        sample = pd.concat([groups[s] for s in rng.choice(subjects, len(subjects))])
        m = scalar_metrics(evaluate(sample.task_5, sample.prediction, LABELS))
        for k in draws:
            draws[k].append(m[k])
    return {
        k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        for k, v in draws.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-root", type=Path, default=Path("."))
    parser.add_argument(
        "--splits", type=Path, default=Path("configs/splits/sc_subjects_v1.csv")
    )
    parser.add_argument(
        "--research-root", type=Path, default=Path("var/research/sc-w30-v2")
    )
    parser.add_argument("--output", type=Path, default=Path("var/test-evaluation"))
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepara datos y modelo sin calcular métricas de prueba.",
    )
    parser.add_argument("--allow-reevaluation", action="store_true")
    args = parser.parse_args()
    root = args.source_root.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "test_metrics.json"
    if result_path.exists() and not args.allow_reevaluation and not args.prepare_only:
        raise SystemExit(f"El test ya fue evaluado: {result_path}. No se repite.")

    mlflow.set_tracking_uri(f"sqlite:///{(out / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("sleep-cassette-final-test")

    # 1. Desarrollo
    dev_path = build_research_dataset(
        root, root / args.splits, args.research_root.resolve()
    )
    dev_raw = pd.read_parquet(dev_path)
    columns = signal_columns(dev_raw)
    if len(columns) != 145:
        raise ValueError(f"Se esperaban 145 variables y hay {len(columns)}.")
    dev = with_context(dev_raw, columns)
    model_columns = [c for c in dev.columns if c not in META]
    data = SequenceData.from_frame(dev, model_columns)
    print(
        f"DEV epochs={len(dev)} targets={len(data.targets)} columns={len(model_columns)}",
        flush=True,
    )

    # 2. Reentrenamiento del campeón congelado
    cfg = Parameters(seed=42, lgbm_threads=THREADS, lgbm_patience=40, min_delta=0.0001)
    identity = fingerprint(
        {
            "dataset": hash_file(dev_path),
            "candidate": CHAMPION.model_dump(),
            "iterations": ITERATIONS,
            "seed": SEED,
        }
    )
    fit = fit_booster(
        data,
        cfg,
        CHAMPION,
        sorted(set(dev.subject_id)),
        [],
        out / "champion",
        identity,
        SEED,
        ITERATIONS,
    )
    print(
        f"CHAMPION sha256={fit['model_sha256']} trees={fit['best_epoch']}", flush=True
    )

    # 3. Variables de prueba
    test_raw = test_features(root, root / args.splits, out / "test-records")
    if signal_columns(test_raw) != columns:
        raise ValueError("Las columnas de prueba no coinciden con las de desarrollo.")
    if set(test_raw.subject_id) & set(dev_raw.subject_id):
        raise ValueError("Fuga de sujetos entre desarrollo y prueba.")
    test = with_context(test_raw, columns)
    if args.prepare_only:
        print("Preparación terminada; no se calcularon métricas de prueba.")
        return

    # 4. Evaluación única
    import joblib

    model = joblib.load(out / "champion" / "model.joblib")
    held = test[test.task_5.notna()].copy()
    held["prediction"] = model.predict(held[model_columns])
    overall = evaluate(held.task_5, held.prediction, LABELS)
    per_subject = (
        held.groupby("subject_id")
        .apply(
            lambda g: pd.Series(
                scalar_metrics(evaluate(g.task_5, g.prediction, LABELS))
            ),
            include_groups=False,
        )
        .reset_index()
    )
    start = time.monotonic()
    ci = subject_bootstrap(held, args.bootstrap, seed=42)
    report = overall["report"]
    result = {
        "evaluated_at": datetime.now(UTC).isoformat(),
        "reevaluation": bool(result_path.exists()),
        "model": {
            "candidate": CHAMPION.model_dump(),
            "iterations": fit["best_epoch"],
            "seed": SEED,
            "model_sha256": fit["model_sha256"],
            "training_targets": int(len(data.targets)),
        },
        "test": {
            "subjects": int(held.subject_id.nunique()),
            "recordings": int(held.recording_id.nunique()),
            "epochs": int(len(held)),
            "class_support": held.task_5.value_counts()
            .reindex(LABELS)
            .astype(int)
            .to_dict(),
        },
        "metrics": scalar_metrics(overall),
        "bootstrap_95ci_subjects": ci,
        "bootstrap_resamples": args.bootstrap,
        "f1_per_stage": {c: float(report[c]["f1-score"]) for c in LABELS},
        "recall_per_stage": {c: float(report[c]["recall"]) for c in LABELS},
        "precision_per_stage": {c: float(report[c]["precision"]) for c in LABELS},
        "confusion": overall["confusion"],
        "per_subject_summary": {
            k: {
                "mean": float(per_subject[k].mean()),
                "min": float(per_subject[k].min()),
                "max": float(per_subject[k].max()),
            }
            for k in ["f1_macro", "accuracy", "kappa"]
        },
        "bootstrap_seconds": round(time.monotonic() - start, 1),
        "versions": {"python_pid": os.getpid()},
    }
    held[["epoch_id", "subject_id", "recording_id", "task_5", "prediction"]].to_parquet(
        out / "test_predictions.parquet", index=False
    )
    per_subject.to_csv(out / "test_per_subject.csv", index=False)
    plot_test_confusion(overall, out / "test_confusion.png")
    write_json(result_path, result)
    with mlflow.start_run(run_name="final-test", tags={"test_evaluated": "true"}):
        mlflow.log_metrics({"test_" + k: v for k, v in result["metrics"].items()})
        mlflow.log_artifact(str(result_path))
    print(
        json.dumps(
            {"metrics": result["metrics"], "ci": ci, "f1": result["f1_per_stage"]},
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
