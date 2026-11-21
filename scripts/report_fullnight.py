"""Derive paper tables from saved predictions and counts without rerunning a model."""

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from sleep_staging.fullnight.data import LABELS
from sleep_staging.fullnight.metrics import state_slices
from sleep_staging.fullnight.training import ALL, REFERENCE
from sleep_staging.provenance import atomic_path, fingerprint, hash_file


def n1_errors(path: Path) -> list[dict]:
    cols = [
        "record",
        "subject",
        "cohort",
        "model",
        "onset_s",
        "target",
        "eligible",
        *["prob_" + c for c in LABELS[:5]],
    ]
    table = pq.read_table(
        path,
        columns=cols,
        filters=[
            ("profile", "=", REFERENCE),
            ("model", "in", ["blend", "triple", "spectrogram"]),
        ],
    )
    rows = []
    for (cohort, subject, record, model), night in table.to_pandas().groupby(
        ["cohort", "subject", "record", "model"], sort=False
    ):
        night = night.sort_values("onset_s")
        if not np.all(np.diff(night.onset_s) == 30):
            raise ValueError(f"Faltan épocas adquiridas en {record}.")
        y = night.target.to_numpy()
        transition, stable = state_slices(y)
        p = night[[f"prob_{c}" for c in LABELS[:5]]].to_numpy()
        pred = p.argmax(axis=1)
        for name, mask in [
            ("transition", transition),
            ("stable", stable),
            ("without_two_neighbours", ~(transition | stable)),
        ]:
            used = mask & (y == 1) & night.eligible.to_numpy()
            rows.append(
                dict(
                    cohort=cohort,
                    subject=subject,
                    record=record,
                    model=model,
                    context=name,
                    n1=int(used.sum()),
                    **{
                        f"as_{c}": int(np.sum(used & (pred == i)))
                        for i, c in enumerate(LABELS[:5])
                    },
                )
            )
    return rows


def save(frame: pd.DataFrame, path: Path) -> None:
    with atomic_path(path) as tmp:
        frame.to_csv(tmp, index=False)


def report(run: Path) -> None:
    dest = run / "tables"
    partitions = [run]
    nightly = sorted(run.glob("fold-*/predictions.parquet"))
    if (run / "reserved/completed.json").is_file():
        partitions.append(run / "reserved")
        nightly.append(run / "reserved/predictions.parquet")
    metrics = pd.concat([pd.read_csv(path / "metrics.csv") for path in partitions])
    full = metrics.loc[
        metrics.window == "full",
        [
            "cohort",
            "profile",
            "model",
            "scored",
            "coverage",
            "accuracy",
            "macro_f1",
            "n1_f1",
        ],
    ]
    ref = full.loc[full.profile == ALL, ["cohort", "model", "macro_f1"]].rename(
        columns={"macro_f1": "all_channels_macro_f1"}
    )
    profiles = full.merge(ref, on=["cohort", "model"], validate="many_to_one")
    profiles["delta_macro_f1_vs_all"] = (
        profiles.macro_f1 - profiles.all_channels_macro_f1
    )
    save(profiles, dest / "profiles.csv")
    stage_rows = []
    for partition in partitions:
        for key, values in json.loads(
            (partition / "confusion.json").read_text()
        ).items():
            cohort, profile, model, window = key.split("/")
            cm = np.asarray(values["counts"])
            total = cm.sum()
            expected = float(cm.sum(0) @ cm.sum(1)) / max(float(total) ** 2, 1)
            agreement = float(cm.trace()) / max(total, 1)
            for i, label in enumerate(values["classes"]):
                actual, called = cm[i].sum(), cm[:, i].sum()
                stage_rows.append(
                    dict(
                        cohort=cohort,
                        profile=profile,
                        model=model,
                        window=window,
                        stage=label,
                        support=actual,
                        predicted=called,
                        recall=cm[i, i] / max(actual, 1),
                        precision=cm[i, i] / max(called, 1),
                        f1=2 * cm[i, i] / max(actual + called, 1),
                        kappa=(agreement - expected) / (1 - expected)
                        if expected < 1
                        else np.nan,
                    )
                )
    save(pd.DataFrame(stage_rows), dest / "stages.csv")
    errors = pd.DataFrame([r for path in nightly for r in n1_errors(path)])
    save(errors, dest / "n1-by-night.csv")
    save(
        errors.groupby(["cohort", "model", "context"], as_index=False)[
            ["n1", *["as_" + c for c in LABELS[:5]]]
        ].sum(),
        dest / "n1-errors.csv",
    )
    mov = []
    for path in nightly:
        rows = json.loads(path.with_name("counts.json").read_text())
        for r in rows:
            if r["profile"] == ALL and r["model"] == "six_direct":
                cm = np.asarray(r["windows"]["full"]["matrix"])
                mov.append(
                    dict(
                        cohort=r["cohort"],
                        subject=r["subject"],
                        record=r["record"],
                        fold=r["fold"],
                        m_scored=cm[5].sum(),
                        m_correct=cm[5, 5],
                        m_false_positives=cm[:, 5].sum() - cm[5, 5],
                    )
                )
    save(pd.DataFrame(mov), dest / "movement-by-night.csv")


def package(run: Path, output: Path) -> None:
    freeze = json.loads((run / "freeze.json").read_text())
    protocol = json.loads((run / "protocol.json").read_text())
    if freeze["smoke"] or fingerprint(protocol) != freeze["run"]:
        raise ValueError("Se requiere una campaña de desarrollo completa y congelada.")
    files = [
        run / name for name in ("protocol.json", "freeze.json", "training-source.zip")
    ]
    for name, digest in freeze["weights"].items():
        weight = run / "refit" / name
        if hash_file(weight) != digest:
            raise ValueError(f"Peso incompatible con el manifiesto: {name}")
        files.append(weight)
    for experiment in [
        run,
        *sorted(p.parent for p in run.glob("usleep/*/freeze.json")),
    ]:
        if experiment != run:
            weight = experiment / "refit/model.pt"
            frozen = json.loads((experiment / "freeze.json").read_text())
            if hash_file(weight) != frozen["model_sha256"]:
                raise ValueError("Peso de U-Sleep incompatible.")
            files.extend(
                [weight, experiment / "protocol.json", experiment / "freeze.json"]
            )
        files.extend(experiment.glob("mlflow-runs.json"))
        for portion in [experiment, experiment / "reserved"]:
            for name in (
                "metrics.csv",
                "confusion.json",
                "paired-subject-bootstrap.json",
            ):
                files.append(portion / name)
        files.append(experiment / "reserved/completed.json")
    files.extend(sorted((run / "tables").glob("*.csv")))
    for name in ("edf-upload-check.json", "five-stage-edf-parity.json"):
        files.extend(run.glob(name))
    with (
        atomic_path(output) as tmp,
        zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as archive,
    ):
        for file in files:
            archive.write(file, str(Path(freeze["run"]) / file.relative_to(run)))
        archive.writestr(
            "LEEME.txt",
            f"Desde la raíz del repositorio con las dependencias instaladas, use el directorio extraído para la inferencia.\n\n.venv/bin/python scripts/serve_fullnight.py --run /ruta/{freeze['run']} --device cpu\n\nPara puntuar cinco estadios desde un EDF, añada --task five --edf /ruta/noche.edf --profile eeg_fpz_cz+eeg_pz_oz+eog --output /tmp/predicciones.csv.\nLas tablas contienen métricas agregadas por perfil, estadio y cohorte. La guía de reproducción está en docs/ibio-registro-completo.md del repositorio.\n",
        )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run", type=Path, required=True)
    cli.add_argument(
        "--bundle", type=Path, help="ZIP con pesos congelados y tablas agregadas."
    )
    args = cli.parse_args()
    report(args.run)
    if args.bundle:
        package(args.run, args.bundle)
