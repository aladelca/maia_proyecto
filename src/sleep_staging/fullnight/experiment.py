"""Five frozen development folds, production refit and one locked evaluation."""

import json
import subprocess
import sys
import zipfile
from collections import defaultdict
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from filelock import FileLock

from sleep_staging.fullnight.cohort import development, prepare
from sleep_staging.fullnight.data import LABELS, PROFILES
from sleep_staging.fullnight.metrics import score, summarize_counts
from sleep_staging.fullnight.training import (
    ALL,
    NETWORKS,
    REFERENCE,
    Night,
    Settings,
    encode_night,
    load_model,
    predict_encoded,
    save_model,
    train_network,
)
from sleep_staging.provenance import atomic_path, fingerprint, hash_file, write_json


def protocol_code() -> dict:
    pkg = Path(__file__).parents[1]
    sources = [
        *Path(__file__).parent.glob("*.py"),
        pkg / "data/edf.py",
        pkg / "data/labels.py",
        pkg / "models/tcn.py",
        pkg / "config.py",
        pkg / "sensor_profiles.py",
        pkg / "research_data.py",
    ]
    return {str(p.relative_to(pkg)): hash_file(p) for p in sorted(sources)}


def archive_code(folder: Path, sources: dict) -> None:
    pkg = Path(__file__).parents[1]
    with (
        atomic_path(folder / "training-source.zip") as tmp,
        zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as archive,
    ):
        for relative, digest in sorted(sources.items()):
            source = pkg / relative
            if hash_file(source) != digest:
                raise ValueError("Cambió el código antes de iniciar el entrenamiento.")
            archive.write(source, "src/sleep_staging/" + relative)


def fold_models(
    folder: Path,
    train: list[Night],
    cfg: Settings,
    device: torch.device,
    seed: int,
    run_id: str,
) -> dict:
    result = {}
    for name in NETWORKS:
        path = folder / (name + ".pt")
        if not path.is_file():
            print(f"{folder.name}: entrenando {name}", flush=True)
            net = train_network(name, train, cfg, device, seed)
            save_model(path, net, cfg, [n.subject for n in train], run_id)
        result[name] = load_model(path, device)[0]
    return result


def boosters(
    folder: Path, train: list[Night], validation: list[Night], cfg: Settings, seed: int
) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    req = {
        "train": [n.record for n in train],
        "validation": [n.record for n in validation],
        "paths": {n.record: str(n.path.resolve()) for n in train + validation},
        "output": str(folder.resolve()),
        "seed": seed,
        "trees": cfg.trees,
        "window": cfg.training_window,
    }
    request = folder / "request.json"
    write_json(request, req)
    subprocess.run(
        [sys.executable, "-m", "sleep_staging.fullnight.trees", str(request)],
        check=True,
    )


def evaluate_nights(
    folder: Path,
    tree_folder: Path,
    nets: dict,
    validation: list[Night],
    device: torch.device,
    conditions: dict,
) -> list:
    records = []
    with atomic_path(folder / "predictions.parquet") as tmp:
        writer = None
        try:
            for night in validation:
                x = night.inp
                embeddings = {
                    name: encode_night(net, x, device) for name, net in nets.items()
                }
                onsets = np.arange(len(x.available), dtype=np.int32) * 30
                for profile in PROFILES:
                    tree = np.load(tree_folder / night.record / (profile + ".npy"))
                    wave = predict_encoded(nets["wave"], embeddings["wave"], x, profile)
                    probs = {
                        "lightgbm": tree,
                        "wave": wave,
                        "blend": 0.5 * tree + 0.5 * wave,
                    }
                    if profile == REFERENCE:
                        spec = predict_encoded(
                            nets["spectrogram"], embeddings["spectrogram"], x, profile
                        )
                        probs.update(
                            spectrogram=spec,
                            triple=0.5 * tree + 0.25 * wave + 0.25 * spec,
                        )
                    for name in ("six_direct", "six_movement"):
                        probs[name] = predict_encoded(
                            nets[name], embeddings[name], x, profile
                        )
                    for name, p in probs.items():
                        common = {
                            "record": night.record,
                            "subject": night.subject,
                            "fold": night.fold,
                            "cohort": conditions.get(night.record, "development"),
                            "profile": profile,
                            "model": name,
                        }
                        records.append(
                            {**common, "windows": score(night.y, p, night.w30)}
                        )
                        columns = {
                            **{k: [v] * len(p) for k, v in common.items()},
                            "onset_s": onsets,
                            "target": night.y,
                            "w30": night.w30,
                            "eligible": np.isfinite(p[:, 0]),
                        }
                        for i, label in enumerate(LABELS):
                            columns["prob_" + label] = (
                                p[:, i]
                                if i < p.shape[1]
                                else np.full(len(p), np.nan, dtype=np.float32)
                            )
                        table = pa.table(columns)
                        if writer is None:
                            writer = pq.ParquetWriter(
                                tmp, table.schema, compression="zstd"
                            )
                        writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()
    write_json(folder / "counts.json", records)
    return records


def reports(folder: Path, records: list, development_run: bool) -> pd.DataFrame:
    groups = defaultdict(list)
    individual: dict[tuple[str, str, str, str], dict[str, np.ndarray]] = defaultdict(
        dict
    )
    for r in records:
        for win, w in r["windows"].items():
            k = (r["cohort"], r["profile"], r["model"], win)
            groups[k].append((r, w))
            mats = individual[k]
            m = np.asarray(w["matrix"])
            mats[r["subject"]] = m + mats.get(r["subject"], np.zeros_like(m))
    rows = []
    matrices = {}
    for key, entries in groups.items():
        frames = np.array([w["matrix"] for _, w in entries])
        total = frames.sum(0)
        row = summarize_counts(total, len(entries))
        row.update(
            dict(zip(("cohort", "profile", "model", "window"), key, strict=True))
        )
        denom = sum(w["reference"] for _, w in entries)
        row["coverage"] = float(total.sum() / max(denom, 1))
        for name in ("transition", "stable"):
            correct, n, miss = np.sum([w[name] for _, w in entries], axis=0)
            row[name + "_accuracy"] = float(correct / max(n, 1))
            row[name + "_missed_n1"] = int(miss)
        if development_run:
            folds = np.array([r["fold"] for r, _ in entries])
            scores = [
                summarize_counts(frames[folds == f].sum(0), int((folds == f).sum()))[
                    "macro_f1"
                ]
                for f in np.unique(folds)
            ]
            row["mean_fold_macro_f1"] = float(np.mean(scores))
            row["fold_sd"] = float(np.std(scores, ddof=1)) if len(scores) > 1 else None
        matrices["/".join(key)] = {
            "classes": entries[0][1]["classes"],
            "counts": total.tolist(),
            "row_normalized": (
                total / np.maximum(total.sum(1, keepdims=True), 1)
            ).tolist(),
        }
        rows.append(row)
    result = pd.DataFrame(rows)
    with atomic_path(folder / "metrics.csv") as tmp:
        result.to_csv(tmp, index=False)
    write_json(folder / "confusion.json", matrices)
    intervals = []
    for cohort in sorted(result.cohort.unique()):
        for profile, first, second in [
            (REFERENCE, "triple", "blend"),
            (ALL, "six_movement", "six_direct"),
        ]:
            key = (cohort, profile, first, "full")
            other = (cohort, profile, second, "full")
            a, b = individual[key], individual[other]
            ids = sorted(a)
            if not ids or set(b) != set(ids):
                continue
            A = np.stack([a[s] for s in ids])
            B = np.stack([b[s] for s in ids])
            draws = np.random.default_rng(42).integers(
                0, len(ids), size=(1000, len(ids))
            )

            def f1(mat: np.ndarray) -> np.ndarray:
                return (
                    2
                    * np.diagonal(mat, axis1=-2, axis2=-1)
                    / np.maximum(mat.sum(-2) + mat.sum(-1), 1)
                )

            diff = f1(A[draws].sum(1)) - f1(B[draws].sum(1))
            obs = f1(A.sum(0)) - f1(B.sum(0))
            for name, dist, delta in [
                ("macro_f1", diff.mean(1), obs.mean()),
                ("n1_f1", diff[:, 1], obs[1]),
            ]:
                intervals.append(
                    dict(
                        cohort=cohort,
                        profile=profile,
                        candidate=first,
                        reference=second,
                        metric=name,
                        delta=float(delta),
                        ci95=np.quantile(dist, [0.025, 0.975]).tolist(),
                        subjects=len(ids),
                    )
                )
    write_json(folder / "paired-subject-bootstrap.json", intervals)
    return result


def fit_development(
    repo: Path, source: Path, output: Path, cfg: Settings, device: torch.device
) -> Path:
    split = repo / "configs/splits/sc_subjects_v1.csv"
    rows = development(source, split, cfg.smoke)
    dirs = prepare(source, output / "cache", rows)
    protocol: dict = {
        "settings": asdict(cfg),
        "split_sha256": hash_file(split),
        "code": protocol_code(),
        "nights": {r: d.name for r, d in dirs.items()},
        "device": str(device),
        "versions": {
            p: version(p)
            for p in ("torch", "lightgbm", "numpy", "scipy", "mne", "pandas", "pyarrow")
        },
        "windows": ["full", "w30"],
        "reference": REFERENCE,
        "classes": LABELS,
    }
    identity = fingerprint(protocol)
    folder = output / ("smoke" if cfg.smoke else "development") / identity
    folder.mkdir(parents=True, exist_ok=True)
    nights = [
        Night.load(r.recording_id, r.subject_id, int(r.fold), dirs[r.recording_id])
        for r in rows.itertuples()
    ]
    print(f"Campaña {folder}, {len(nights)} registros en {device}", flush=True)
    with FileLock(str(folder / "campaign.lock")):
        write_json(folder / "protocol.json", protocol)
        if not (folder / "training-source.zip").is_file():
            archive_code(folder, protocol["code"])
        records = []
        for fold in range(5):
            dest = folder / f"fold-{fold}"
            if (dest / "counts.json").exists() and (
                dest / "predictions.parquet"
            ).exists():
                records.extend(json.loads((dest / "counts.json").read_text()))
                continue
            train = [n for n in nights if n.fold != fold]
            val = [n for n in nights if n.fold == fold]
            assert {n.subject for n in train}.isdisjoint(n.subject for n in val)
            nets = fold_models(dest, train, cfg, device, cfg.seed + fold, identity)
            boosters(dest / "trees", train, val, cfg, cfg.seed + fold)
            records.extend(evaluate_nights(dest, dest / "trees", nets, val, device, {}))
            print(f"Pliegue {fold + 1}/5 completo.", flush=True)
        table = reports(folder, records, True)
        primary = table[
            (table.profile == ALL)
            & (table.window == "full")
            & table.model.isin(["six_direct", "six_movement"])
        ]
        best = (
            primary.sort_values(
                ["mean_fold_macro_f1", "m_f1", "model"], ascending=[False, False, True]
            )
            .iloc[0]
            .model
        )
        refit = folder / "refit"
        fold_models(refit, nights, cfg, device, cfg.seed + 5, identity)
        boosters(refit / "trees", nights, [], cfg, cfg.seed + 5)
        results = {
            p.relative_to(refit).as_posix(): hash_file(p)
            for p in [*refit.glob("*.pt"), *sorted((refit / "trees").glob("*.txt"))]
        }
        write_json(
            folder / "freeze.json",
            {
                "run": identity,
                "smoke": cfg.smoke,
                "edf_model": best + ".pt",
                "selection": "mean full-night six-class fold macro-F1, then movement F1",
                "weights": results,
                "subjects": sorted({n.subject for n in nights}),
                "records": len(nights),
            },
        )
    return folder


def reserved_recordings(
    repo: Path, source: Path, people: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    sc = pd.read_parquet(source / "data/manifests/sleep-cassette/recordings.parquet")
    sc = sc.merge(
        people.loc[people.partition.eq("test")], on="subject_id", validate="many_to_one"
    )
    st = pd.read_parquet(source / "data/manifests/sleep-telemetry/recordings.parquet")
    treatment = pd.read_csv(repo / "configs/splits/st_conditions_v1.csv")
    if set(st.recording_id) != set(treatment.recording_id):
        raise ValueError("Faltan condiciones de Telemetry.")
    st = st.merge(treatment, on="recording_id", validate="one_to_one")
    if (
        set(st.condition) != {"placebo", "temazepam"}
        or not st.groupby("subject_id").condition.nunique().eq(2).all()
    ):
        raise ValueError("Condiciones de Telemetry no emparejadas.")
    records = pd.concat([sc, st], ignore_index=True)
    conditions = {
        **dict.fromkeys(sc.recording_id, "cassette_test"),
        **dict(zip(st.recording_id, st.condition, strict=True)),
    }
    return records, conditions


def result_hashes(folder: Path) -> dict[str, str]:
    files = (
        "predictions.parquet",
        "counts.json",
        "metrics.csv",
        "confusion.json",
        "paired-subject-bootstrap.json",
    )
    return {name: hash_file(folder / name) for name in files}


def evaluate_frozen(
    repo: Path, source: Path, output: Path, folder: Path, device: torch.device
) -> Path:
    freeze = json.loads((folder / "freeze.json").read_text())
    protocol = json.loads((folder / "protocol.json").read_text())
    split = repo / "configs/splits/sc_subjects_v1.csv"
    if (
        freeze["smoke"]
        or fingerprint(protocol) != freeze["run"]
        or protocol["split_sha256"] != hash_file(split)
    ):
        raise ValueError(
            "Se requiere una campaña completa congelada con la misma partición."
        )
    for path, digest in freeze["weights"].items():
        if hash_file(folder / "refit" / path) != digest:
            raise ValueError("Cambió un modelo congelado.")
    people = pd.read_csv(split)
    dev = set(people.loc[people.partition.eq("development"), "subject_id"])
    if dev != set(freeze["subjects"]):
        raise ValueError("Desarrollo congelado incompatible.")
    conditions_hash = hash_file(repo / "configs/splits/st_conditions_v1.csv")
    dest = folder / "reserved"
    dest.mkdir(exist_ok=True)
    with FileLock(str(dest / "evaluation.lock")):
        marker = dest / "completed.json"
        if marker.is_file():
            published = json.loads(marker.read_text())
            if (
                published["run"] != freeze["run"]
                or published["conditions_sha256"] != conditions_hash
            ):
                raise ValueError(
                    "La evaluación guardada difiere de la campaña o las condiciones."
                )
            measured = result_hashes(dest)
            # Earlier releases stored the checksums in a separate file.
            seal = dest / "result-integrity.json"
            expected = published.get("artifacts")
            if expected is None and seal.is_file():
                expected = json.loads(seal.read_text())
            if expected is not None and expected != measured:
                raise ValueError("Cambió un resultado de la evaluación congelada.")
            if expected is None:
                write_json(seal, measured)
            return dest
        if protocol["code"] != protocol_code():
            raise ValueError(
                "Se requiere el código congelado para una evaluación nueva."
            )
        records, conditions = reserved_recordings(repo, source, people)
        dirs = prepare(source, output / "cache", records)
        nights = [
            Night.load(r.recording_id, r.subject_id, -1, dirs[r.recording_id])
            for r in records.itertuples()
        ]
        nets = {
            name: load_model(folder / "refit" / (name + ".pt"), device)[0]
            for name in NETWORKS
        }
        tree_folder = dest / "trees"
        tree_folder.mkdir(exist_ok=True)
        for profile in PROFILES:
            with atomic_path(tree_folder / (profile + ".txt")) as tmp:
                tmp.write_bytes(
                    (folder / "refit/trees" / (profile + ".txt")).read_bytes()
                )
        boosters(tree_folder, [], nights, Settings(**protocol["settings"]), 0)
        counts = evaluate_nights(dest, tree_folder, nets, nights, device, conditions)
        combined = [
            {**r, "cohort": "telemetry"}
            for r in counts
            if r["cohort"] in {"placebo", "temazepam"}
        ]
        reports(dest, counts + combined, False)
        write_json(
            marker,
            {
                "run": freeze["run"],
                "nights": len(nights),
                "cohort_counts": records.groupby("cohort").size().to_dict(),
                "conditions_sha256": conditions_hash,
                "artifacts": result_hashes(dest),
            },
        )
    return dest
