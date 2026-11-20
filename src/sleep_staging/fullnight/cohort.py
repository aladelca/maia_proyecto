"""Versioned per-night caches and frozen participant partitions."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from filelock import FileLock

from sleep_staging.data.download import download_file
from sleep_staging.data.edf import read_hypnogram, read_psg
from sleep_staging.fullnight import data
from sleep_staging.fullnight.data import make_inputs, targets
from sleep_staging.provenance import fingerprint, hash_file, write_json
from sleep_staging.research_data import development_recordings


def research_root(repo: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    if (repo / "data/manifests/sleep-cassette").is_dir():
        return repo
    prior = repo / "var/research/sc-w30-v2/work-state.json"
    if prior.is_file():
        return Path(json.loads(prior.read_text())["source_root"])
    raise ValueError("Indica --source-root con los manifiestos de Sleep-EDFx.")


def development(source: Path, split: Path, smoke: bool) -> pd.DataFrame:
    rows, _ = development_recordings(source, split)
    if smoke:
        rows = (
            rows.sort_values("recording_id")
            .drop_duplicates("subject_id")
            .groupby("fold")
            .head(2)
        )
    return rows.sort_values("recording_id").reset_index(drop=True)


def source_file(source: Path, cache: Path, path: str, row: pd.Series) -> Path:
    local = source / path
    if local.is_file():
        st = local.stat()
        if (
            not getattr(st, "st_flags", 0) & 0x40000000
            and st.st_size == row["bytes"]
            and hash_file(local) == row.sha256
        ):
            return local
    dest = cache / "sources" / row.sha256 / local.name
    download_file(row.url, dest, row.sha256)
    return dest


def prepare(source: Path, cache: Path, records: pd.DataFrame) -> dict[str, Path]:
    from sleep_staging.data import edf, labels

    manifest = pd.read_parquet(
        source / "data/manifests/sleep-edfx-complete/files.parquet"
    ).set_index("path", verify_integrity=True)
    files = [m.__file__ for m in (data, edf, labels)]
    if not all(files):
        raise RuntimeError("No se encontró el código de preprocesamiento.")
    version = fingerprint([hash_file(Path(p)) for p in files if p is not None])
    directories = {}
    for rec in records.itertuples():
        raw_hashes = {
            p: str(manifest.loc[p, "sha256"]) for p in (rec.psg_file, rec.hyp_file)
        }
        key = fingerprint([version, raw_hashes])
        parent = cache / "nights" / rec.recording_id
        parent.mkdir(parents=True, exist_ok=True)
        dest = parent / key
        with FileLock(str(dest) + ".lock"):
            if not (dest / "metadata.json").is_file():
                psg = source_file(
                    source, cache, rec.psg_file, manifest.loc[rec.psg_file]
                )
                hyp = source_file(
                    source, cache, rec.hyp_file, manifest.loc[rec.hyp_file]
                )
                signals, start, duration = read_psg(psg)
                inp = make_inputs(signals, duration)
                y, crop = targets(read_hypnogram(hyp, start), duration)
                with tempfile.TemporaryDirectory(dir=parent) as tmp:
                    stage = Path(tmp) / "night"
                    stage.mkdir()
                    inp.save(stage)
                    np.save(stage / "labels.npy", y)
                    np.save(stage / "w30.npy", crop)
                    write_json(
                        stage / "metadata.json",
                        {
                            "id": rec.recording_id,
                            "subject": rec.subject_id,
                            "inputs": raw_hashes,
                            "preprocessing": version,
                            "epochs": len(y),
                            "seconds": duration,
                            "channels": {
                                k: {
                                    "name": v.original_name,
                                    "rate": v.sfreq,
                                    "kind": v.kind,
                                }
                                for k, v in signals.items()
                            },
                        },
                    )
                    stage.rename(dest)
                print(f"Preparado {rec.recording_id} ({len(y)} épocas)", flush=True)
        directories[rec.recording_id] = dest
    return directories
