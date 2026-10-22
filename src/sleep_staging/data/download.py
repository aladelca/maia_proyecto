"""Descarga atómica con SHA-256 publicado por PhysioNet."""

import re
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd
import requests
from filelock import FileLock
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from sleep_staging.config import DataConfig
from sleep_staging.provenance import atomic_path, hash_file

BASE_URL = "https://physionet-open.s3.amazonaws.com/sleep-edfx/1.0.0/"
CHECKSUM_URL = "https://physionet.org/files/sleep-edfx/1.0.0/SHA256SUMS.txt"


def http_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def selected_files(entries: dict[str, str], config: DataConfig) -> list[dict[str, Any]]:
    names = [
        name
        for name in sorted(entries)
        if name.startswith(config.source + "/") and name.endswith("-PSG.edf")
    ]
    pairs = []
    for name in names:
        recording = PurePosixPath(name).name[:6]
        if config.records and recording not in config.records:
            continue
        hyp = [
            item
            for item in entries
            if item.startswith(config.source + "/" + recording)
            and item.endswith("-Hypnogram.edf")
        ]
        if len(hyp) != 1:
            raise ValueError(f"Se esperaba un hipnograma único para {recording}.")
        pairs.append(
            {
                "recording_id": recording,
                "subject_id": config.source + ":" + recording[3:5],
                "night": recording[5],
                "cohort": config.source,
                "psg_file": name,
                "hyp_file": hyp[0],
            }
        )
    missing = set(config.records) - {row["recording_id"] for row in pairs}
    if missing or not pairs:
        raise ValueError(f"Registros no encontrados: {sorted(missing)}")
    return pairs


def download_file(url: str, path: Path, expected_sha256: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock"):
        if path.exists():
            if hash_file(path) != expected_sha256:
                raise ValueError(
                    f"checksum incorrecto: {path}. Mueve el archivo corrupto antes de repetir."
                )
            return
        # Restart partial transfers rather than append potentially incompatible ranges.
        with atomic_path(path) as temp:
            with (
                http_session() as session,
                session.get(url, stream=True, timeout=(15, 120)) as response,
            ):
                response.raise_for_status()
                with temp.open("wb") as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        stream.write(chunk)
            if hash_file(temp) != expected_sha256:
                raise ValueError(f"checksum incorrecto en descarga: {url}")


def download_dataset(config: DataConfig, root: Path) -> Path:
    base = root / "data/raw/sleep-edfx/1.0.0"
    checksum_path = base / "SHA256SUMS.txt"
    if not checksum_path.exists():
        with http_session() as session:
            response = session.get(CHECKSUM_URL, timeout=(15, 120))
            response.raise_for_status()
        with atomic_path(checksum_path) as temp:
            temp.write_text(response.text)
    entries = {}
    for line in checksum_path.read_text().splitlines():
        checksum, name = line.split(maxsplit=1)
        relative = PurePosixPath(name)
        if (
            not re.fullmatch(r"[0-9a-f]{64}", checksum)
            or relative.is_absolute()
            or ".." in relative.parts
        ):
            raise ValueError("Manifiesto de descarga inválido.")
        entries[name] = checksum
    pairs = selected_files(entries, config)
    files = []
    for row in pairs:
        for key in ["psg_file", "hyp_file"]:
            name = row[key]
            path = base / name
            print(f"Verificando/descargando {name}", flush=True)
            download_file(BASE_URL + name, path, entries[name])
            files.append(
                {
                    "url": BASE_URL + name,
                    "path": str(path.relative_to(root)),
                    "sha256": entries[name],
                    "bytes": path.stat().st_size,
                    "status": "verified",
                    "dataset_version": "1.0.0",
                }
            )
            row[key] = str(path.relative_to(root))
    directory = root / "data/manifests" / config.name
    with atomic_path(directory / "files.parquet") as temp:
        pd.DataFrame(files).to_parquet(temp, index=False)
    with atomic_path(directory / "recordings.parquet") as temp:
        pd.DataFrame(pairs).to_parquet(temp, index=False)
    return directory
