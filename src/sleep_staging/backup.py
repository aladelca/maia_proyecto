import json
import shutil
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from filelock import FileLock

from sleep_staging.provenance import hash_file, write_json


def backup_mlflow(root: Path) -> Path:
    source = root / "var/mlflow"
    with (
        FileLock(str(source / "service.lock"), timeout=0),
        FileLock(str(root / "var/experiment.lock"), timeout=0),
    ):
        if not (source / "mlflow.db").exists():
            raise ValueError("No existe la base de MLflow.")
        with sqlite3.connect(source / "mlflow.db") as db:
            if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("Falló la integridad de SQLite.")
        target = root / "var/backups" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        target.mkdir(parents=True)
        with (
            sqlite3.connect(source / "mlflow.db") as origin,
            sqlite3.connect(target / "mlflow.db") as destination,
        ):
            origin.backup(destination)
        if (source / "artifacts").exists():
            shutil.copytree(source / "artifacts", target / "artifacts")
        hashes = {
            str(p.relative_to(target)): hash_file(p)
            for p in target.rglob("*")
            if p.is_file()
        }
        write_json(target / "manifest.json", {"schema_version": 1, "files": hashes})
        verify_backup(target)
        return target


def verify_backup(path: Path) -> None:
    manifest = json.loads((path / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        file = (path / name).resolve()
        if (
            not file.is_relative_to(path.resolve())
            or not file.is_file()
            or hash_file(file) != expected
        ):
            raise ValueError(f"Falló integridad del respaldo: {name}")
    with sqlite3.connect(f"file:{path / 'mlflow.db'}?mode=ro", uri=True) as db:
        if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("Falló integridad de SQLite del respaldo.")
