import sqlite3

import pytest
from filelock import FileLock, Timeout

from sleep_staging.backup import backup_mlflow, verify_backup


def test_backup_validates_artifacts_and_refuses_running_service(tmp_path):
    directory = tmp_path / "var/mlflow"
    directory.mkdir(parents=True)
    with sqlite3.connect(directory / "mlflow.db") as db:
        db.execute("create table sample (id integer)")
        db.execute("insert into sample values (1)")
    artifact = directory / "artifacts/example.txt"
    artifact.parent.mkdir()
    artifact.write_text("model artifact")
    with FileLock(str(directory / "service.lock")):
        with pytest.raises(Timeout):
            backup_mlflow(tmp_path)
    path = backup_mlflow(tmp_path)
    verify_backup(path)
    (path / "artifacts/example.txt").write_text("corrupted")
    with pytest.raises(ValueError, match="integridad"):
        verify_backup(path)
