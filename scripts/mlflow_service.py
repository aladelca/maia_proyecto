"""Servicio en primer plano con bloqueo compartido con el respaldo."""

import os
import signal
import subprocess
import sys
from types import FrameType
from urllib.parse import urlparse

from filelock import FileLock

from sleep_staging.config import local_config, project_root


def main() -> None:
    root = project_root()
    uri = urlparse(local_config().tracking_uri)
    if uri.scheme != "http" or uri.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("MLflow debe escuchar únicamente en localhost.")
    directory = root / "var/mlflow"
    (directory / "artifacts").mkdir(parents=True, exist_ok=True)
    args = [
        sys.executable,
        "-m",
        "mlflow",
        "server",
        "--host",
        "127.0.0.1",
        "--port",
        str(uri.port or 5001),
        "--workers",
        "1",
        "--backend-store-uri",
        f"sqlite:///{directory}/mlflow.db",
        "--serve-artifacts",
        "--artifacts-destination",
        str(directory / "artifacts"),
    ]

    def interrupted(signum: int, frame: FrameType | None) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    with FileLock(str(directory / "service.lock"), timeout=0):
        process = subprocess.Popen(args, cwd=root, start_new_session=True)
        try:
            raise SystemExit(process.wait())
        except KeyboardInterrupt:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


if __name__ == "__main__":
    main()
