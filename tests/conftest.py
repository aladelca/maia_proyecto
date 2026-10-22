import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests


@pytest.fixture(scope="session")
def tracking_server(tmp_path_factory):
    directory = tmp_path_factory.mktemp("tracking")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    uri = f"http://127.0.0.1:{port}"
    log = (directory / "server.log").open("w")
    args = [
        sys.executable,
        "-m",
        "mlflow",
        "server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--workers",
        "1",
        "--backend-store-uri",
        f"sqlite:///{directory}/mlflow.db",
        "--serve-artifacts",
        "--artifacts-destination",
        str(directory / "artifacts"),
    ]
    process = subprocess.Popen(
        args, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )
    try:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if process.poll() is not None:
                pytest.fail((directory / "server.log").read_text())
            try:
                if requests.get(uri + "/health", timeout=1).ok:
                    break
            except requests.RequestException:
                pass
            time.sleep(0.2)
        else:
            pytest.fail("MLflow no arrancó: " + (directory / "server.log").read_text())
        yield uri
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=15)
        log.close()


@pytest.fixture(scope="session")
def real_dataset(tmp_path_factory):
    import pandas as pd

    from sleep_staging.config import data_config, project_root
    from sleep_staging.provenance import hash_file

    project = project_root()
    data = data_config(project / "configs/data/pilot.yaml")
    manifest = project / "data/manifests" / data.name
    if not (manifest / "recordings.parquet").exists():
        pytest.skip(
            "Descarga explícita necesaria: sleep-stage download --config configs/data/pilot.yaml"
        )
    root = tmp_path_factory.mktemp("real-data")
    target = root / "data/manifests" / data.name
    target.mkdir(parents=True)
    for name in ["recordings", "files"]:
        frame = pd.read_parquet(manifest / f"{name}.parquet")
        columns = ["psg_file", "hyp_file"] if name == "recordings" else ["path"]
        for column in columns:
            frame[column] = frame[column].map(lambda value: str(project / value))
        if name == "files":
            assert all(
                hash_file(Path(row.path)) == row.sha256 for row in frame.itertuples()
            )
        frame.to_parquet(target / f"{name}.parquet", index=False)
    return data, root
