import os
from urllib.parse import urlparse

import mlflow
import requests


def require_tracking(uri: str) -> None:
    parsed = urlparse(uri)
    if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1"}:
        raise ValueError("Este entorno requiere MLflow HTTP en localhost.")
    try:
        response = requests.get(uri.rstrip("/") + "/health", timeout=3)
        response.raise_for_status()
    except requests.RequestException as error:
        raise ValueError(
            f"MLflow no está disponible en {uri}; ejecuta make mlflow."
        ) from error
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "30")
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)


def init_tracking(uri: str, experiment: str) -> None:
    require_tracking(uri)
    mlflow.set_experiment(experiment)
