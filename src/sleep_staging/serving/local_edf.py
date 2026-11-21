"""Loopback inference for the same EDF contract as the cloud worker."""

import json
import tempfile
from pathlib import Path

import torch
from flask import Flask, Response, jsonify, request, send_from_directory
from werkzeug.exceptions import RequestEntityTooLarge

from sleep_staging.fullnight.model import FullNightNet
from sleep_staging.sensor_profiles import MAX_EDF_BYTES, validate_edf_profile
from sleep_staging.serving.edf import edf_summary, predict_edf


def create_app(
    net: FullNightNet, meta: dict, device: torch.device, frontend: Path
) -> Flask:
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = MAX_EDF_BYTES + 1024 * 1024

    @app.get("/")
    def index() -> Response:
        return send_from_directory(frontend, "index.html")

    @app.get("/assets/<path:asset>")
    def assets(asset: str) -> Response:
        return send_from_directory(frontend / "assets", asset)

    @app.get("/config.json")
    def config() -> Response:
        return jsonify(mode="local")

    @app.post("/v1/edf")
    def infer() -> Response | tuple[Response, int]:
        incoming = request.files.get("file")
        if incoming is None:
            return jsonify(error="Selecciona un archivo EDF."), 400
        try:
            profile = validate_edf_profile(request.form.get("profile", "auto"))
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "input.edf"
                incoming.save(path)
                frame, used = predict_edf(path, net, profile, "night", device)
        except (ValueError, EOFError, OverflowError, OSError):
            return jsonify(
                error="No se pudo leer el EDF. Comprueba el archivo y los canales seleccionados."
            ), 400
        return jsonify(
            **edf_summary(frame, used, meta),
            epochs=json.loads(frame.to_json(orient="records", double_precision=15)),
        )

    @app.errorhandler(413)
    def oversized(_: RequestEntityTooLarge) -> tuple[Response, int]:
        return jsonify(error="El EDF supera 128 MiB."), 413

    return app
