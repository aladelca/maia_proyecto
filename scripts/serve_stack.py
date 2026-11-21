"""Run the published stack and its web interface on loopback with real EDF files."""

import argparse
import json
import tempfile
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory

from sleep_staging.sensor_profiles import MAX_EDF_BYTES
from sleep_staging.serving.stacked_edf import EXAMPLE_PATH, score, summary


def create_app(frontend: Path) -> Flask:
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
        return jsonify(mode="local-stack")

    def result(path: Path) -> Response:
        frame = score(path, "night")
        return jsonify(
            **summary(frame),
            epochs=json.loads(frame.to_json(orient="records", double_precision=15)),
        )

    @app.post("/v1/edf/example")
    def example() -> Response:
        return result(EXAMPLE_PATH)

    @app.post("/v1/edf")
    def upload() -> tuple[Response, int] | Response:
        incoming = request.files.get("file")
        if incoming is None:
            return jsonify(error="Selecciona una polisomnografía EDF."), 400
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "input.edf"
            incoming.save(path)
            if path.stat().st_size > MAX_EDF_BYTES:
                return jsonify(error="El EDF supera 128 MiB."), 413
            try:
                return result(path)
            except (ValueError, EOFError, OverflowError):
                return jsonify(
                    error="Comprueba el EDF y los cuatro canales requeridos."
                ), 400

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8775)
    args = parser.parse_args()
    directory = Path(__file__).resolve().parents[1] / "frontend/dist"
    if not (directory / "index.html").is_file():
        parser.error("Compila la web con npm --prefix frontend run build.")
    create_app(directory).run(host="127.0.0.1", port=args.port, threaded=False)
