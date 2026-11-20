"""Serve a frozen IBIO release locally or predict a single unannotated EDF."""

import argparse
from pathlib import Path

import torch

from sleep_staging.models.tcn import research_device
from sleep_staging.provenance import atomic_path
from sleep_staging.serving.edf import frozen_model, predict_edf, predict_edf_five


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run",
        type=Path,
        required=True,
        help="Campaña que contiene freeze.json y refit/.",
    )
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda", "auto"])
    p.add_argument("--edf", type=Path)
    p.add_argument("--profile", default="auto")
    p.add_argument(
        "--task",
        choices=["five", "six"],
        default="six",
        help="Para un EDF: cinco estadios en mezcla o seis salidas con movimiento.",
    )
    p.add_argument("--output", type=Path, help="CSV de predicciones para --edf.")
    p.add_argument("--port", type=int, default=8765)
    a = p.parse_args()
    if bool(a.edf) != bool(a.output):
        p.error("--edf requiere --output y viceversa.")
    if a.task == "five" and not a.edf:
        p.error("--task five requiere el archivo --edf y su --output.")
    torch.set_num_threads(2)
    device = research_device(a.device)
    if a.edf:
        if a.task == "five":
            result, profile, method = predict_edf_five(
                a.edf, a.run, a.profile, "night", device
            )
        else:
            model, meta = frozen_model(a.run, device)
            result, profile = predict_edf(a.edf, model, a.profile, "night", device)
            method = meta["head"]
        with atomic_path(a.output) as tmp:
            result.to_csv(tmp, index=False)
        print(f"{len(result)} épocas, perfil {profile}, modelo {method}: {a.output}")
        return
    from sleep_staging.serving.local_edf import create_app

    frontend = Path(__file__).resolve().parents[1] / "frontend/dist"
    if not (frontend / "index.html").is_file():
        p.error("Compila la interfaz con npm --prefix frontend run build.")
    model, meta = frozen_model(a.run, device)
    create_app(model, meta, device, frontend).run(
        host="127.0.0.1", port=a.port, threaded=False
    )


if __name__ == "__main__":
    main()
