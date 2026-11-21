"""Whole-recording IBIO comparisons on MPS, CUDA or CPU."""

import argparse
import os
from pathlib import Path

import torch

from sleep_staging.fullnight.cohort import research_root
from sleep_staging.fullnight.experiment import evaluate_frozen, fit_development
from sleep_staging.fullnight.training import Settings
from sleep_staging.models.tcn import research_device

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    p.add_argument(
        "--source-root", type=Path, default=os.getenv("SLEEP_RESEARCH_DATA_ROOT")
    )
    p.add_argument("--output", type=Path, default=ROOT / "var/research/ibio-fullnight")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Prueba técnica de 10 sujetos de desarrollo.",
    )
    p.add_argument(
        "--evaluate-frozen",
        type=Path,
        help="Directorio de una campaña completa ya congelada.",
    )
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--trees", type=int, default=64)
    p.add_argument("--width", type=int, default=24)
    p.add_argument("--chunk", type=int, default=192)
    p.add_argument(
        "--chunks-per-night",
        type=int,
        default=0,
        help="0 usa toda la señal para entrenar. La evaluación siempre usa el registro completo.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--training-window", choices=["full", "w30"], default="full")
    a = p.parse_args()
    if a.evaluate_frozen and a.smoke:
        p.error("No se puede evaluar test con --smoke.")
    device = research_device(a.device)
    torch.set_num_threads(2)
    source = research_root(ROOT, a.source_root)
    cfg = Settings(
        epochs=1 if a.smoke else a.epochs,
        trees=3 if a.smoke else a.trees,
        width=8 if a.smoke else a.width,
        chunk=64 if a.smoke else a.chunk,
        chunks_per_night=3 if a.smoke else a.chunks_per_night,
        seed=a.seed,
        training_window=a.training_window,
        smoke=a.smoke,
    )
    report = (
        evaluate_frozen(ROOT, source, a.output, a.evaluate_frozen, device)
        if a.evaluate_frozen
        else fit_development(ROOT, source, a.output, cfg, device)
    )
    print(report, flush=True)


if __name__ == "__main__":
    main()
