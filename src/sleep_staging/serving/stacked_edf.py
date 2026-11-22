"""Score the frozen, two-branch Sleep Cassette model on an unannotated EDF."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sleep_staging.provenance import hash_file
from sleep_staging.sensor_profiles import LGBM_PROFILE
from sleep_staging.serving.contract import KEYS, LABELS, context_matrix
from sleep_staging.stack_release import RUN_ID, VERSION

MODEL_DIR = Path(
    os.environ.get(
        "STACK_MODEL_DIR", Path(__file__).resolve().parents[3] / "modelos/stack_five"
    )
)
EXAMPLE_PATH = Path(
    os.environ.get(
        "STACK_EXAMPLE_EDF",
        Path(__file__).resolve().parents[3] / "output/demo-sueno-sintetico-10min.edf",
    )
)
CHECKPOINTS = {
    "stack_five.joblib": "4a7021f0b6d147a1c64b86204489230aac1a469982d15780ca1189b4e431fb02",
    "six_direct.pt": "4e5171ee040d42bfd389964150241d7ba618a73aec50573251ada67c76addfa3",
}


def checkpoint(name: str) -> Path:
    path = MODEL_DIR / name
    if hash_file(path) != CHECKPOINTS[name]:
        raise RuntimeError("Falló la verificación del modelo publicado.")
    return path


@lru_cache(maxsize=1)
def classifier() -> Any:
    import joblib

    return joblib.load(checkpoint("stack_five.joblib"))


def neural(path: Path, output: Path) -> None:
    import torch

    from sleep_staging.fullnight.training import load_model
    from sleep_staging.serving.edf import predict_edf

    torch.set_num_threads(2)
    net, meta = load_model(checkpoint("six_direct.pt"), torch.device("cpu"))
    if meta["head"] != "six" or meta["classes"] != [*LABELS, "M"]:
        raise RuntimeError("Clases neuronales incompatibles.")
    frame, _ = predict_edf(path, net, LGBM_PROFILE, "night", torch.device("cpu"))
    # Keep the float32 neural predictions used when training the second level.
    np.savez(
        output,
        onset=frame.onset_s.to_numpy(),
        stages=frame[["prob_" + s for s in LABELS]].to_numpy(dtype=np.float32),
    )


def score(path: Path, recording_id: str) -> pd.DataFrame:
    from sleep_staging.serving.edf_lgbm import prepare_edf

    raw = prepare_edf(path, recording_id)
    result = raw[KEYS].copy()
    result["prediction"] = pd.Series(None, index=result.index, dtype=object)
    prob_columns = ["prob_" + s for s in LABELS]
    result[prob_columns] = np.nan
    valid = raw[raw.eligible]
    if valid.empty:
        return result
    model = classifier()
    columns = list(model.feature_names_in_)
    x = context_matrix(valid, columns[:145])
    # Torch and LightGBM have incompatible OpenMP runtimes on some operators' Macs.
    # Release neural-process memory before applying the tabular classifier.
    with tempfile.TemporaryDirectory() as folder:
        output = Path(folder) / "probabilities.npz"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "sleep_staging.serving.stacked_edf",
                str(path),
                str(output),
                "--neural",
            ],
            check=True,
        )
        with np.load(output, allow_pickle=False) as data:
            if not np.array_equal(data["onset"], raw.onset_s):
                raise RuntimeError("Desalineación entre las dos ramas del modelo.")
            stages = data["stages"][raw.eligible.to_numpy()]
    mass = stages.sum(axis=1, keepdims=True)
    if not np.isfinite(stages).all() or (mass <= 0).any():
        raise ValueError(
            "El EDF contiene épocas sin señal compatible en la rama neuronal."
        )
    for i, label in enumerate(LABELS):
        x["fullnight_" + label] = stages[:, i] / mass[:, 0]
    if list(x) != columns or set(model.classes_) != set(LABELS):
        raise RuntimeError("Esquema del clasificador híbrido incompatible.")
    p = model.predict_proba(x)[:, [list(model.classes_).index(s) for s in LABELS]]
    if not np.isfinite(p).all() or not np.allclose(p.sum(1), 1, atol=1e-6):
        raise RuntimeError("Probabilidades finales inválidas.")
    result.loc[raw.eligible, prob_columns] = p
    result.loc[raw.eligible, "prediction"] = np.asarray(LABELS)[p.argmax(1)]
    return result


def summary(frame: pd.DataFrame) -> dict:
    return {
        "model_version": VERSION,
        "mlflow_run_id": RUN_ID,
        "classes": LABELS,
        "preprocessing_id": "full-edf-context435-fullnight5-v1",
        "profile": LGBM_PROFILE,
        "window": "full",
        "rows": len(frame),
        "eligible": int(frame.eligible.sum()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("edf", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--neural", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.neural:
        neural(args.edf, args.output)
    else:
        result = score(args.edf, "night")
        result.to_csv(args.output, index=False)
        print(json.dumps(summary(result)))
