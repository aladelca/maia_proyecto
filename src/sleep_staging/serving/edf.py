"""Versioned, label-free EDF inference shared by local and hosted uploads."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sleep_staging.data.edf import ALIASES, read_psg, validated_edf_channels
from sleep_staging.fullnight.data import CHANNELS, LABELS, PROFILES, Inputs, make_inputs
from sleep_staging.fullnight.model import FullNightNet
from sleep_staging.fullnight.training import (
    REFERENCE,
    encode_night,
    load_model,
    predict_encoded,
)
from sleep_staging.fullnight.trees import tree_features
from sleep_staging.provenance import hash_file
from sleep_staging.sensor_profiles import validate_edf_profile


def validate_edf(path: Path, profile: str) -> str:
    validate_edf_profile(profile)
    infos = validated_edf_channels(path)
    names = [ALIASES[i["name"]] for i in infos if i["name"] in ALIASES]
    if profile == "auto":
        profile = validate_edf_profile("+".join(ch for ch in CHANNELS if ch in names))
    if not set(PROFILES[profile]).issubset(names):
        raise ValueError("El EDF carece de un canal solicitado.")
    if any(names.count(ch) > 1 for ch in PROFILES[profile]):
        raise ValueError("Canales EDF duplicados.")
    return profile


def frozen_component(run: Path, freeze: dict, name: str) -> Path:
    path = run / "refit" / name
    if hash_file(path) != freeze["weights"][name]:
        raise ValueError("Checksum del modelo EDF incorrecto.")
    return path


def frozen_net(
    run: Path, freeze: dict, name: str, device: torch.device
) -> tuple[FullNightNet, dict]:
    net, meta = load_model(frozen_component(run, freeze, name), device)
    if meta["run_id"] != freeze["run"]:
        raise ValueError("Identidad de entrenamiento incompatible.")
    return net, meta


def frozen_model(run: Path, device: torch.device) -> tuple[FullNightNet, dict]:
    freeze = json.loads((run / "freeze.json").read_text())
    name = freeze["edf_model"]
    if name not in {"six_direct.pt", "six_movement.pt"}:
        raise ValueError("Cabezal de movimiento incompatible.")
    return frozen_net(run, freeze, name, device)


def selected_inputs(path: Path, profile: str) -> tuple[Inputs, str]:
    selection = validate_edf(path, profile)
    signals, _, duration = read_psg(path, channels=PROFILES[selection])
    return make_inputs(signals, duration), selection


def predict_edf(
    path: Path, net: FullNightNet, profile: str, record_id: str, device: torch.device
) -> tuple[pd.DataFrame, str]:
    x, selection = selected_inputs(path, profile)
    enc = encode_night(net, x, device)
    prob = predict_encoded(net, enc, x, selection)
    return epoch_frame(prob, record_id), selection


def tree_probabilities(checkpoint: Path, features: np.ndarray) -> np.ndarray:
    # Keep the native LightGBM and PyTorch runtimes in separate processes, as in training.
    with tempfile.TemporaryDirectory() as folder:
        source, output = Path(folder) / "features.npy", Path(folder) / "prob.npy"
        np.save(source, features)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "sleep_staging.fullnight.trees",
                "--features",
                str(checkpoint),
                str(source),
                str(output),
            ],
            check=True,
        )
        return np.load(output)


def predict_edf_five(
    path: Path, run: Path, profile: str, record_id: str, device: torch.device
) -> tuple[pd.DataFrame, str, str]:
    x, selected = selected_inputs(path, profile)
    freeze = json.loads((run / "freeze.json").read_text())
    tree = frozen_component(run, freeze, f"trees/{selected}.txt")
    probs = 0.5 * tree_probabilities(tree, tree_features(x, selected))
    spectral = selected == REFERENCE
    branches = [("wave", 0.25), ("spectrogram", 0.25)] if spectral else [("wave", 0.5)]
    for name, weight in branches:
        net, _ = frozen_net(run, freeze, name + ".pt", device)
        probs += weight * predict_encoded(
            net, encode_night(net, x, device), x, selected
        )
    return epoch_frame(probs, record_id), selected, "triple" if spectral else "blend"


def epoch_frame(prob: np.ndarray, record_id: str) -> pd.DataFrame:
    eligible = np.isfinite(prob).all(1)
    labels = np.asarray(LABELS[: prob.shape[1]])
    pred = np.full(len(prob), None, dtype=object)
    pred[eligible] = labels[prob[eligible].argmax(1)]
    onset = np.arange(len(prob)) * 30
    frame = pd.DataFrame(
        {
            "recording_id": record_id,
            "epoch_id": [f"{record_id}:{t * 100}" for t in onset],
            "onset_s": onset,
            "eligible": eligible,
            "prediction": pred,
        }
    )
    for i, name in enumerate(labels):
        frame["prob_" + name] = prob[:, i]
    return frame


def edf_summary(frame: pd.DataFrame, profile: str, meta: dict) -> dict:
    return {
        "profile": profile,
        "classes": meta["classes"],
        "rows": len(frame),
        "eligible": int(frame.eligible.sum()),
        "model_version": f"{meta['run_id']}/{meta['head']}",
        "preprocessing_id": meta["schema"],
    }
