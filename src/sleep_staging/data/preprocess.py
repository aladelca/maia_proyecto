import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from filelock import FileLock
from scipy.signal import butter, sosfiltfilt

from sleep_staging.config import DataConfig, ExperimentConfig
from sleep_staging.data.edf import read_recording
from sleep_staging.data.labels import epoch_annotations
from sleep_staging.features.extract import signal_features
from sleep_staging.provenance import (
    atomic_path,
    code_fingerprint,
    environment,
    fingerprint,
    hash_file,
    write_json,
)


def preprocessing_parameters(cfg: ExperimentConfig) -> dict:
    return {
        key: getattr(cfg, key)
        for key in [
            "eeg_low",
            "eeg_high",
            "max_epochs_per_recording",
            "wake_margin_minutes",
        ]
    }


def prepare_dataset(data: DataConfig, cfg: ExperimentConfig, root: Path) -> Path:
    manifest = root / "data/manifests" / data.name
    if not (manifest / "recordings.parquet").exists():
        raise ValueError("Falta catálogo: ejecuta download primero.")
    files = pd.read_parquet(manifest / "files.parquet")
    for item in files.to_dict("records"):
        if hash_file(root / item["path"]) != item["sha256"]:
            raise ValueError(f"Checksum cambiado: {item['path']}")
    records = pd.read_parquet(manifest / "recordings.parquet")
    inputs = {
        "schema_version": 1,
        "data": data.model_dump(),
        "files": files.to_dict("records"),
        "preprocessing": preprocessing_parameters(cfg),
        "source": code_fingerprint(),
        "environment": environment(),
    }
    key = fingerprint(inputs)
    destination = root / "data/interim" / key
    destination.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(destination) + ".lock"):
        if (destination / "metadata.json").exists():
            return destination
        start = time.monotonic()
        all_epochs, channels = [], []
        for record in records.to_dict("records"):
            print(f"Preparando {record['recording_id']}", flush=True)
            signals, annotations, duration = read_recording(
                root / record["psg_file"], root / record["hyp_file"]
            )
            epochs = epoch_annotations(annotations, duration)
            epochs["recording_id"] = record["recording_id"]
            epochs["subject_id"] = record["subject_id"]
            epochs["cohort"] = record["cohort"]
            epochs["epoch_id"] = (
                epochs.recording_id + ":" + epochs.onset_sample_eeg.astype(str)
            )
            if cfg.wake_margin_minutes is not None:
                sleep = epochs[epochs.task_5.isin(["N1", "N2", "N3", "REM"])]
                if sleep.empty:
                    raise ValueError(
                        "No se puede recortar vigilia sin épocas de sueño."
                    )
                lower = sleep.onset_s.min() - cfg.wake_margin_minutes * 60
                upper = sleep.onset_s.max() + cfg.wake_margin_minutes * 60
                outside = ~epochs.onset_s.between(lower, upper)
                epochs.loc[outside & epochs.reason.eq(""), "reason"] = "wake_crop"
            if (
                cfg.max_epochs_per_recording
                and len(epochs) > cfg.max_epochs_per_recording
            ):
                chosen = np.linspace(
                    0, len(epochs) - 1, cfg.max_epochs_per_recording, dtype=int
                )
                epochs.loc[
                    ~epochs.index.isin(chosen) & epochs.reason.eq(""), "reason"
                ] = "pilot_subsample"
            # Keep the audit table for every interval; arrays cover selected complete epochs.
            eligible = epochs[epochs.reason.eq("")].copy()
            for canonical, signal in signals.items():
                expected_samples = int(30 * signal.sfreq)
                channel_dir = destination / record["recording_id"]
                signal_values = np.asarray(signal.values, dtype="float64")
                if not np.isfinite(signal_values).all():
                    raise ValueError(f"NaN/Inf en {record['recording_id']}:{canonical}")
                if signal.kind in {"eeg", "eog", "emg_raw"}:
                    low, high = (
                        (cfg.eeg_low, cfg.eeg_high)
                        if signal.kind == "eeg"
                        else (0.3, 15)
                        if signal.kind == "eog"
                        else (10, 45)
                    )
                    if high >= signal.sfreq / 2:
                        raise ValueError(
                            f"Filtro incompatible con Nyquist: {canonical}"
                        )
                    signal_values = sosfiltfilt(
                        butter(
                            4,
                            [low, high],
                            fs=signal.sfreq,
                            btype="bandpass",
                            output="sos",
                        ),
                        signal_values,
                    )
                arrays = []
                for row in eligible.itertuples():
                    onset = int(round(row.onset_s * signal.sfreq))
                    chunk = signal_values[onset : onset + expected_samples]
                    if len(chunk) != expected_samples:
                        raise ValueError(
                            f"Duración de canal incompatible: {record['recording_id']}:{canonical}"
                        )
                    arrays.append(chunk.astype("float32"))
                array = (
                    np.stack(arrays)
                    if arrays
                    else np.empty((0, expected_samples), dtype="float32")
                )
                path = channel_dir / f"{canonical}.npy"
                with atomic_path(path) as temp, temp.open("wb") as stream:
                    np.save(stream, array)
                epochs.loc[eligible.index, "array_row"] = np.arange(len(eligible))
                epochs[f"has_{canonical}"] = True
                # Flags are reported per channel and only required channels affect eligibility.
                flags = (
                    np.ptp(array, axis=1) <= 1e-10
                    if len(array)
                    else np.array([], dtype=bool)
                )
                epochs.loc[eligible.index, f"flat_{canonical}"] = flags
                channels.append(
                    {
                        "recording_id": record["recording_id"],
                        "canonical": canonical,
                        "original_name": signal.original_name,
                        "unit": signal.unit,
                        "sfreq": signal.sfreq,
                        "kind": signal.kind,
                        "path": str(path.relative_to(root)),
                    }
                )
            all_epochs.append(epochs)
        combined = pd.concat(all_epochs, ignore_index=True)
        for channel in ["eeg_fpz_cz", "eeg_pz_oz", "eog", "emg"]:
            for prefix in ["has_", "flat_"]:
                column = prefix + channel
                combined[column] = (
                    combined[column].eq(True) if column in combined else False
                )
        for filename, frame in [
            ("epochs", combined),
            ("channels", pd.DataFrame(channels)),
        ]:
            with atomic_path(destination / f"{filename}.parquet") as temp:
                frame.to_parquet(temp, index=False)
        write_json(
            destination / "metadata.json",
            {
                **inputs,
                "preprocessing_id": key,
                "elapsed_seconds": time.monotonic() - start,
                "rss_bytes": psutil.Process().memory_info().rss,
                "exclusions": combined.reason.value_counts().to_dict(),
            },
        )
    return destination


def extract_dataset_features(prepared: Path, cfg: ExperimentConfig, root: Path) -> Path:
    feature_key = fingerprint(
        {"preprocessing": prepared.name, "source": code_fingerprint(), "features": "v1"}
    )
    destination = (
        root / "data/processed" / prepared.name / feature_key / "features.parquet"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(destination) + ".lock"):
        if destination.exists():
            return destination
        epochs = pd.read_parquet(prepared / "epochs.parquet")
        channels = pd.read_parquet(prepared / "channels.parquet")
        rows = []
        for recording, group in epochs[epochs.reason.eq("")].groupby(
            "recording_id", sort=True
        ):
            sources = {
                item.canonical: (np.load(root / item.path, mmap_mode="r"), item)
                for item in channels[channels.recording_id.eq(recording)].itertuples()
            }
            for epoch in group.itertuples():
                features = {"epoch_id": epoch.epoch_id}
                for canonical, (array, info) in sources.items():
                    values = signal_features(
                        array[int(epoch.array_row)], info.sfreq, info.kind
                    )
                    features.update(
                        {
                            f"{canonical}__{info.kind}__{name}": value
                            for name, value in values.items()
                        }
                    )
                if "eeg_fpz_cz" in sources and "eeg_pz_oz" in sources:
                    a = sources["eeg_fpz_cz"][0][int(epoch.array_row)]
                    b = sources["eeg_pz_oz"][0][int(epoch.array_row)]
                    features["cross_eeg__correlation"] = (
                        float(np.corrcoef(a, b)[0, 1])
                        if a.std() > 0 and b.std() > 0
                        else 0.0
                    )
                rows.append(features)
        if not rows:
            raise ValueError("No hay épocas válidas para extraer características.")
        with atomic_path(destination) as temp:
            result = pd.DataFrame(rows)
            result.attrs = {
                "preprocessing_id": prepared.name,
                "feature_set_id": feature_key,
            }
            result.to_parquet(temp, index=False)
        write_json(
            destination.with_suffix(".json"),
            {
                "schema_version": 1,
                "preprocessing_id": prepared.name,
                "feature_set_id": feature_key,
                "parameters": json.loads((prepared / "metadata.json").read_text())[
                    "preprocessing"
                ],
            },
        )
    return destination
