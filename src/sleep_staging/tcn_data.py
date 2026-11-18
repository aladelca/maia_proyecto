"""Development cohort, transition labels and disk-backed native-rate waveforms."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from filelock import FileLock
from scipy.signal import butter, sosfiltfilt

from sleep_staging.config import CLASSES
from sleep_staging.data.edf import read_recording
from sleep_staging.lstm_data import SequenceData, sequence_index
from sleep_staging.provenance import atomic_path, fingerprint, hash_file, write_json
from sleep_staging.research_data import development_recordings

WAVE_CHANNELS = ("eeg_fpz_cz", "eeg_pz_oz", "eog")
KEYS = ["epoch_id", "subject_id", "recording_id", "onset_s", "fold"]


def smoke_subset(frame: pd.DataFrame) -> pd.DataFrame:
    """Five small fragments per fold with genuine labels, times and subject IDs."""
    keep = []
    for _, fold in frame.groupby("fold"):
        for _, night in fold.groupby("recording_id"):
            if set(night.task_5.dropna()) != set(CLASSES["task_5"]):
                continue
            night = night.sort_values("onset_s")
            for label in CLASSES["task_5"]:
                positions = np.flatnonzero(night.task_5.eq(label).fillna(False))
                center = positions[len(positions) // 2]
                keep.extend(night.index[max(0, center - 8) : center + 9])
            break
        else:
            raise ValueError("The smoke run needs one five-stage night per fold.")
    return frame.loc[sorted(set(keep))].copy()


def load_cohort(features: Path, split: Path, smoke: bool) -> tuple[SequenceData, dict]:
    meta = json.loads(features.with_name("metadata.json").read_text())
    if (
        hash_file(features) != meta["features_sha256"]
        or hash_file(split) != meta["split_sha256"]
    ):
        raise ValueError("Feature cache or frozen split checksum mismatch.")
    frame = pd.read_parquet(features)
    partitions = pd.read_csv(split).set_index("subject_id", verify_integrity=True)
    dev = partitions.loc[partitions.partition.eq("development"), "fold"]
    if (
        set(frame.subject_id) != set(dev.index)
        or set(dev) != set(range(5))
        or not frame.fold.eq(frame.subject_id.map(dev)).all()
    ):
        raise ValueError(
            "Research features must match the five frozen development folds."
        )
    data = SequenceData.from_frame(smoke_subset(frame) if smoke else frame)
    if len(data.columns) != 145:
        raise ValueError("Expected the campaign's 145 per-epoch signal features.")
    return data, meta


def boundary_targets(data: SequenceData) -> np.ndarray:
    """-1 means a neighboring expert label is unavailable, including gaps and M."""
    neighbors, pad = sequence_index(data.frame, 3, centered=True)
    stages = data.labels[neighbors]
    usable = ~pad.any(axis=1) & (stages >= 0).all(axis=1)
    target = np.full(len(stages), -1, dtype=np.int64)
    target[usable] = (stages[usable] != stages[usable, 1:2]).any(axis=1)
    return target


def wave_cache(
    data: SequenceData, meta: dict, split: Path, source: Path, cache: Path
) -> Path:
    # Rows are already sorted by SequenceData; this key also distinguishes smoke subsets.
    identity = fingerprint(
        {
            "features": meta["features_sha256"],
            "sources": meta["files"],
            "epochs": data.frame.epoch_id.tolist(),
            "channels": WAVE_CHANNELS,
            "reader": hash_file(Path(__file__)),
            "edf": hash_file(Path(__file__).parent / "data/edf.py"),
        }
    )
    directory = cache / identity
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "waveforms.npy"
    manifest = directory / "manifest.json"
    with FileLock(str(directory / "cache.lock")):
        if manifest.exists():
            if hash_file(path) != json.loads(manifest.read_text())["sha256"]:
                raise ValueError("Waveform cache checksum mismatch.")
            return path
        recordings, _ = development_recordings(source, split)
        catalog = recordings.set_index("recording_id")
        hashes = {entry["path"]: entry["sha256"] for entry in meta["files"]}
        with atomic_path(path) as tmp:
            waves = np.lib.format.open_memmap(
                tmp,
                mode="w+",
                dtype="float32",
                shape=(len(data.frame), len(WAVE_CHANNELS), 3000),
            )
            for recording, night in data.frame.groupby("recording_id", sort=False):
                record = catalog.loc[recording]
                if (
                    not night.subject_id.eq(record.subject_id).all()
                    or not night.fold.eq(record.fold).all()
                ):
                    raise ValueError(
                        f"Recording and features disagree for {recording}."
                    )
                for rel in (record.psg_file, record.hyp_file):
                    if hash_file(source / rel) != hashes[rel]:
                        raise ValueError(f"Development EDF checksum mismatch: {rel}")
                signals, _, _ = read_recording(
                    source / record.psg_file, source / record.hyp_file
                )
                onsets = night.onset_s.to_numpy() * 100
                if (onsets < 0).any() or not np.array_equal(onsets, onsets.astype(int)):
                    raise ValueError(f"Invalid epoch offset in {recording}.")
                samples = onsets.astype(int)
                for i, name in enumerate(WAVE_CHANNELS):
                    signal = signals[name]
                    if signal.sfreq != 100 or samples.max() + 3000 > len(signal.values):
                        raise ValueError(
                            f"Incompatible signal timing for {recording}/{name}."
                        )
                    filt = butter(
                        4,
                        (0.3, 15 if name == "eog" else 35),
                        fs=100,
                        btype="bandpass",
                        output="sos",
                    )
                    values = sosfiltfilt(filt, signal.values.astype(np.float64))
                    for row, start in zip(night.index, samples, strict=True):
                        epoch = values[start : start + 3000]
                        if not np.isfinite(epoch).all():
                            raise ValueError(
                                f"Non-finite waveform in {recording}/{name}."
                            )
                        waves[row, i] = epoch / 50.0  # fixed scale in microvolts
            waves.flush()
            del waves
        write_json(
            manifest,
            {
                "sha256": hash_file(path),
                "channels": WAVE_CHANNELS,
                "sfreq": 100,
                "scale_uv": 50,
                "rows": len(data.frame),
            },
        )
    return path
