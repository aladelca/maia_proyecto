"""Development-only extraction with bounded memory and resumable record caches."""

import json
from pathlib import Path

import pandas as pd
from filelock import FileLock

from sleep_staging.data.edf import read_recording
from sleep_staging.data.labels import epoch_annotations
from sleep_staging.features.recording import (
    FPZ_FLAT,
    SIGNAL_CHANNELS,
    recording_features,
)
from sleep_staging.provenance import (
    atomic_path,
    environment,
    fingerprint,
    hash_file,
    write_json,
)


def development_recordings(
    source_root: Path, split_file: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subjects = pd.read_csv(split_file, dtype={"subject_id": str})
    if subjects.subject_id.duplicated().any() or subjects.isna().any().any():
        raise ValueError("Duplicate or null subjects in frozen splits.")
    if set(subjects.partition) != {"development", "test"}:
        raise ValueError("Expected development and reserved test partitions.")
    dev = subjects[subjects.partition.eq("development")]
    test = subjects[subjects.partition.eq("test")]
    if set(dev.fold) != set(range(5)) or not test.fold.eq(-1).all():
        raise ValueError("Expected five frozen development folds and test fold -1.")
    records = pd.read_parquet(
        source_root / "data/manifests/sleep-cassette/recordings.parquet"
    )
    if records.recording_id.duplicated().any() or set(records.subject_id) != set(
        subjects.subject_id
    ):
        raise ValueError("Recording catalogue and frozen subjects are incompatible.")
    selected = records.merge(dev, on="subject_id", validate="many_to_one")
    # Only selected paths are opened or hashed downstream; test EDFs stay untouched.
    return selected.sort_values("recording_id").reset_index(drop=True), subjects


def _record_features(record: dict, source_root: Path, margin: int) -> pd.DataFrame:
    signals, annotations, duration = read_recording(
        source_root / record["psg_file"], source_root / record["hyp_file"]
    )
    if set(SIGNAL_CHANNELS) - set(signals):
        raise ValueError(f"Missing channels in {record['recording_id']}.")
    epochs = epoch_annotations(annotations, duration)
    sleep = epochs[epochs.task_5.isin(["N1", "N2", "N3", "REM"])]
    if sleep.empty:
        raise ValueError("No scored sleep for the existing annotation-dependent crop.")
    selected = epochs.reason.eq("") & epochs.onset_s.between(
        sleep.onset_s.min() - margin * 60, sleep.onset_s.max() + margin * 60
    )
    epochs = epochs[selected].copy().reset_index(drop=True)
    epochs["recording_id"] = record["recording_id"]
    epochs["subject_id"] = record["subject_id"]
    epochs["fold"] = int(record["fold"])
    epochs["epoch_id"] = (
        record["recording_id"] + ":" + epochs.onset_sample_eeg.astype(str)
    )
    features = recording_features(signals, epochs.onset_s.to_numpy())
    result = pd.concat([epochs, features], axis=1)
    return result[~features[FPZ_FLAT].astype(bool)].reset_index(drop=True)


def build_research_dataset(
    source_root: Path, split_file: Path, output_root: Path
) -> Path:
    records, subjects = development_recordings(source_root, split_file)
    manifest = pd.read_parquet(
        source_root / "data/manifests/sleep-cassette/files.parquet"
    )
    selected_paths = set(records.psg_file) | set(records.hyp_file)
    files = manifest[manifest.path.isin(selected_paths)]
    if set(files.path) != selected_paths or files.path.duplicated().any():
        raise ValueError("Missing or duplicate development files in checksum manifest.")
    package = Path(__file__).parent
    inputs = {
        "schema": 1,
        "split_sha256": hash_file(split_file),
        "files": files[["path", "sha256"]].sort_values("path").to_dict("records"),
        "source": {
            name: hash_file(package / name)
            for name in [
                "research_data.py",
                "features/research.py",
                "features/recording.py",
                "features/extract.py",
                "data/edf.py",
                "data/labels.py",
            ]
        },
        "environment": environment(),
        "wake_margin_minutes": 30,
        "channels": SIGNAL_CHANNELS,
        "subject_ids": sorted(records.subject_id.unique()),
    }
    destination = output_root / "datasets" / fingerprint(inputs)
    destination.mkdir(parents=True, exist_ok=True)
    with FileLock(str(destination / "dataset.lock")):
        if (destination / "metadata.json").exists():
            metadata = json.loads((destination / "metadata.json").read_text())
            features = destination / "features.parquet"
            if (
                not features.is_file()
                or hash_file(features) != metadata["features_sha256"]
            ):
                raise ValueError("Research dataset cache checksum mismatch.")
            return destination / "features.parquet"
        for file in files.itertuples():
            if hash_file(source_root / file.path) != file.sha256:
                raise ValueError(f"Checksum mismatch: {file.path}.")
        frames = []
        for number, record in enumerate(records.to_dict("records"), 1):
            path = destination / "records" / f"{record['recording_id']}.parquet"
            if not path.exists():
                frame = _record_features(record, source_root, margin=30)
                with atomic_path(path) as temporary:
                    frame.to_parquet(temporary, index=False)
            frames.append(pd.read_parquet(path))
            print(
                f"EXTRACT {number}/{len(records)} {record['recording_id']} epochs={len(frames[-1])}",
                flush=True,
            )
        result = pd.concat(frames, ignore_index=True)
        if result.epoch_id.duplicated().any() or not set(result.subject_id).issubset(
            set(inputs["subject_ids"])
        ):
            raise ValueError("Invalid development-only epoch identities.")
        with atomic_path(destination / "features.parquet") as temporary:
            result.to_parquet(temporary, index=False)
        supervised = result[result.task_5.notna()]
        write_json(
            destination / "metadata.json",
            {
                **inputs,
                "records": len(records),
                "development_subjects": supervised.subject_id.nunique(),
                "reserved_test_subjects": subjects[
                    subjects.partition.eq("test")
                ].subject_id.nunique(),
                "supervised_epochs": len(supervised),
                "class_support": supervised.task_5.value_counts().to_dict(),
                "features_sha256": hash_file(destination / "features.parquet"),
            },
        )
    return destination / "features.parquet"
