from pathlib import Path

import mne
import pandas as pd

from sleep_staging.config import DataConfig
from sleep_staging.data.edf import edf_channels
from sleep_staging.provenance import atomic_path, write_json


def audit_dataset(data: DataConfig, root: Path) -> dict:
    manifest = root / "data/manifests" / data.name
    records = pd.read_parquet(manifest / "recordings.parquet")
    channel_rows: list[dict] = []
    annotation_rows: list[dict] = []
    for record in records.to_dict("records"):
        for name in ["psg_file", "hyp_file"]:
            if not (root / record[name]).is_file():
                raise ValueError(f"Falta {record[name]}")
        channel_rows.extend(
            {"recording_id": record["recording_id"], **item}
            for item in edf_channels(root / record["psg_file"])
        )
        annotations = mne.read_annotations(root / record["hyp_file"])
        for text, duration in zip(
            annotations.description, annotations.duration, strict=True
        ):
            annotation_rows.append(
                {
                    "recording_id": record["recording_id"],
                    "annotation": str(text),
                    "duration_s": float(duration),
                }
            )
    for name, rows in [("channels", channel_rows), ("annotations", annotation_rows)]:
        with atomic_path(manifest / f"{name}.parquet") as temp:
            pd.DataFrame(rows).to_parquet(temp, index=False)
    summary = {
        "dataset": data.name,
        "source": data.source,
        "records": len(records),
        "subjects": int(records.subject_id.nunique()),
        "annotation_seconds": pd.DataFrame(annotation_rows)
        .groupby("annotation")
        .duration_s.sum()
        .to_dict(),
        "manifest": str(manifest),
    }
    write_json(manifest / "audit.json", summary)
    return summary
