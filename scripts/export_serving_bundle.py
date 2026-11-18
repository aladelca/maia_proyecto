"""Export the frozen CPU champion and a label-free development night."""

import argparse
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from sleep_staging.provenance import hash_file, write_json
from sleep_staging.serving.contract import KEYS, LABELS, SCHEMA, predictions, read_night


def export(root: Path, output: Path) -> None:
    research = json.loads((root / "sc-w30-v2/campaign.json").read_text())
    result = json.loads((root / "sc-w30-refinement-v1/lgbm.json").read_text())
    refit = result["candidate"]["refit"]
    model_path = Path(refit["directory"]) / "model.joblib"
    if (
        hash_file(model_path) != refit["model_sha256"]
        or hash_file(Path(research["dataset"]))
        != result["provenance"]["dataset_sha256"]
    ):
        raise ValueError("Invalid research source checksum")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(model_path, output / "model.joblib")
    model = joblib.load(model_path)
    raw = result["provenance"]["columns"]
    manifest = {
        "release_id": "lgbm-" + refit["model_sha256"][:16],
        "model_sha256": refit["model_sha256"],
        "raw_columns": raw,
        "model_columns": list(model.feature_names_in_),
        "classes": LABELS,
        "preprocessing_id": "sc-w30-context435-v1",
        "input_schema": SCHEMA,
        "run_id": result["candidate"]["mlflow_run_id"],
        "dataset_sha256": result["provenance"]["dataset_sha256"],
        "split_sha256": result["provenance"]["split_sha256"],
        "training_config": refit["configuration"],
        "iterations": refit["best_epoch"],
        "versions": {
            **result["provenance"]["versions"],
            "lightgbm": result["provenance"]["lightgbm_version"],
        },
    }
    df = pd.read_parquet(research["dataset"])
    with np.load(
        Path(result["directory"]) / "candidate/example-predictions.npz",
        allow_pickle=False,
    ) as gold:
        ids = gold["epoch_id"]
        record = df.loc[df.epoch_id.eq(ids[0]), "recording_id"].item()
        night = df.loc[df.recording_id.eq(record)].copy()
        night["eligible"] = night.task_5.notna()
        table = pa.Table.from_pandas(
            night[KEYS + raw], preserve_index=False
        ).replace_schema_metadata(
            {
                b"input_schema": SCHEMA.encode(),
                b"preprocessing_id": manifest["preprocessing_id"].encode(),
            }
        )
        pq.write_table(table, output / "example.parquet")
        frame = read_night(output / "example.parquet", manifest)
        pred = predictions(model, manifest, frame)
        np.testing.assert_allclose(
            pred.set_index("epoch_id")
            .loc[ids, ["prob_" + c for c in LABELS]]
            .to_numpy(),
            gold["probabilities"],
            atol=1e-6,
            rtol=0,
        )
        pred.to_parquet(output / "expected.parquet", index=False)
    manifest["example_sha256"] = hash_file(output / "example.parquet")
    write_json(output / "manifest.json", manifest)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--research-root", type=Path, default=Path("var/research"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    export(args.research_root, args.output)
