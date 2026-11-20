"""Record a frozen IBIO experiment and its evaluation in a local MLflow store."""

import argparse
import json
from pathlib import Path

import pandas as pd
from filelock import FileLock
from mlflow import MlflowClient

from sleep_staging.provenance import fingerprint, hash_file, write_json


def sync(run: Path, store: Path) -> dict:
    run, store = run.resolve(), store.resolve()
    store.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((run / "protocol.json").read_text())
    freeze = json.loads((run / "freeze.json").read_text())
    if fingerprint(protocol) != freeze["run"]:
        raise ValueError("La campaña no coincide con el protocolo.")
    if "edf_model" in freeze:
        model = run / "refit" / freeze["edf_model"]
        model_id, profile, digest = (
            model.stem,
            "eeg_fpz_cz+eeg_pz_oz+eog+emg",
            freeze["weights"][model.name],
        )
    else:
        model = run / "refit/model.pt"
        model_id, profile, digest = (
            "usleep",
            protocol["profile"],
            freeze["model_sha256"],
        )
    if hash_file(model) != digest:
        raise ValueError("Checksum del modelo final incompatible.")
    client = MlflowClient(tracking_uri="sqlite:///" + str(store / "mlflow.db"))
    name = "ibio-fullnight"
    ids = {}
    with FileLock(str(store / "sync.lock")):
        exp = client.get_experiment_by_name(name)
        exp_id = (
            exp.experiment_id
            if exp
            else client.create_experiment(
                name, artifact_location=(store / "artifacts").as_uri()
            )
        )
        for stage, folder in (("development", run), ("reserved", run / "reserved")):
            if stage == "reserved" and not (folder / "completed.json").is_file():
                continue
            tag = fingerprint(
                [
                    freeze["run"],
                    stage,
                    hash_file(folder / "metrics.csv"),
                    hash_file(folder / "confusion.json"),
                ]
            )
            prior = client.search_runs(
                [exp_id],
                f"tags.ibio_export = '{tag}' AND attributes.status = 'FINISHED'",
                max_results=1,
            )
            if prior:
                ids[stage] = prior[0].info.run_id
                continue
            r = client.create_run(
                exp_id,
                tags={"ibio_export": tag, "campaign": freeze["run"], "stage": stage},
                run_name=stage + "-" + freeze["run"],
            )
            rid = r.info.run_id
            try:
                for key, value in {
                    **protocol["settings"],
                    "device": protocol["device"],
                    "model": model_id,
                    "profile": profile,
                }.items():
                    client.log_param(rid, key, value)
                table = pd.read_csv(folder / "metrics.csv")
                main = table.loc[
                    (table["window"] == "full")
                    & (table["profile"] == profile)
                    & (table["model"] == model_id)
                ]
                for row in main.itertuples():
                    for key in ("accuracy", "macro_f1", "n1_f1", "m_f1", "coverage"):
                        if hasattr(row, key):
                            client.log_metric(
                                rid, f"{row.cohort}_{key}", float(getattr(row, key))
                            )
                for artifact in (
                    "metrics.csv",
                    "confusion.json",
                    "paired-subject-bootstrap.json",
                ):
                    client.log_artifact(rid, str(folder / artifact))
                if stage == "development":
                    for path in (run / "protocol.json", run / "freeze.json", model):
                        client.log_artifact(rid, str(path))
                else:
                    client.log_artifact(rid, str(folder / "completed.json"))
                client.set_terminated(rid)
                ids[stage] = rid
            except Exception:
                client.set_terminated(rid, status="FAILED")
                raise
        write_json(run / "mlflow-runs.json", ids)
    return ids


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument(
        "--store", type=Path, default=Path("var/research/ibio-fullnight/tracking")
    )
    a = p.parse_args()
    print(sync(a.run, a.store))
