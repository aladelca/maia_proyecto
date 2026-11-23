"""Install a cloud-trained hybrid after reproducing its example from the EDF. Release it with a PR."""

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any
from uuid import UUID

import boto3
import joblib
import numpy as np
import pandas as pd

from sleep_staging.provenance import hash_file
from sleep_staging.serving.contract import LABELS
from sleep_staging.serving.stacked_edf import EXAMPLE_PATH, score
from sleep_staging.stack_release import (
    CHECKPOINTS,
    EXAMPLE_SHA256,
    MODEL,
    VERSION,
)

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "src/sleep_staging/stack_release.py"
MODEL_DIR = ROOT / "modelos/stack_five"
# The web refuses to analyse EDF until the API reports this exact version.
# Only the version is bumped: run IDs elsewhere describe the evaluation campaign.
PINNED = [
    "frontend/src/main.tsx",
    "frontend/src/LocalLab.tsx",
    "frontend/tests/local-edf.spec.ts",
    "frontend/tests/ux.spec.ts",
    "modelos/stack_five/README.md",
    "README.md",
]
FILES = {"manifest", "stack_five.joblib", "expected.parquet", "stack_example"}
PROBS = ["prob_" + s for s in LABELS]


def release_module(run_id: str, checkpoints: dict) -> str:
    return f'''"""Public identity of the immutable five-stage EDF release.

scripts/promote_stack_candidate.py rewrites this file when it installs a new candidate.
"""

MODEL = "{MODEL}"
RUN_ID = "{run_id}"
VERSION = "stack-five-" + RUN_ID[:12]
EXAMPLE_SHA256 = "{EXAMPLE_SHA256}"
CHECKPOINTS = {{
    "stack_five.joblib": "{checkpoints["stack_five.joblib"]}",
    "six_direct.pt": "{checkpoints["six_direct.pt"]}",
}}
'''


def check_provenance(job: str, manifest: dict, candidate: dict, inputs: dict) -> None:
    objects = inputs["objects"]
    if (
        manifest.get("model") != MODEL
        or manifest["training_job"] != job
        or manifest["model_sha256"] != candidate["stack_five.joblib"]["sha256"]
        or manifest["dataset_sha256"] != objects["dataset"]["sha256"]
        or manifest["split_sha256"] != objects["splits"]["sha256"]
        or manifest["fullnight_oof_sha256"] != objects["fullnight_oof"]["sha256"]
        or manifest["stack_example_sha256"] != objects["stack_example"]["sha256"]
    ):
        raise ValueError("Procedencia del candidato incompatible.")
    if manifest["fullnight_checkpoint_sha256"] != CHECKPOINTS["six_direct.pt"]:
        raise ValueError("El candidato usa probabilidades de otra red FullNightNet.")


def download(s3: Any, job: str, inputs: dict, root: Path) -> dict:
    b = inputs["buckets"]
    prefix = "candidates/" + str(UUID(job)) + "/"
    progress = json.loads(
        s3.get_object(Bucket=b["research"], Key=prefix + "progress.json")["Body"].read()
    )
    candidate = progress.get("candidate", {})
    if set(candidate) != FILES:
        raise ValueError(
            "La campaña no dejó un candidato híbrido; para LightGBM usar promote_cloud_candidate.py."
        )
    for name, ref in candidate.items():
        shared = name == "stack_example"
        if (shared and ref != inputs["objects"][name]) or (
            not shared and not ref["key"].startswith(prefix)
        ):
            raise ValueError("Objeto inesperado en el candidato.")
        s3.download_file(
            b["research"],
            ref["key"],
            str(root / name),
            ExtraArgs={"VersionId": ref["version"]},
        )
        if hash_file(root / name) != ref["sha256"]:
            raise ValueError("Checksum del candidato incompatible.")
    return candidate


def promote(job: str, baseline: Path) -> None:
    inputs = json.loads(baseline.read_text())
    if hash_file(EXAMPLE_PATH) != EXAMPLE_SHA256:
        raise ValueError("El EDF de ejemplo no coincide con la release.")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        candidate = download(boto3.client("s3"), job, inputs, root)
        manifest = json.loads((root / "manifest").read_text())
        check_provenance(job, manifest, candidate, inputs)
        result = score(EXAMPLE_PATH, "night", joblib.load(root / "stack_five.joblib"))
        scored = result[result.eligible].reset_index(drop=True)
        expected = pd.read_parquet(root / "expected.parquet")
        if (
            scored.onset_s.tolist() != expected.onset_s.tolist()
            or scored.prediction.tolist() != expected.prediction.tolist()
        ):
            raise ValueError("El EDF de ejemplo no reproduce las etapas del candidato.")
        np.testing.assert_allclose(
            scored[PROBS].to_numpy(float), expected[PROBS], rtol=0, atol=1e-6
        )
        if manifest["model_sha256"] == CHECKPOINTS["stack_five.joblib"]:
            print(
                f"El candidato {manifest['release_id']} es idéntico al modelo servido "
                f"({VERSION}); no hay cambios que instalar."
            )
            return
        shutil.copyfile(root / "stack_five.joblib", MODEL_DIR / "stack_five.joblib")
    result.to_csv(MODEL_DIR / "example-expected.csv", index=False)
    run_id = manifest["run_id"]
    RELEASE.write_text(
        release_module(
            run_id, {**CHECKPOINTS, "stack_five.joblib": manifest["model_sha256"]}
        )
    )
    version = "stack-five-" + run_id[:12]
    for name in PINNED:
        path = ROOT / name
        text = path.read_text()
        bumped = text.replace(VERSION, version)
        if bumped != text:
            path.write_text(bumped)
    print(
        json.dumps(
            {"installed": version, "replaces": VERSION, "run_id": run_id}, indent=2
        )
    )
    print(
        "Revisar git diff, ejecutar make test y abrir un PR. Al mergearlo, deploy.yml "
        "publica el worker y comprueba la paridad. La corrida, los datos y las métricas "
        "de modelos/README.md, docs/cloud-operations.md y frontend/src/project.tsx "
        "describen la evaluación y se revisan a mano."
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("job")
    p.add_argument("--baseline", type=Path, default=Path("var/cloud/inputs.json"))
    a = p.parse_args()
    promote(a.job, a.baseline)
