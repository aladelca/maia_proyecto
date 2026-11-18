"""Prepare Terraform variables from immutable artifacts produced by the build."""

import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--origin", required=True)
p.add_argument("--inputs", type=Path, default=Path("var/cloud/inputs.json"))
p.add_argument("--existing-role-arn", default="")
a = p.parse_args()
root = Path("var/cloud")
data = {
    "inputs": json.loads(a.inputs.read_text()),
    "images": {
        k: (root / (k + "-image.txt")).read_text().strip() for k in ("api", "worker")
    },
    "existing_role_arn": a.existing_role_arn,
    "origin": a.origin.rstrip("/"),
}
Path("infra/aws/deploy.auto.tfvars.json").write_text(json.dumps(data))
