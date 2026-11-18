"""Derive the training settings from the Terraform network and immutable inputs."""

import argparse
import json
import subprocess
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--batch", action="store_true")
p.add_argument(
    "--sqlite",
    action="store_true",
    help="Persist per-campaign MLflow SQLite snapshots and S3 artifacts",
)
p.add_argument("--instance-profile", default="")
p.add_argument("--job-role", default="")
p.add_argument("--batch-service-role")
args = p.parse_args()
net = json.loads(
    subprocess.check_output(["terraform", "-chdir=infra/network", "output", "-json"])
)
root = Path("var/cloud")
settings = dict(
    subnets=net["subnets"]["value"],
    vpc_id=net["vpc_id"]["value"],
    standalone_security_group=net["security_group"]["value"],
    inputs=json.loads((root / "inputs.json").read_text()),
    image=(root / "training-image.txt").read_text().strip(),
    use_batch=args.batch,
    managed_mlflow=not args.sqlite,
    instance_profile=args.instance_profile,
    job_role=args.job_role,
    batch_service_role=args.batch_service_role,
)
Path("infra/training/deploy.auto.tfvars.json").write_text(json.dumps(settings))
