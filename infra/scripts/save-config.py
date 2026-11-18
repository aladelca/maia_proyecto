"""Back up non-secret release references used by CodeBuild and by a fresh checkout."""

import json
from pathlib import Path

import boto3

fields = {
    "aws": {
        "origin",
        "images",
        "inputs",
        "existing_role_arn",
        "live_version_arn",
        "region",
    },
    "vercel": {"project_name", "team_id", "publish"},
    "training": {
        "region",
        "subnets",
        "vpc_id",
        "standalone_security_group",
        "instance_profile",
        "job_role",
        "batch_service_role",
        "image",
        "inputs",
        "managed_mlflow",
        "use_batch",
        "instance_type",
        "run_id",
        "mode",
    },
}
s3 = boto3.client("s3")
values = {
    name: json.loads(Path("infra", name, "deploy.auto.tfvars.json").read_text())
    for name in fields
}
for name, data in values.items():
    if set(data) - fields[name]:
        raise ValueError(f"Unknown config field in {name}")
    if name == "training" and data.get("run_id"):
        raise ValueError("Clear run_id before backing up deployment settings")
bucket = values["aws"]["inputs"]["buckets"]["research"]
for name, data in values.items():
    key = f"delivery/config/{name}.json"
    resp = s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(data).encode(),
        ContentType="application/json",
    )
    print(name, resp["VersionId"])
