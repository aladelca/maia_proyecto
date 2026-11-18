"""Create a retained bucket, or verify ownership of an existing deterministic name."""

import json
import subprocess
import sys

name, region = sys.argv[1:]


def aws(*args):
    return subprocess.run(["aws", *args], capture_output=True, text=True)


owner = json.loads(aws("sts", "get-caller-identity").stdout)["Account"]
head = aws("s3api", "head-bucket", "--bucket", name, "--expected-bucket-owner", owner)
if head.returncode:
    options = (
        []
        if region == "us-east-1"
        else ["--create-bucket-configuration", f"LocationConstraint={region}"]
    )
    create = aws(
        "s3api", "create-bucket", "--bucket", name, "--region", region, *options
    )
    if create.returncode:
        sys.exit(create.stderr)
location = json.loads(
    aws(
        "s3api",
        "get-bucket-location",
        "--bucket",
        name,
        "--expected-bucket-owner",
        owner,
    ).stdout
)["LocationConstraint"]
if (location or "us-east-1") != region:
    sys.exit("Bucket region does not match configuration")
