"""Only update built image references in the approved deployment settings."""

import json
from pathlib import Path

for stack, names in {"training": ["training"], "aws": ["api", "worker"]}.items():
    path = Path("infra") / stack / "deploy.auto.tfvars.json"
    settings = json.loads(path.read_text())
    images = {
        name: Path(f"var/cloud/{name}-image.txt").read_text().strip() for name in names
    }
    if stack == "training":
        if settings.get("run_id"):
            raise ValueError(
                "Delivery cannot submit or replace a running training instance"
            )
        settings["image"] = images["training"]
    else:
        settings["images"] = images
    path.write_text(json.dumps(settings))
