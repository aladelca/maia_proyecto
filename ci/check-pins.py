"""Keep the package pins baked into the containers identical to uv.lock."""

import re
import sys
import tomllib
from pathlib import Path


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


locked = {
    p["name"]: p["version"]
    for p in tomllib.loads(Path("uv.lock").read_text())["package"]
}
drift = [
    f"{path}: {name}=={pinned}, uv.lock has {locked.get(normalize(name), 'nothing')}"
    for path in sorted(Path("containers").iterdir())
    for name, pinned in re.findall(
        r"([A-Za-z0-9][\w.-]*)==([\w.+!-]+)", path.read_text()
    )
    if locked.get(normalize(name)) != pinned
]
if drift:
    sys.exit("Container pins differ from uv.lock:\n" + "\n".join(drift))
