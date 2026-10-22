import hashlib
import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from typing import Any


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str, allow_nan=False).encode()
    ).hexdigest()[:20]


def code_fingerprint() -> str:
    package = Path(__file__).parent
    return fingerprint(
        {
            str(p.relative_to(package)): hash_file(p)
            for p in sorted(package.rglob("*.py"))
        }
    )


def environment() -> dict[str, str]:
    packages = ["numpy", "scipy", "pandas", "mne", "scikit-learn", "mlflow", "pyarrow"]
    return {name: version(name) for name in packages}


@contextmanager
def atomic_path(path: Path) -> Iterator[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".tmp-", suffix=path.suffix, dir=path.parent
    )
    os.close(descriptor)
    temp = Path(temporary)
    try:
        yield temp
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def write_json(path: Path, value: Any) -> None:
    with atomic_path(path) as temp:
        temp.write_text(
            json.dumps(value, indent=2, default=str, allow_nan=False) + "\n"
        )
