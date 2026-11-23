import importlib.util
from pathlib import Path

import pytest

from sleep_staging import stack_release

pytest.importorskip("boto3")

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "maia_promote_stack", ROOT / "scripts/promote_stack_candidate.py"
)
assert spec and spec.loader
promotion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(promotion)


def test_promotion_regenerates_the_committed_release_module():
    text = promotion.release_module(stack_release.RUN_ID, stack_release.CHECKPOINTS)
    assert text == Path(stack_release.__file__).read_text()


def test_web_gate_pins_the_served_release():
    for name in promotion.PINNED[:4]:
        assert stack_release.VERSION in (ROOT / name).read_text(), name


def candidate() -> tuple[dict, dict, dict]:
    objects = {
        name: {"key": name, "version": "v", "sha256": name + "-sha"}
        for name in ("dataset", "splits", "fullnight_oof", "stack_example")
    }
    model = {"key": "candidates/job/fits/stack/stack_five.joblib", "sha256": "m"}
    manifest = {
        "model": stack_release.MODEL,
        "training_job": "job",
        "model_sha256": "m",
        "dataset_sha256": "dataset-sha",
        "split_sha256": "splits-sha",
        "fullnight_oof_sha256": "fullnight_oof-sha",
        "stack_example_sha256": "stack_example-sha",
        "fullnight_checkpoint_sha256": stack_release.CHECKPOINTS["six_direct.pt"],
    }
    return manifest, {"stack_five.joblib": model}, {"objects": objects}


def test_promotion_rejects_foreign_provenance():
    manifest, files, inputs = candidate()
    promotion.check_provenance("job", manifest, files, inputs)
    for field, value in [
        ("model", "lightgbm"),
        ("training_job", "other"),
        ("model_sha256", "x"),
        ("dataset_sha256", "x"),
        ("fullnight_oof_sha256", "x"),
        ("fullnight_checkpoint_sha256", "x"),
    ]:
        with pytest.raises(ValueError):
            promotion.check_provenance("job", {**manifest, field: value}, files, inputs)
