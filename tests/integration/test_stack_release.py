"""Check the public worker, embedded demo and frozen model on CPU in an isolated process."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_frozen_stack_worker():
    for package in ("boto3", "torch", "lightgbm"):
        pytest.importorskip(package)
    subprocess.run(
        [sys.executable, __file__],
        check=True,
        env={**os.environ, "JOBS_BUCKET": "jobs"},
    )


if __name__ == "__main__":
    import io
    import json
    from unittest.mock import Mock, patch

    import numpy as np
    import pandas as pd

    from sleep_staging.serving.worker import handler
    from sleep_staging.stack_release import EXAMPLE_SHA256, MODEL, RUN_ID, VERSION

    s3 = Mock()
    event = {
        "format": "edf",
        "model": MODEL,
        "example": True,
        "input_key": "bundled/example.edf",
        "input_version": VERSION,
        "sha256": EXAMPLE_SHA256,
        "job": "job",
        "owner": "guest",
    }
    with patch("boto3.client", return_value=s3):
        result = handler(event, None)
    assert result["state"] == "succeeded" and result["mlflow_run_id"] == RUN_ID
    assert result["model_version"] == VERSION and result["eligible"] == 20
    s3.get_object.assert_not_called()
    body = json.loads(s3.put_object.call_args_list[0].kwargs["Body"])
    csv = pd.read_csv(io.BytesIO(s3.put_object.call_args_list[1].kwargs["Body"]))
    expected = pd.read_csv(ROOT / "modelos/stack_five/example-expected.csv")
    assert csv.onset_s.equals(expected.onset_s) and csv.prediction.equals(
        expected.prediction
    )
    cols = ["prob_" + s for s in result["classes"]]
    np.testing.assert_allclose(csv[cols], expected[cols], rtol=0, atol=1e-6)
    np.testing.assert_allclose(
        pd.DataFrame(body["epochs"])[cols], expected[cols], rtol=0, atol=1e-6
    )
