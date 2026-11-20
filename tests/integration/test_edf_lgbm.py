"""Parity against the original development EDF and frozen release bundle."""

import json
import os
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest


@pytest.mark.real_data
def test_edf_lgbm_sc4001_frozen_parity():
    source = os.environ.get("MAIA_EDF_LGBM_REFERENCE")
    if not source:
        pytest.skip("Set MAIA_EDF_LGBM_REFERENCE to SC4001E0-PSG.edf")
    # On macOS torch and LightGBM can load incompatible OpenMP runtimes into
    # the same pytest process during collection. Serving the model needs only one.
    subprocess.run([sys.executable, __file__, source], check=True)


def verify_reference(source):
    from sleep_staging.serving.contract import LABELS, predictions
    from sleep_staging.serving.edf_lgbm import prepare_edf, score_night

    bundle = Path(os.environ.get("MAIA_LGBM_BUNDLE", "var/cloud/bundle"))
    manifest = json.loads((bundle / "manifest.json").read_text())
    model = joblib.load(bundle / "model.joblib")
    frame = prepare_edf(Path(source), "SC4001")
    assert frame.onset_s.tolist() == list(range(0, 79500, 30))
    example = pd.read_parquet(bundle / "example.parquet")
    rows = frame.set_index("onset_s").loc[example.onset_s].reset_index()
    raw = manifest["raw_columns"]
    pd.testing.assert_frame_equal(rows[raw], example[raw], check_exact=True)
    rows["eligible"] = example.eligible
    result = predictions(model, manifest, rows)
    expected = pd.read_parquet(bundle / "expected.parquet")
    assert result.epoch_id.tolist() == expected.epoch_id.tolist()
    assert result.prediction.tolist() == expected.prediction.tolist()
    probs = ["prob_" + c for c in LABELS]
    np.testing.assert_allclose(result[probs], expected[probs], atol=1e-6, rtol=0)
    full = score_night(model, manifest, frame)
    assert full.shape[0] == 2650
    np.testing.assert_allclose(full.loc[full.eligible, probs].sum(axis=1), 1)
    assert full.loc[~full.eligible, probs].isna().all().all()
    # Run the production handler against version-pinned input and in-memory S3.
    import hashlib
    import io
    from unittest.mock import Mock, patch

    from botocore.response import StreamingBody

    from sleep_staging.serving import worker

    data = Path(source).read_bytes()
    storage = Mock()
    storage.get_object.return_value = {
        "Body": StreamingBody(io.BytesIO(data), len(data)),
        "ContentLength": len(data),
    }
    event = {
        "owner": "fixture",
        "job": "job1",
        "example": False,
        "format": "edf",
        "model": "lightgbm",
        "profile": "auto",
        "input_key": "fixture.edf",
        "input_version": "v1",
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    with (
        patch.dict(os.environ, JOBS_BUCKET="jobs"),
        patch.object(worker.boto3, "client", return_value=storage),
        patch.object(worker, "load_release", return_value=(manifest, model)),
    ):
        info = worker.handler(event, None)
    assert (
        info["state"] == "succeeded" and info["model_version"] == manifest["release_id"]
    )
    assert info["window"] == "full" and info["classes"] == LABELS
    out = json.loads(storage.put_object.call_args_list[0].kwargs["Body"])
    actual = pd.DataFrame(out["epochs"])
    assert actual.onset_s.equals(full.onset_s)
    assert actual.prediction.fillna("").equals(full.prediction.fillna(""))
    np.testing.assert_allclose(
        actual[probs], full[probs], atol=1e-6, rtol=0, equal_nan=True
    )


if __name__ == "__main__":
    verify_reference(sys.argv[1])
