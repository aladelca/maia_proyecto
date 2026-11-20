import numpy as np
import pytest
import torch


def edf(path, names=("EEG Pz-Oz",), epochs=3):
    """A small native-rate EDF with a valid header and real EDF data-record layout."""

    def field(value, width):
        return str(value).encode("ascii").ljust(width)

    c = len(names)
    rates = [1 if "EMG" in n else 100 for n in names]
    size = 256 * (1 + c)
    h = b"".join(
        [
            field(0, 8),
            field("test", 80),
            field("test", 80),
            field("01.01.00", 8),
            field("00.00.00", 8),
            field(size, 8),
            field("", 44),
            field(epochs, 8),
            field(30, 8),
            field(c, 4),
        ]
    )
    for values, width in [
        (names, 16),
        ([""] * c, 80),
        (["uV"] * c, 8),
        ([-100] * c, 8),
        ([100] * c, 8),
        ([-32768] * c, 8),
        ([32767] * c, 8),
        ([""] * c, 80),
        ([r * 30 for r in rates], 8),
        ([""] * c, 32),
    ]:
        h += b"".join(field(v, width) for v in values)
    record = b"".join(
        (np.sin(np.arange(r * 30) * 0.31) * 1500).astype("<i2").tobytes() for r in rates
    )
    path.write_bytes(h + record * epochs)
    return path


def test_real_edf_format_no_hypnogram_and_user_profile(tmp_path):
    from sleep_staging.fullnight.model import FullNightNet
    from sleep_staging.serving.edf import predict_edf

    torch.set_num_threads(1)
    net = FullNightNet("wave", "movement", 8).eval()
    path = edf(tmp_path / "test.edf", ("EEG Pz-Oz", "EOG horizontal"))
    result, profile = predict_edf(path, net, "eeg_pz_oz", "demo", torch.device("cpu"))
    assert profile == "eeg_pz_oz" and result.eligible.all()
    assert result.onset_s.tolist() == [0, 30, 60]
    np.testing.assert_allclose(result.filter(like="prob_").sum(axis=1), 1, atol=1e-6)
    with pytest.raises(ValueError):
        predict_edf(path, net, "eeg_fpz_cz", "demo", torch.device("cpu"))
    assert (
        predict_edf(path, net, "auto", "demo", torch.device("cpu"))[1]
        == "eeg_pz_oz+eog"
    )


def test_invalid_edf_and_no_eeg(tmp_path):
    from sleep_staging.serving.edf import validate_edf

    p = edf(tmp_path / "input.edf", ("EOG horizontal",))
    with pytest.raises(ValueError):
        validate_edf(p, "auto")
    good = edf(p)
    good.write_bytes(good.read_bytes()[:-6])
    with pytest.raises(ValueError):
        validate_edf(good, "auto")


def test_unsupported_auxiliary_channel_cannot_block_selected_eeg(tmp_path):
    from sleep_staging.fullnight.model import FullNightNet
    from sleep_staging.serving.edf import predict_edf

    p = edf(tmp_path / "subset.edf", ("EEG Pz-Oz", "EMG submental"))
    raw = bytearray(p.read_bytes())
    unit = 256 + 96 * 2 + 8
    raw[unit : unit + 8] = b"percent "
    p.write_bytes(raw)
    net = FullNightNet("wave", "six", 8).eval()
    f, used = predict_edf(p, net, "eeg_pz_oz", "one", torch.device("cpu"))
    assert f.eligible.all() and used == "eeg_pz_oz"
    with pytest.raises(ValueError, match="Unidad"):
        predict_edf(p, net, "auto", "one", torch.device("cpu"))


def test_five_stage_edf_uses_fixed_spectral_blend_only_at_reference_profile(
    tmp_path, monkeypatch
):
    from sleep_staging.fullnight.training import REFERENCE
    from sleep_staging.serving import edf as serving

    raw = edf(tmp_path / "input.edf", ("EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"))
    from sleep_staging.fullnight.model import FullNightNet
    from sleep_staging.fullnight.training import Settings, save_model
    from sleep_staging.provenance import hash_file, write_json

    for name, view in [("wave", "wave"), ("spectrogram", "spectrum")]:
        save_model(
            tmp_path / "refit" / (name + ".pt"),
            FullNightNet(view, "five", 8),
            Settings(),
            ["s"],
            "fixture",
        )
    tree_dir = tmp_path / "refit/trees"
    tree_dir.mkdir()
    for name in (REFERENCE, "eeg_pz_oz"):
        (tree_dir / (name + ".txt")).write_text("fixture tree")
    weights = {
        str(p.relative_to(tmp_path / "refit")): hash_file(p)
        for p in (tmp_path / "refit").rglob("*")
        if p.is_file()
    }
    write_json(tmp_path / "freeze.json", {"run": "fixture", "weights": weights})

    def fake_tree(checkpoint, features):
        return np.tile([0.0, 0.0, 1.0, 0.0, 0.0], (len(features), 1))

    monkeypatch.setattr(serving, "tree_probabilities", fake_tree)
    with_spec, p, method = serving.predict_edf_five(
        raw, tmp_path, REFERENCE, "x", torch.device("cpu")
    )
    assert p == REFERENCE and method == "triple" and "prob_M" not in with_spec
    assert np.all(with_spec.prob_N2 >= 0.5)
    pair, p, method = serving.predict_edf_five(
        raw, tmp_path, "eeg_pz_oz", "x", torch.device("cpu")
    )
    assert p == "eeg_pz_oz" and method == "blend"
    np.testing.assert_allclose(pair.filter(like="prob_").sum(axis=1), 1, atol=1e-6)


def test_local_upload_and_cloud_worker_return_six_stage_hypnogram(
    tmp_path, monkeypatch
):
    import json
    from hashlib import sha256
    from io import BytesIO
    from unittest.mock import Mock

    from botocore.response import StreamingBody

    from sleep_staging.fullnight.model import FullNightNet
    from sleep_staging.serving import worker
    from sleep_staging.serving.local_edf import create_app

    device = torch.device("cpu")
    torch.set_num_threads(1)
    net = FullNightNet("wave", "movement", 8).eval()
    meta = {
        "run_id": "unit",
        "schema": "ibio-edf-v1",
        "classes": ["W", "N1", "N2", "N3", "REM", "M"],
        "head": "movement",
    }
    p = edf(tmp_path / "test.edf")
    data = p.read_bytes()
    api = create_app(net, meta, device, tmp_path).test_client()
    ans = api.post(
        "/v1/edf", data={"profile": "auto", "file": (BytesIO(data), "test.edf")}
    )
    assert (
        ans.status_code == 200
        and ans.json["profile"] == "eeg_pz_oz"
        and ans.json["rows"] == 3
    )
    assert ans.json["classes"][-1] == "M"
    assert (
        api.post(
            "/v1/edf", data={"profile": "eog", "file": (BytesIO(data), "test.edf")}
        ).status_code
        == 400
    )
    with monkeypatch.context() as local:
        local.setattr(
            "sleep_staging.serving.local_edf.predict_edf",
            Mock(side_effect=OSError("internal filesystem details")),
        )
        unreadable = api.post("/v1/edf", data={"file": (BytesIO(data), "test.edf")})
        assert unreadable.status_code == 400
        assert unreadable.json == {
            "error": "No se pudo leer el EDF. Comprueba el archivo y los canales seleccionados."
        }
    s3 = Mock()
    s3.get_object.return_value = {
        "ContentLength": len(data),
        "Body": StreamingBody(BytesIO(data), len(data)),
    }
    monkeypatch.setattr(worker.boto3, "client", lambda _: s3)
    monkeypatch.setattr(worker, "load_edf_release", lambda: (net, meta))
    monkeypatch.setenv("JOBS_BUCKET", "jobs")
    out = worker.handler(
        {
            "job": "r1",
            "owner": "u1",
            "example": False,
            "format": "edf",
            "profile": "auto",
            "input_key": "users/u1/uploads/r1.edf",
            "input_version": "v1",
            "sha256": sha256(data).hexdigest(),
        },
        None,
    )
    assert out["state"] == "succeeded" and out["profile"] == ans.json["profile"]
    payload = json.loads(s3.put_object.call_args_list[0].kwargs["Body"])
    assert (
        payload["rows"] == 3
        and payload["epochs"][0]["prob_M"] == ans.json["epochs"][0]["prob_M"]
    )


def test_lgbm_edf_features_and_mount(tmp_path):
    from sleep_staging.serving.edf_lgbm import prepare_edf

    names = ("EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal", "EMG submental")
    source = edf(tmp_path / "demo.edf", names, epochs=1)
    frame = prepare_edf(source, "demo")
    assert frame.shape == (1, 149)
    assert frame.epoch_id.tolist() == ["demo:0"]
    assert frame.eligible.all()
    assert frame.filter(like="emg__emg_envelope__").shape[1] == 19
    with pytest.raises(ValueError, match="EMG submental"):
        prepare_edf(edf(source, names[:3]), "demo")
    with pytest.raises(ValueError, match="duplicad"):
        prepare_edf(edf(source, (*names, names[0])), "demo")
    source.write_bytes(source.read_bytes()[:-10])
    with pytest.raises(ValueError, match="Cabecera"):
        prepare_edf(source, "demo")


def test_edf_lgbm_worker_and_flat_epochs(tmp_path, monkeypatch):
    import hashlib
    import json
    from io import BytesIO
    from unittest.mock import Mock

    from botocore.response import StreamingBody

    from sleep_staging.serving import worker
    from sleep_staging.serving.contract import KEYS, LABELS, SCHEMA, context_matrix
    from sleep_staging.serving.edf_lgbm import prepare_edf, score_night

    names = ("EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal", "EMG submental")
    source = edf(tmp_path / "test.edf", names)
    frame = prepare_edf(source, "test")
    cols = [c for c in frame if c not in KEYS]
    model = Mock(
        classes_=np.array(LABELS), feature_names_in_=list(context_matrix(frame, cols))
    )
    model.predict_proba.side_effect = lambda x: np.tile([0, 0, 1, 0, 0], (len(x), 1))
    manifest = {
        "raw_columns": cols,
        "model_columns": model.feature_names_in_,
        "classes": LABELS,
        "input_schema": SCHEMA,
        "preprocessing_id": "sc-w30-context435-v1",
        "release_id": "lgbm-test",
    }
    s3 = Mock()
    data = source.read_bytes()
    s3.get_object.return_value = {
        "ContentLength": len(data),
        "Body": StreamingBody(BytesIO(data), len(data)),
    }
    monkeypatch.setenv("JOBS_BUCKET", "jobs")
    monkeypatch.setattr(worker.boto3, "client", lambda _: s3)
    monkeypatch.setattr(worker, "load_release", lambda: (manifest, model))
    event = {
        "job": "test",
        "owner": "u",
        "example": False,
        "format": "edf",
        "model": "lightgbm",
        "profile": "auto",
        "sha256": hashlib.sha256(data).hexdigest(),
        "input_key": "test.edf",
        "input_version": "1",
    }
    summary = worker.handler(event, None)
    assert summary["classes"] == LABELS and summary["window"] == "full"
    rows = json.loads(s3.put_object.call_args_list[0].kwargs["Body"])["epochs"]
    assert len(rows) == 3 and all(r["prob_N2"] == 1 and "prob_M" not in r for r in rows)

    frame.loc[1, "eligible"] = False
    result = score_night(model, manifest, frame)
    assert (
        result.onset_s.tolist() == [0, 30, 60]
        and result.loc[[0, 2], "prediction"].eq("N2").all()
    )
    assert result.loc[1].filter(like="prob_").isna().all()
    passed = model.predict_proba.call_args.args[0]
    assert passed.equals(context_matrix(frame.loc[[0, 2]], cols))

    # Replace all samples with a constant digital value and keep the EDF header.
    source.write_bytes(data[:1280] + bytes(len(data) - 1280))
    flat = prepare_edf(source, "test")
    assert not flat.eligible.any()
    model.reset_mock()
    assert score_night(model, manifest, flat).prediction.isna().all()
    model.predict_proba.assert_not_called()


def test_edf_lgbm_requires_native_envelope_and_ignores_partial_epoch(tmp_path):
    from sleep_staging.serving.edf_lgbm import prepare_edf

    names = ("EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal", "EMG submental")
    p = edf(tmp_path / "test.edf", names, epochs=2)
    data = bytearray(p.read_bytes())
    # Make two 20-second records at native rates, leaving a 10-second tail.
    data[244:252] = b"20      "
    start = 256 + 216 * 4
    for i, n in enumerate((2000, 2000, 2000, 20)):
        data[start + 8 * i : start + 8 * (i + 1)] = str(n).encode().ljust(8)
    p.write_bytes(data[: 1280 + 2 * 2 * 6020])
    assert prepare_edf(p, "test").onset_s.tolist() == [0]
    data[start + 24 : start + 32] = b"2000    "
    p.write_bytes(data[: 1280 + 2 * 2 * 8000])
    with pytest.raises(ValueError, match="EMG submental a 1 Hz"):
        prepare_edf(p, "test")
