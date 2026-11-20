"""The HTTP adapter derives ownership from trusted JWT claims for each request."""

import hashlib
import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

import pytest

pytest.importorskip("boto3")
pytest.importorskip("flask")

from botocore.exceptions import ClientError

spec = importlib.util.spec_from_file_location(
    "maia_http", Path(__file__).resolve().parents[2] / "services/api/app.py"
)
assert spec and spec.loader
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)


def invoke(method, path, subject=None, body=None, headers=None):
    return api.handler(
        {
            "requestContext": {
                "http": {"method": method},
                "authorizer": {"jwt": {"claims": {"sub": subject}}},
            },
            "rawPath": path,
            "headers": {"content-type": "application/json", **(headers or {})},
            "body": json.dumps(body),
        },
        None,
    )


def test_idempotency_and_owner(monkeypatch):
    machine = "arn:aws:states:us-west-2:111111111111:stateMachine:demo"
    for key, value in {
        "MACHINE_ARN": machine,
        "EXAMPLE_KEY": "example",
        "EXAMPLE_VERSION": "version",
        "EXAMPLE_SHA256": "a" * 64,
    }.items():
        monkeypatch.setenv(key, value)
    states = Mock()
    monkeypatch.setattr(api, "client", lambda _: states)
    job = uuid4().hex
    path = "/v1/predictions"
    body = {"example": "default"}
    headers = {"Idempotency-Key": job}
    assert (
        invoke("POST", path, None, body, {**headers, "x-user": "forged"})["statusCode"]
        == 401
    )
    assert invoke("POST", path, "person-a", body, headers)["statusCode"] == 202
    params = states.start_execution.call_args.kwargs
    assert params["stateMachineArn"] == machine + ":live"
    owner = hashlib.sha256(b"person-a").hexdigest()[:32]
    assert params["name"] == owner + "-" + job
    states.start_execution.side_effect = ClientError(
        {"Error": {"Code": "ExecutionAlreadyExists", "Message": ""}}, "StartExecution"
    )
    states.describe_execution.return_value = {"input": params["input"]}
    assert invoke("POST", path, "person-a", body, headers)["statusCode"] == 202
    monkeypatch.setenv("EXAMPLE_VERSION", "new-published-example")
    assert invoke("POST", path, "person-a", body, headers)["statusCode"] == 202
    states.describe_execution.return_value = {"input": json.dumps({"example": False})}
    assert invoke("POST", path, "person-a", body, headers)["statusCode"] == 409
    states.describe_execution.return_value = {"status": "RUNNING"}
    assert invoke("GET", path + "/" + job, "person-b")["statusCode"] == 200
    assert owner not in states.describe_execution.call_args.kwargs["executionArn"]


def test_validation_precedes_signing(monkeypatch):
    s3 = Mock()
    monkeypatch.setattr(api, "client", lambda _: s3)
    monkeypatch.setenv("JOBS_BUCKET", "jobs")
    for size in (0, -4, True, 32 * 1024**2 + 1):
        assert (
            invoke("POST", "/v1/uploads", "person", {"size": size, "sha256": "a" * 64})[
                "statusCode"
            ]
            == 400
        )
    assert not s3.generate_presigned_post.called
    s3.generate_presigned_post.return_value = {
        "url": "https://storage.example",
        "fields": {},
    }
    assert (
        invoke("POST", "/v1/uploads", "person", {"size": 123, "sha256": "a" * 64})[
            "statusCode"
        ]
        == 201
    )
    assert [
        "content-length-range",
        123,
        123,
    ] in s3.generate_presigned_post.call_args.kwargs["Conditions"]


def test_unconfigured_edf_release_returns_before_upload_or_execution(monkeypatch):
    monkeypatch.setenv("EDF_ENABLED", "0")
    client = Mock()
    monkeypatch.setattr(api, "client", client)
    body = {"format": "edf", "size": 123, "sha256": "a" * 64}
    assert invoke("POST", "/v1/uploads", "p", body)["statusCode"] == 503
    assert (
        invoke("POST", "/v1/predictions", "p", body, {"Idempotency-Key": uuid4().hex})[
            "statusCode"
        ]
        == 503
    )
    client.assert_not_called()


def test_edf_upload_and_selected_profile_are_bound_to_job(monkeypatch):
    s3, states = Mock(), Mock()
    monkeypatch.setenv("EDF_ENABLED", "1")
    monkeypatch.setattr(api, "client", lambda name: s3 if name == "s3" else states)
    monkeypatch.setenv("JOBS_BUCKET", "jobs")
    monkeypatch.setenv("MACHINE_ARN", "arn:example")
    s3.generate_presigned_post.return_value = {
        "url": "https://s3.example",
        "fields": {},
    }
    uploaded = invoke(
        "POST",
        "/v1/uploads",
        "person",
        {"format": "edf", "size": 52_000_000, "sha256": "a" * 64},
    )
    assert uploaded["statusCode"] == 201
    uid = json.loads(uploaded["body"])["upload_id"]
    args = s3.generate_presigned_post.call_args.kwargs
    assert args["Key"].endswith(uid + ".edf")
    s3.head_object.return_value = {
        "ContentLength": 52_000_000,
        "Metadata": {"sha256": "a" * 64},
    }
    body = {
        "format": "edf",
        "profile": "eeg_pz_oz",
        "upload_id": uid,
        "version_id": "version",
    }
    headers = {"Idempotency-Key": uuid4().hex}
    assert (
        invoke("POST", "/v1/predictions", "person", body, headers)["statusCode"] == 202
    )
    event = json.loads(states.start_execution.call_args.kwargs["input"])
    assert (
        event["profile"] == "eeg_pz_oz"
        and event["format"] == "edf"
        and event["input_key"] == args["Key"]
    )
    assert (
        invoke(
            "POST", "/v1/predictions", "person", {**body, "profile": "emg"}, headers
        )["statusCode"]
        == 400
    )
