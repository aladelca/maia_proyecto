"""Cognito-authenticated API Gateway HTTP v2 adapter; jobs run in Step Functions."""

import base64
import hashlib
import json
import os
import re
import uuid
from functools import lru_cache
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from flask import Blueprint, Flask, g, jsonify, request
from werkzeug.datastructures import Headers
from werkzeug.exceptions import HTTPException

MAX_BYTES = 32 * 1024**2
routes = Blueprint("v1", __name__, url_prefix="/v1")


@lru_cache(maxsize=3)
def client(service: str) -> Any:
    if service == "s3":
        region = os.environ["AWS_REGION"]
        return boto3.client(
            "s3",
            region_name=region,
            endpoint_url=f"https://s3.{region}.amazonaws.com",
            config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}),
        )
    return boto3.client(service)


def uid(value: str) -> str:
    try:
        return uuid.UUID(value).hex
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError("Se requiere un identificador UUID.") from exc


def job_name(job: str) -> str:
    return f"{g.owner}-{uid(job)}"


def execution(job: str) -> str:
    return (
        os.environ["MACHINE_ARN"].replace(":stateMachine:", ":execution:")
        + ":"
        + job_name(job)
    )


def obj_head(key: str, version: str, bucket: str) -> dict:
    if not isinstance(version, str) or not 0 < len(version) < 1025:
        raise ValueError("Versión de archivo inválida.")
    return client("s3").head_object(Bucket=bucket, Key=key, VersionId=version)


@routes.post("/uploads")
def uploads() -> Any:
    b = request.get_json()
    if (
        not isinstance(b, dict)
        or type(b.get("size")) is not int
        or not 0 < b["size"] <= MAX_BYTES
        or not re.fullmatch("[a-f0-9]{64}", str(b.get("sha256", "")))
    ):
        raise ValueError("Tamaño o SHA-256 inválido.")
    key_id = uuid.uuid4().hex
    key = f"users/{g.owner}/uploads/{key_id}.parquet"
    meta = {
        "x-amz-meta-sha256": b["sha256"],
        "Content-Type": "application/octet-stream",
    }
    signed = client("s3").generate_presigned_post(
        Bucket=os.environ["JOBS_BUCKET"],
        Key=key,
        Fields={**meta, "success_action_status": "201"},
        Conditions=[
            {"x-amz-meta-sha256": b["sha256"]},
            {"Content-Type": "application/octet-stream"},
            {"success_action_status": "201"},
            ["content-length-range", b["size"], b["size"]],
        ],
        ExpiresIn=600,
    )
    return jsonify(upload_id=key_id, **signed), 201


@routes.get("/examples/default")
def example_info() -> Any:
    return jsonify(
        example="default", description="Noche pública Sleep Cassette de desarrollo."
    )


@routes.post("/predictions")
def start() -> Any:
    job = uid(request.headers.get("Idempotency-Key", ""))
    b = request.get_json()
    if not isinstance(b, dict):
        raise ValueError("Solicitud inválida.")
    example = b == {"example": "default"}
    if example:
        key, version, sha = (
            os.environ[k] for k in ["EXAMPLE_KEY", "EXAMPLE_VERSION", "EXAMPLE_SHA256"]
        )
    else:
        upload = uid(b.get("upload_id", ""))
        key = f"users/{g.owner}/uploads/{upload}.parquet"
        version = b.get("version_id", "")
        obj = obj_head(key, version, os.environ["JOBS_BUCKET"])
        if not 0 < obj["ContentLength"] <= MAX_BYTES:
            raise ValueError("Tamaño de archivo inválido.")
        sha = obj["Metadata"].get("sha256", "")
        if not re.fullmatch("[a-f0-9]{64}", sha):
            raise ValueError("Falta el checksum de la carga.")
    params = {
        "owner": g.owner,
        "job": job,
        "example": example,
        "input_key": key,
        "input_version": version,
        "sha256": sha,
    }
    payload = json.dumps(params, sort_keys=True, separators=(",", ":"))
    states = client("stepfunctions")
    try:
        states.start_execution(
            stateMachineArn=os.environ["MACHINE_ARN"] + ":live",
            name=job_name(job),
            input=payload,
        )
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "ExecutionAlreadyExists":
            raise
        existing = states.describe_execution(executionArn=execution(job))
        if existing["input"] != payload:
            return jsonify(error="Esta clave ya identifica otra solicitud."), 409
    return jsonify(job_id=job, status_url=f"/v1/predictions/{job}"), 202


@routes.get("/predictions/<job>")
def status(job: str) -> Any:
    job = uid(job)
    result = client("stepfunctions").describe_execution(executionArn=execution(job))
    if result["status"] == "RUNNING":
        return jsonify(job_id=job, state="processing")
    out = (
        json.loads(result["output"])
        if result["status"] == "SUCCEEDED"
        else {"state": "failed", "reason": "Error al procesar la noche."}
    )
    if out["state"] == "succeeded":
        bucket = os.environ["JOBS_BUCKET"]
        prefix = f"users/{g.owner}/results/{job}"
        try:
            client("s3").head_object(Bucket=bucket, Key=prefix + ".json")
        except ClientError as exc:
            if exc.response["Error"]["Code"] in ("404", "NoSuchKey"):
                return jsonify(error="Resultado expirado."), 410
            raise
        out["files"] = {
            ext: client("s3").generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": prefix + "." + ext},
                ExpiresIn=300,
            )
            for ext in ["json", "csv"]
        }
    return jsonify({**out, "job_id": job})


def create_app() -> Flask:
    app = Flask(__name__)
    app.register_blueprint(routes)

    @app.before_request
    def owner() -> Any:
        sub = request.environ.get("maia.subject")
        if not sub:
            return jsonify(error="Autenticación requerida."), 401
        g.owner = hashlib.sha256(sub.encode()).hexdigest()[:32]

    @app.errorhandler(ValueError)
    def invalid(exc: ValueError) -> Any:
        return jsonify(error=str(exc)), 400

    @app.errorhandler(HTTPException)
    def http_error(exc: HTTPException) -> Any:
        return jsonify(error=exc.name), exc.code

    @app.errorhandler(ClientError)
    def aws_error(exc: ClientError) -> Any:
        code = exc.response["Error"]["Code"]
        if code in (
            "404",
            "NoSuchKey",
            "NoSuchVersion",
            "ExecutionDoesNotExist",
            "AccessDenied",
        ):
            return jsonify(error="Recurso no disponible."), 404
        app.logger.error("AWS operation failed: %s", code)
        return jsonify(error="Servicio temporalmente no disponible."), 503

    return app


app = create_app()


def handler(event: dict, context: Any) -> dict:
    ctx = event["requestContext"]
    sub = ctx.get("authorizer", {}).get("jwt", {}).get("claims", {}).get("sub")
    body = event.get("body") or ""
    if event.get("isBase64Encoded"):
        body = base64.b64decode(body)
    with app.test_client() as api:
        res = api.open(
            event["rawPath"],
            method=ctx["http"]["method"],
            headers=Headers(event.get("headers", {})),
            data=body,
            environ_overrides={"maia.subject": sub},
        )
        return {
            "statusCode": res.status_code,
            "headers": {
                "content-type": "application/json",
                "cache-control": "no-store",
            },
            "body": res.get_data(as_text=True),
        }
