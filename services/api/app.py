"""HTTP API for private guest and Cognito job namespaces."""

import base64
import hashlib
import json
import os
import re
import uuid
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from flask import (
    Blueprint,
    Flask,
    Response,
    current_app,
    g,
    jsonify,
    request,
    url_for,
)
from werkzeug.datastructures import Headers
from werkzeug.exceptions import HTTPException, ServiceUnavailable

from sleep_staging.sensor_profiles import (
    MAX_EDF_BYTES,
    MAX_PARQUET_BYTES,
    validate_edf_profile,
    validate_lgbm_profile,
)
from sleep_staging.stack_release import EXAMPLE_SHA256, MODEL, VERSION

MAX_BYTES = MAX_PARQUET_BYTES
SPEC = Path(__file__).with_name("openapi.json")
REQUIRED_ENV = (
    "AWS_REGION",
    "JOBS_BUCKET",
    "MACHINE_ARN",
    "EXAMPLE_KEY",
    "EXAMPLE_VERSION",
    "EXAMPLE_SHA256",
)
SWAGGER = "https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.17.14"
DOCS = f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>API de estadificación del sueño</title>
<link rel="stylesheet" href="{SWAGGER}/swagger-ui.css" crossorigin="anonymous"
  integrity="sha384-wxLW6kwyHktdDGr6Pv1zgm/VGJh99lfUbzSn6HNHBENZlCN7W602k9VkGdxuFvPn">
</head>
<body>
<div id="docs"></div>
<script src="{SWAGGER}/swagger-ui-bundle.js" crossorigin="anonymous"
  integrity="sha384-wmyclcVGX/WhUkdkATwhaK1X1JtiNrr2EoYJ+diV3vj4v6OC5yCeSu+yW13SYJep"></script>
<script>SwaggerUIBundle({{url: "openapi.json", dom_id: "#docs"}});</script>
</body>
</html>
"""
routes = Blueprint("v1", __name__, url_prefix="/v1")
# Unauthenticated operational endpoints, outside the versioned job API.
service = Blueprint("service", __name__)


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


def upload_contract(body: dict) -> tuple[str, int, str | None]:
    if os.getenv("ACTIVE_EDF_MODEL") == MODEL:
        if body.get("format") != "edf" or body.get("model", MODEL) != MODEL:
            raise ValueError("El modelo publicado requiere una polisomnografía EDF.")
        return "edf", MAX_EDF_BYTES, validate_lgbm_profile(body.get("profile", "auto"))
    fmt = body.get("format", "parquet")
    if fmt == "parquet":
        if body.get("model", "lightgbm") != "lightgbm" or "profile" in body:
            raise ValueError("Modelo o perfil incompatible con Parquet.")
        return fmt, MAX_BYTES, None
    if fmt != "edf":
        raise ValueError("Formato de entrada desconocido.")
    model = body.get("model", "fullnight")
    requested = body.get("profile", "auto")
    if model == "lightgbm":
        profile = validate_lgbm_profile(requested)
        enabled = os.getenv("EDF_LGBM_ENABLED")
    elif model == "fullnight":
        profile = validate_edf_profile(requested)
        enabled = os.getenv("EDF_ENABLED")
    else:
        raise ValueError("Modelo EDF desconocido.")
    if enabled != "1":
        raise ServiceUnavailable(
            "El modelo EDF seleccionado aún no está publicado en este entorno."
        )
    return fmt, MAX_EDF_BYTES, profile


@routes.post("/uploads")
def uploads() -> Any:
    b = request.get_json()
    if not isinstance(b, dict):
        raise ValueError("Solicitud inválida.")
    fmt, limit, _ = upload_contract(b)
    if (
        type(b.get("size")) is not int
        or not 0 < b["size"] <= limit
        or not re.fullmatch("[a-f0-9]{64}", str(b.get("sha256", "")))
    ):
        raise ValueError("Tamaño o SHA-256 inválido.")
    key_id = uuid.uuid4().hex
    key = f"users/{g.owner}/uploads/{key_id}.{fmt}"
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
    if os.getenv("ACTIVE_EDF_MODEL") == MODEL:
        return jsonify(
            example="default",
            model=MODEL,
            model_version=VERSION,
            format="edf",
            description="Polisomnografía sintética de 10 minutos con cuatro canales.",
        )
    return jsonify(
        example="default", description="Noche pública Sleep Cassette de desarrollo."
    )


@routes.post("/predictions")
def start() -> Any:
    job = uid(request.headers.get("Idempotency-Key", ""))
    b = request.get_json()
    if not isinstance(b, dict):
        raise ValueError("Solicitud inválida.")
    hybrid = os.getenv("ACTIVE_EDF_MODEL") == MODEL
    example = b == {"example": "default"} or (
        hybrid and b == {"example": "default", "model": MODEL}
    )
    if example and hybrid:
        key, version, sha = "bundled/example.edf", VERSION, EXAMPLE_SHA256
        fmt, _, profile = upload_contract({"format": "edf", "model": MODEL})
    elif example:
        key, version, sha = (
            os.environ[k] for k in ["EXAMPLE_KEY", "EXAMPLE_VERSION", "EXAMPLE_SHA256"]
        )
    else:
        fmt, limit, profile = upload_contract(b)
        upload = uid(b.get("upload_id", ""))
        key = f"users/{g.owner}/uploads/{upload}.{fmt}"
        version = b.get("version_id", "")
        obj = obj_head(key, version, os.environ["JOBS_BUCKET"])
        if not 0 < obj["ContentLength"] <= limit:
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
    if hybrid:
        params.update(format="edf", model=MODEL, profile=profile)
    elif not example and fmt == "edf":
        params.update(format=fmt, profile=profile)
        if b.get("model") == "lightgbm":
            params["model"] = "lightgbm"
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
        original = json.loads(existing["input"])
        same_example = example and original.get("example") is True
        if original != params and not same_example:
            return jsonify(error="Esta clave ya identifica otra solicitud."), 409
    return jsonify(
        job_id=job, status_url=url_for(f"{request.blueprint}.status", job=job)
    ), 202


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


def features() -> dict[str, bool]:
    # While the published stack is active it replaces every other input.
    stack = os.getenv("ACTIVE_EDF_MODEL") == MODEL
    return {
        "parquet": not stack,
        "edf_fullnight": not stack and os.getenv("EDF_ENABLED") == "1",
        "edf_lightgbm": not stack and os.getenv("EDF_LGBM_ENABLED") == "1",
        "edf_stack_five": stack,
    }


@service.get("/health")
def health() -> Any:
    # Liveness only: it answers whenever the function runs and never calls AWS.
    return jsonify(
        status="ok",
        service="maia-sleep-api",
        version=os.getenv("API_VERSION", "local"),
        features=features(),
    )


def storage_ready() -> bool:
    versioning = client("s3").get_bucket_versioning(Bucket=os.environ["JOBS_BUCKET"])
    # Jobs pin uploads by S3 version ID, so an unversioned bucket cannot serve them.
    return versioning.get("Status") == "Enabled"


def workflow_ready() -> bool:
    client("stepfunctions").describe_state_machine_alias(
        stateMachineAliasArn=os.environ["MACHINE_ARN"] + ":live"
    )
    return True


def probe(name: str, check: Callable[[], bool]) -> str:
    try:
        return "ok" if check() else "error"
    except (BotoCoreError, ClientError) as exc:
        code = (
            exc.response["Error"]["Code"]
            if isinstance(exc, ClientError)
            else type(exc).__name__
        )
        current_app.logger.error("Readiness check %s failed: %s", name, code)
        return "error"


@service.get("/health/ready")
def ready() -> Any:
    if all(os.getenv(name) for name in REQUIRED_ENV):
        checks = {
            "config": "ok",
            "storage": probe("storage", storage_ready),
            "workflow": probe("workflow", workflow_ready),
        }
    else:
        checks = {"config": "error", "storage": "skipped", "workflow": "skipped"}
    ok = all(state == "ok" for state in checks.values())
    return jsonify(status="ready" if ok else "unavailable", checks=checks), (
        200 if ok else 503
    )


@service.get("/openapi.json")
def openapi() -> Response:
    return Response(SPEC.read_bytes(), mimetype="application/json")


@service.get("/docs")
def docs() -> Response:
    return Response(DOCS, mimetype="text/html")


def create_app() -> Flask:
    app = Flask(__name__)
    app.register_blueprint(service)
    app.register_blueprint(routes)
    app.register_blueprint(routes, name="guest", url_prefix="/public/v1")

    @app.before_request
    def owner() -> Any:
        if request.blueprint == "service":
            return None
        if request.blueprint == "guest":
            if request.endpoint == "guest.example_info":
                return None
            token = request.headers.get("X-Guest-Token", "")
            if not re.fullmatch(r"[a-f0-9]{64}", token):
                return jsonify(
                    error="Se requiere una clave temporal del navegador."
                ), 401
            # A job ID is safe to display. Only the independent guest secret grants access.
            g.owner = hashlib.sha256(("guest:" + token).encode()).hexdigest()[:32]
            return None
        sub = request.environ.get("maia.subject")
        if not sub:
            return jsonify(error="Autenticación requerida."), 401
        g.owner = hashlib.sha256(sub.encode()).hexdigest()[:32]

    @app.errorhandler(ValueError)
    def invalid(exc: ValueError) -> Any:
        return jsonify(error=str(exc)), 400

    @app.errorhandler(HTTPException)
    def http_error(exc: HTTPException) -> Any:
        return jsonify(
            error=exc.description if isinstance(exc, ServiceUnavailable) else exc.name
        ), exc.code

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
                "content-type": res.content_type,
                "cache-control": "no-store",
            },
            "body": res.get_data(as_text=True),
        }
