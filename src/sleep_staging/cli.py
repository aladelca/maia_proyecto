"""CLI local; ningún comando descarga datos de forma implícita al entrenar."""

import json
import platform
import shutil
from pathlib import Path
from typing import Annotated

import pandas as pd
import psutil
import typer

from sleep_staging.config import (
    DataConfig,
    ExperimentConfig,
    data_config,
    experiment_config,
    local_config,
    project_root,
    read_yaml,
)
from sleep_staging.data.catalog import audit_dataset
from sleep_staging.data.download import download_dataset
from sleep_staging.data.preprocess import extract_dataset_features, prepare_dataset
from sleep_staging.experiments import create_splits, run_experiment
from sleep_staging.inference import predict_features
from sleep_staging.provenance import atomic_path, environment
from sleep_staging.tracking import require_tracking

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
ConfigOption = Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)]


def configurations(path: Path) -> tuple[DataConfig, ExperimentConfig]:
    root = project_root()
    if "source" in read_yaml(path):
        return data_config(path), ExperimentConfig(data_config=str(path.resolve()))
    cfg = experiment_config(path)
    cfg.workers = min(cfg.workers, local_config().workers)
    return data_config(root / cfg.data_config), cfg


@app.command("init-dirs")
def init_dirs() -> None:
    root = project_root()
    for relative in [
        "data/manifests",
        "data/raw",
        "data/interim",
        "data/processed",
        "data/splits",
        "var/mlflow/artifacts",
        "var/reports",
        "var/backups",
    ]:
        (root / relative).mkdir(parents=True, exist_ok=True)
    typer.echo(str(root))


@app.command()
def doctor(check_mlflow: bool = False) -> None:
    root = project_root()
    cfg = local_config()
    try:
        import torch

        mps = torch.backends.mps.is_available()
    except ImportError:
        mps = False
    typer.echo(
        json.dumps(
            {
                "root": str(root),
                "python": platform.python_version(),
                "machine": platform.machine(),
                "ram_gib": round(psutil.virtual_memory().total / 1024**3, 1),
                "free_disk_gib": round(shutil.disk_usage(root).free / 1024**3, 1),
                "mps_available": mps,
                "tracking_uri": cfg.tracking_uri,
                "workers": cfg.workers,
                "versions": environment(),
            },
            indent=2,
        )
    )
    if check_mlflow:
        try:
            require_tracking(cfg.tracking_uri)
        except ValueError as error:
            typer.echo(str(error), err=True)
            raise typer.Exit(1) from error
        typer.echo("MLflow disponible.")


@app.command()
def download(config: ConfigOption) -> None:
    data, _ = configurations(config)
    typer.echo(str(download_dataset(data, project_root())))


@app.command()
def audit(config: ConfigOption) -> None:
    data, _ = configurations(config)
    typer.echo(json.dumps(audit_dataset(data, project_root()), indent=2))


@app.command()
def prepare(config: ConfigOption) -> None:
    data, cfg = configurations(config)
    typer.echo(str(prepare_dataset(data, cfg, project_root())))


@app.command()
def features(config: ConfigOption) -> None:
    data, cfg = configurations(config)
    root = project_root()
    prepared = prepare_dataset(data, cfg, root)
    typer.echo(str(extract_dataset_features(prepared, cfg, root)))


@app.command()
def split(config: ConfigOption) -> None:
    data, cfg = configurations(config)
    root = project_root()
    typer.echo(str(create_splits(prepare_dataset(data, cfg, root), cfg, root)))


@app.command()
def experiment(config: ConfigOption) -> None:
    data, cfg = configurations(config)
    result = run_experiment(data, cfg, project_root(), local_config().tracking_uri)
    typer.echo(result["leaderboard"].to_string(index=False))
    typer.echo(f"Run: {result['run_id']}\nModel URI: {result['model_uri']}")


@app.command()
def report(
    run_id: str | None = None,
    config: Annotated[Path | None, typer.Option("--config")] = None,
) -> None:
    from mlflow.tracking import MlflowClient

    root = project_root()
    if not run_id:
        if config is None:
            raise typer.BadParameter("Indica --run-id o --config.")
        data, cfg = configurations(config)
        uri = local_config().tracking_uri
        require_tracking(uri)
        client = MlflowClient(tracking_uri=uri)
        found = client.get_experiment_by_name(cfg.name)
        if found is None:
            raise typer.BadParameter("No hay experimentos con ese nombre.")
        runs = client.search_runs(
            [found.experiment_id],
            filter_string=f"tags.dataset = '{data.name}' AND attributes.status = 'FINISHED'",
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        if not runs:
            raise typer.BadParameter("No hay corridas completadas.")
        run_id = runs[0].info.run_id
    typer.echo((root / "var/reports" / run_id / "summary.json").read_text())


@app.command()
def predict(
    model_uri: Annotated[str, typer.Option("--model-uri")],
    input: Annotated[Path, typer.Option("--input", exists=True)],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    result = predict_features(
        model_uri, pd.read_parquet(input), local_config().tracking_uri
    )
    with atomic_path(output) as temp:
        result.to_parquet(temp, index=False)
    typer.echo(f"{len(result)} predicciones: {output}")
