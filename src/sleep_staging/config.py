"""Contratos de configuración; rutas relativas siempre a la raíz del proyecto."""

import os
from pathlib import Path
from typing import Literal, Self

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, model_validator

CHANNELS = {"eeg_fpz_cz", "eeg_pz_oz", "eog", "emg"}
CLASSES = {
    "task_5": ["W", "N1", "N2", "N3", "REM"],
    "task_6": ["W", "N1", "N2", "N3", "REM", "M"],
}


def project_root() -> Path:
    if value := os.environ.get("SLEEP_STAGING_ROOT"):
        return Path(value).resolve()
    for directory in [Path.cwd(), *Path.cwd().parents]:
        if (directory / "pyproject.toml").exists():
            return directory
    raise ValueError("Ejecuta desde el proyecto o define SLEEP_STAGING_ROOT.")


def channel_profiles() -> dict[str, list[str]]:
    profiles = {}
    for eeg in [["eeg_fpz_cz"], ["eeg_pz_oz"], ["eeg_fpz_cz", "eeg_pz_oz"]]:
        for extra in [[], ["eog"], ["emg"], ["eog", "emg"]]:
            channels = eeg + extra
            profiles["+".join(channels)] = channels
    return profiles


class LocalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tracking_uri: str = "http://127.0.0.1:5001"
    workers: int = Field(default=2, ge=1, le=2)


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(pattern=r"^[a-z0-9-]+$")
    source: Literal["sleep-cassette", "sleep-telemetry"]
    records: list[str] = Field(default_factory=list)


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "sleep-staging"
    data_config: str = "configs/data/pilot.yaml"
    channels: list[str] = Field(default_factory=lambda: ["eeg_fpz_cz"])
    task: Literal["task_5", "task_6"] = "task_6"
    models: list[Literal["dummy", "rf", "cnn"]] = ["dummy", "rf"]
    seed: int = 42
    n_splits: int = Field(default=5, ge=2)
    test_fraction: float = Field(default=0.2, gt=0, lt=0.5)
    workers: int = Field(default=2, ge=1, le=2)
    eeg_low: float = Field(default=0.3, gt=0)
    eeg_high: float = Field(default=35, gt=0, lt=50)
    rf_trees: int = Field(default=64, ge=1)
    cnn_epochs: int = Field(default=3, ge=1)
    batch_size: int = Field(default=32, ge=1)
    device: Literal["cpu", "mps", "auto"] = "cpu"
    max_epochs_per_recording: int | None = Field(default=None, ge=6)
    wake_margin_minutes: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_channels(self) -> Self:
        if not set(self.channels) & {"eeg_fpz_cz", "eeg_pz_oz"}:
            raise ValueError("El perfil requiere al menos un EEG.")
        if (
            len(set(self.channels)) != len(self.channels)
            or set(self.channels) - CHANNELS
        ):
            raise ValueError("Canales desconocidos o duplicados.")
        if self.eeg_low >= self.eeg_high:
            raise ValueError("La banda EEG debe tener low < high.")
        if not self.models or len(set(self.models)) != len(self.models):
            raise ValueError("Define modelos únicos y al menos uno.")
        if "cnn" in self.models and self.channels not in [
            ["eeg_fpz_cz"],
            ["eeg_pz_oz"],
        ]:
            raise ValueError("La primera CNN admite exactamente un EEG.")
        return self


def read_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Se esperaba un objeto YAML: {path}")
    return value


def local_config() -> LocalConfig:
    root = project_root()
    load_dotenv(root / ".env", override=False)
    path = root / "configs/local.yaml"
    values = read_yaml(path) if path.exists() else {}
    if uri := os.environ.get("MLFLOW_TRACKING_URI"):
        values["tracking_uri"] = uri
    return LocalConfig.model_validate(values)


def experiment_config(path: Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(read_yaml(path))


def data_config(path: Path) -> DataConfig:
    return DataConfig.model_validate(read_yaml(path))
