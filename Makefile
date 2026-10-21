.PHONY: install init-dirs mlflow doctor smoke test check notebook backup

install:
	uv sync --locked --group dev --group notebook --extra dl

init-dirs:
	uv run --locked sleep-stage init-dirs

mlflow:
	uv run --locked python scripts/mlflow_service.py

doctor:
	uv run --locked sleep-stage doctor --check-mlflow

smoke:
	uv run --locked sleep-stage download --config configs/data/pilot.yaml
	uv run --locked sleep-stage experiment --config configs/experiments/smoke.yaml

test:
	uv run --locked pytest tests -q

check:
	uv run --locked ruff format --check src tests scripts
	uv run --locked ruff check src tests scripts
	uv run --locked mypy src/sleep_staging scripts

notebook:
	uv run --locked --group notebook jupyter lab --ip 127.0.0.1 --no-browser

backup:
	uv run --locked python scripts/backup_mlflow.py
