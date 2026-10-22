import json

from typer.testing import CliRunner

from sleep_staging.cli import app


def test_doctor_reports_local_environment_and_missing_server(monkeypatch, tmp_path):
    monkeypatch.setenv("SLEEP_STAGING_ROOT", str(tmp_path))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:1")
    runner = CliRunner()
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["root"] == str(tmp_path)
    failed = runner.invoke(app, ["doctor", "--check-mlflow"])
    assert failed.exit_code == 1
    assert "MLflow" in failed.output


def test_prediction_cli_exposes_documented_options():
    result = CliRunner().invoke(app, ["predict", "--help"])
    assert result.exit_code == 0
    assert "--model-uri" in result.output
    assert "--output" in result.output
