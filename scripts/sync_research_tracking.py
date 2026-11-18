"""Expose cached inner trials, the fixed ensemble and the candidate in MLflow."""

import argparse
import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from sleep_staging.config import CLASSES, project_root
from sleep_staging.evaluation import evaluate, scalar_metrics
from sleep_staging.provenance import fingerprint, hash_file, write_json
from sleep_staging.research import research_views
from sleep_staging.research_tracking import ensemble_identity, validated_candidate


def sync_tracking(root: Path) -> dict:
    root = root.resolve()
    campaign = json.loads((root / "campaign.json").read_text())
    bundle = validated_candidate(campaign, root)
    board = pd.read_csv(root / "leaderboard.csv")
    summaries = {
        row["name"]: json.loads((Path(row["run_dir"]) / "summary.json").read_text())
        for row in board.to_dict("records")
    }
    uri = f"sqlite:///{root / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    client = MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name("sleep-cassette-literature-v2")
    if experiment is None:
        raise ValueError("Research experiment not found.")
    experiment_id = experiment.experiment_id
    trials = []
    tuned = summaries.get("catboost_tuned_context")
    if tuned:
        for path in sorted(Path(tuned["run_dir"]).glob("fold-*.json")):
            fold = json.loads(path.read_text())
            for trial in fold["training"]["trials"]:
                identity = (
                    f"{tuned['mlflow_run_id']}-{fold['fold']}-{trial['candidate']}"
                )
                found = client.search_runs(
                    [experiment_id],
                    filter_string=f"tags.research_trial_id = '{identity}' AND attributes.status = 'FINISHED'",
                )
                if found:
                    trials.append(found[0].info.run_id)
                    continue
                run = client.create_run(
                    experiment_id,
                    tags={
                        "mlflow.runName": f"inner-fold-{fold['fold']}-candidate-{trial['candidate']}",
                        "mlflow.parentRunId": tuned["mlflow_run_id"],
                        "research_trial_id": identity,
                        "purpose": "inner_subject_tuning",
                        "test_evaluated": "false",
                    },
                )
                run_id = run.info.run_id
                params = {
                    **trial["params"],
                    "maximum_iterations": tuned["identity"]["tuning"][
                        trial["candidate"]
                    ]["iterations"],
                    "selected_iterations": trial["params"]["iterations"],
                    "outer_fold": fold["fold"],
                    "candidate": trial["candidate"],
                    "seed": 42 + fold["fold"],
                }
                for key, value in params.items():
                    client.log_param(run_id, key, str(value))
                for key in ["f1_macro", "accuracy", "kappa", "elapsed_seconds"]:
                    client.log_metric(run_id, key, float(trial[key]))
                client.log_artifact(run_id, str(path), artifact_path="outer_fold_audit")
                client.set_terminated(run_id, status="FINISHED")
                trials.append(run_id)

    ensemble = root / "fixed-ensemble"
    ensemble.mkdir(exist_ok=True)
    components = ["lightgbm_context_offline", "catboost_context_offline"]
    component_identity = []
    for name in components:
        summary = summaries[name]
        oof_path = Path(summary["run_dir"]) / "oof.parquet"
        if hash_file(oof_path) != summary["oof_sha256"]:
            raise ValueError("Ensemble OOF checksum mismatch.")
        component_identity.append(
            {
                "name": name,
                "run_id": summary["mlflow_run_id"],
                "oof_sha256": summary["oof_sha256"],
            }
        )
    ensemble_key = ensemble_identity(component_identity, [0.5, 0.5])
    a, b = [
        pd.read_parquet(Path(summaries[name]["run_dir"]) / "oof.parquet")
        .sort_values("epoch_id")
        .reset_index(drop=True)
        for name in components
    ]
    if not a[["epoch_id", "subject_id", "y_true", "fold"]].equals(
        b[["epoch_id", "subject_id", "y_true", "fold"]]
    ):
        raise ValueError("Ensemble components have different OOF identities.")
    classes = CLASSES["task_5"]
    columns = ["prob_" + c for c in classes]
    probability = 0.5 * a[columns].to_numpy() + 0.5 * b[columns].to_numpy()
    oof = a[
        ["epoch_id", "subject_id", "recording_id", "onset_s", "fold", "y_true"]
    ].copy()
    oof["y_pred"] = np.asarray(classes)[probability.argmax(axis=1)]
    oof[columns] = probability
    oof.to_parquet(ensemble / "oof.parquet", index=False)
    folds = []
    for fold, group in oof.groupby("fold", sort=True):
        metrics = evaluate(group.y_true.to_numpy(), group.y_pred.to_numpy(), classes)
        folds.append({"fold": int(fold), **scalar_metrics(metrics)})
    ensemble_summary: dict[str, Any] = {
        "name": "soft_vote_fixed_50_50",
        "components": components,
        "weights": [0.5, 0.5],
        "evaluation_only": True,
        "f1_macro": float(np.mean([f["f1_macro"] for f in folds])),
        "f1_std": float(np.std([f["f1_macro"] for f in folds])),
        "accuracy": float(np.mean([f["accuracy"] for f in folds])),
        "kappa": float(np.mean([f["kappa"] for f in folds])),
        "folds": folds,
    }
    write_json(ensemble / "summary.json", ensemble_summary)
    write_json(
        ensemble / "pooled_metrics.json",
        evaluate(oof.y_true.to_numpy(), oof.y_pred.to_numpy(), classes),
    )
    found = client.search_runs(
        [experiment_id],
        filter_string=f"tags.research_ensemble_id = '{ensemble_key}' AND attributes.status = 'FINISHED'",
    )
    if found:
        ensemble_id = found[0].info.run_id
    else:
        run = client.create_run(
            experiment_id,
            tags={
                "mlflow.runName": "soft_vote_fixed_50_50",
                "mlflow.parentRunId": campaign["mlflow_parent_run_id"],
                "research_ensemble_id": ensemble_key,
                "purpose": "fixed_oof_ensemble_evaluation",
                "test_evaluated": "false",
            },
        )
        ensemble_id = run.info.run_id
        for key, value in {
            "weights": "0.5,0.5",
            "components": ",".join(components),
            "evaluation_only": True,
            "outer_folds": 5,
        }.items():
            client.log_param(ensemble_id, key, str(value))
        for key in ["f1_macro", "f1_std", "accuracy", "kappa"]:
            client.log_metric(ensemble_id, key, ensemble_summary[key])
        for row in folds:
            for key in ["f1_macro", "accuracy", "kappa"]:
                client.log_metric(ensemble_id, f"fold_{row['fold']}_{key}", row[key])
        for path in ensemble.iterdir():
            client.log_artifact(ensemble_id, str(path), artifact_path="evaluation")
        client.set_terminated(ensemble_id, status="FINISHED")

    frame, views = research_views(pd.read_parquet(campaign["dataset"]))
    example = views[bundle["spec"]["feature_view"]].iloc[:5].copy()
    package = project_root() / "src/sleep_staging"
    protocol = {
        "view": bundle["spec"]["feature_view"],
        "wake_margin_minutes": 30,
        "offline": True,
        "source": {
            name: hash_file(package / name)
            for name in [
                "features/research.py",
                "features/extract.py",
                "research_data.py",
                "data/edf.py",
                "data/labels.py",
            ]
        },
    }
    protocol_id = fingerprint(protocol)
    example.attrs["preprocessing_id"] = protocol_id
    example.to_parquet(root / "candidate-example-input.parquet", index=False)
    prediction = np.asarray(bundle["model"].predict(example)).reshape(-1)
    candidate_identity = campaign["mlflow_parent_run_id"]
    found = client.search_runs(
        [experiment_id],
        filter_string=f"tags.research_candidate_id = '{candidate_identity}' AND attributes.status = 'FINISHED'",
    )
    if found:
        model_uri = found[0].data.tags["model_uri"]
        registry_uri = found[0].data.tags["registry_uri"]
    else:
        with mlflow.start_run(
            experiment_id=experiment_id, run_name="candidate_refit_full_development"
        ) as run:
            mlflow.set_tags(
                {
                    "mlflow.parentRunId": campaign["mlflow_parent_run_id"],
                    "research_candidate_id": candidate_identity,
                    "purpose": "candidate_full_development_refit",
                    "test_evaluated": "false",
                }
            )
            mlflow.log_params(
                {
                    "selected_model": campaign["winner"]["name"],
                    "subjects": campaign["subjects"],
                    "epochs": campaign["epochs"],
                    "features": example.shape[1],
                }
            )
            mlflow.log_metrics(
                {
                    "selected_cv_" + k: campaign["winner"][k]
                    for k in ["f1_macro", "accuracy", "kappa"]
                }
            )
            model = mlflow.sklearn.log_model(
                bundle["model"],
                name="candidate",
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
                signature=infer_signature(example, prediction),
                input_example=example,
                pip_requirements=[
                    f"{p}=={version(p)}"
                    for p in [
                        "mlflow",
                        "scikit-learn",
                        "numpy",
                        "pandas",
                        "scipy",
                        "lightgbm",
                        "catboost",
                        "joblib",
                        "cloudpickle",
                    ]
                ],
                metadata={
                    "feature_columns": bundle["feature_columns"],
                    "preprocessing_id": protocol_id,
                    "preprocessing": protocol,
                    "task": "task_5",
                    "channels": ["eeg_fpz_cz", "eeg_pz_oz", "eog", "emg"],
                    "input_kind": "features",
                    "feature_view": bundle["spec"]["feature_view"],
                    "requires_offline_feature_engineering": True,
                    "test_evaluated": False,
                },
            )
            registered = mlflow.register_model(
                model.model_uri, "sleep_sc_task5_literature_v2"
            )
            client.set_registered_model_alias(
                "sleep_sc_task5_literature_v2", "candidate", registered.version
            )
            model_uri, registry_uri = (
                model.model_uri,
                "models:/sleep_sc_task5_literature_v2@candidate",
            )
            mlflow.set_tags({"model_uri": model_uri, "registry_uri": registry_uri})
            mlflow.log_artifact(str(root / "campaign.json"))
            mlflow.log_artifact(str(root / "candidate.joblib"))
            mlflow.log_artifact(str(root / "candidate-example-input.parquet"))
    reloaded = mlflow.sklearn.load_model(model_uri)
    if not np.array_equal(
        np.asarray(reloaded.predict(example)).reshape(-1), prediction
    ):
        raise ValueError("Reloaded model predictions differ.")
    probability = reloaded.predict_proba(example)
    np.testing.assert_allclose(probability.sum(axis=1), 1, atol=1e-6)
    validation = {
        "model_uri": model_uri,
        "registry_uri": registry_uri,
        "input_rows": len(example),
        "input_features": example.shape[1],
        "predictions_equal_after_reload": True,
        "probabilities_sum_to_one": True,
        "candidate_sha256": hash_file(root / "candidate.joblib"),
        "preprocessing_id": protocol_id,
        "test_evaluated": False,
    }
    write_json(root / "artifact-validation.json", validation)
    audit = {
        "experiment_id": experiment_id,
        "experiment_name": experiment.name,
        "url": "http://127.0.0.1:5002",
        "model_runs": {n: s["mlflow_run_id"] for n, s in summaries.items()},
        "inner_trial_runs": trials,
        "fixed_ensemble_run": ensemble_id,
        **validation,
    }
    write_json(root / "tracking-audit.json", audit)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", type=Path, default=Path("var/research/sc-w30-v2")
    )
    args = parser.parse_args()
    print(json.dumps(sync_tracking(args.output_root), indent=2))


if __name__ == "__main__":
    main()
