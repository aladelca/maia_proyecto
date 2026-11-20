"""Summarize paired W30 predictions and register them in both historical MLflow stores."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from compare_fullnight_lgbm import (
    ARMS,
    BUNDLE,
    CAMPAIGN,
    OUT,
    PROBS,
    RESEARCH,
    ROOT,
    SIX,
    SPLIT,
    read_json,
)
from filelock import FileLock
from mlflow import MlflowClient
from sklearn.metrics import confusion_matrix

from sleep_staging.provenance import environment, fingerprint, hash_file, write_json
from sleep_staging.serving.contract import LABELS

NATIVE = "fullnight_native"
STORES = {
    "lightgbm": (
        RESEARCH / "sc-w30-v2/mlflow.db",
        "sleep-cassette-lgbm-bilstm-refinement-v1",
    ),
    "fullnight": (RESEARCH / "ibio-fullnight/tracking/mlflow.db", "ibio-fullnight"),
}
REPORT = ROOT / "output/evidencia/resumen/fullnight-lgbm-w30"


def from_counts(m: np.ndarray) -> dict:
    # Six prediction columns retain native M decisions as errors; reference rows are five stages.
    n = m.sum()
    d = np.diag(m)
    row, col = m.sum(1), m.sum(0)[:5]
    f1 = np.divide(2 * d, row + col, out=np.zeros(5), where=(row + col) > 0)
    agree = (row @ col) / n**2
    return {
        "macro_f1": float(f1.mean()),
        "accuracy": float(d.sum() / n),
        "kappa": float((d.sum() / n - agree) / (1 - agree)) if agree < 1 else 0.0,
        **{"f1_" + label: float(v) for label, v in zip(LABELS, f1, strict=True)},
        "movement_calls": int(m[:, 5].sum()),
        "epochs": int(n),
    }


def assessments(df: pd.DataFrame, cohort: str, metrics: list, matrices: dict) -> dict:
    subjects = sorted(df.subject_id.unique())
    true = pd.Categorical(df.task_5, categories=LABELS).codes
    people = df.subject_id.to_numpy()
    all_counts = {}
    for name in [*ARMS, NATIVE]:
        cols = (
            ["six_direct_" + c for c in SIX]
            if name == NATIVE
            else [name + "_" + c for c in PROBS]
        )
        pred = df[cols].to_numpy().argmax(1)
        counts = np.stack(
            [
                confusion_matrix(true[people == s], pred[people == s], labels=range(6))[
                    :5
                ]
                for s in subjects
            ]
        )
        total = counts.sum(0)
        metrics.append(
            {
                "sample": cohort,
                "arm": name,
                "subjects": len(subjects),
                "nights": df.recording_id.nunique(),
                **from_counts(total),
            }
        )
        matrices[cohort + "/" + name] = {
            "rows": LABELS,
            "columns": [*LABELS, "M"],
            "counts": total.tolist(),
        }
        all_counts[name] = counts
    return all_counts


def bootstrap(counts: dict, sample: str) -> list[dict]:
    # Reuse the same participant draws for every paired contrast and both metrics.
    n = len(counts[ARMS[0]])
    draws = np.random.default_rng(42).integers(0, n, size=(2000, n))
    intervals = []
    for arm, base in [(a, ARMS[0]) for a in ARMS[1:]] + [(ARMS[-1], ARMS[-2])]:
        for metric in ["macro_f1", "f1_N1"]:
            a, b = counts[arm], counts[base]
            differences = [
                from_counts(x)[metric] - from_counts(y)[metric]
                for x, y in zip(a[draws].sum(1), b[draws].sum(1), strict=True)
            ]
            lo, hi = np.quantile(differences, [0.025, 0.975])
            intervals.append(
                {
                    "sample": sample,
                    "candidate": arm,
                    "reference": base,
                    "metric": metric,
                    "delta": from_counts(a.sum(0))[metric]
                    - from_counts(b.sum(0))[metric],
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "resamples": len(draws),
                    "subjects": n,
                }
            )
    return intervals


def report(out: Path, dest: Path) -> pd.DataFrame:
    dest.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    matrices: dict = {}
    intervals = []
    folds = []
    for f in range(5):
        fold = pd.read_parquet(out / "predictions" / f"fold-{f}.parquet")
        if not fold.fold.eq(f).all():
            raise ValueError("Mismatched outer fold")
        assessments(fold, f"fold-{f}", rows, matrices)
        folds.append(fold)
    dev, test = pd.concat(folds), pd.read_parquet(out / "predictions/reserved.parquet")
    if (
        dev.epoch_id.duplicated().any()
        or test.epoch_id.duplicated().any()
        or set(dev.subject_id) & set(test.subject_id)
        or not test.fold.eq(-1).all()
    ):
        raise ValueError("Invalid pooled evaluation assignments")
    whole = pd.read_parquet(out / "predictions/whole_edf.parquet")
    common = ["epoch_id", "subject_id", "task_5"]
    if whole.epoch_id.duplicated().any() or not whole[whole.w30][common].sort_values(
        "epoch_id"
    ).reset_index(drop=True).equals(
        test[common].sort_values("epoch_id").reset_index(drop=True)
    ):
        raise ValueError(
            "Whole-EDF W30 labels or epochs differ from the historical denominator"
        )
    for name, data in [
        ("development", dev),
        ("reserved", test),
        ("whole_edf", whole),
        ("whole_edf_w30", whole[whole.w30]),
    ]:
        intervals.extend(bootstrap(assessments(data, name, rows, matrices), name))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(dest / "metrics.csv", index=False)
    pd.DataFrame(intervals).to_csv(dest / "paired_intervals.csv", index=False)
    write_json(dest / "confusion.json", matrices)
    foldrows = metrics[metrics["sample"].str.startswith("fold-")]
    summary = (
        foldrows.groupby("arm")
        .macro_f1.agg(cv_macro_f1="mean", cv_sd=lambda folds: folds.std(ddof=0))
        .join(
            metrics[metrics["sample"].eq("reserved")]
            .set_index("arm")[["macro_f1", "accuracy", "f1_N1", "kappa"]]
            .add_prefix("test_")
        )
        .join(
            metrics[metrics["sample"].eq("whole_edf")]
            .set_index("arm")[["macro_f1", "f1_N1"]]
            .add_prefix("edf_")
        )
        .sort_values("cv_macro_f1", ascending=False)
    )
    historical = read_json(RESEARCH / "sc-w30-refinement-v1/lgbm.json")
    np.testing.assert_allclose(
        summary.loc["lightgbm_release", "cv_macro_f1"],
        historical["f1_macro"],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        summary.loc["lightgbm_release", "cv_sd"], historical["f1_std"], atol=1e-12
    )
    summary.to_csv(dest / "leaderboard.csv")
    return summary


def sync(out: Path, dest: Path, store: str, summary: pd.DataFrame) -> dict:
    db, name = STORES[store]
    if not db.is_file():
        raise FileNotFoundError(f"Restore historical MLflow store {db}")
    client = MlflowClient(tracking_uri="sqlite:///" + str(db.resolve()))
    exp = client.get_experiment_by_name(name)
    if exp is None:
        raise ValueError(f"Missing historical MLflow experiment {name}")
    inputs = read_json(out / "inputs.json")
    metrics = pd.read_csv(dest / "metrics.csv")
    intervals = pd.read_csv(dest / "paired_intervals.csv")
    scripts = [
        ROOT / "scripts" / n
        for n in (
            "stack_fullnight_folds.py",
            "compare_fullnight_lgbm.py",
            "compare_fullnight_edf.py",
            "report_fullnight_lgbm.py",
        )
    ]
    protocol = {
        "task": "W30, 5 expert stages, entire night grouped by subject, four channels",
        "primary": "unweighted mean of five outer fold macro-F1 scores; confirm on reserved Cassette",
        "fullnight_decision": "argmax of five conditional stage probabilities; companion native M counted as error",
        "hybrids": "5 conditional probabilities or same 5 + raw P(M); historical per-fold and refit booster recipes",
        "retrospective": "W30 label-dependent reference crop for LightGBM context; FullNightNet receives the whole waveform",
        "secondary": "Frozen models, whole-EDF context, pooled across all scored epochs and the W30 subset",
        "bootstrap": "paired 2000 subject resamples seed 42, CI percentile; single training seed schedule",
        "reference_lgbm_run": inputs["release"]["run_id"],
        "reference_fullnight_runs": inputs["fullnight_run_ids"],
        "source": {p.name: hash_file(p) for p in scripts},
        "versions": environment(),
        "torch_lightgbm_versions": inputs["release"]["versions"],
        "predictions": {
            p.name: hash_file(p)
            for p in sorted((out / "predictions").glob("*.parquet"))
        },
        "inner_fits": {
            p.parent.name: read_json(p)["identity"]
            for p in sorted((out / "inner").glob("*/audit.json"))
        },
    }
    campaign = fingerprint(protocol)
    prior = client.search_runs(
        [exp.experiment_id],
        f"tags.w30_stack = '{campaign}' AND tags.role = 'comparison' AND attributes.status = 'FINISHED'",
        max_results=1,
    )
    if prior:
        return read_json_from_run(client, prior[0].info.run_id)
    parent = client.create_run(
        exp.experiment_id,
        run_name="W30 · LightGBM publicado, FullNightNet e híbridos",
        tags={"w30_stack": campaign, "role": "comparison"},
    ).info.run_id
    ids = {
        "experiment_id": exp.experiment_id,
        "experiment_name": name,
        "parent": parent,
        "arms": {},
    }
    try:
        client.log_dict(parent, protocol, "protocol.json")
        eligible = summary.loc[ARMS]
        cv_winner = str(eligible.cv_macro_f1.idxmax())
        client.set_tag(parent, "cv_winner", cv_winner)
        client.log_metric(
            parent, "winning_cv_macro_f1", float(eligible.loc[cv_winner, "cv_macro_f1"])
        )
        client.log_metric(
            parent,
            "winning_arm_reserved_macro_f1",
            float(eligible.loc[cv_winner, "test_macro_f1"]),
        )
        client.log_param(parent, "development_subjects", len(inputs["dev_subjects"]))
        client.log_param(parent, "reserved_subjects", len(inputs["test_subjects"]))
        client.log_param(parent, "inner_fullnight_fits", len(protocol["inner_fits"]))
        for path in [
            out / "inputs.json",
            out / "serving-parity.json",
            CAMPAIGN / "reserved/result-integrity.json",
            SPLIT,
            ROOT / "uv.lock",
            *scripts,
            BUNDLE / "manifest.json",
            CAMPAIGN / "freeze.json",
            CAMPAIGN / "protocol.json",
        ]:
            group = "fullnight" if path.parent == CAMPAIGN else "reproduction"
            client.log_artifact(parent, str(path), group)
        booster = read_json(RESEARCH / "sc-w30-refinement-v1/lgbm.json")
        roots = RESEARCH / "sc-w30-refinement-v1/runs" / booster["identity"] / "lgbm"
        for fold in [*range(5), -1]:
            name_fold = f"fold-{fold}" if fold >= 0 else "candidate"
            part = roots / name_fold / "refit"
            for artifact in ["summary.json", "audit.json"]:
                client.log_artifact(
                    parent, str(part / artifact), "booster_recipes/" + name_fold
                )
            if fold >= 0:
                client.log_artifact(
                    parent, str(part / "model.joblib"), "baseline_folds/" + name_fold
                )
                client.log_artifact(
                    parent,
                    str(CAMPAIGN / name_fold / "six_direct.pt"),
                    "baseline_folds/" + name_fold,
                )
        client.log_artifacts(parent, str(out / "inner"), "nested_fullnight")
        client.log_artifacts(parent, str(out / "predictions"), "paired_predictions")
        for file in (
            "metrics.csv",
            "leaderboard.csv",
            "paired_intervals.csv",
            "confusion.json",
        ):
            client.log_artifact(parent, str(dest / file), "results")
        for arm in [*ARMS, NATIVE]:
            run = client.create_run(
                exp.experiment_id,
                run_name="W30 · " + arm,
                tags={"mlflow.parentRunId": parent, "w30_stack": campaign, "role": arm},
            ).info.run_id
            try:
                row = summary.loc[arm]
                for key, value in row.items():
                    client.log_metric(run, key, float(value))
                for r in metrics[metrics.arm.eq(arm)].to_dict("records"):
                    for k in (
                        "macro_f1",
                        "accuracy",
                        "kappa",
                        "f1_W",
                        "f1_N1",
                        "f1_N2",
                        "f1_N3",
                        "f1_REM",
                        "epochs",
                        "movement_calls",
                    ):
                        client.log_metric(
                            run, r["sample"].replace("-", "_") + "_" + k, float(r[k])
                        )
                client.log_param(
                    run,
                    "lightgbm_columns",
                    {
                        "lightgbm_release": 435,
                        "stack_five": 440,
                        "stack_five_movement": 441,
                    }.get(arm, 0),
                )
                client.log_param(
                    run, "channel_profile", "2 EEG + horizontal EOG + EMG envelope"
                )
                client.log_param(
                    run, "historical_release", inputs["release"]["release_id"]
                )
                intervals_arm = intervals[intervals.candidate.eq(arm)]
                client.log_dict(
                    run, intervals_arm.to_dict("records"), "paired_intervals.json"
                )
                if arm in [ARMS[0], *ARMS[2:]]:
                    for k, v in inputs["release"]["training_config"].items():
                        if k != "n_estimators":
                            client.log_param(run, "refit_" + k, v)
                    client.log_param(
                        run, "refit_iterations", inputs["release"]["iterations"]
                    )
                if arm != ARMS[0]:
                    client.log_param(
                        run,
                        "fullnight_checkpoint",
                        inputs["fullnight_checkpoint_sha256"],
                    )
                for diff in intervals_arm.query(
                    "sample == 'reserved' and metric == 'macro_f1'"
                ).itertuples():
                    for k in ("delta", "ci95_low", "ci95_high"):
                        client.log_metric(
                            run,
                            f"reserved_vs_{diff.reference}_{k}",
                            float(getattr(diff, k)),
                        )
                if arm in ARMS[2:]:
                    for path in sorted((out / "models").glob("*/" + arm + ".joblib")):
                        client.log_artifact(
                            run, str(path), "models/" + path.parent.name
                        )
                elif arm == "lightgbm_release":
                    client.log_artifact(
                        run, str(BUNDLE / "model.joblib"), "models/reserved"
                    )
                else:
                    client.log_artifact(
                        run, str(CAMPAIGN / "refit/six_direct.pt"), "models/reserved"
                    )
                client.set_terminated(run)
                ids["arms"][arm] = run
            except BaseException:
                client.set_terminated(run, status="FAILED")
                raise
        client.log_dict(parent, ids, "run_ids.json")
        client.set_terminated(parent)
    except BaseException:
        client.set_terminated(parent, status="FAILED")
        raise
    return ids


def read_json_from_run(client: MlflowClient, run: str) -> dict:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        return read_json(Path(client.download_artifacts(run, "run_ids.json", tmp)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=OUT)
    p.add_argument("--report", type=Path, default=REPORT)
    args = p.parse_args()
    with FileLock(str(args.output / "tracking.lock")):
        summary = report(args.output, args.report)
        print(summary.to_string(), flush=True)
        ids = {
            store: sync(args.output, args.report, store, summary) for store in STORES
        }
        write_json(args.report / "mlflow-runs.json", ids)
        print(ids, flush=True)


if __name__ == "__main__":
    main()
