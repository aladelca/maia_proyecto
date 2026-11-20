"""Compare the published LightGBM, FullNightNet and two stacked models on W30."""

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from filelock import FileLock
from sklearn.base import clone

from sleep_staging.fullnight.cohort import research_root, source_file
from sleep_staging.provenance import atomic_path, fingerprint, hash_file, write_json
from sleep_staging.research_data import _record_features
from sleep_staging.serving.contract import LABELS, context_matrix

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "var/research"
CAMPAIGN = RESEARCH / "ibio-fullnight/development/681a52f21d273bf190ae"
OUT = RESEARCH / "fullnight-lgbm-w30"
SPLIT = ROOT / "configs/splits/sc_subjects_v1.csv"
BUNDLE = ROOT / "var/cloud/bundle"
PROBS = ["prob_" + s for s in LABELS]
SIX = PROBS + ["prob_M"]
KEYS = ["recording_id", "subject_id", "onset_s", "fold"]
PROFILE = "eeg_fpz_cz+eeg_pz_oz+eog+emg"
ARMS = ["lightgbm_release", "fullnight_five", "stack_five", "stack_five_movement"]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def reserved_data(source: Path, output: Path, split: pd.DataFrame) -> pd.DataFrame:
    """Build reference features with the same wake crop and flat-channel rule as training."""
    catalogue = source / "data/manifests/sleep-cassette"
    records = pd.read_parquet(catalogue / "recordings.parquet").merge(
        split[split.partition.eq("test")], on="subject_id", validate="many_to_one"
    )
    files = pd.read_parquet(catalogue / "files.parquet").set_index(
        "path", verify_integrity=True
    )
    cache = RESEARCH / "ibio-fullnight/cache"
    package = ROOT / "src/sleep_staging"
    code = [
        hash_file(package / module)
        for module in (
            "features/recording.py",
            "features/research.py",
            "features/extract.py",
            "research_data.py",
            "data/edf.py",
            "data/labels.py",
        )
    ]
    frames = []
    for rec in records.sort_values("recording_id").to_dict("records"):
        hashes = [files.loc[rec[key], "sha256"] for key in ("psg_file", "hyp_file")]
        dest = (
            output
            / "w30"
            / fingerprint([hashes, code])
            / (rec["recording_id"] + ".parquet")
        )
        if not dest.is_file():
            for key in ("psg_file", "hyp_file"):
                rec[key] = str(
                    source_file(source, cache, rec[key], files.loc[rec[key]]).resolve()
                )
            frame = _record_features(rec, source, margin=30)
            with atomic_path(dest) as tmp:
                frame.to_parquet(tmp, index=False)
            print("Prepared", rec["recording_id"], flush=True)
        frames.append(pd.read_parquet(dest))
    return pd.concat(frames, ignore_index=True)


def matrices(
    raw: pd.DataFrame, manifest: dict, people: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = raw.sort_values(["recording_id", "onset_s"]).reset_index(drop=True)
    expected = people.set_index("subject_id", verify_integrity=True).fold
    if (
        set(raw.subject_id) != set(expected.index)
        or not raw.fold.eq(raw.subject_id.map(expected)).all()
    ):
        raise ValueError("Unexpected subjects or folds")
    x = context_matrix(raw, manifest["raw_columns"])
    if list(x) != manifest["model_columns"] or raw.epoch_id.duplicated().any():
        raise ValueError("Historical feature or epoch contract mismatch")
    keep = raw.task_5.isin(LABELS)
    return (
        raw.loc[keep, ["epoch_id", *KEYS, "task_5"]].reset_index(drop=True),
        x.loc[keep].reset_index(drop=True),
    )


def frozen_prob(folder: Path, cohort: str) -> pd.DataFrame:
    return pd.read_parquet(
        folder / "predictions.parquet",
        filters=[
            ("model", "==", "six_direct"),
            ("profile", "==", PROFILE),
            ("cohort", "==", cohort),
            ("w30", "==", True),
            ("target", ">=", 0),
            ("target", "<", 5),
        ],
        columns=["record", "subject", "onset_s", "fold", "target", *SIX],
    ).rename(columns={"record": "recording_id", "subject": "subject_id"})


def aligned(meta: pd.DataFrame, predictions: pd.DataFrame) -> np.ndarray:
    merged = meta.merge(
        predictions, on=KEYS, how="left", validate="one_to_one", sort=False
    )
    if not merged.target.eq(
        merged.task_5.map(dict(zip(LABELS, range(5), strict=True)))
    ).all():
        raise ValueError(
            "Missing probability or mismatched target/epoch in a paired comparison"
        )
    p = merged[SIX].to_numpy(dtype=np.float32)
    if (
        not np.isfinite(p).all()
        or (p < 0).any()
        or not np.allclose(p.sum(1), 1, atol=1e-5)
    ):
        raise ValueError("Invalid FullNight probabilities on LightGBM-eligible epochs")
    return p


def conditional(p: np.ndarray) -> np.ndarray:
    total = p[:, :5].sum(1, keepdims=True)
    if (total <= 0).any():
        raise ValueError("Undefined conditional probabilities")
    return p[:, :5] / total


def stacked(x: pd.DataFrame, p: np.ndarray, movement: bool) -> pd.DataFrame:
    added = pd.DataFrame(
        conditional(p), index=x.index, columns=["fullnight_" + c for c in LABELS]
    )
    if movement:
        added["fullnight_M"] = p[:, 5]
    return pd.concat([x, added], axis=1)


def predict(model: Any, x: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(x)[:, [list(model.classes_).index(s) for s in LABELS]]


def read_checked_model(path: Path, digest: str) -> Any:
    if hash_file(path) != digest:
        raise ValueError(f"Unexpected checkpoint {path}")
    return joblib.load(path)


def run_comparison(source: Path, out: Path) -> None:
    split = pd.read_csv(SPLIT)
    dev_people, test_people = (
        split[split.partition.eq("development")],
        split[split.partition.eq("test")],
    )
    release = read_json(BUNDLE / "manifest.json")
    ref = read_json(RESEARCH / "sc-w30-refinement-v1/lgbm.json")
    frozen = read_json(CAMPAIGN / "freeze.json")
    history = RESEARCH / "sc-w30-v2"
    dataset_id = Path(read_json(history / "campaign.json")["dataset"]).parent.name
    dataset = history / "datasets" / dataset_id / "features.parquet"
    reference_dir = RESEARCH / "sc-w30-refinement-v1/runs" / ref["identity"] / "lgbm"
    if (
        hash_file(dataset) != release["dataset_sha256"]
        or hash_file(SPLIT) != release["split_sha256"]
        or read_json(CAMPAIGN / "protocol.json")["split_sha256"]
        != release["split_sha256"]
        or frozen["edf_model"] != "six_direct.pt"
        or hash_file(CAMPAIGN / "refit/six_direct.pt")
        != frozen["weights"]["six_direct.pt"]
        or set(frozen["subjects"]) != set(dev_people.subject_id)
        or ref["candidate"]["refit"]["model_sha256"] != release["model_sha256"]
        or hash_file(reference_dir / "oof.parquet") != ref["oof_sha256"]
    ):
        raise ValueError("Inconsistent baseline, dataset or subject split")
    holdout_folder = CAMPAIGN / "reserved"
    if (
        hash_file(holdout_folder / "predictions.parquet")
        != read_json(holdout_folder / "result-integrity.json")["predictions.parquet"]
    ):
        raise ValueError("Corrupt FullNight reserved predictions")
    dev, xd = matrices(pd.read_parquet(dataset), release, dev_people)
    test, xt = matrices(reserved_data(source, out, split), release, test_people)
    fn_dev = aligned(
        dev,
        pd.concat(
            [frozen_prob(CAMPAIGN / f"fold-{f}", "development") for f in range(5)]
        ),
    )
    fn_test = aligned(test, frozen_prob(holdout_folder, "cassette_test"))
    history_oof = pd.read_parquet(reference_dir / "oof.parquet")
    baseline = dev.merge(
        history_oof, on=["epoch_id", *KEYS], how="left", validate="one_to_one"
    )
    if not baseline.y_true.eq(baseline.task_5).all() or len(baseline) != len(
        history_oof
    ):
        raise ValueError("Historical OOF targets differ from W30 epochs")
    pred_folder = out / "predictions"
    for f in [*range(5), -1]:
        train = dev.fold.ne(f)
        if f == -1:
            valid, pv, xv = test, fn_test, xt
            ptrain = fn_dev
            original = read_checked_model(
                BUNDLE / "model.joblib", release["model_sha256"]
            )
            base_p = predict(original, xv)
            name = "reserved"
        else:
            valid, pv, xv = dev[~train], fn_dev[~train], xd[~train]
            tables = []
            for other in sorted(set(range(5)) - {f}):
                a, b = sorted([f, other])
                folder = out / "inner" / f"{a}-{b}"
                audit = read_json(folder / "audit.json")
                expected = set(
                    dev_people.loc[~dev_people.fold.isin([a, b]), "subject_id"]
                )
                if (
                    set(audit["training_subjects"]) != expected
                    or audit["excluded_folds"] != [a, b]
                    or audit["campaign"] != frozen["run"]
                    or hash_file(folder / "probabilities.parquet")
                    != audit["predictions_sha256"]
                ):
                    raise ValueError("Invalid nested cross-fit provenance")
                tables.append(
                    pd.read_parquet(
                        folder / "probabilities.parquet",
                        filters=[("fold", "==", other)],
                    )
                )
            ptrain = aligned(dev[train], pd.concat(tables))
            original = read_checked_model(
                reference_dir / f"fold-{f}/refit/model.joblib",
                ref["folds"][f]["refit"]["model_sha256"],
            )
            base_p = baseline.loc[~train, PROBS].to_numpy()
            np.testing.assert_allclose(predict(original, xv), base_p, atol=1e-6, rtol=0)
            name = f"fold-{f}"
        audit_dir = reference_dir / ("candidate" if f == -1 else f"fold-{f}") / "refit"
        if set(read_json(audit_dir / "audit.json")["training_subjects"]) != set(
            dev.loc[train, "subject_id"]
        ):
            raise ValueError("Historical booster trained on unexpected subjects")
        outputs = {"lightgbm_release": base_p, "fullnight_five": conditional(pv)}
        record = valid.copy()
        for idx, col in enumerate(SIX):
            record["six_direct_" + col] = pv[:, idx]
        for arm in ARMS[2:]:
            movement = arm == "stack_five_movement"
            model = clone(original)
            print(f"Fitting {name} {arm}", flush=True)
            model.fit(stacked(xd[train], ptrain, movement), dev.loc[train, "task_5"])
            path = out / "models" / name / (arm + ".joblib")
            with atomic_path(path) as tmp:
                joblib.dump(model, tmp, compress=3)
            inputs = stacked(xv, pv, movement)
            outputs[arm] = predict(model, inputs)
            np.testing.assert_allclose(
                predict(joblib.load(path), inputs.iloc[:32]),
                outputs[arm][:32],
                rtol=0,
                atol=1e-12,
            )
        for arm, prob in outputs.items():
            if not np.isfinite(prob).all() or not np.allclose(
                prob.sum(1), 1, atol=1e-5
            ):
                raise ValueError("Invalid five-stage probabilities")
            for i, col in enumerate(PROBS):
                record[arm + "_" + col] = prob[:, i]
        with atomic_path(pred_folder / (name + ".parquet")) as tmp:
            record.to_parquet(tmp, index=False)
    write_json(
        out / "inputs.json",
        {
            "release": release,
            "fullnight": frozen["run"],
            "fullnight_checkpoint_sha256": frozen["weights"]["six_direct.pt"],
            "training_dataset_sha256": hash_file(dataset),
            "split_sha256": hash_file(SPLIT),
            "historical_oof_sha256": ref["oof_sha256"],
            "historical_run": ref["mlflow_run_id"],
            "fullnight_run_ids": read_json(CAMPAIGN / "mlflow-runs.json"),
            "dev_subjects": sorted(dev_people.subject_id),
            "test_subjects": sorted(test_people.subject_id),
            "development_epochs": len(dev),
            "reserved_epochs": len(test),
        },
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path)
    p.add_argument("--output", type=Path, default=OUT)
    p.add_argument("--prepare-test", action="store_true")
    args = p.parse_args()
    source = research_root(ROOT, args.source_root)
    if args.prepare_test:
        reserved_data(source, args.output, pd.read_csv(SPLIT))
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        with FileLock(str(args.output / "comparison.lock"), timeout=0):
            run_comparison(source, args.output)


if __name__ == "__main__":
    main()
