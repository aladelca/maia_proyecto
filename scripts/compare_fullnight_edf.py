"""Secondary paired test of frozen W30 models through whole-EDF preprocessing."""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from compare_fullnight_lgbm import (
    ARMS,
    BUNDLE,
    CAMPAIGN,
    OUT,
    PROBS,
    PROFILE,
    RESEARCH,
    ROOT,
    SIX,
    SPLIT,
    conditional,
    predict,
    read_checked_model,
    read_json,
    stacked,
)

from sleep_staging.fullnight.cohort import research_root, source_file
from sleep_staging.provenance import atomic_path, fingerprint, hash_file, write_json
from sleep_staging.serving.contract import LABELS, context_matrix
from sleep_staging.serving.edf_lgbm import prepare_edf


def verify_serving(
    edf: Path,
    record: str,
    scored: pd.DataFrame,
    tree: np.ndarray,
    six: np.ndarray,
    out: Path,
) -> None:
    audit = {
        "recording": record,
        "paired_epochs": len(scored),
        "edf_sha256": hash_file(edf),
    }
    with tempfile.TemporaryDirectory() as directory:
        for name, script, args, cols, expected in [
            ("lightgbm", "predict_edf_lgbm.py", ["--bundle", str(BUNDLE)], PROBS, tree),
            (
                "fullnight",
                "serve_fullnight.py",
                ["--run", str(CAMPAIGN), "--device", "cpu", "--profile", PROFILE],
                SIX,
                six,
            ),
        ]:
            csv = Path(directory) / (name + ".csv")
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / script),
                    "--edf",
                    str(edf),
                    "--output",
                    str(csv),
                    *args,
                ],
                check=True,
            )
            served = (
                pd.read_csv(csv)
                .set_index("onset_s", verify_integrity=True)
                .loc[scored.onset_s, cols]
                .to_numpy()
            )
            np.testing.assert_allclose(served, expected, atol=2e-5, rtol=0)
            audit[name + "_max_abs_probability_error"] = float(
                np.abs(served - expected).max()
            )
            audit[name + "_label_agreement"] = float(
                (served.argmax(1) == expected.argmax(1)).mean()
            )
    write_json(out / "serving-parity.json", audit)


def score_files(source: Path, out: Path, prepare_only: bool) -> None:
    catalogue = source / "data/manifests/sleep-cassette"
    files = pd.read_parquet(catalogue / "files.parquet").set_index(
        "path", verify_integrity=True
    )
    people = pd.read_csv(SPLIT).query("partition == 'test'")
    records = pd.read_parquet(catalogue / "recordings.parquet").merge(
        people, on="subject_id", validate="many_to_one"
    )
    manifest = read_json(BUNDLE / "manifest.json")
    parent = CAMPAIGN / "reserved"
    if (
        hash_file(parent / "predictions.parquet")
        != read_json(parent / "result-integrity.json")["predictions.parquet"]
    ):
        raise ValueError("FullNight probabilities checksum mismatch")
    truth = pd.read_parquet(
        parent / "predictions.parquet",
        filters=[
            ("model", "==", "six_direct"),
            ("cohort", "==", "cassette_test"),
            ("profile", "==", PROFILE),
        ],
        columns=["record", "subject", "onset_s", "target", "w30", *SIX],
    )
    if set(truth.record) != set(records.recording_id) or set(truth.subject) != set(
        people.subject_id
    ):
        raise ValueError("Incomplete FullNight reserved subject or recording set")
    code_hashes = [
        hash_file(ROOT / "src/sleep_staging" / name)
        for name in (
            "serving/edf_lgbm.py",
            "features/recording.py",
            "features/research.py",
            "features/extract.py",
            "data/edf.py",
        )
    ]
    models = (
        {}
        if prepare_only
        else {
            arm: joblib.load(out / "models/reserved" / (arm + ".joblib"))
            for arm in ARMS[2:]
        }
    )
    baseline = read_checked_model(BUNDLE / "model.joblib", manifest["model_sha256"])
    predictions: list[pd.DataFrame] = []
    for r in records.sort_values("recording_id").itertuples():
        entry = files.loc[r.psg_file]
        dest = (
            out
            / "whole_edf"
            / fingerprint([entry.sha256, code_hashes])
            / (r.recording_id + ".parquet")
        )
        if not dest.exists():
            edf = source_file(
                source, RESEARCH / "ibio-fullnight/cache", r.psg_file, entry
            )
            with atomic_path(dest) as tmp:
                prepare_edf(edf, r.recording_id).to_parquet(tmp, index=False)
            print("Prepared EDF", r.recording_id, flush=True)
        if prepare_only:
            continue
        raw = pd.read_parquet(dest)
        night = truth.loc[truth.record.eq(r.recording_id)].set_index(
            "onset_s", verify_integrity=True
        )
        if not night.subject.eq(r.subject_id).all() or set(raw.onset_s) != set(
            night.index
        ):
            raise ValueError("FullNight and EDF timelines differ")
        raw = raw[raw.eligible].reset_index(drop=True)
        # Compute context before consulting expert annotations, exactly as in score_night.
        x = context_matrix(raw, manifest["raw_columns"])
        s = night.loc[raw.onset_s].reset_index()
        use = s.target.between(0, 4)
        x, s = x[use], s[use]
        pr = s[SIX].to_numpy(dtype=np.float32)
        if not np.isfinite(pr).all() or not np.allclose(pr.sum(1), 1, atol=1e-5):
            raise ValueError("FullNight is missing a production-eligible epoch")
        frame = raw.loc[use, ["epoch_id", "recording_id", "onset_s"]].copy()
        frame["subject_id"], frame["fold"] = r.subject_id, -1
        frame["task_5"], frame["w30"] = np.asarray(LABELS)[s.target], s.w30.to_numpy()
        for i, col in enumerate(SIX):
            frame["six_direct_" + col] = pr[:, i]
        probs = {ARMS[0]: predict(baseline, x), ARMS[1]: conditional(pr)}
        if not predictions:
            edf = source_file(
                source, RESEARCH / "ibio-fullnight/cache", r.psg_file, entry
            )
            verify_serving(edf, r.recording_id, frame, probs[ARMS[0]], pr, out)
        for arm, model in models.items():
            probs[arm] = predict(model, stacked(x, pr, arm == ARMS[-1]))
        for arm, p in probs.items():
            for i, col in enumerate(PROBS):
                frame[arm + "_" + col] = p[:, i]
        predictions.append(frame)
    if not prepare_only:
        with atomic_path(out / "predictions/whole_edf.parquet") as tmp:
            pd.concat(predictions, ignore_index=True).to_parquet(tmp, index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-root", type=Path)
    ap.add_argument("--output", type=Path, default=OUT)
    ap.add_argument("--prepare-only", action="store_true")
    a = ap.parse_args()
    score_files(research_root(ROOT, a.source_root), a.output, a.prepare_only)
