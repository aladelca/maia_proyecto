"""Cross-fit FullNightNet inside each historical outer subject fold for stacking."""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sleep_staging.fullnight.data import LABELS
from sleep_staging.fullnight.training import (
    ALL,
    Night,
    Settings,
    encode_night,
    load_model,
    predict_encoded,
    save_model,
    train_network,
)
from sleep_staging.models.tcn import research_device
from sleep_staging.provenance import atomic_path, fingerprint, hash_file, write_json

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "var/research/ibio-fullnight/development/681a52f21d273bf190ae"
OUT = ROOT / "var/research/fullnight-lgbm-w30"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", type=Path, default=CAMPAIGN)
    p.add_argument("--output", type=Path, default=OUT)
    p.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    args = p.parse_args()
    device = research_device(args.device)
    torch.set_num_threads(2)
    campaign, out = args.campaign, args.output
    protocol = json.loads((campaign / "protocol.json").read_text())
    freeze = json.loads((campaign / "freeze.json").read_text())
    split = ROOT / "configs/splits/sc_subjects_v1.csv"
    if (
        fingerprint(protocol) != freeze["run"]
        or freeze["smoke"]
        or hash_file(split) != protocol["split_sha256"]
    ):
        raise ValueError("Incompatible FullNight campaign or split")
    subjects = pd.read_csv(split).set_index("subject_id", verify_integrity=True)
    subjects = subjects[subjects.partition.eq("development")]
    all_subjects = set(subjects.index)
    if set(freeze["subjects"]) != all_subjects:
        raise ValueError("Incompatible development subjects")
    cfg = Settings(**protocol["settings"])
    nights = []
    for record, key in sorted(protocol["nights"].items()):
        path = campaign.parents[1] / "cache/nights" / record / key
        meta = json.loads((path / "metadata.json").read_text())
        subject = meta["subject"]
        nights.append(
            Night.load(record, subject, int(subjects.loc[subject, "fold"]), path)
        )
    if {n.subject for n in nights} != all_subjects:
        raise ValueError("Incomplete night cache")
    for fold in [*range(5), -1]:
        folder = campaign / (f"fold-{fold}" if fold >= 0 else "refit")
        _, saved = load_model(folder / "six_direct.pt", torch.device("cpu"))
        if (
            set(saved["training_subjects"])
            != set(subjects.index[subjects.fold.ne(fold)])
            or saved["run_id"] != freeze["run"]
        ):
            raise ValueError("Original FullNight checkpoint training overlap")
    for a, b in combinations(range(5), 2):
        folder = out / "inner" / f"{a}-{b}"
        seed = cfg.seed + 1000 + 10 * a + b
        training = [n for n in nights if n.fold not in (a, b)]
        validation = [n for n in nights if n.fold in (a, b)]
        audit = {
            "excluded_folds": [a, b],
            "campaign": freeze["run"],
            "settings": protocol["settings"],
            "seed": seed,
            "device": str(device),
            "source_sha256": hash_file(Path(__file__)),
            "training_subjects": sorted({n.subject for n in training}),
            "prediction_subjects": sorted({n.subject for n in validation}),
        }
        if set(audit["training_subjects"]) & set(audit["prediction_subjects"]):
            raise ValueError("Inner subject leakage")
        identity = fingerprint(audit)
        checkpoint = folder / "six_direct.pt"
        predictions = folder / "probabilities.parquet"
        manifest = folder / "audit.json"
        if manifest.is_file():
            prior = json.loads(manifest.read_text())
            if (
                prior["identity"] == identity
                and prior["checkpoint_sha256"] == hash_file(checkpoint)
                and prior["predictions_sha256"] == hash_file(predictions)
            ):
                continue
            raise ValueError(f"Stale inner fold {folder}; choose another output")
        print(f"Training inner FullNight folds {a}-{b} on {device}", flush=True)
        net = train_network("six_direct", training, cfg, device, seed)
        save_model(checkpoint, net, cfg, audit["training_subjects"], identity)
        frames = []
        for n in validation:
            pr = predict_encoded(net, encode_night(net, n.inp, device), n.inp, ALL)
            use = n.w30 & (n.y >= 0) & (n.y < 5)
            df = pd.DataFrame(pr[use], columns=["prob_" + s for s in LABELS])
            df["recording_id"], df["onset_s"] = n.record, np.flatnonzero(use) * 30
            df["subject_id"], df["fold"], df["target"] = n.subject, n.fold, n.y[use]
            frames.append(df)
        with atomic_path(predictions) as tmp:
            pd.concat(frames, ignore_index=True).to_parquet(tmp, index=False)
        write_json(
            manifest,
            {
                **audit,
                "identity": identity,
                "checkpoint_sha256": hash_file(checkpoint),
                "predictions_sha256": hash_file(predictions),
            },
        )
    print("Ten inner networks complete", flush=True)


if __name__ == "__main__":
    main()
