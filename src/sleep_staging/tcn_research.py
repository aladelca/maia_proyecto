"""Three controlled comparisons over the frozen Sleep Cassette development folds."""

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from filelock import FileLock

from sleep_staging.booster_selection import split_subjects
from sleep_staging.config import CLASSES
from sleep_staging.evaluation import evaluate
from sleep_staging.lstm_data import FeatureScaler, SequenceData, sequence_index
from sleep_staging.provenance import (
    atomic_path,
    code_fingerprint,
    environment,
    fingerprint,
    hash_file,
    write_json,
)
from sleep_staging.tcn_data import (
    KEYS,
    WAVE_CHANNELS,
    boundary_targets,
    load_cohort,
    wave_cache,
)
from sleep_staging.tcn_training import (
    Settings,
    fit_head,
    learn_embeddings,
    predict,
    save_network,
)

STAGES = CLASSES["task_5"]
PROBS = ["prob_" + c for c in STAGES]


@dataclass(frozen=True)
class Variant:
    name: str
    kind: str
    view: str
    length: int = 15
    auxiliary: bool = False


FEATURE = [
    Variant(f"features_{head}_{length}", head, "features", length)
    for head in ("tcn", "bilstm")
    for length in (1, 15)
]
MATCHED = Variant("eeg_eog_features_tcn_15", "tcn", "matched")
WAVE = Variant("wave_cnn_tcn_15", "tcn", "wave")
ATTENTION = [
    Variant("wave_transformer", "transformer", "wave"),
    Variant("wave_transformer_aux", "transformer", "wave", auxiliary=True),
]


def fit_variant(
    spec: Variant,
    data: SequenceData,
    waves: np.ndarray | None,
    boundary: np.ndarray,
    root: Path,
    cfg: Settings,
    device: torch.device,
) -> pd.DataFrame:
    windows, _ = sequence_index(data.frame, spec.length, centered=True)
    parts = []
    values = data.values
    columns = data.columns
    if spec.view == "matched":
        columns = [c for c in data.columns if c.split("__")[0] in WAVE_CHANNELS]
        values = data.frame[columns].to_numpy(dtype="float32")
    for fold in range(5):
        directory = root / spec.name / f"fold-{fold}"
        manifest, result, checkpoint = (
            directory / f for f in ("fit.json", "oof.parquet", "model.pt")
        )
        if manifest.exists():
            saved = json.loads(manifest.read_text())
            if (
                hash_file(result) != saved["oof_sha256"]
                or hash_file(checkpoint) != saved["model_sha256"]
            ):
                raise ValueError(f"Invalid fold cache: {directory}")
            if (
                "encoder" in saved
                and hash_file(root / saved["encoder"]) != saved["encoder_sha256"]
            ):
                raise ValueError(f"Invalid encoder checkpoint for {directory}")
            parts.append(pd.read_parquet(result))
            continue
        seed = cfg.seed + fold
        inner, validation, external = split_subjects(data.frame, fold, seed)
        selected_epochs = cfg.epochs
        history: list[float] = []
        for phase, training, scoring in (
            ("inner", inner, validation),
            ("refit", inner + validation, external),
        ):
            train_mask = data.frame.subject_id.isin(training).to_numpy()
            score_mask = data.frame.subject_id.isin(scoring).to_numpy()
            train = data.targets[train_mask[data.targets]]
            score = data.targets[score_mask[data.targets]]
            extra: dict = {}
            if spec.view == "wave":
                assert waves is not None
                encoder = root / "encoders" / f"fold-{fold}" / phase
                x = learn_embeddings(
                    waves,
                    data.labels,
                    train,
                    np.flatnonzero(train_mask | score_mask),
                    encoder,
                    cfg,
                    device,
                    seed,
                )
                extra["encoder"] = str((encoder / "encoder.pt").relative_to(root))
                extra["encoder_sha256"] = hash_file(encoder / "encoder.pt")
            else:
                scaler = FeatureScaler.fit(values[train_mask])
                x = scaler.transform(values)
                extra = {"scaler": scaler.to_dict(), "columns": columns}
            model, n, scores = fit_head(
                spec.kind,
                x,
                windows,
                data.labels,
                boundary,
                train,
                score if phase == "inner" else np.array([], dtype=int),
                cfg,
                device,
                seed,
                spec.auxiliary,
                selected_epochs,
            )
            if phase == "inner":
                selected_epochs, history = n, scores
            else:
                prob = predict(model, x, windows, score, cfg.batch_size, device)
                table = data.frame.iloc[score][KEYS].copy()
                table["y_true"] = data.frame.iloc[score].task_5
                table[PROBS] = prob
                save_network(
                    model,
                    checkpoint,
                    {
                        "variant": asdict(spec),
                        "width": cfg.width,
                        "dropout": cfg.dropout,
                        "input_dim": x.shape[1],
                        "stages": STAGES,
                        **extra,
                    },
                )
                with atomic_path(result) as tmp:
                    table.to_parquet(tmp, index=False)
                write_json(
                    manifest,
                    {
                        "selected_epochs": selected_epochs,
                        "inner_macro_f1": history,
                        "inner_training": inner,
                        "inner_validation": validation,
                        "external": external,
                        **(
                            {k: extra[k] for k in ("encoder", "encoder_sha256")}
                            if spec.view == "wave"
                            else {}
                        ),
                        "model_sha256": hash_file(checkpoint),
                        "oof_sha256": hash_file(result),
                    },
                )
                parts.append(table)
        print(f"{spec.name} fold {fold + 1}/5, epochs={selected_epochs}", flush=True)
    return pd.concat(parts, ignore_index=True)


def aligned(data: SequenceData, table: pd.DataFrame) -> pd.DataFrame:
    target = data.frame.iloc[data.targets][KEYS + ["task_5"]].rename(
        columns={"task_5": "y_true"}
    )
    keys = target.set_index("epoch_id", verify_integrity=True)
    table = table.set_index("epoch_id", verify_integrity=True)
    if set(keys.index) != set(table.index):
        raise ValueError(
            "OOF predictions do not cover exactly the development targets."
        )
    table = table.loc[keys.index]
    pd.testing.assert_frame_equal(table[keys.columns], keys, check_dtype=False)
    prob = table[PROBS].to_numpy()
    if not np.isfinite(prob).all() or (prob < 0).any() or (prob > 1).any():
        raise ValueError("Invalid class probabilities.")
    np.testing.assert_allclose(prob.sum(axis=1), 1, atol=1e-5)
    return table.reset_index()


def evaluate_oof(table: pd.DataFrame, boundaries: np.ndarray) -> dict:
    predicted = np.asarray(STAGES)[table[PROBS].to_numpy().argmax(axis=1)]
    truth = table.y_true.to_numpy()
    folds = []
    for _, group in table.groupby("fold", sort=True):
        idx = group.index.to_numpy()
        folds.append(evaluate(truth[idx], predicted[idx], STAGES))
    f1 = np.array([f["f1_macro"] for f in folds])
    pooled = evaluate(truth, predicted, STAGES)
    slices = {}
    for name, mask in (("transition", boundaries == 1), ("stable", boundaries == 0)):
        slices[name] = {
            "n": int(mask.sum()),
            "metrics": evaluate(truth[mask], predicted[mask], STAGES)
            if mask.any()
            else None,
        }
    n1 = pooled["report"]["N1"]
    return {
        "macro_f1": f1.mean(),
        "fold_sd": f1.std(),
        "n1_f1": n1["f1-score"],
        "n1_precision": n1["precision"],
        "n1_recall": n1["recall"],
        "accuracy": np.mean([f["accuracy"] for f in folds]),
        "pooled": pooled,
        "slices": slices,
        "folds": folds,
    }


def run(
    combination: str,
    features: Path,
    split: Path,
    source: Path,
    output: Path,
    cfg: Settings,
    device: torch.device,
    smoke: bool,
    champion: Path | None = None,
) -> Path:
    data, meta = load_cohort(features, split, smoke)
    identity = {
        "features": meta["features_sha256"],
        "champion_summary": hash_file(champion) if champion and not smoke else None,
        "split": meta["split_sha256"],
        "files": fingerprint(meta["files"]),
        "code": code_fingerprint(),
        "settings": asdict(cfg),
        "device": device.type,
        "smoke": smoke,
        "versions": {
            **environment(),
            "torch": str(torch.__version__),
            "lightgbm": version("lightgbm"),
        },
        "platform": platform.platform(),
        "test_evaluated": False,
    }
    root = output / ("smoke" if smoke else "runs") / fingerprint(identity)
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / "campaign.lock")):
        write_json(root / "protocol.json", identity)
        boundaries = boundary_targets(data)
        variants = []
        if combination in {"1", "all"}:
            variants.extend(FEATURE)
        if combination in {"2", "all"}:
            variants.append(MATCHED)
        waves = None
        if combination != "1":
            variants.append(WAVE)
            waves = np.load(
                wave_cache(data, meta, split, source, output / "cache"), mmap_mode="r"
            )
        if combination in {"3", "all"}:
            variants.extend(ATTENTION)
        predictions = {
            v.name: aligned(
                data, fit_variant(v, data, waves, boundaries, root, cfg, device)
            )
            for v in variants
        }
        if combination in {"2", "all"}:
            directory = root / "matched_lightgbm"
            request = directory / "request.json"
            path = directory / "oof.parquet"
            receipt = directory / "complete.json"
            if not receipt.exists():
                write_json(
                    request,
                    {
                        "directory": str(directory.resolve()),
                        "features": str(features.resolve()),
                        "split": str(split.resolve()),
                        "seed": cfg.seed,
                        "smoke": smoke,
                    },
                )
                subprocess.run(
                    [sys.executable, "-m", "sleep_staging.tcn_booster", str(request)],
                    check=True,
                )
                write_json(receipt, {"sha256": hash_file(path)})
            if hash_file(path) != json.loads(receipt.read_text())["sha256"]:
                raise ValueError("LightGBM OOF checksum mismatch.")
            tree = aligned(data, pd.read_parquet(path))
            predictions["matched_lightgbm"] = tree
            blend = tree.copy()
            blend[PROBS] = torch.lerp(
                torch.tensor(tree[PROBS].to_numpy(), dtype=torch.float64),
                torch.tensor(
                    predictions[WAVE.name][PROBS].to_numpy(), dtype=torch.float64
                ),
                0.5,
            ).numpy()
            predictions["wave_tcn_lightgbm_50_50"] = aligned(data, blend)
            if champion is not None and not smoke:
                summary = json.loads(champion.read_text())
                provenance = summary["provenance"]
                if (
                    summary["family"] != "lgbm"
                    or summary["test_evaluated"]
                    or provenance["dataset_sha256"] != meta["features_sha256"]
                    or provenance["split_sha256"] != meta["split_sha256"]
                ):
                    raise ValueError(
                        "Historical champion used a different development protocol."
                    )
                saved = (
                    champion.parent
                    / "runs"
                    / summary["identity"]
                    / "lgbm"
                    / "oof.parquet"
                )
                if hash_file(saved) != summary["oof_sha256"]:
                    raise ValueError("Historical champion OOF checksum mismatch.")
                predictions["historical_lightgbm_four_channels"] = aligned(
                    data, pd.read_parquet(saved)
                )
        board = []
        for name, table in predictions.items():
            table["y_pred"] = np.asarray(STAGES)[table[PROBS].to_numpy().argmax(axis=1)]
            with atomic_path(root / name / "predictions.parquet") as tmp:
                table.to_parquet(tmp, index=False)
            metrics = evaluate_oof(table, boundaries[data.targets])
            write_json(root / name / "metrics.json", metrics)
            board.append(
                {
                    "model": name,
                    **{
                        k: v
                        for k, v in metrics.items()
                        if k not in {"folds", "slices", "pooled"}
                    },
                    **{
                        f"{s}_accuracy": m["metrics"]["accuracy"]
                        if m["metrics"]
                        else None
                        for s, m in metrics["slices"].items()
                    },
                }
            )
        board_frame = pd.DataFrame(board).sort_values("macro_f1", ascending=False)
        with atomic_path(root / f"leaderboard-{combination}.csv") as tmp:
            board_frame.to_csv(tmp, index=False)
        print(board_frame.to_string(index=False), flush=True)
    return root
