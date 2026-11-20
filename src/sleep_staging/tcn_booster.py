"""Matched EEG/EOG LightGBM control in a CPU process isolated from PyTorch."""

import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from sleep_staging.booster_selection import split_subjects
from sleep_staging.config import CLASSES
from sleep_staging.features.research import temporal_features
from sleep_staging.provenance import atomic_path, write_json
from sleep_staging.refinement_data import class_weights, macro_f1_eval
from sleep_staging.tcn_data import KEYS, WAVE_CHANNELS, load_cohort


def main(request: dict) -> None:
    root = Path(request["directory"])
    data, _ = load_cohort(
        Path(request["features"]), Path(request["split"]), request["smoke"]
    )
    columns = [c for c in data.columns if c.split("__")[0] in WAVE_CHANNELS]
    x = pd.concat([data.frame[columns], temporal_features(data.frame, columns)], axis=1)
    parts = []
    for fold in range(5):
        seed = request["seed"] + fold
        train, val, test = split_subjects(data.frame, fold, seed)

        def rows(subjects: list) -> np.ndarray:
            return data.targets[
                data.frame.subject_id.isin(subjects).to_numpy()[data.targets]
            ]

        inner, holdout, outer, external = map(rows, (train, val, train + val, test))
        params = dict(
            num_leaves=63,
            min_child_samples=2 if request["smoke"] else 200,
            learning_rate=0.05,
            max_bin=63,
            reg_lambda=5,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            n_jobs=4,
            verbosity=-1,
            metric="None",
            random_state=seed,
            deterministic=True,
            force_col_wise=True,
        )

        def weights(indices: np.ndarray) -> np.ndarray:
            y = data.labels[indices]
            return class_weights(y, 1)[y]

        model = lgb.LGBMClassifier(
            n_estimators=4 if request["smoke"] else 450, **params
        )
        model.fit(
            x.iloc[inner],
            data.labels[inner],
            sample_weight=weights(inner),
            eval_set=[(x.iloc[holdout], data.labels[holdout])],
            eval_metric=macro_f1_eval,
            callbacks=[lgb.early_stopping(40, verbose=False)],
        )
        iterations = model.best_iteration_ or model.n_estimators_
        model = lgb.LGBMClassifier(n_estimators=iterations, **params)
        model.fit(x.iloc[outer], data.labels[outer], sample_weight=weights(outer))
        np.testing.assert_array_equal(model.classes_, np.arange(5))
        path = root / f"fold-{fold}"
        with atomic_path(path / "model.joblib") as tmp:
            joblib.dump(model, tmp)
        write_json(
            path / "fit.json",
            {
                "iterations": iterations,
                "columns": list(x.columns),
                "inner_training": train,
                "inner_validation": val,
                "external": test,
            },
        )
        result = data.frame.iloc[external][KEYS].copy()
        result["y_true"] = data.frame.iloc[external].task_5
        result[["prob_" + c for c in CLASSES["task_5"]]] = model.predict_proba(
            x.iloc[external]
        )
        parts.append(result)
    with atomic_path(root / "oof.parquet") as tmp:
        pd.concat(parts, ignore_index=True).to_parquet(tmp, index=False)


if __name__ == "__main__":
    main(json.loads(Path(sys.argv[1]).read_text()))
