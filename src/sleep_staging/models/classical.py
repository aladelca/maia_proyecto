from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sleep_staging.config import ExperimentConfig


def classical_model(family: str, cfg: ExperimentConfig) -> Pipeline:

    if family == "dummy":
        estimator = DummyClassifier(
            strategy="most_frequent"
        )

    elif family == "rf":
        estimator = RandomForestClassifier(
            n_estimators=cfg.rf_trees,
            random_state=cfg.seed,
            class_weight="balanced",
            n_jobs=cfg.workers,
            min_samples_leaf=2,
        )

    elif family == "lgbm":
        from lightgbm import LGBMClassifier

        estimator = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=15,
            class_weight="balanced",
            random_state=cfg.seed,
            n_jobs=cfg.workers,
            verbosity=-1,
        )

    elif family == "catboost":
        from catboost import CatBoostClassifier

        estimator = CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=5,
            auto_class_weights="Balanced",
            random_seed=cfg.seed,
            thread_count=cfg.workers,
            verbose=False,
            allow_writing_files=False,
        )

    else:
        raise ValueError(f"Modelo desconocido: {family}")

    return Pipeline(
        [
            (
                "impute",
                SimpleImputer(
                    strategy="median",
                    keep_empty_features=True,
                ),
            ),
            ("scale", StandardScaler()),
            ("classifier", estimator),
        ]
    )