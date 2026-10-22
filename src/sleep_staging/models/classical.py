from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sleep_staging.config import ExperimentConfig


def classical_model(family: str, cfg: ExperimentConfig) -> Pipeline:
    estimator = (
        DummyClassifier(strategy="most_frequent")
        if family == "dummy"
        else RandomForestClassifier(
            n_estimators=cfg.rf_trees,
            random_state=cfg.seed,
            class_weight="balanced",
            n_jobs=cfg.workers,
            min_samples_leaf=2,
        )
    )
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("classifier", estimator),
        ]
    )
