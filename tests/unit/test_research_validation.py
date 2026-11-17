import numpy as np
import pandas as pd
import pytest

from sleep_staging.research import (
    ResearchSpec,
    grouped_inner_split,
    paired_subject_bootstrap,
    research_identity,
    tune_catboost,
)


def test_cached_scores_invalidate_when_feature_values_change():
    spec = ResearchSpec("example", "rf", "eeg_v1")
    features = pd.DataFrame({"eeg_fpz_cz__eeg__std": [1.0, 2.0]})
    original = research_identity(spec, features, "same-dataset")
    changed = features.copy()
    changed.loc[0, "eeg_fpz_cz__eeg__std"] = 3.0
    assert original != research_identity(spec, changed, "same-dataset")


def test_cached_scores_invalidate_when_a_package_dependency_changes(monkeypatch):
    spec = ResearchSpec("example", "rf", "eeg_v1")
    features = pd.DataFrame({"eeg_fpz_cz__eeg__std": [1.0, 2.0]})
    monkeypatch.setattr(
        "sleep_staging.research.code_fingerprint", lambda: "initial-code"
    )
    original = research_identity(spec, features, "same-dataset")
    monkeypatch.setattr(
        "sleep_staging.research.code_fingerprint",
        lambda: "changed-temporal-or-estimator-code",
    )
    assert original != research_identity(spec, features, "same-dataset")


def test_inner_validation_keeps_all_epochs_of_each_subject_together():
    groups = np.repeat(np.arange(20), 10)
    training, validation = grouped_inner_split(groups, seed=42)
    assert not set(groups[training]) & set(groups[validation])
    assert len(set(groups[validation])) == 4
    assert set(training) | set(validation) == set(range(len(groups)))
    np.testing.assert_array_equal(training, grouped_inner_split(groups, seed=42)[0])


def test_catboost_tuning_uses_a_disjoint_inner_holdout_and_refits_all_supplied_subjects():
    pytest.importorskip("catboost")
    groups = np.repeat(np.arange(10), 20)
    y = np.tile(np.repeat(["W", "N1", "N2", "N3", "REM"], 4), 10)
    X = pd.DataFrame(
        np.random.default_rng(42).normal(size=(200, 3)), columns=["a", "b", "c"]
    )
    candidates = [
        {"iterations": 3, "depth": 2, "auto_class_weights": weights}
        for weights in ["Balanced", None]
    ]
    model, trials, audit = tune_catboost(X, y, groups, 42, candidates=candidates)
    assert not set(audit["training_subjects"]) & set(audit["validation_subjects"])
    assert set(audit["training_subjects"]) | set(audit["validation_subjects"]) == set(
        groups
    )
    assert len(trials) == 2
    assert set(model.classes_) == set(y)
    assert 1 <= model.tree_count_ <= 3


def test_subject_bootstrap_pairs_the_same_subjects_and_preserves_positive_improvement():
    reference = pd.DataFrame(
        {
            "epoch_id": [f"{i}" for i in range(25)],
            "subject_id": np.repeat(np.arange(5), 5),
            "y_true": ["W", "N1", "N2", "N3", "REM"] * 5,
            "y_pred": ["W"] * 25,
        }
    )
    candidate = reference.copy()
    candidate["y_pred"] = candidate.y_true
    result = paired_subject_bootstrap(reference, candidate.sample(frac=1), repeats=100)
    assert result["subjects"] == 5
    assert result["ci95"][0] > 0.8
    with pytest.raises(ValueError, match="identical"):
        paired_subject_bootstrap(reference, candidate.iloc[:-1])
