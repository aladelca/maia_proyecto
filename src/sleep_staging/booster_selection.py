"""Subject-level selection shared by local studies and cloud CPU training."""

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.model_selection import GroupShuffleSplit


class BoosterCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    n_estimators: int = Field(default=450, ge=1)
    learning_rate: float = Field(default=0.05, gt=0)
    num_leaves: int = Field(default=31, ge=2)
    min_child_samples: int = Field(default=100, ge=1)
    max_bin: int = Field(default=63, ge=2)
    reg_alpha: float = Field(default=0, ge=0)
    reg_lambda: float = Field(default=3, ge=0)
    colsample_bytree: float = Field(default=0.8, gt=0, le=1)
    subsample: float = Field(default=0.8, gt=0, le=1)
    subsample_freq: int = Field(default=1, ge=0)
    weight_power: float = Field(default=1, ge=0, le=1)


def split_subjects(
    frame: pd.DataFrame, fold: int, seed: int
) -> tuple[list, list, list]:
    available = frame.loc[frame.fold.ne(fold)] if fold >= 0 else frame
    groups = available.subject_id.to_numpy()
    a, b = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(
            groups, groups=groups
        )
    )
    training = sorted(set(groups[a]))
    validation = sorted(set(groups[b]))
    external = (
        sorted(set(frame.loc[frame.fold.eq(fold), "subject_id"])) if fold >= 0 else []
    )
    if set(training) & set(validation) or (set(training) | set(validation)) & set(
        external
    ):
        raise ValueError("Subject leakage in nested validation.")
    return training, validation, external


def choose_trial(trials: list[dict]) -> dict:
    return sorted(
        trials, key=lambda t: (-t["best_f1_macro"], t["best_epoch"], t["candidate"])
    )[0]
