"""Second-level inputs of the five-stage hybrid: 435 tabular columns and five FullNightNet probabilities."""

from pathlib import Path

import numpy as np
import pandas as pd

from sleep_staging.serving.contract import LABELS

PROBS = ["prob_" + s for s in LABELS]
SIX = PROBS + ["prob_M"]
KEYS = ["recording_id", "subject_id", "onset_s", "fold"]
PROFILE = "eeg_fpz_cz+eeg_pz_oz+eog+emg"


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


def fullnight_oof(campaign: Path) -> pd.DataFrame:
    """Each development epoch scored by the FullNightNet fold that did not train on it."""
    return pd.concat(
        [frozen_prob(campaign / f"fold-{f}", "development") for f in range(5)],
        ignore_index=True,
    )


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


def stacked(x: pd.DataFrame, p: np.ndarray, movement: bool = False) -> pd.DataFrame:
    added = pd.DataFrame(
        conditional(p), index=x.index, columns=["fullnight_" + c for c in LABELS]
    )
    if movement:
        added["fullnight_M"] = p[:, 5]
    return pd.concat([x, added], axis=1)
