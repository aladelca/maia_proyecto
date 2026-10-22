import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold


def assign_subjects(
    epochs: pd.DataFrame, n_splits: int = 5, seed: int = 42, test_fraction: float = 0.2
) -> pd.DataFrame:
    # Always use the same task-independent subject universe, including M.
    subjects = np.array(sorted(epochs.subject_id.unique()))
    if len(subjects) < n_splits + 2:
        raise ValueError(
            f"Se requieren al menos {n_splits + 2} sujetos para reservar test y {n_splits} folds."
        )
    development, test = next(
        GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed).split(
            subjects, groups=subjects
        )
    )
    if len(development) < n_splits:
        raise ValueError("No hay suficientes sujetos de desarrollo.")
    selected = (
        epochs[epochs.subject_id.isin(subjects[development]) & epochs.task_6.notna()]
        .sort_values(["subject_id", "recording_id", "task_6"])
        .reset_index(drop=True)
    )
    if selected.subject_id.nunique() < n_splits:
        raise ValueError("No hay suficientes sujetos con etiquetas válidas.")
    assignments = {
        str(subject): {"partition": "test", "fold": -1} for subject in subjects[test]
    }
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, val) in enumerate(
        splitter.split(selected, selected.task_6, groups=selected.subject_id)
    ):
        for subject in selected.iloc[val].subject_id.unique():
            assignments[str(subject)] = {"partition": "development", "fold": fold}
    if set(assignments) != set(subjects):
        raise ValueError(
            "Hay sujetos sin épocas supervisadas; revisar catálogo antes de partir."
        )
    return pd.DataFrame(
        [{"subject_id": subject, **assignments[str(subject)]} for subject in subjects]
    )
