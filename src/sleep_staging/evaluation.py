from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)


def evaluate(y: np.ndarray, prediction: np.ndarray, classes: list[str]) -> dict:
    kappa = float(cohen_kappa_score(y, prediction, labels=classes))
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "f1_macro": float(
            f1_score(y, prediction, labels=classes, average="macro", zero_division=0)
        ),
        "kappa": kappa if np.isfinite(kappa) else None,
        "classes": classes,
        "report": classification_report(
            y, prediction, labels=classes, output_dict=True, zero_division=0
        ),
        "confusion": confusion_matrix(y, prediction, labels=classes).tolist(),
        "confusion_normalized": confusion_matrix(
            y, prediction, labels=classes, normalize="true"
        ).tolist(),
    }


def scalar_metrics(result: dict) -> dict[str, float]:
    return {
        key: float(result[key])
        for key in ["accuracy", "f1_macro", "kappa"]
        if result[key] is not None
    }


def plot_confusion(result: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for axis, key, title in zip(
        axes,
        ["confusion", "confusion_normalized"],
        ["Conteos OOF", "Proporción por estadio real"],
        strict=True,
    ):
        values = np.array(result[key])
        axis.imshow(values, cmap="Blues")
        axis.set_xticks(range(len(result["classes"])), result["classes"])
        axis.set_yticks(range(len(result["classes"])), result["classes"])
        axis.set(xlabel="Predicción", ylabel="Estadio real", title=title)
        for (i, j), value in np.ndenumerate(values):
            axis.text(
                j,
                i,
                f"{value:.2f}" if key.endswith("normalized") else str(int(value)),
                ha="center",
                va="center",
                color="black",
            )
    figure.savefig(path, dpi=150)
    plt.close(figure)
