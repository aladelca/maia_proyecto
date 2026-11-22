"""Paired, subject-grouped evaluation on identical scored epochs."""

import numpy as np

from sleep_staging.fullnight.data import LABELS


def state_slices(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = (y >= 0) & (y < 5)
    neighbors = np.zeros_like(valid)
    neighbors[1:-1] = valid[1:-1] & valid[:-2] & valid[2:]
    same = np.zeros_like(valid)
    same[1:-1] = (y[:-2] == y[1:-1]) & (y[1:-1] == y[2:])
    return neighbors & ~same, neighbors & same


def summarize_counts(mat: np.ndarray, nights: int) -> dict:
    diagonal = mat.diagonal().astype(float)
    support = mat.sum(axis=1)
    called = mat.sum(axis=0)

    def ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.divide(a, b, out=np.zeros_like(diagonal), where=b != 0)

    f1 = ratio(2 * diagonal, support + called)
    result = {
        "scored": int(mat.sum()),
        "accuracy": float(diagonal.sum() / max(mat.sum(), 1)),
        "macro_f1": float(f1.mean()),
        "n1_f1": float(f1[1]),
        "n1_precision": float(ratio(diagonal, called)[1]),
        "n1_recall": float(ratio(diagonal, support)[1]),
        "n1_as_w": int(mat[1, 0]),
        "n1_as_n2": int(mat[1, 2]),
    }
    if len(mat) == 6:
        result.update(
            m_support=int(support[5]),
            m_precision=float(ratio(diagonal, called)[5]),
            m_recall=float(ratio(diagonal, support)[5]),
            m_f1=float(f1[5]),
            m_false_positives_per_night=float((called[5] - diagonal[5]) / nights),
        )
    return result


def score(y: np.ndarray, p: np.ndarray, w30: np.ndarray) -> dict:
    width = p.shape[1]
    valid = (y >= 0) & (y < width)
    eligible = np.isfinite(p).all(1)
    if not np.allclose(p[eligible].sum(1), 1, atol=2e-5) or (p[eligible] < 0).any():
        raise ValueError("Probabilidades incompatibles.")
    pred = p.argmax(1)
    transition, stable = state_slices(y)
    out = {}
    for name, window in (("full", np.ones(len(y), bool)), ("w30", w30)):
        use = valid & eligible & window
        primary = np.bincount(
            y[use].astype(int) * width + pred[use], minlength=width**2
        ).reshape(width, width)
        row = {
            "matrix": primary.tolist(),
            "classes": LABELS[:width],
            "reference": int((valid & window).sum()),
        }
        for desc, subset in (("transition", transition), ("stable", stable)):
            mask = use & subset
            row[desc] = [
                int((pred[mask] == y[mask]).sum()),
                int(mask.sum()),
                int(((y == 1) & (pred != 1) & mask).sum()),
            ]
        out[name] = row
    return out
