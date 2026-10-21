import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis, skew

BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "sigma": (12, 16),
    "beta": (16, 30),
}


def signal_features(values: np.ndarray, sfreq: float, kind: str) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError(
            "Se requiere una señal finita unidimensional con al menos dos muestras."
        )
    variance = float(np.var(x))
    derivative = np.diff(x)
    mobility = np.sqrt(np.var(derivative) / variance) if variance > 1e-20 else 0.0
    derivative_var = float(np.var(derivative))
    complexity = (
        np.sqrt(np.var(np.diff(derivative)) / derivative_var) / mobility
        if derivative_var > 1e-20 and mobility > 0 and len(x) > 2
        else 0.0
    )
    result = {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "rms": float(np.sqrt(np.mean(x**2))),
        "q10": float(np.quantile(x, 0.1)),
        "median": float(np.median(x)),
        "q90": float(np.quantile(x, 0.9)),
        "ptp": float(np.ptp(x)),
        "skew": float(skew(x)) if variance > 1e-20 else 0.0,
        "kurtosis": float(kurtosis(x)) if variance > 1e-20 else 0.0,
        "zero_crossings": float(np.mean(np.diff(np.signbit(x)))),
        "hjorth_mobility": float(mobility),
        "hjorth_complexity": float(complexity),
        "diff_std": float(derivative.std()),
    }
    if kind == "emg_envelope":
        return result
    frequencies, power = welch(x, fs=sfreq, nperseg=min(len(x), int(4 * sfreq)))
    resolution = frequencies[1] - frequencies[0]
    total = float(
        np.sum(power[(frequencies >= 0.5) & (frequencies < min(30, sfreq / 2))])
        * resolution
    )
    for name, (low, high) in BANDS.items():
        if high >= sfreq / 2:
            continue
        absolute = float(
            np.sum(power[(frequencies >= low) & (frequencies < high)]) * resolution
        )
        result[f"power_{name}"] = absolute
        result[f"relative_{name}"] = absolute / total if total > 1e-20 else 0.0
    probability = power / power.sum() if power.sum() > 0 else np.zeros_like(power)
    result["spectral_entropy"] = float(
        -np.sum(probability * np.log2(np.maximum(probability, 1e-30)))
    )
    result["delta_beta_ratio"] = result.get("power_delta", 0) / max(
        result.get("power_beta", 0), 1e-12
    )
    return result
