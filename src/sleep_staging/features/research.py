"""Vectorized, label-independent features for offline sleep-staging research."""

import math

import numpy as np
import pandas as pd
from scipy.ndimage import convolve1d
from scipy.signal import welch
from scipy.stats import kurtosis, skew

from sleep_staging.features.extract import BANDS


def batch_signal_features(
    values: np.ndarray, sfreq: float, kind: str
) -> dict[str, np.ndarray]:
    """Preserve the original 25 EEG features and add robust/nonlinear descriptors."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < 3 or not np.isfinite(x).all():
        raise ValueError("Expected finite epochs with at least three samples.")
    variance = x.var(axis=1)
    derivative = np.diff(x, axis=1)
    derivative_var = derivative.var(axis=1)
    mobility = np.sqrt(
        np.divide(
            derivative_var,
            variance,
            out=np.zeros_like(variance),
            where=variance > 1e-20,
        )
    )
    complexity = np.divide(
        np.sqrt(
            np.divide(
                np.diff(derivative, axis=1).var(axis=1),
                derivative_var,
                out=np.zeros_like(variance),
                where=derivative_var > 1e-20,
            )
        ),
        mobility,
        out=np.zeros_like(variance),
        where=mobility > 0,
    )
    quantiles = np.quantile(x, [0.1, 0.25, 0.5, 0.75, 0.9], axis=1)
    nonconstant = variance > 1e-20
    asymmetry, peakedness = np.zeros(len(x)), np.zeros(len(x))
    if nonconstant.any():
        asymmetry[nonconstant] = skew(x[nonconstant], axis=1)
        peakedness[nonconstant] = kurtosis(x[nonconstant], axis=1)
    result = {
        "mean": x.mean(axis=1),
        "std": x.std(axis=1),
        "rms": np.sqrt(np.mean(x**2, axis=1)),
        "q10": quantiles[0],
        "median": quantiles[2],
        "q90": quantiles[4],
        "ptp": np.ptp(x, axis=1),
        "skew": asymmetry,
        "kurtosis": peakedness,
        "zero_crossings": np.mean(np.diff(np.signbit(x), axis=1), axis=1),
        "hjorth_mobility": mobility,
        "hjorth_complexity": complexity,
        "diff_std": derivative.std(axis=1),
        "iqr": quantiles[3] - quantiles[1],
        "mad": np.median(np.abs(x - quantiles[2, :, None]), axis=1),
        "line_length": np.mean(np.abs(derivative), axis=1),
        "flat": (np.ptp(x, axis=1) <= 1e-10).astype(float),
    }
    # Stable ordinal patterns, order 3, delay 1; no labels or learned thresholds.
    triples = np.lib.stride_tricks.sliding_window_view(x, 3, axis=1)
    ordinal = np.argsort(triples, axis=-1, kind="stable")
    codes = ordinal @ np.array([1, 3, 9])
    probabilities = np.stack(
        [(codes == code).mean(axis=1) for code in [21, 15, 19, 7, 11, 5]], axis=1
    )
    result["permutation_entropy"] = -np.sum(
        probabilities * np.log2(np.maximum(probabilities, 1e-30)), axis=1
    ) / math.log2(6)
    changes = np.count_nonzero(np.diff(np.signbit(derivative), axis=1), axis=1)
    length = x.shape[1]
    result["petrosian_fd"] = np.log10(length) / (
        np.log10(length) + np.log10(length / (length + 0.4 * changes))
    )
    if kind == "emg_envelope":
        return result
    frequencies, power = welch(
        x, fs=sfreq, nperseg=min(x.shape[1], int(4 * sfreq)), axis=1
    )
    resolution = frequencies[1] - frequencies[0]
    mask = (frequencies >= 0.5) & (frequencies < min(30, sfreq / 2))
    total = power[:, mask].sum(axis=1) * resolution
    for name, (low, high) in BANDS.items():
        if high >= sfreq / 2:
            continue
        absolute = (
            power[:, (frequencies >= low) & (frequencies < high)].sum(axis=1)
            * resolution
        )
        result[f"power_{name}"] = absolute
        result[f"relative_{name}"] = np.divide(
            absolute, total, out=np.zeros_like(total), where=total > 1e-20
        )
        result[f"log_power_{name}"] = np.log10(np.maximum(absolute, 1e-12))
    probability = np.divide(
        power,
        power.sum(axis=1, keepdims=True),
        out=np.zeros_like(power),
        where=power.sum(axis=1, keepdims=True) > 0,
    )
    result["spectral_entropy"] = -np.sum(
        probability * np.log2(np.maximum(probability, 1e-30)), axis=1
    )
    result["delta_beta_ratio"] = result.get("power_delta", 0) / np.maximum(
        result.get("power_beta", 0), 1e-12
    )
    result["log_total_power"] = np.log10(np.maximum(total, 1e-12))
    for numerator, denominator in [
        ("theta", "alpha"),
        ("sigma", "beta"),
        ("delta", "theta"),
    ]:
        result[f"log_ratio_{numerator}_{denominator}"] = np.log10(
            np.maximum(result[f"power_{numerator}"], 1e-12)
        ) - np.log10(np.maximum(result[f"power_{denominator}"], 1e-12))
    cumulative = np.cumsum(power[:, mask], axis=1)
    indices = (cumulative >= cumulative[:, -1:] * 0.95).argmax(axis=1)
    result["spectral_edge_95"] = frequencies[mask][indices]
    # Short-bin variability can reveal bursts hidden by an epoch-wide average.
    bins = x[:, : (x.shape[1] // 6) * 6].reshape(len(x), 6, -1)
    rms_bins = np.sqrt(np.mean(bins**2, axis=2))
    result["rms_bin_cv"] = rms_bins.std(axis=1) / np.maximum(
        rms_bins.mean(axis=1), 1e-12
    )
    return result


def temporal_features(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Offline, within-record context; never bridge missing 30-second epochs.

    Raw feature values, not labels/predictions, enter past-four and centered-15
    windows. Each smoothed feature is robustly normalized within its own night.
    This is retrospective/transductive preprocessing, not real-time inference.
    """
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("Expected unique feature columns.")
    if any("__" not in c or c in {"task_5", "task_6"} for c in columns):
        raise ValueError("Only signal features may enter temporal context.")
    if frame.duplicated(["recording_id", "onset_s"]).any():
        raise ValueError("Duplicate recording/onset in temporal features.")
    if not frame.index.is_unique:
        raise ValueError("Temporal feature rows must have a unique index.")
    out = np.empty((len(frame), 2 * len(columns)), dtype=np.float32)
    positions = pd.Series(np.arange(len(frame)), index=frame.index)
    weights = np.r_[np.arange(1, 9), np.arange(7, 0, -1)].astype(float)
    for _, night in frame.groupby("recording_id", sort=True):
        night = night.sort_values("onset_s")
        blocks = night.onset_s.diff().ne(30).cumsum()
        smooth = np.empty((len(night), 2 * len(columns)), dtype=np.float64)
        local_positions = pd.Series(np.arange(len(night)), index=night.index)
        for _, group in night.groupby(blocks, sort=False):
            values = group[columns].to_numpy(dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError("Context requires finite input features.")
            past = (
                pd.DataFrame(values)
                .shift(1)
                .rolling(4, min_periods=1)
                .mean()
                .to_numpy()
            )
            past[0] = values[0]
            center = convolve1d(values, weights, axis=0, mode="constant", cval=0.0)
            divisor = convolve1d(
                np.ones(len(values)), weights, mode="constant", cval=0.0
            )
            center /= divisor[:, None]
            smooth[local_positions.loc[group.index].to_numpy()] = np.c_[past, center]
        quantiles = np.quantile(smooth, [0.05, 0.5, 0.95], axis=0)
        scale = np.maximum((quantiles[2] - quantiles[0]) / 3.29, 1e-6)
        normalized = np.clip((smooth - quantiles[1]) / scale, -50, 50)
        out[positions.loc[night.index].to_numpy()] = normalized.astype(np.float32)
    names = [
        f"{c}__context_{suffix}" for suffix in ["past4", "center15"] for c in columns
    ]
    return pd.DataFrame(out, index=frame.index, columns=names)
