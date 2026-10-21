"""Lectura nativa por canal, sin elevar la frecuencia del EMG envolvente."""

from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np

ALIASES = {
    "EEG Fpz-Cz": "eeg_fpz_cz",
    "EEG Pz-Oz": "eeg_pz_oz",
    "EOG horizontal": "eog",
    "EMG submental": "emg",
}


@dataclass
class Signal:
    values: np.ndarray
    sfreq: float
    kind: str
    original_name: str
    unit: str = "uV"


def edf_channels(path: Path) -> list[dict]:
    with path.open("rb") as stream:
        header = stream.read(256)
        duration = float(header[244:252])
        count = int(header[252:256])
        block = stream.read(count * 256)
    if duration <= 0:
        raise ValueError(f"Duración de registro EDF inválida: {path}")
    names = [
        block[i * 16 : (i + 1) * 16].decode("latin-1").strip() for i in range(count)
    ]
    units = [
        block[count * 96 + i * 8 : count * 96 + (i + 1) * 8].decode("latin-1").strip()
        for i in range(count)
    ]
    rates = [
        int(block[count * 216 + i * 8 : count * 216 + (i + 1) * 8]) / duration
        for i in range(count)
    ]
    return [
        {"name": name, "unit": unit, "sfreq": rate}
        for name, unit, rate in zip(names, units, rates, strict=True)
    ]


def read_recording(
    psg: Path, hyp: Path
) -> tuple[dict[str, Signal], list[tuple[float, float, str]], float]:
    signals = {}
    if psg.suffix.lower() != ".edf" or hyp.suffix.lower() != ".edf":
        raise ValueError(
            "Este proyecto requiere PSG e hipnograma EDF reales de Sleep-EDFx."
        )
    original_time = None
    for info in edf_channels(psg):
        if info["name"] not in ALIASES:
            continue
        if info["unit"] not in {"uV", "µV", "μV", "mV", "V"}:
            raise ValueError(f"Unidad no soportada para {info['name']}: {info['unit']}")
        canonical = ALIASES[info["name"]]
        raw = mne.io.read_raw_edf(
            psg,
            include=[info["name"]],
            infer_types=False,
            preload=False,
            verbose="ERROR",
        )
        if original_time is None:
            original_time = raw.info["meas_date"]
        rate = float(raw.info["sfreq"])
        if canonical.startswith("eeg") and rate != 100:
            raise ValueError("Este protocolo requiere EEG a 100 Hz.")
        kind = (
            "emg_envelope"
            if canonical == "emg" and rate == 1
            else "emg_raw"
            if canonical == "emg"
            else "eeg"
            if canonical.startswith("eeg")
            else "eog"
        )
        if canonical == "emg" and rate not in {1, 100}:
            raise ValueError("Frecuencia EMG no soportada por este protocolo.")
        signals[canonical] = Signal(
            (raw.get_data()[0] * 1e6).astype("float32"), rate, kind, info["name"]
        )
        raw.close()
    eeg = next((s for key, s in signals.items() if key.startswith("eeg")), None)
    if eeg is None:
        raise ValueError(f"No hay un EEG compatible: {psg}")
    annotation = mne.read_annotations(hyp)
    shift = (
        (annotation.orig_time - original_time).total_seconds()
        if annotation.orig_time and original_time
        else 0.0
    )
    annotations = [
        (float(onset) + shift, float(length), str(text))
        for onset, length, text in zip(
            annotation.onset, annotation.duration, annotation.description, strict=True
        )
    ]
    return signals, annotations, len(eeg.values) / eeg.sfreq
