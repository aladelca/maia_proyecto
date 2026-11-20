"""Single-night, label-free interface to the retrospective Sleep Cassette model."""

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from sleep_staging.features.research import temporal_features

SCHEMA = "sleep-cassette-night-v1"
LABELS = ["W", "N1", "N2", "N3", "REM"]
KEYS = ["recording_id", "epoch_id", "onset_s", "eligible"]
MAX_BYTES = 32 * 1024**2
MAX_ROWS = 3000


def validate_night(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if (
        not 0 < len(frame) <= MAX_ROWS
        or not frame.columns.is_unique
        or set(frame) != set(KEYS + columns)
    ):
        raise ValueError("Columnas o número de épocas incompatibles.")
    for key in ["epoch_id", "recording_id"]:
        if (
            not frame[key]
            .map(
                lambda x: (
                    isinstance(x, str)
                    and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9:_.-]{0,159}", x)
                    is not None
                )
            )
            .all()
        ):
            raise ValueError("Identificador de época o registro inválido.")
    if frame.recording_id.nunique() != 1 or frame.epoch_id.duplicated().any():
        raise ValueError("Se requiere una sola noche e identificadores únicos.")
    if (
        not pd.api.types.is_integer_dtype(frame.onset_s)
        or frame.onset_s.isna().any()
        or frame.onset_s.lt(0).any()
        or (frame.onset_s % 30 != frame.onset_s.iloc[0] % 30).any()
        or frame.onset_s.duplicated().any()
    ):
        raise ValueError("Tiempos inválidos o duplicados.")
    if not pd.api.types.is_bool_dtype(frame.eligible) or frame.eligible.isna().any():
        raise ValueError("La máscara de elegibilidad debe ser booleana.")
    if (
        not all(pd.api.types.is_float_dtype(frame[c]) for c in columns)
        or not np.isfinite(frame[columns].to_numpy()).all()
    ):
        raise ValueError("Las variables deben ser números flotantes finitos.")
    return frame.sort_values("onset_s").reset_index(drop=True)


def context_matrix(ordered: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return pd.concat(
        [ordered[columns], temporal_features(ordered, columns)], axis=1
    ).astype(np.float32)


def read_night(path: Path, manifest: dict) -> pd.DataFrame:
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("El archivo supera el máximo permitido.")
    parquet = pq.ParquetFile(
        path, thrift_string_size_limit=1024**2, thrift_container_size_limit=100000
    )
    meta = parquet.metadata
    if (
        meta.num_rows > MAX_ROWS
        or meta.num_row_groups > 128
        or meta.num_columns != len(KEYS) + len(manifest["raw_columns"])
        or sum(meta.row_group(i).total_byte_size for i in range(meta.num_row_groups))
        > 64 * 1024**2
    ):
        raise ValueError("El contenido Parquet supera los límites.")
    schema = parquet.schema_arrow
    tags = schema.metadata or {}
    if (
        tags.get(b"input_schema") != SCHEMA.encode()
        or tags.get(b"preprocessing_id") != manifest["preprocessing_id"].encode()
    ):
        raise ValueError("Protocolo de preparación incompatible.")
    for f in schema:
        ok = (
            pa.types.is_floating(f.type)
            if f.name in manifest["raw_columns"]
            else pa.types.is_integer(f.type)
            if f.name == "onset_s"
            else pa.types.is_boolean(f.type)
            if f.name == "eligible"
            else (pa.types.is_string(f.type) or pa.types.is_large_string(f.type))
            and f.name in ["recording_id", "epoch_id"]
        )
        if not ok:
            raise ValueError("Tipo de columna incompatible.")
    return validate_night(
        parquet.read().replace_schema_metadata(None).to_pandas(),
        manifest["raw_columns"],
    )


def predictions(model: Any, manifest: dict, frame: pd.DataFrame) -> pd.DataFrame:
    x = context_matrix(frame, manifest["raw_columns"])
    if (
        list(x) != manifest["model_columns"]
        or list(model.feature_names_in_) != list(x)
        or set(model.classes_) != set(LABELS)
    ):
        raise ValueError("El modelo y el esquema son incompatibles.")
    result = frame[KEYS].copy()
    result["prediction"] = pd.Series([None] * len(frame), dtype=object)
    probs = ["prob_" + c for c in LABELS]
    result[probs] = np.nan
    mask = frame.eligible
    if mask.any():
        native = model.predict_proba(x[mask])
        p = native[:, [list(model.classes_).index(c) for c in LABELS]]
        if (
            not np.isfinite(p).all()
            or (p < 0).any()
            or not np.allclose(p.sum(axis=1), 1, atol=1e-6)
        ):
            raise ValueError("Probabilidades inválidas.")
        result.loc[mask, "prediction"] = np.asarray(model.classes_)[
            native.argmax(axis=1)
        ]
        result.loc[mask, probs] = p
    return result
