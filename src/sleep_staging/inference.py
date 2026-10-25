import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sleep_staging.tracking import require_tracking


def predict_features(
    model_uri: str, frame: pd.DataFrame, tracking_uri: str
) -> pd.DataFrame:
    require_tracking(tracking_uri)
    info = mlflow.models.get_model_info(model_uri)
    metadata = info.metadata or {}
    columns = metadata.get("feature_columns", [])
    missing = set(columns) - set(frame.columns)
    if not columns or missing:
        raise ValueError(
            f"Faltan características requeridas por el modelo: {sorted(missing)}"
        )
    if frame.attrs.get("preprocessing_id") != metadata.get("preprocessing_id"):
        raise ValueError(
            "Las características no corresponden al preprocesamiento del modelo."
        )
    model = mlflow.sklearn.load_model(model_uri)
    prediction = np.asarray(
        model.predict(frame[columns])
    ).reshape(-1)

    if len(prediction) != len(frame):
        raise ValueError("Número de predicciones incompatible con la entrada")
    probability = model.predict_proba(frame[columns])
    result = (
        frame[["epoch_id"]].copy()
        if "epoch_id" in frame
        else pd.DataFrame(index=frame.index)
    )
    result["prediction"] = prediction
    for i, label in enumerate(model.classes_):
        result[f"prob_{label}"] = probability[:, i]
    return result
