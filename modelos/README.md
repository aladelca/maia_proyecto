# Modelo publicado

El clasificador híbrido combina 435 características del EDF con cinco probabilidades de FullNightNet. [Pesos, versión y ejecución local](stack_five/README.md) · [Evaluación emparejada](../docs/comparacion-fullnight-lightgbm.md)

## Referencia histórica de LightGBM


El LightGBM base fue el primer modelo desplegado (hasta el 20 de noviembre de 2026) y es la base del híbrido actual. Su archivo no se versiona en Git: se publica en un bucket S3 versionado y cada versión queda identificada por su SHA-256.

## LightGBM base

| Atributo | Valor |
| --- | --- |
| Algoritmo | LightGBM multiclase (`LGBMClassifier` dentro de un `Pipeline` de scikit-learn), configuración `large_regularized` |
| Tarea | Cinco estadios por época de 30 s: W, N1, N2, N3, REM |
| Entrada | 145 variables por época (EEG Fpz-Cz, EEG Pz-Oz, EOG, envolvente EMG); el servicio deriva otras 290 de contexto temporal, para un total de 435 |
| Hiperparámetros | 261 iteraciones, 63 hojas, mínimo 200 muestras por hoja, tasa de aprendizaje 0,05, 63 bins, L2 = 5, 80 % de filas y columnas, pesos inversos de clase, semilla 542 |
| Entrenamiento | 62 sujetos de desarrollo (122 noches, 152.052 épocas); los 16 sujetos de prueba quedan reservados |
| Desempeño (validación cruzada, 5 folds por sujeto) | F1 macro 0,7893 ± 0,0179 · exactitud 0,8300 · kappa 0,7691 |
| Versión publicada | `lgbm-49f74e944ce5457b` (los 16 caracteres corresponden al inicio del SHA-256 del archivo) |
| Archivos | `model.joblib` (pipeline serializado) y `manifest.json` (columnas, clases, procedencia, hashes y versiones) |
| Tamaño en disco | 9,3 MB (reconstrucción local con predicciones idénticas a la versión publicada) |
| Desempeño en prueba (16 sujetos, W30) | F1 macro 0,788 · exactitud 0,834 · kappa 0,768 ([evaluación única](../scripts/evaluate_test.py)) |

## Requerimientos de hardware y tiempos

- La inferencia corre en CPU; no requiere GPU. En AWS se ejecuta en una función Lambda con 3.072 MB de memoria, un límite de 300 s y dos hilos de OpenMP.
- Una noche de 841 épocas se resuelve en 1,83 s con el servicio activo. Con cinco solicitudes simultáneas, el percentil 95 del tiempo total fue de 6,51 s.
- El worker descarga el modelo una vez por contenedor y verifica su SHA-256 antes de cargarlo. Las invocaciones siguientes reutilizan el modelo en memoria (carga diferida).
- El reentrenamiento completo con 261 árboles tarda unos 5 minutos en una instancia `m5.xlarge`.

## Opción 1. Descargar la versión publicada

El modelo aprobado se guarda en el bucket de releases definido por Terraform. Con credenciales de AWS del proyecto:

```bash
python3 -c 'import json; print(json.dumps(json.load(open("infra/aws/deploy.auto.tfvars.json"))["inputs"], indent=2))'
aws s3api get-object --bucket "<buckets.releases>" --key "<objects.model.key>" \
  --version-id "<objects.model.version>" modelos/model.joblib
sha256sum modelos/model.joblib   # debe coincidir con objects.model.sha256
```

El procedimiento para restaurar `deploy.auto.tfvars.json` está en [docs/cloud-operations.md](../docs/cloud-operations.md#restaurar-y-automatizar-la-entrega).

Para evaluadores sin credenciales de AWS, el modelo desplegado (el híbrido) está versionado directamente en [stack_five/](stack_five/README.md); este LightGBM base se reconstruye con la Opción 2 o con `scripts/evaluate_test.py`.

## Opción 2. Reconstruir el modelo localmente

Requiere el dataset Sleep-EDFx descargado (unos 8 GiB) y los resultados de la campaña de variables (`var/research/sc-w30-v2/`). Desde la raíz del repositorio:

```bash
uv sync --locked --group dev --group notebook --extra dl --extra boosting
uv run --locked --extra boosting python scripts/run_research.py --source-root .
uv run --locked --extra dl --extra boosting python scripts/run_refinement.py --family lgbm
uv run --locked --extra cloud --extra boosting python scripts/export_serving_bundle.py --output var/cloud/bundle
```

El exportador verifica el SHA-256 del modelo y del dataset y genera en `var/cloud/bundle/` los archivos `model.joblib`, `manifest.json` y una noche de ejemplo (`example.parquet`).

## Opción 3. Reentrenar en la nube

```bash
infra/scripts/run-training.sh refit
```

El script lanza una instancia EC2 efímera que descarga los datos y la configuración por versión y SHA-256, reajusta los 261 árboles con la semilla 542, registra la corrida en MLflow y deja un candidato en S3. La promoción a producción se hace con `scripts/promote_cloud_candidate.py`. Ese script solo copia el candidato si reproduce las predicciones de referencia.

## Cargar el modelo y predecir

```python
import json
from pathlib import Path

import joblib

from sleep_staging.serving.contract import predictions, read_night

manifest = json.loads(Path("var/cloud/bundle/manifest.json").read_text())
model = joblib.load("var/cloud/bundle/model.joblib")
night = read_night(Path("var/cloud/bundle/example.parquet"), manifest)
result = predictions(model, manifest, night)
print(result[["epoch_id", "prediction", "prob_W", "prob_N1", "prob_N2", "prob_N3", "prob_REM"]].head())
```
