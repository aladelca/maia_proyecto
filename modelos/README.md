# Modelo publicado: híbrido `stack_five`

La aplicación desplegada usa el híbrido **FullNightNet + LightGBM** (`stack_five`). Clasifica cada época de 30 s en cinco estadios: W, N1, N2, N3 y REM. Los pesos están en esta carpeta ([`stack_five/`](stack_five/)), se versionan en Git y se copian en la imagen del worker. No hace falta descargarlos de otro sitio. [Ejecución local](stack_five/README.md) · [Evaluación emparejada](../docs/comparacion-fullnight-lightgbm.md) · [Notebook](../notebooks/10_hibrido_fullnight_lightgbm.ipynb)

## Ficha

| Atributo | Valor |
| --- | --- |
| Versión servida | `stack-five-a64415fc2bd6`, que la API devuelve en `model_version` |
| Corrida MLflow | `a64415fc2bd649b39c6fd158994b0479`, brazo `stack_five` de la corrida padre `bb7fa78abcfb41ab87c86ee89755ae87` (experimento `sleep-cassette-lgbm-bilstm-refinement-v1`) |
| Entrada | EDF con EEG Fpz-Cz, EEG Pz-Oz y EOG horizontal a 100 Hz, y envolvente EMG submental a 1 Hz. Hasta 128 MiB y 48 horas. |
| Rama tabular | 145 variables por época de los cuatro canales y 290 resúmenes de contexto temporal: 435 columnas |
| Rama neuronal | FullNightNet (`six_direct.pt`) recorre la noche completa y estima seis probabilidades (W, N1, N2, N3, REM y movimiento). Se descarta la de movimiento y las cinco restantes se renormalizan. |
| Clasificador final | `Pipeline` de scikit-learn: imputación por mediana, estandarización y `LGBMClassifier` sobre las 440 columnas |
| Hiperparámetros | 261 árboles, 63 hojas, mínimo 200 muestras por hoja, tasa de aprendizaje 0,05, 63 bins, L2 = 5, 80 % de filas y columnas, pesos inversos de clase, semilla 542. Son los del LightGBM tabular aprobado, que se clona para el ajuste. |
| Entrenamiento | 62 participantes de desarrollo, 122 noches, 152.052 épocas etiquetadas. Las probabilidades de FullNightNet de cada época provienen del pliegue que no entrenó con ella. Los 16 participantes de prueba quedan fuera. |
| Desempeño | Desarrollo W30, cinco pliegues por sujeto: F1 macro 0,7940 ± 0,0165. Prueba reservada (16 participantes, 31 noches), EDF completo: F1 macro 0,7806 y F1 de N1 0,4841, frente a 0,7630 y 0,4530 del LightGBM solo. |
| Archivos | `stack_five.joblib` (4.361.420 bytes), `six_direct.pt` (51.145 bytes) y `example-expected.csv` (salida esperada del EDF de ejemplo) |
| Integridad | Los SHA-256 de ambos archivos están en [`src/sleep_staging/stack_release.py`](../src/sleep_staging/stack_release.py). El worker los comprueba antes de deserializarlos. |

## Hardware y tiempos

- Corre en CPU y no necesita GPU. En AWS se ejecuta en una función Lambda con 3.072 MB, dos hilos y un límite de 300 s; Step Functions corta el flujo a los 420 s.
- **Carga diferida.** El clasificador se carga una vez por contenedor y queda en memoria (`lru_cache`). La red, de 50 KiB, se verifica y se carga en un subproceso por trabajo, porque torch y LightGBM usan runtimes de OpenMP incompatibles. Torch solo se importa en ese subproceso.
- **Tiempos medidos el 23 de noviembre de 2026** con el ejemplo público (10 minutos, 20 épocas): la primera solicitud tardó 10,3 s de punta a punta (5,6 s de inferencia) y las dos siguientes 4,9 s (4,05 s de inferencia). En local, en un Apple M4 Max, el mismo EDF tarda 5,1 s.
- No hay una medición del híbrido con una noche completa. Como referencia de la rama neuronal sola, FullNightNet procesó en AWS una noche de 2.786 épocas en 48,4 s en frío y 2,1 s en caliente ([evidencia](../docs/cloud-operations.md#evidencia-del-despliegue-de-desarrollo)).
- El reajuste del clasificador final con las 152.052 épocas tardó 86 s en el mismo equipo local.

## Usar el modelo

Desde la raíz, con el entorno instalado (`uv sync --locked --group dev --extra dl --extra boosting --extra cloud`):

```sh
.venv/bin/python -m sleep_staging.serving.stacked_edf output/demo-sueno-sintetico-10min.edf hipnograma.csv
```

Imprime el resumen de la release y escribe una fila por época con `prediction` y `prob_W` … `prob_REM`. Con el EDF de ejemplo, la salida coincide con [`stack_five/example-expected.csv`](stack_five/example-expected.csv). Desde Python:

```python
from pathlib import Path

from sleep_staging.serving.stacked_edf import score, summary

result = score(Path("noche-PSG.edf"), "noche")
print(summary(result))
print(result[["onset_s", "prediction", "prob_W", "prob_N1", "prob_N2", "prob_N3", "prob_REM"]].head())
```

Para probar con una noche real, [SC4011E0-PSG.edf](https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/SC4011E0-PSG.edf) (51 MB) pertenece a un participante de la partición de prueba, así que el modelo no la vio al entrenar.

## Reentrenar

### En la nube

`train_cloud.py --mode stack`, que es el modo por defecto, reajusta el clasificador final en una instancia EC2 efímera y registra la corrida en MLflow:

```sh
uv run --extra cloud --extra boosting --extra dl python scripts/publish_cloud_inputs.py \
  --edf-run var/research/ibio-fullnight/development/681a52f21d273bf190ae \
  --stack-run var/research/ibio-fullnight/development/681a52f21d273bf190ae
python3 infra/scripts/configure-training.py   # en el laboratorio, con --instance-profile y --job-role
terraform -chdir=infra/training apply
infra/scripts/run-training.sh stack
uv run --extra cloud --extra boosting --extra dl python scripts/promote_stack_candidate.py "$CAMPAIGN_ID"
```

- **Entradas.** `--stack-run` publica dos entradas en el bucket de investigación, cada una con versión S3 y SHA-256: las probabilidades fuera de pliegue de FullNightNet (`fullnight_oof`) y las 440 entradas del EDF de ejemplo (`stack_example`). El entrenamiento descarga además el dataset de variables, la partición y el LightGBM aprobado, cuya configuración clona.
- **Promoción.** `promote_stack_candidate.py` verifica la procedencia y los SHA-256, puntúa el EDF de ejemplo con el candidato y exige las mismas etapas y probabilidades dentro de 10⁻⁶. Si el candidato es idéntico al modelo servido, no cambia nada. Si no lo es, instala el clasificador en `stack_five/`, regenera `example-expected.csv` y `stack_release.py`, y actualiza la versión que fija la web. El cambio se publica con un PR: al mergearlo, `deploy.yml` construye el worker y repite la prueba de paridad en AWS.
- **FullNightNet.** La red queda congelada: entrenarla requiere GPU o MPS y la caché de señales de la campaña IBIO ([guía](../docs/ibio-registro-completo.md)).

### En local

Con el dataset Sleep-EDFx descargado y las campañas previas en `var/research/`, que no se versionan, la evaluación completa se repite así ([detalle](../docs/comparacion-fullnight-lightgbm.md)):

```sh
.venv/bin/python scripts/stack_fullnight_folds.py --device auto   # 10 redes internas para los pliegues
.venv/bin/python scripts/compare_fullnight_lgbm.py                # ajusta y evalúa los híbridos
.venv/bin/python scripts/compare_fullnight_edf.py
.venv/bin/python scripts/report_fullnight_lgbm.py
```

`compare_fullnight_lgbm.py` deja el clasificador servido en `var/research/fullnight-lgbm-w30/models/reserved/stack_five.joblib`. El 22 de noviembre de 2026 se reajustó en local con el código de `train_cloud.py --mode stack` y se obtuvo el mismo SHA-256 que el archivo publicado (`4a7021f0…`).

## Componente tabular: LightGBM `lgbm-49f74e944ce5457b`

Es el LightGBM de 435 columnas que precedió al híbrido: F1 macro de desarrollo 0,7893 ± 0,0179 y 0,7630 en la prueba con EDF completo. Sus hiperparámetros son la base del híbrido. Se guarda en el bucket de releases y lo reentrenan `run-training.sh refit` (261 árboles) y `run-training.sh tune` (calibración de 12 configuraciones), que se promueven con `promote_cloud_candidate.py`. La API desplegada no lo sirve: la imagen fija `ACTIVE_EDF_MODEL=stack_five` y rechaza la ruta Parquet de ese modelo. [Resultados del ajuste](../docs/refinement-results.md)
