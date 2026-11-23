# Release del híbrido de cinco etapas

Esta carpeta contiene el modelo que sirve la aplicación, `stack-five-a64415fc2bd6`. La identidad de la release (corrida MLflow, versión y SHA-256 de cada archivo) está en [`src/sleep_staging/stack_release.py`](../../src/sleep_staging/stack_release.py). El worker la comprueba antes de deserializar los pesos. [Ficha del modelo](../README.md)

| Archivo | Contenido |
| --- | --- |
| `stack_five.joblib` | Clasificador final: imputación, estandarización y LightGBM sobre 435 columnas tabulares y cinco probabilidades de FullNightNet |
| `six_direct.pt` | FullNightNet reajustada con los 62 participantes de desarrollo. Aporta las probabilidades de etapa. |
| `example-expected.csv` | Salida esperada para el EDF sintético de diez minutos del repositorio (`output/demo-sueno-sintetico-10min.edf`, 20 épocas) |

Para puntuar un EDF en CPU desde la raíz:

```sh
.venv/bin/python -m sleep_staging.serving.stacked_edf noche.edf hipnograma.csv
```

La API publicada admite únicamente EDF con ambos EEG, EOG horizontal y EMG submental en envolvente. Las solicitudes de carga y de predicción envían `format="edf"` y pueden fijar `model="stack_five"`. El ejemplo se ejecuta con el mismo worker y devuelve 20 épocas. [Resultados de la evaluación pareada](../../docs/comparacion-fullnight-lightgbm.md)

Para usar el modelo con la interfaz en este equipo, ejecuta `npm --prefix frontend run build` y `.venv/bin/python scripts/serve_stack.py`, y abre el laboratorio local en `http://127.0.0.1:8775`.

Un nuevo clasificador llega a esta carpeta solo mediante `scripts/promote_stack_candidate.py`, después de entrenarse con `infra/scripts/run-training.sh stack`. El script reemplaza `stack_five.joblib` y `example-expected.csv` y regenera `stack_release.py`. [Reentrenar](../README.md#reentrenar)
