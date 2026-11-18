# Estadificación del sueño: entorno local

Procesamiento y experimentación con **Sleep-EDFx real**: EDF/hipnogramas → épocas de 30 segundos → características o EEG para CNN → validación por sujeto → MLflow.

El dataset completo está descargado y auditado localmente: **197 PSG con sus 197 hipnogramas** —153 de Sleep Cassette y 44 de Sleep Telemetry—, más cuatro archivos de metadatos. Los 398 archivos se verificaron con SHA-256 y ocupan **8,117 GiB**. El inventario completo está en `data/manifests/sleep-edfx-complete/summary.json`.

El piloto usa ocho sujetos de Sleep Cassette, una noche por sujeto y hasta 240 épocas distribuidas a lo largo de cada registro. Reserva dos sujetos para test y utiliza cinco folds sobre los seis sujetos restantes. Los resultados del piloto sirven para comprobar el flujo y preparar la campaña completa.

## Inicio rápido

Desde la raíz del proyecto, con `uv` disponible:

```bash
uv sync --locked --group dev --group notebook --extra dl
cp .env.example .env
make init-dirs
```

En una primera terminal, mantener MLflow en primer plano:

```bash
make mlflow
```

Abrir [MLflow local](http://127.0.0.1:5001). En otra terminal:

```bash
uv run --locked sleep-stage doctor --check-mlflow
uv run --locked sleep-stage download --config configs/data/pilot.yaml
uv run --locked sleep-stage audit --config configs/data/pilot.yaml
uv run --locked sleep-stage experiment --config configs/experiments/rf-eeg.yaml
uv run --locked sleep-stage experiment --config configs/experiments/cnn-eeg.yaml
uv run --locked sleep-stage report --config configs/experiments/rf-eeg.yaml
```

`experiment` prepara datos, features y particiones cuando no existe una caché compatible. La descarga es explícita. Los registros provienen del espejo público oficial de PhysioNet en S3 y se verifican con sus SHA-256 publicados.

La **[guía de ejecución local](docs/local-operations.md)** detalla requisitos, configuración, ejecución por etapas, notebooks, predicción, detención, respaldos y solución de problemas. Los [contratos de datos](docs/data-contract.md) y el [protocolo experimental](docs/experiment-protocol.md) explican la trazabilidad y la separación por sujeto.

## Resultado del piloto real

Validación de desarrollo con cinco folds por sujeto, cinco clases y EEG Fpz-Cz:

| Modelo | F1 macro medio ± desviación | Accuracy media |
| --- | --- | --- |
| Random Forest | **0,5830 ± 0,1398** | 0,7800 |
| CNN 1D, tres épocas máximas | 0,2837 ± 0,0455 | 0,4625 |
| Baseline mayoritario | 0,1651 ± 0,0041 | 0,7033 |

Random Forest es el ganador por F1 macro en este piloto. Los dos sujetos reservados para test mantienen su evaluación final pendiente. La campaña ampliada de desarrollo se describe en la sección siguiente.

## Nuevas variables y ajuste de CatBoost

La campaña de mejora se revisa en [notebook/04_literature_catboost.ipynb](notebook/04_literature_catboost.ipynb). Compara los 62 sujetos de desarrollo con cinco folds fijos: variables originales, nuevos descriptores, canales adicionales, contexto temporal retrospectivo y ajuste interno de CatBoost por sujeto. El [protocolo y la revisión de literatura](docs/literature-model-improvements.md) documentan los experimentos y sus límites. Los 16 sujetos reservados quedan fuera de la preparación y de la evaluación.

```bash
uv sync --locked --group dev --group notebook --extra boosting --extra dl
uv run --locked --extra boosting python scripts/run_research.py \
  --source-root /ruta/al/proyecto/que/contiene/data
```

Resultado de la campaña: **LightGBM con contexto, F1 macro 0,7822 ± 0,0161**, frente a 0,6544 ± 0,0212 de RF recalculado sobre los mismos segmentos. El [resumen de resultados](docs/research-results-20261116.md) incluye ablaciones, incertidumbre y limitaciones.

Las nuevas corridas, cachés y el candidato se guardan en `var/research/sc-w30-v2/`, con tracking MLflow propio y checkpoints por fold. El notebook consulta los resultados guardados; activar `RUN_TRAINING` permite entrenar explícitamente. Ejecutar `scripts/sync_research_tracking.py` después de la campaña expone los 15 ensayos internos, el ensamble y el candidato nativo en MLflow; la UI de esta campaña está en [localhost:5002](http://127.0.0.1:5002).

## Validación

```bash
make check
make test
```

Las pruebas de integración de señales y entrenamiento usan los EDF reales del piloto. Si aún no se descargaron, se marcan como omitidas; `make test` informa cuántas corrieron. Las pruebas unitarias de fórmulas y validación de metadatos no sustituyen un dataset experimental.

El [plan de implementación](plans/20261020-1943-infraestructura-local-sleep-edfx-mlflow.md) conserva el avance y las ampliaciones pendientes. La evaluación final del test reservado, el análisis Telemetry y la integración para EDF nuevos permanecen pendientes.

Datos y licencia: [Sleep-EDFx 1.0.0 en PhysioNet](https://physionet.org/content/sleep-edfx/1.0.0/), Open Data Commons Attribution License 1.0. Conservar las citas y atribución indicadas por el repositorio de origen al publicar resultados.
