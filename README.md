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

## LSTM y BiLSTM en MPS

La campaña de [notebook/05_lstm_mps.ipynb](notebook/05_lstm_mps.ipynb) compara un control neuronal de un segmento, una LSTM con 15 segmentos pasados y una BiLSTM con 15 segmentos centrados. Usa las 145 variables multicanal por segmento, los mismos sujetos y objetivos de desarrollo y cinco folds fijos. Dentro de cada entrenamiento externo se seleccionan learning rate y número de épocas con un holdout de sujetos; después se reentrena sin consultar el fold externo.

```bash
env -u SLEEP_STAGING_ROOT PYTORCH_ENABLE_MPS_FALLBACK=0 \
  uv run --locked --extra dl --extra boosting python scripts/run_lstm_research.py \
  --mode preflight
env -u SLEEP_STAGING_ROOT PYTORCH_ENABLE_MPS_FALLBACK=0 \
  uv run --locked --extra dl --extra boosting python scripts/run_lstm_research.py \
  --mode run --resume
```

Requiere el dataset y los resultados de la campaña anterior. Entrenamiento y predicción de la red exigen MPS; el preprocesamiento y las métricas usan CPU. Cada ensayo, fold, curva y checkpoint se registra en el experimento MLflow `sleep-cassette-lstm-mps-v1`, visible en la misma UI del puerto 5002. Los artefactos nuevos quedan en `var/research/sc-w30-lstm-mps-v1/`. El [informe de resultados](docs/lstm-mps-results.md) compara las redes con LightGBM y documenta su contrato de secuencias y sus limitaciones.

Resultado: la mejor red fue **BiLSTM, F1 macro 0,7687 ± 0,0211**. En esa campaña, LightGBM obtuvo el mejor F1 macro de desarrollo, 0,7822 ± 0,0161. Se completaron 48 entrenamientos neuronales con ajuste interno y refit, sin evaluar el test reservado.

## Ajuste de LightGBM y BiLSTM

LightGBM ajustado obtuvo el mayor F1 macro en desarrollo, **0,7893**. Frente a su referencia, LightGBM pasó de 0,7822 a 0,7893 y BiLSTM de 0,7687 a 0,7800. Se compararon 12 configuraciones LightGBM y seis BiLSTM en cinco folds por sujeto. Las 120 corridas de entrenamiento están en [MLflow local](http://127.0.0.1:5002/#/experiments/3). El [informe](docs/refinement-results.md) incluye hiperparámetros finales y comandos de reproducción. El [notebook ejecutado](notebook/06_lgbm_bilstm_tuning.ipynb) muestra las curvas y el F1 por estadio.

## Validación

```bash
make check
make test
```

GitHub Actions ejecuta estos mismos comandos en cada pull request. Al integrar en `main`, publica la inferencia y el frontend tras una prueba de paridad. [Integración y entrega continuas](docs/cloud-operations.md#integración-y-entrega-continuas)

Las pruebas de integración de señales y entrenamiento usan los EDF reales del piloto. Si aún no se descargaron, se marcan como omitidas; `make test` informa cuántas corrieron. Las pruebas unitarias de fórmulas y validación de metadatos no sustituyen un dataset experimental.

El [plan de implementación inicial](plans/20261020-1943-infraestructura-local-sleep-edfx-mlflow.md) conserva el historial del piloto. La campaña IBIO posterior completó la evaluación reservada, la transferencia a Telemetry y la carga de EDF completos.

## Estadificación de un EDF completo

La [guía de ejecución](docs/ibio-registro-completo.md) permite cargar una PSG, elegir entre 12 combinaciones de EEG, EOG y EMG y descargar un hipnograma con seis probabilidades por época. El [informe experimental](output/pdf/resultados-ibio-registro-completo.pdf) incluye las matrices de cinco y seis clases, los errores de N1 y la referencia U-Sleep reentrenada. En los 16 participantes reservados de Cassette, añadir el espectrograma elevó el F1 macro de cinco clases de 0,740 a 0,752. El modelo de movimiento detectó 5 de 13 épocas M.

Datos y licencia: [Sleep-EDFx 1.0.0 en PhysioNet](https://physionet.org/content/sleep-edfx/1.0.0/), Open Data Commons Attribution License 1.0. Conservar las citas y atribución indicadas por el repositorio de origen al publicar resultados.

## Laboratorio desplegado

El frontend en Vercel permite procesar una noche preparada, ver su hipnograma y descargar las probabilidades por época. AWS ejecuta la inferencia con el campeón versionado en S3. El entrenamiento bajo demanda se realiza en EC2 y registra cada ajuste en MLflow. [Abrir el laboratorio](https://maia-sueno-dev-60306.vercel.app) · [Despliegue y operación](docs/cloud-operations.md)
