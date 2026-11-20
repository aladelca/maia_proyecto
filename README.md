# Clasificación automática de estadios del sueño

Proyecto de grado de la Maestría en Inteligencia Artificial de la Universidad de los Andes. Integrantes: Carlos Adrián Alarcón Delgado, Andrés Enrique Bautista Pabón y John Janer Buila Ferrin.

**[Abrir la aplicación desplegada](https://maia-sueno-dev-60306.vercel.app)** · [Operación en la nube](docs/cloud-operations.md) · [Modelo final](modelos/stack_five/README.md) · [Resultados](docs/comparacion-fullnight-lightgbm.md) · [Modelos, predicciones y MLflow de IBIO](output/evidencia/README.md) · [Inventario de datos](docs/epoch-inventory.md) · [Dataset Sleep-EDFx](https://physionet.org/content/sleep-edfx/1.0.0/)

## Problema y solución

Para estadificar el sueño, un especialista revisa la polisomnografía (EEG, EOG y EMG) y asigna un estadio a cada época de 30 segundos. Una noche tiene cerca de 900 épocas, así que el proceso toma horas, y la concordancia entre evaluadores ronda el 83 %. En el estadio de transición N1 la concordancia es aún menor.

Este proyecto entrena un modelo que clasifica cada época en cinco estadios (W, N1, N2, N3 y REM) y lo publica como una aplicación web que muestra el hipnograma de la noche:

1. **Datos:** Sleep Cassette de Sleep-EDFx, con 153 noches de 78 sujetos. Se separan 62 sujetos para desarrollo y 16 para la prueba final, sin mezclar noches de una misma persona.
2. El EDF aporta 145 características por época y 290 resúmenes temporales. FullNightNet aporta cinco probabilidades de etapa.
3. LightGBM combina las 440 entradas. En EDF completo obtuvo 0,7806 de F1 macro en 16 sujetos reservados. La mejora sobre el LightGBM tabular fue de 1,76 puntos (IC 95 % de +0,91 a +2,67).
4. **Despliegue:** frontend React en Vercel; API, orquestación e inferencia sin servidor en AWS (API Gateway, Lambda, Step Functions y S3); reentrenamiento bajo demanda en EC2 con MLflow. Toda la infraestructura está definida con Terraform.

## Uso de la aplicación

La aplicación es pública y no requiere usuario ni contraseña. Cada pestaña del navegador genera una clave aleatoria de invitado, de modo que un visitante solo puede ver sus propios resultados.

1. Abrir [la aplicación](https://maia-sueno-dev-60306.vercel.app).
2. Seleccionar **Probar ejemplo público** para procesar diez minutos de señal sintética, o cargar un EDF y pulsar **Generar predicción**.
3. Revisar el hipnograma y descargar las probabilidades de vigilia, N1, N2, N3 y REM por época. La consulta conserva su enlace al recargar la pestaña.

El modelo requiere EEG Fpz-Cz, EEG Pz-Oz y EOG horizontal a 100 Hz, y la envolvente de EMG submental a 1 Hz. El límite es 128 MiB y 48 horas. La API identifica el modelo como stack-five-a64415fc2bd6. [Ejemplo y contrato](docs/cloud-operations.md#probar-la-aplicación) · [Pesos y prueba local](modelos/stack_five/README.md)

La API REST (`/public/v1`), sus endpoints de salud (`/health`, `/health/ready`) y la decisión de usar Flask se documentan en [docs/api.md](docs/api.md). La API desplegada sirve su especificación OpenAPI en `/openapi.json` y una documentación interactiva en `/docs`.

## Consideraciones del modelo

- **Tamaño y hardware:** el modelo final es un LightGBM de 261 iteraciones que corre en CPU y no necesita GPU. El worker usa 3.072 MB de memoria y dos hilos.
- **Tiempos:** una noche de 841 épocas se procesa en 1,83 s. Con cinco noches simultáneas, el percentil 95 del tiempo total fue de 6,51 s.
- **Carga diferida:** el worker descarga el modelo una vez por contenedor, verifica su SHA-256 y lo reutiliza en las invocaciones siguientes.
- **Escalabilidad:** Lambda escala de forma automática con las solicitudes y el costo es proporcional al uso. No hay servidores encendidos de forma permanente.
- **Artefactos:** el modelo no se versiona en Git. [modelos/README.md](modelos/README.md) explica cómo descargarlo, reconstruirlo o reentrenarlo.
- **Limitaciones:** el recorte de vigilia de ±30 minutos usa las anotaciones de referencia. Por eso el desempeño corresponde a un análisis retrospectivo de noches preparadas, y la evaluación sobre los 16 sujetos de prueba sigue pendiente.

## Parametrización

| Dónde | Parámetro | Uso |
| --- | --- | --- |
| `.env` (a partir de `.env.example`) | `MLFLOW_TRACKING_URI` | Servidor de MLflow para la experimentación local |
| `configs/data/*.yaml` | Cohorte, canales, filtro, `wake_margin_minutes`, `max_epochs_per_recording` | Preparación de los datos |
| `configs/experiments/*.yaml` | Tarea, canales, modelos, folds, épocas y tamaño de lote | Experimentos de la CLI `sleep-stage` |
| `configs/research/*.yaml` | Espacios de búsqueda de LightGBM, CatBoost y BiLSTM, paciencia y semilla | Campañas de ajuste |
| `configs/splits/sc_subjects_v1.csv` | Partición fija por sujeto | Separación desarrollo/prueba |
| `infra/<stack>/deploy.auto.tfvars.json` | `region`, `origin`, `images`, `inputs` (versión y SHA-256 del modelo), `monthly_budget` | Despliegue con Terraform |
| Variables de la Lambda API | `JOBS_BUCKET`, `MACHINE_ARN`, `EXAMPLE_KEY`, `EXAMPLE_VERSION`, `EXAMPLE_SHA256`, `API_VERSION` | Bucket de trabajos, flujo, noche de ejemplo y versión que informa `/health` |
| Variables de la Lambda worker | `RELEASE_BUCKET`, `MANIFEST_KEY`, `MANIFEST_VERSION`, `MANIFEST_SHA256`, `OMP_NUM_THREADS` | Versión exacta del modelo y número de hilos |
| `frontend/dist/config.json` | `api`, `origin`, `edfEnabled` y los datos de Cognito (`issuer`, `clientId`, `login`), tomados de la salida pública de Terraform y guardados en Vercel como `MAIA_PUBLIC_CONFIG` | Configuración del frontend |

La versión de la imagen fija ambos checkpoints y el ejemplo del híbrido. Para revertir el servicio se restauran juntas las imágenes anteriores de API y worker.

## Reproducir el despliegue

Requisitos: AWS CLI, Terraform 1.10 o superior, Docker con buildx, Node 22 y uv. Las dependencias de Python están fijadas en `pyproject.toml` y `uv.lock`, y las del frontend en `frontend/package-lock.json`. El procedimiento completo está en [docs/cloud-operations.md](docs/cloud-operations.md):

1. Crear el estado remoto (`infra/scripts/bootstrap.sh`) y aplicar los stacks `platform` y `network`.
2. Construir y publicar las imágenes `api`, `worker` y `training` (`infra/scripts/push-image.sh`).
3. Exportar y publicar el modelo (`scripts/export_serving_bundle.py` y `scripts/publish_cloud_inputs.py`).
4. Aplicar los stacks `training`, `aws` y `vercel`, y compilar el frontend.
5. Verificar el servicio con `uv run --extra cloud python scripts/smoke_cloud.py --guest`.

Las credenciales de AWS se cargan desde un `.env` local, que no se versiona. El repositorio no contiene secretos.

## Estructura del repositorio

| Carpeta | Contenido |
| --- | --- |
| `src/sleep_staging/` | Paquete: datos, características, modelos, evaluación y servicio de inferencia (`serving/`) |
| `services/api/` | API de la aplicación |
| `frontend/` | Interfaz web (React + Vite) y pruebas end-to-end con Playwright |
| `notebooks/` | Auditoría de datos, análisis de variables y campañas de experimentación |
| `modelos/` | Ficha del modelo final e instrucciones de descarga y reconstrucción |
| `configs/` | Configuración de datos, experimentos, búsquedas y partición |
| `scripts/` | Campañas, exportación, publicación y pruebas del servicio |
| `infra/`, `containers/` | Terraform, scripts de despliegue y Dockerfiles |
| `docs/`, `tests/` | Documentación, resultados y pruebas automatizadas |

## Experimentación local

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

La campaña de mejora se revisa en [notebooks/04_literature_catboost.ipynb](notebooks/04_literature_catboost.ipynb). Compara los 62 sujetos de desarrollo con cinco folds fijos: variables originales, nuevos descriptores, canales adicionales, contexto temporal retrospectivo y ajuste interno de CatBoost por sujeto. El [protocolo y la revisión de literatura](docs/literature-model-improvements.md) documentan los experimentos y sus límites. Los 16 sujetos reservados quedan fuera de la preparación y de la evaluación.

```bash
uv sync --locked --group dev --group notebook --extra boosting --extra dl
uv run --locked --extra boosting python scripts/run_research.py \
  --source-root /ruta/al/proyecto/que/contiene/data
```

Resultado de la campaña: **LightGBM con contexto, F1 macro 0,7822 ± 0,0161**, frente a 0,6544 ± 0,0212 de RF recalculado sobre los mismos segmentos. El [resumen de resultados](docs/research-results-20261116.md) incluye ablaciones, incertidumbre y limitaciones.

Las nuevas corridas, cachés y el candidato se guardan en `var/research/sc-w30-v2/`, con tracking MLflow propio y checkpoints por fold. El notebook consulta los resultados guardados; activar `RUN_TRAINING` permite entrenar explícitamente. Ejecutar `scripts/sync_research_tracking.py` después de la campaña expone los 15 ensayos internos, el ensamble y el candidato nativo en MLflow; la UI de esta campaña está en [localhost:5002](http://127.0.0.1:5002).

## LSTM y BiLSTM en MPS

La campaña de [notebooks/05_lstm_mps.ipynb](notebooks/05_lstm_mps.ipynb) compara un control neuronal de un segmento, una LSTM con 15 segmentos pasados y una BiLSTM con 15 segmentos centrados. Usa las 145 variables multicanal por segmento, los mismos sujetos y objetivos de desarrollo y cinco folds fijos. Dentro de cada entrenamiento externo se seleccionan learning rate y número de épocas con un holdout de sujetos; después se reentrena sin consultar el fold externo.

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

LightGBM ajustado obtuvo el mayor F1 macro en desarrollo, **0,7893**. Frente a su referencia, LightGBM pasó de 0,7822 a 0,7893 y BiLSTM de 0,7687 a 0,7800. Se compararon 12 configuraciones LightGBM y seis BiLSTM en cinco folds por sujeto. Las 120 corridas de entrenamiento están en [MLflow local](http://127.0.0.1:5002/#/experiments/3). El [informe](docs/refinement-results.md) incluye hiperparámetros finales y comandos de reproducción. El [notebook ejecutado](notebooks/06_lgbm_bilstm_tuning.ipynb) muestra las curvas y el F1 por estadio.

## Validación

```bash
make check
make test
```

GitHub Actions ejecuta estos mismos comandos en cada pull request. Vercel compila `frontend/` y publica la web directamente en cada merge a `main`. GitHub Actions despliega los cambios del backend en AWS y comprueba la paridad de las predicciones. [Integración y entrega continuas](docs/cloud-operations.md#integración-y-entrega-continuas)

Las pruebas de integración de señales y entrenamiento usan los EDF reales del piloto. Si aún no se descargaron, se marcan como omitidas; `make test` informa cuántas corrieron. Las pruebas unitarias de fórmulas y validación de metadatos no sustituyen un dataset experimental.

El [plan de implementación inicial](plans/20261020-1943-infraestructura-local-sleep-edfx-mlflow.md) conserva el historial del piloto. La campaña IBIO posterior completó la evaluación reservada, la transferencia a Telemetry y la carga de EDF completos.

## Experimento histórico de seis clases

La [guía de ejecución](docs/ibio-registro-completo.md) permite cargar una PSG, elegir entre 12 combinaciones de EEG, EOG y EMG y descargar un hipnograma con seis probabilidades por época. El [informe experimental](output/pdf/resultados-ibio-registro-completo.pdf) incluye las matrices de cinco y seis clases, los errores de N1 y la referencia U-Sleep reentrenada. En los 16 participantes reservados de Cassette, añadir el espectrograma elevó el F1 macro de cinco clases de 0,740 a 0,752. El modelo de movimiento detectó 5 de 13 épocas M.

Datos y licencia: [Sleep-EDFx 1.0.0 en PhysioNet](https://physionet.org/content/sleep-edfx/1.0.0/), Open Data Commons Attribution License 1.0. Conservar las citas y atribución indicadas por el repositorio de origen al publicar resultados.

## Laboratorio desplegado

El frontend en Vercel permite procesar un EDF, ver su hipnograma y descargar las probabilidades por época. AWS ejecuta la inferencia con los dos checkpoints del híbrido incluidos en la imagen. El entrenamiento bajo demanda se realiza en EC2 y registra cada ajuste en MLflow. [Abrir el laboratorio](https://maia-sueno-dev-60306.vercel.app) · [Despliegue y operación](docs/cloud-operations.md)
