# Clasificación automática de estadios del sueño

Proyecto de grado de la Maestría en Inteligencia Artificial de la Universidad de los Andes. Integrantes: Carlos Adrián Alarcón Delgado, Andrés Enrique Bautista Pabón y John Janer Buila Ferrin.

**[Abrir la aplicación desplegada](https://maia-sueno-dev-60306.vercel.app)** · [Operación en la nube](docs/cloud-operations.md) · [API](docs/api.md) · [Modelo final](modelos/README.md) · [Resultados](docs/comparacion-fullnight-lightgbm.md) · [Modelos, predicciones y MLflow de IBIO](output/evidencia/README.md) · [Inventario de datos](docs/epoch-inventory.md) · [Dataset Sleep-EDFx](https://physionet.org/content/sleep-edfx/1.0.0/)

## Problema y solución

Para estadificar el sueño, un especialista revisa la polisomnografía (EEG, EOG y EMG) y asigna un estadio a cada época de 30 segundos. Una noche tiene cerca de 900 épocas, así que el proceso toma horas, y la concordancia entre evaluadores ronda el 83 %. En el estadio de transición N1 la concordancia es aún menor.

Este proyecto entrena un modelo que clasifica cada época en cinco estadios (W, N1, N2, N3 y REM) y lo publica como una aplicación web que muestra el hipnograma de la noche:

1. **Datos:** Sleep Cassette de Sleep-EDFx, con 153 noches de 78 sujetos. Se separan 62 sujetos para desarrollo y 16 para la prueba final, sin mezclar noches de una misma persona.
2. **Modelo:** el híbrido `stack_five` tiene dos ramas. Del EDF se extraen 145 características por época y 290 resúmenes temporales. La red FullNightNet recorre la noche completa y aporta cinco probabilidades de etapa. Un LightGBM clasifica las 440 entradas.
3. **Resultado:** en los 16 sujetos reservados, puntuando el EDF completo, obtuvo 0,7806 de F1 macro. La mejora sobre el LightGBM tabular fue de 1,76 puntos (IC 95 % de +0,91 a +2,67).
4. **Despliegue:** frontend React en Vercel; API, orquestación e inferencia sin servidor en AWS (API Gateway, Lambda, Step Functions y S3); reentrenamiento bajo demanda del clasificador en EC2 con MLflow. Toda la infraestructura está definida con Terraform.

## Uso de la aplicación

La aplicación es pública y no requiere usuario ni contraseña. Cada pestaña del navegador genera una clave aleatoria de invitado, de modo que un visitante solo puede ver sus propios resultados.

1. Abrir [la aplicación](https://maia-sueno-dev-60306.vercel.app).
2. Seleccionar **Probar ejemplo público** para procesar diez minutos de señal sintética, o cargar un EDF y pulsar **Generar predicción**. Para probar con una noche real, descargar [SC4011E0-PSG.edf](https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/SC4011E0-PSG.edf) (51 MB) de PhysioNet. Pertenece a un participante de la partición de prueba, que el modelo no vio al entrenar.
3. Revisar el hipnograma, el tiempo en cada estadio y el resumen de la noche, y descargar las probabilidades de vigilia, N1, N2, N3 y REM por época. La consulta conserva su enlace al recargar la pestaña.

El modelo requiere EEG Fpz-Cz, EEG Pz-Oz y EOG horizontal a 100 Hz, y la envolvente de EMG submental a 1 Hz. El límite es 128 MiB y 48 horas. La API identifica el modelo como `stack-five-a64415fc2bd6`. [Ficha del modelo](modelos/README.md)

**Disponibilidad.** La web en Vercel está siempre en línea. La API corre en un laboratorio de AWS Academy y responde HTTP 500 mientras la sesión del laboratorio está detenida. En ese caso, la web avisa que el servicio no confirma el modelo y solo permite el modo demo. Antes de una evaluación hay que iniciar la sesión y comprobar `GET {api}/health/ready`.

La API REST (`/public/v1`), sus endpoints de salud (`/health`, `/health/ready`) y la decisión de usar Flask se documentan en [docs/api.md](docs/api.md). La API desplegada sirve su especificación OpenAPI en `/openapi.json` y una documentación interactiva en `/docs`.

### Ejemplo de entrada y salida

La entrada es un EDF. El ejemplo público usa el EDF sintético del repositorio ([output/demo-sueno-sintetico-10min.edf](output/demo-sueno-sintetico-10min.edf)), con los cuatro canales y 20 épocas. Como invitado, desde la terminal:

```bash
api=$(curl -s https://maia-sueno-dev-60306.vercel.app/config.json | jq -r .api)
token=$(openssl rand -hex 32)
status=$(curl -s -X POST "$api/public/v1/predictions" \
  -H "X-Guest-Token: $token" -H "Idempotency-Key: $(uuidgen)" \
  -H 'content-type: application/json' -d '{"example":"default","model":"stack_five"}' | jq -r .status_url)
curl -s "$api$status" -H "X-Guest-Token: $token"   # repetir hasta que state sea succeeded
```

Fragmento de la respuesta final, obtenida el 23 de noviembre de 2026:

```json
{
  "state": "succeeded",
  "model_version": "stack-five-a64415fc2bd6",
  "mlflow_run_id": "a64415fc2bd649b39c6fd158994b0479",
  "profile": "eeg_fpz_cz+eeg_pz_oz+eog+emg",
  "window": "full",
  "classes": ["W", "N1", "N2", "N3", "REM"],
  "rows": 20,
  "eligible": 20,
  "inference_seconds": 4.055,
  "files": {"csv": "https://…", "json": "https://…"}
}
```

`files` contiene enlaces temporales. El CSV trae una fila por época, con la etapa predicha y las cinco probabilidades:

```csv
recording_id,epoch_id,onset_s,eligible,prediction,prob_W,prob_N1,prob_N2,prob_N3,prob_REM
<job>,<job>:0,0,True,W,0.9623689050052214,0.03316560511942778,0.0012077137064328723,0.0001259363470790804,0.003131839821838886
<job>,<job>:3000,30,True,W,0.8316801326078458,0.15743967819534452,0.008462318837443512,0.00020933191790500365,0.0022085384414612845
```

Para un archivo propio se pide `POST /public/v1/uploads` con su tamaño y SHA-256, se sube a S3 con el formulario firmado y se inicia la predicción con `upload_id` y `version_id`. [Contrato completo](docs/api.md#ejemplo-de-uso-como-invitado)

## Consideraciones del modelo

- **Tamaño:** el clasificador ocupa 4,2 MiB (`stack_five.joblib`) y la red FullNightNet 50 KiB (`six_direct.pt`). Ambos se versionan en Git en [modelos/stack_five/](modelos/stack_five/) y se copian en la imagen del worker, así que no hay descargas externas. El worker verifica sus SHA-256 antes de cargarlos.
- **Hardware:** corre en CPU y no necesita GPU. El worker es una Lambda con 3.072 MB de memoria, dos hilos y un límite de 300 s.
- **Tiempos:** con el ejemplo público de 20 épocas, la primera solicitud tardó 10,3 s de punta a punta y las siguientes 4,9 s (4,05 s de inferencia), medidos el 23 de noviembre de 2026. No hay una medición del híbrido con una noche completa. Como referencia, FullNightNet sola procesó en AWS una noche de 2.786 épocas en 48,4 s en frío y 2,1 s en caliente.
- **Carga diferida:** el clasificador se carga una vez por contenedor y se reutiliza en las invocaciones siguientes. La red se carga en un subproceso por trabajo, y torch solo se importa allí.
- **Escalabilidad:** cada análisis es un trabajo asíncrono de Step Functions. Lambda escala con las solicitudes y el costo es proporcional al uso. API Gateway limita la API a 10 solicitudes/s con ráfagas de 15. No hay servidores encendidos de forma permanente.
- **Limitaciones:** las métricas de desarrollo usan el recorte W30, que conserva 30 minutos de vigilia alrededor del sueño anotado y por eso depende de las anotaciones. La aplicación puntúa el EDF completo, que en la prueba reservada dio 0,7806 frente a 0,7970 con W30. N1 sigue siendo la etapa más difícil (F1 de 0,4841). El modelo se entrenó con Sleep Cassette: otros montajes, o una EMG cruda en lugar de su envolvente, no son equivalentes.

## Parametrización

| Dónde | Parámetro | Uso |
| --- | --- | --- |
| `.env` (a partir de `.env.example`) | `MLFLOW_TRACKING_URI`; `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, `AWS_REGION`; `VERCEL_API_TOKEN`, `TF_VAR_project_name`, `TF_VAR_team_id` | MLflow local, credenciales de la sesión de AWS y proyecto de Vercel. No se versiona. |
| `configs/data/*.yaml` | Cohorte, canales, filtro, `wake_margin_minutes`, `max_epochs_per_recording` | Preparación de los datos |
| `configs/experiments/*.yaml` | Tarea, canales, modelos, folds, épocas y tamaño de lote | Experimentos de la CLI `sleep-stage` |
| `configs/research/*.yaml` | Espacios de búsqueda de LightGBM, CatBoost y BiLSTM, paciencia y semilla | Campañas de ajuste |
| `configs/splits/sc_subjects_v1.csv` | Partición fija por sujeto | Separación desarrollo/prueba |
| `src/sleep_staging/stack_release.py` | `RUN_ID`, `VERSION`, `CHECKPOINTS`, `EXAMPLE_SHA256` | Identidad y SHA-256 del modelo servido. Lo regenera `scripts/promote_stack_candidate.py`. |
| `infra/<stack>/deploy.auto.tfvars.json` | `region`, `origin`, `images`, `inputs` (versión y SHA-256 de cada entrada), `edf_lgbm_enabled`, `monthly_budget` | Despliegue con Terraform |
| Imagen y variables de la Lambda API | `ACTIVE_EDF_MODEL=stack_five` (en la imagen); `JOBS_BUCKET`, `MACHINE_ARN`, `EXAMPLE_KEY`, `EXAMPLE_VERSION`, `EXAMPLE_SHA256`, `API_VERSION`, `EDF_ENABLED`, `EDF_LGBM_ENABLED` | Con `ACTIVE_EDF_MODEL=stack_five` la API solo acepta EDF con el híbrido. Las demás variables apuntan al bucket, al flujo y a las rutas históricas. |
| Imagen y variables de la Lambda worker | `STACK_MODEL_DIR`, `STACK_EXAMPLE_EDF` (en la imagen); `JOBS_BUCKET`, `OMP_NUM_THREADS` | Carpeta de pesos del híbrido, EDF de ejemplo, bucket de trabajos e hilos. `RELEASE_BUCKET`, `MANIFEST_*` y `EDF_MODEL_*` solo sirven a las rutas históricas de Parquet y FullNightNet. |
| `infra/scripts/run-training.sh` | Modo `stack` (por defecto), `refit` o `tune`; `CAMPAIGN_ID` para reanudar; `TRAIN_INSTANCE_TYPE` (por defecto `t3.large`) | Reentrenamiento en EC2. El contenedor recibe las entradas en `CLOUD_INPUTS` y el tracking en `MLFLOW_TRACKING_URI`. |
| `frontend/dist/config.json`, tomado de Vercel `MAIA_PUBLIC_CONFIG` | `api` | URL de la API que usa la web. `edfEnabled`, `edfLgbmEnabled` y los datos de Cognito se publican para las integraciones en `/v1` y para el smoke test. |

## Reproducir el despliegue

Requisitos: AWS CLI, Terraform 1.10 o superior, Docker con buildx, Node 22 y uv. Las dependencias de Python están fijadas en `pyproject.toml` y `uv.lock` (Python 3.12), las de los contenedores en `containers/science.txt` y las del frontend en `frontend/package-lock.json`. El procedimiento completo está en [docs/cloud-operations.md](docs/cloud-operations.md):

1. Crear el estado remoto (`infra/scripts/bootstrap.sh`) y aplicar los stacks `platform` y `network`.
2. Construir y publicar las imágenes `api`, `worker` y `training` (`infra/scripts/push-image.sh`). La del worker incluye los pesos de `modelos/stack_five/` y el EDF de ejemplo.
3. Publicar las entradas versionadas con `scripts/export_serving_bundle.py` y `scripts/publish_cloud_inputs.py --edf-run … --stack-run …`. Terraform las necesita para el bucket y el ejemplo, y el entrenamiento para reajustar el híbrido. Requieren las campañas locales de `var/research/`; un entorno existente se restaura desde S3.
4. Aplicar los stacks `training`, `aws` y `vercel`, y cargar en Vercel `MAIA_PUBLIC_CONFIG` con la salida `public_config` de `infra/aws`.
5. Verificar el servicio con `uv run --extra cloud python scripts/smoke_cloud.py --guest`.

Las credenciales de AWS se cargan desde un `.env` local, que no se versiona. El repositorio no contiene secretos.

## Reentrenar el modelo

`infra/scripts/run-training.sh stack` reajusta el clasificador del híbrido en una instancia EC2 efímera. Usa las entradas publicadas por versión y SHA-256 y registra la corrida en MLflow. `scripts/promote_stack_candidate.py` verifica el candidato con el EDF de ejemplo y lo instala en `modelos/stack_five/`, y un PR lo publica. FullNightNet queda congelada porque entrenarla requiere GPU o MPS. [Pasos y verificación](modelos/README.md#reentrenar)

## Estructura del repositorio

| Carpeta | Contenido |
| --- | --- |
| `src/sleep_staging/` | Paquete: datos, características, modelos, evaluación, entrenamiento del híbrido (`stack_training.py`) y servicio de inferencia (`serving/`) |
| `services/api/` | API de la aplicación |
| `frontend/` | Interfaz web (React + Vite) y pruebas end-to-end con Playwright |
| `notebooks/` | Auditoría de datos, análisis de variables y campañas de experimentación. `10_hibrido_fullnight_lightgbm.ipynb` evalúa el modelo final. |
| `modelos/` | Ficha del modelo final y sus pesos (`stack_five/`) |
| `configs/` | Configuración de datos, experimentos, búsquedas y partición |
| `scripts/` | Campañas, exportación, publicación, entrenamiento en la nube, promoción y pruebas del servicio |
| `infra/`, `containers/` | Terraform, scripts de despliegue y Dockerfiles |
| `output/` | EDF de ejemplo, informes en PDF y evidencia. Los zip de `output/evidencia/` (≈1,5 GB) se guardan con Git LFS. |
| `docs/`, `tests/` | Documentación, resultados y pruebas automatizadas |

## Experimentación local

Procesamiento y experimentación con **Sleep-EDFx real**: EDF/hipnogramas → épocas de 30 segundos → características o EEG para CNN → validación por sujeto → MLflow.

El dataset completo está descargado y auditado localmente: **197 PSG con sus 197 hipnogramas** —153 de Sleep Cassette y 44 de Sleep Telemetry—, más cuatro archivos de metadatos. Los 398 archivos se verificaron con SHA-256 y ocupan **8,117 GiB**. El inventario completo está en `data/manifests/sleep-edfx-complete/summary.json`.

El piloto usa ocho sujetos de Sleep Cassette, una noche por sujeto y hasta 240 épocas distribuidas a lo largo de cada registro. Reserva dos sujetos para test y utiliza cinco folds sobre los seis sujetos restantes. Los resultados del piloto sirven para comprobar el flujo y preparar la campaña completa.

## Inicio rápido

Para clonar sin descargar los ≈1,5 GB de evidencia en Git LFS, usar `GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/aladelca/maia_proyecto.git`. Un archivo concreto se baja después con `git lfs pull --include output/evidencia/<archivo>.zip`.

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

Abrir MLflow local en `http://127.0.0.1:5001`, que solo existe en el equipo que ejecuta `make mlflow`. En otra terminal:

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

Las nuevas corridas, cachés y el candidato se guardan en `var/research/sc-w30-v2/`, con tracking MLflow propio y checkpoints por fold. El notebook consulta los resultados guardados; activar `RUN_TRAINING` permite entrenar explícitamente. Ejecutar `scripts/sync_research_tracking.py` después de la campaña expone los 15 ensayos internos, el ensamble y el candidato nativo en MLflow; la UI local de esta campaña queda en `http://127.0.0.1:5002`.

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

LightGBM ajustado obtuvo el mayor F1 macro en desarrollo, **0,7893**. Frente a su referencia, LightGBM pasó de 0,7822 a 0,7893 y BiLSTM de 0,7687 a 0,7800. Se compararon 12 configuraciones LightGBM y seis BiLSTM en cinco folds por sujeto. Las 120 corridas de entrenamiento están en el experimento 3 del MLflow local (`http://127.0.0.1:5002`). El [informe](docs/refinement-results.md) incluye hiperparámetros finales y comandos de reproducción. El [notebook ejecutado](notebooks/06_lgbm_bilstm_tuning.ipynb) muestra las curvas y el F1 por estadio.

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
