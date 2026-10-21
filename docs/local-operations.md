# Cómo ejecutar el proyecto localmente

## Requisitos y primera instalación

Ejecutar desde la raíz del proyecto. El entorno validado es macOS ARM64 con Python 3.12; CPU funciona para procesamiento y modelos clásicos, y PyTorch admite CPU o MPS. Instalar `uv` siguiendo su [documentación oficial](https://docs.astral.sh/uv/getting-started/installation/) si aún no está disponible.

El equipo de desarrollo tiene 36 GiB de RAM. El piloto ocupa aproximadamente 0,4 GB en EDF más derivados, modelos y entorno; reservar al menos 5 GB para comenzar. Para el conjunto completo, medir expansión y conservar el presupuesto de 40–60 GB indicado en el plan. El código procesa un registro a la vez y limita Random Forest a dos workers.

```bash
uv sync --locked --group dev --group notebook --extra dl
cp .env.example .env
make init-dirs
uv run --locked sleep-stage doctor
```

La primera instalación requiere Internet para descargar paquetes/Python cuando hagan falta. `uv.lock` ya existe: las ejecuciones ordinarias no necesitan `uv lock`. El extra `dl` instala PyTorch. Para trabajar solo con Random Forest puede omitirse, pero no podrá ejecutarse la CNN ni sus pruebas.

`doctor` imprime raíz, versiones, RAM, disco, disponibilidad MPS, límite de workers y URI de tracking. Si aparece un aviso sobre otro `VIRTUAL_ENV`, desactivar ese entorno antes de trabajar; `uv` selecciona `.venv` del proyecto.

## Configuración

- `.env`: variables locales, excluido de Git. CLI, scripts y notebooks cargan el archivo desde la raíz; las variables ya exportadas en el proceso tienen prioridad.
- `MLFLOW_TRACKING_URI`: por defecto `http://127.0.0.1:5001`. El lanzador del servidor y los clientes usan el mismo valor.
- `configs/local.yaml`: URI por defecto y límite global de workers.
- `configs/data/pilot.yaml`: ocho identificadores reales `SC4001`–`SC4071`.
- `configs/experiments/rf-eeg.yaml`: cinco clases, Fpz-Cz, baseline y Random Forest, cinco folds y máximo 240 épocas por registro.
- `configs/experiments/cnn-eeg.yaml`: mismo preprocesamiento/particiones, baseline y CNN de tres épocas máximas; `device: auto` selecciona MPS cuando está disponible.
- `configs/channels.yaml`: catálogo de 12 perfiles. Copiar su lista `channels` a la configuración experimental deseada; la primera CNN admite exactamente un EEG.

Las rutas de datos se resuelven desde la raíz. `SLEEP_STAGING_ROOT` permite indicar otra raíz operativa, principalmente para pruebas/restauración; esa raíz debe contener sus propios manifiestos y configuraciones cuando se soliciten por CLI.

## Iniciar y comprobar MLflow

En una terminal dedicada:

```bash
make mlflow
```

Mantener esa terminal abierta. El servicio escucha en localhost, utiliza un worker y conserva un bloqueo para impedir respaldos simultáneos. Abrir [http://127.0.0.1:5001](http://127.0.0.1:5001).

En otra terminal, desde la misma raíz:

```bash
uv run --locked sleep-stage doctor --check-mlflow
```

Debe terminar con `MLflow disponible.`. Los metadatos viven en `var/mlflow/mlflow.db` y los modelos/artefactos en `var/mlflow/artifacts/`.

## Descargar y auditar el piloto real

```bash
uv run --locked sleep-stage download --config configs/data/pilot.yaml
uv run --locked sleep-stage audit --config configs/data/pilot.yaml
```

La descarga requiere Internet y obtiene ocho pares PSG/hipnograma. Usa el espejo público oficial en S3, sin credenciales AWS, y verifica cada archivo con el SHA-256 publicado por PhysioNet. La auditoría debe informar ocho registros y ocho sujetos, canales/frecuencias y duración de las anotaciones.

Salidas:

- `data/raw/sleep-edfx/1.0.0/sleep-cassette/`: EDF originales.
- `data/raw/sleep-edfx/1.0.0/SHA256SUMS.txt`: sumas publicadas.
- `data/manifests/sc-pilot/`: archivos, registros, canales, anotaciones y resumen de auditoría.

Repetir `download` verifica los archivos existentes y descarga los que falten. Una transferencia fallida se reinicia como archivo temporal; un archivo final con checksum incorrecto produce un error y debe moverse fuera de la carpeta antes de repetir.

## Procesar y entrenar por etapas

Usar la configuración experimental en todas las etapas para conservar exactamente los mismos parámetros:

```bash
uv run --locked sleep-stage prepare --config configs/experiments/rf-eeg.yaml
uv run --locked sleep-stage features --config configs/experiments/rf-eeg.yaml
uv run --locked sleep-stage split --config configs/experiments/rf-eeg.yaml
uv run --locked sleep-stage experiment --config configs/experiments/rf-eeg.yaml
uv run --locked sleep-stage experiment --config configs/experiments/cnn-eeg.yaml
```

| Paso | Entrada | Salida y comprobación |
| --- | --- | --- |
| `prepare` | Manifiestos y EDF verificados | Directorio `data/interim/<id>`; `epochs.parquet`, `channels.parquet`, señales `.npy` y `metadata.json`. Cada época válida dura 30 segundos. |
| `features` | Épocas y señales | Ruta de `features.parquet` bajo `data/processed/`; mantiene `epoch_id` y procedencia. |
| `split` | Índice de épocas | `subjects.parquet` y metadatos bajo `data/splits/<id>`; cada sujeto tiene una única asignación. |
| `experiment` | Datos anteriores y MLflow activo | Run padre, runs por fold, tabla de comparación, ganador y URI de modelo. |

`experiment` ejecuta automáticamente los pasos previos si faltan. Ningún paso de entrenamiento descarga EDF. Si datos/configuración/código relevante cambian, se genera otra versión; una versión compatible reutiliza caché. Los límites del piloto seleccionan épocas distribuidas a lo largo de cada noche y conservan sus tiempos originales.

La tarea de cinco clases excluye M y `?`; la de seis incluye M. Para experimentar con seis clases se debe crear una configuración explícita con `task: task_6` y verificar soporte suficiente por sujeto y fold. Si falta una clase en entrenamiento, la corrida falla en vez de cambiar etiquetas automáticamente. En este piloto hay muy pocas épocas M.

Después de la instalación y descarga, el flujo funciona sin Internet externo con MLflow en localhost. Para exigir a `uv` que no use la red, agregar `--offline` después de `uv run`; los recursos necesarios deben estar preparados.

## Consultar resultados y recargar el modelo

```bash
uv run --locked sleep-stage report --config configs/experiments/rf-eeg.yaml
```

El resumen contiene `run_id`, `winner`, `model_uri`, `features_path`, `preprocessing_id`, `splits_path`, cobertura y métricas. Para una corrida específica:

```bash
uv run --locked sleep-stage report --run-id ID_DE_LA_CORRIDA
```

Sustituir los valores en mayúsculas por los de ese resumen. En MLflow, abrir el experimento `pilot-sleep-cassette`; revisar métricas, matrices de confusión y artefactos. En disco, los informes están en `var/reports/<run_id>/`, con `summary.json`, `leaderboard.csv`, predicciones OOF, métricas por clase e imágenes de confusión. El registro de modelos conserva versiones y un alias `candidate`; el resumen referencia una URI inmutable para reproducir la corrida.

Para un modelo tabular:

```bash
uv run --locked sleep-stage predict \
  --model-uri URI_DEL_MODELO \
  --input RUTA_FEATURES_PARQUET \
  --output var/reports/predictions.parquet
```

La salida contiene `epoch_id`, predicción y probabilidades. El comando valida columnas y versión de preprocesamiento antes de cargar el modelo. Los modelos provienen de este entorno local y se guardan con cloudpickle, incluida la implementación necesaria para recargarlos.

La CNN recibe 3000 muestras por época; su entrada tabular se construye con `waveform_frame` a partir de las señales preparadas. El notebook de revisión muestra el contrato del ganador. Los modelos no cargan directamente archivos PSG nuevos mediante este comando; esa integración corresponde a una etapa posterior.

## Notebooks

```bash
uv run --locked --group notebook python -m ipykernel install \
  --prefix .venv --name sleep-staging-local --display-name 'Sleep staging (.venv)'
make notebook
```

Abrir la URL local que imprime JupyterLab y seleccionar `Sleep staging (.venv)`. Seguir este orden:

1. `notebook/01_data_audit.ipynb`: catálogo, etiquetas y una señal real del piloto.
2. `notebook/02_feature_analysis.ipynb`: épocas, características y particiones existentes; puede preparar caché mediante el paquete.
3. `notebook/03_experiment_review.ipynb`: corridas de MLflow, leaderboard compatible y modelo ganador.

Los notebooks reutilizan el paquete y los datos reales descargados. La comparación se ejecuta con `experiment`; abrir los notebooks no dispara una campaña de entrenamiento.

## Detener, reiniciar y respaldar

Detener JupyterLab y MLflow con Ctrl+C en sus terminales. Esperar que finalicen las corridas antes de detener el servidor. Reiniciar MLflow con `make mlflow`: las corridas y los modelos siguen en disco.

Con MLflow y los experimentos detenidos:

```bash
make backup
```

El comando crea `var/backups/<fecha>/` con SQLite, artefactos y `manifest.json`; verifica hashes e integridad de la base. Si el servicio o un entrenamiento siguen activos, el bloqueo impide el respaldo.

Para comprobar una copia:

```bash
uv run --locked python -c 'from pathlib import Path; from sleep_staging.backup import verify_backup; verify_backup(Path("var/backups/FECHA_DEL_RESPALDO"))'
```

Para restaurar, detener procesos y conservar la carpeta actual `var/mlflow` con otro nombre. Crear una nueva `var/mlflow`, copiar dentro el `mlflow.db` y el directorio `artifacts` del respaldo verificado, y ejecutar `make mlflow`. Comprobar un run previo y recargar su modelo antes de seguir. Restaurar con la versión de MLflow del mismo lockfile. La copia contiene seguimiento/modelos; los EDF y los derivados se conservan aparte o se regeneran desde sus manifiestos.

## Solución de problemas

| Problema | Diagnóstico y acción |
| --- | --- |
| Puerto ocupado | Comprobar `lsof -i :5001`. Detener el proceso que ya no se necesite o cambiar `MLFLOW_TRACKING_URI` en `.env` para servidor y clientes. |
| MLflow no disponible | Ejecutar `doctor --check-mlflow`; iniciar `make mlflow` en otra terminal y revisar su salida. |
| Bloqueo activo | Esperar al servicio/experimento que conserva el bloqueo. La existencia del archivo `.lock` por sí sola no implica un bloqueo activo. |
| Paquetes o kernel incorrectos | Repetir `uv sync --locked --group dev --group notebook --extra dl`; registrar y seleccionar el kernel del proyecto. |
| Download interrumpido | Repetir `download`. Los archivos finales válidos se reutilizan; los temporales no se consideran descargados. |
| Checksum incorrecto | Apartar el archivo identificado y descargarlo otra vez. No editar las sumas para hacerlo pasar. |
| Canal ausente | Revisar `audit` y el perfil `channels`; los registros sin canales obligatorios quedan fuera de la cohorte elegible. |
| RAM/disco insuficientes | Revisar `doctor`; bajar workers/batch size y usar el piloto. Conservar EDF y respaldos; identificar derivados regenerables antes de limpiar. |
| MPS no disponible | Usar `device: cpu`; instalar el extra `dl`. No activar fallback oculto para errores de operadores. |
| Perfil o preprocesamiento incompatible | Usar el archivo de features correspondiente al `preprocessing_id` del modelo; respetar el orden de columnas/muestras del contrato. |
| Clases insuficientes | Revisar soporte por sujeto y fold. Cambiar el protocolo de forma explícita antes de una nueva campaña; nunca partir por época para completar clases. |

## Pasar a la cohorte completa

```bash
uv run --locked sleep-stage download --config configs/data/sleep-cassette.yaml
uv run --locked sleep-stage audit --config configs/data/sleep-cassette.yaml
uv run --locked sleep-stage experiment --config configs/experiments/rf-full.yaml
```

Es una campaña nueva sin límite de épocas. Medir recursos antes de iniciarla; no se ejecuta como parte del piloto. Telemetry dispone de su propia configuración de datos y debe analizarse por separado debido al cambio de representación EMG y condiciones de registro.

## Verificación de esta entrega

El 20 de octubre de 2026 se verificaron la instalación bloqueada, los ocho registros reales y sus SHA-256, las etapas de procesamiento y ambos entrenamientos, los tres notebooks y la predicción por CLI. `make check` pasó y `make test` aprobó 25 pruebas, incluidas las rutas CPU/MPS y la protección del test reservado.

Se comprobó un reinicio de MLflow y se restauró un respaldo en otro servidor local temporal; ambos modelos recargados produjeron probabilidades válidas. La evidencia operativa de esa restauración queda en `var/reports/local-verification.json`. También se completó una corrida con `uv run --offline --locked` después de preparar dependencias y EDF.

Posteriormente se completó la descarga de Sleep-EDFx 1.0.0: 153 registros de Cassette y 44 de Telemetry, con sus 197 hipnogramas y cuatro metadatos auxiliares. La comprobación global validó los SHA-256 de los 398 archivos (8,117 GiB). Ambas cohortes tienen auditoría de canales y anotaciones en sus carpetas de manifiestos. El resumen agregado está en `data/manifests/sleep-edfx-complete/summary.json`; las características y los modelos reportados corresponden al piloto.
