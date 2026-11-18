# Campaña de mejora de modelos basada en literatura

Fecha de revisión: 15 de noviembre de 2026. Notebook: `notebook/04_literature_catboost.ipynb`.

## Decisiones derivadas de la revisión

| Fuente primaria | Qué aporta | Decisión para esta campaña | Límites de comparación |
| --- | --- | --- | --- |
| Vallat y Walker (2021), [YASA, eLife](https://pmc.ncbi.nlm.nih.gov/articles/PMC8516415/), DOI 10.7554/eLife.70092 | Características no lineales y contexto temporal de las señales. | Añadir entropía de permutación, dimensión de Petrosian, variables robustas y versiones suavizadas sobre cuatro segmentos previos y 15 segmentos centrados. | Adaptación de ideas; no reproducción del modelo preentrenado ni de todas sus características. Nuestro EEG es bipolar y el EMG de Cassette es una envolvente a 1 Hz. |
| Supratak et al. (2017), [DeepSleepNet](https://arxiv.org/abs/1703.04046), DOI 10.1109/TNSRE.2017.2721116 | Aprendizaje de dependencias entre segmentos mediante CNN y contexto recurrente. | Evaluar primero el aporte del contexto usando variables explícitas y los mismos sujetos. Una arquitectura secuencial queda como extensión posterior. | La CNN compacta existente no reproduce DeepSleepNet. Sus cifras publicadas no se usan como un objetivo comparable directamente. |
| Mari et al. (2026), [fusión de PSG multicanal, Sensors](https://www.mdpi.com/1424-8220/26/19/6011), DOI 10.3390/s26196011 | Ablaciones de canales, contexto temporal y representaciones de tiempo/frecuencia en Sleep-EDFx. | Comparar por separado Fpz-Cz, canales adicionales, nuevas variables y contexto. | El artículo mantiene otra distribución de vigilia en evaluación y emplea una arquitectura profunda. Sus métricas no equivalen a nuestro protocolo recortado. |
| [Documentación oficial de CatBoost](https://catboost.ai/docs/en/references/training-parameters/) | Parámetros de profundidad, regularización, balanceo y detención temprana. | Buscar parámetros y número de árboles con una validación interna agrupada por sujeto. | No usar el fold externo para escoger parámetros ni la mejor iteración. |

La prioridad práctica es comprobar mejoras con coste local acotado. No se incorporan pesos de modelos preentrenados con una procedencia de entrenamiento no auditada.

## Protocolo congelado

- Tarea: cinco clases W, N1, N2, N3 y REM; F1 macro con lista fija de clases.
- Datos: 62 sujetos y 122 noches de desarrollo; archivo inmutable `configs/splits/sc_subjects_v1.csv`.
- Test: los 16 sujetos y sus EDF/hipnogramas se excluyen antes de leer señales o verificar sus archivos. No hay evaluación final en esta campaña.
- Preprocesamiento: frecuencias y filtros nativos del pipeline existente, recorte de vigilia de 30 minutos definido con anotaciones, sin muestreo de segmentos. Este recorte continúa siendo una limitación de inferencia autónoma.
- Comparación: cinco folds externos idénticos para todos los modelos. Las noches de un sujeto permanecen juntas. Todas las ablaciones comparten exactamente los mismos segmentos.
- Métricas secundarias: accuracy, kappa, F1 por estadio, matriz de confusión y métricas por sujeto. Se distingue el promedio por fold de las métricas OOF agrupadas.
- Semilla: 42. CatBoost y modelos de referencia usan dos threads.

## Nuevas variables y ablaciones

1. Conservar las 25 variables originales de Fpz-Cz para reproducir RF y CatBoost.
2. Añadir a Fpz-Cz IQR, desviación absoluta mediana, longitud media de la señal, indicador de señal plana, entropía de permutación de orden 3 y dimensión fractal de Petrosian.
3. Añadir log-potencias, ratios logarítmicos entre bandas, frecuencia de borde espectral al 95 % y variabilidad del RMS entre seis bloques de cinco segundos.
4. Comparar esas variables con y sin Pz-Oz, EOG y EMG, preservando su representación nativa. El EMG envolvente no recibe características espectrales de una señal a 100 Hz.
5. Añadir medias de los cuatro segmentos previos y medias triangulares centradas de 15 segmentos. Las ventanas se reinician en cada registro y en cada hueco temporal; nunca se rellenan con etiquetas reales ni con predicciones OOF.

Las características temporales suavizadas se normalizan robustamente por noche usando mediana y percentiles 5/95. Su cálculo utiliza la señal completa de la noche, incluidos segmentos futuros: es procesamiento retrospectivo de un EDF completo. No se presenta como una modalidad causal ni en tiempo real. Las etiquetas solo definen la supervisión y el recorte ya existente, no las nuevas variables.

## Ajuste de hiperparámetros

Dentro de cada entrenamiento externo se reserva un 20 % de sus sujetos para validación interna. Sobre esa misma partición interna se comparan tres candidatos CatBoost. La configuración acotada se conserva en `configs/research/catboost-moderate.yaml`:

| Máximo de árboles | Profundidad | Learning rate | L2 | Balanceo | Fracción de variables | Bordes |
| --- | --- | --- | --- | --- | --- | --- |
| 300 | 4 | 0,08 | 3 | Balanced | 0,6 | 64 |
| 400 | 5 | 0,06 | 5 | SqrtBalanced | 0,6 | 64 |
| 500 | 6 | 0,05 | 9 | Sin pesos | 0,6 | 64 |

El presupuesto se acotó después de medir el coste de las ablaciones, antes de evaluar candidatos de ajuste. No se eligió el espacio a partir de sus resultados de validación externa. La pérdida es MultiClass. La selección y detención temprana usan F1 macro sin ponderar, con paciencia de 50 iteraciones. Se elige el candidato y su número de árboles usando únicamente validación interna, se reentrena sobre todo el entrenamiento externo con ese número fijo y se evalúa el fold externo una sola vez. Esto es validación externa de cinco folds con selección interna por holdout de sujetos; no una búsqueda interna de cinco folds.

Para reentrenar el candidato final se repite la selección interna sobre desarrollo y se ajusta el modelo con todos sus sujetos. El procedimiento no consulta el test. La selección de la familia y las variables entre varias ablaciones sigue siendo exploratoria; requiere confirmación posterior sobre el test reservado.

## Ejecución y artefactos

```bash
uv sync --locked --group dev --group notebook --extra boosting --extra dl
uv run --locked --extra boosting python scripts/run_research.py \
  --source-root /ruta/al/proyecto/que/contiene/data
```

Para exponer los ensayos internos como subcorridas, registrar el ensamble y publicar el candidato nativo en MLflow, ejecutar después de la campaña:

```bash
uv run --locked --extra boosting python scripts/sync_research_tracking.py
```

Este paso es idempotente: reutiliza las corridas terminadas y permite reintentar una sincronización interrumpida. La identidad del ensamble incluye las corridas, los hashes de las predicciones OOF y los pesos de sus componentes. Registra el candidato en `sleep_sc_task5_literature_v2`, alias `candidate`, y comprueba la igualdad de predicciones después de recargarlo. Requiere una campaña completa con reentrenamiento y verifica el hash, el modelo seleccionado y el dataset del candidato antes de escribir en MLflow; una ejecución con `--no-refit` no puede publicar pesos de una campaña anterior.

La carpeta de origen se usa solo para lectura. Los nuevos datos derivados, corridas y modelos se guardan bajo `var/research/sc-w30-v2/`:

- `datasets/<fingerprint>/`: caché Parquet por registro, dataset consolidado y manifiesto con hashes y soporte de clases.
- `runs/<fingerprint>/`: checkpoints por fold, selección interna, predicciones OOF, métricas y matrices de confusión.
- `leaderboard.csv` y `campaign.json`: comparación y ganador.
- `paired_bootstrap.json`: diferencia de F1 macro OOF frente a RF con remuestreo emparejado de sujetos; incertidumbre condicional a los modelos OOF ya entrenados.
- `candidate.joblib`: estimador reentrenado en desarrollo y contrato de características. Requiere reconstruir las mismas variables; no es todavía una interfaz de predicción directa de EDF.
- `mlflow.db` y `mlflow-artifacts/`: tracking local independiente de las corridas históricas.
- `tracking-audit.json`: identificadores de las comparaciones, ensayos internos, ensamble y modelo registrado.
- `candidate-example-input.parquet` y `artifact-validation.json`: entrada de ejemplo, contrato del protocolo y comprobación de recarga del candidato.

Para consultar el tracking:

```bash
uv run --locked mlflow ui \
  --backend-store-uri sqlite:///var/research/sc-w30-v2/mlflow.db \
  --host 127.0.0.1 --port 5002
```

El argumento `--tuning-config` permite otra lista explícita de candidatos; los valores realmente utilizados se incluyen en la identidad de cada corrida ajustada.

Para registrar el kernel del notebook en el entorno del proyecto:

```bash
uv run --locked --group notebook python -m ipykernel install \
  --prefix .venv --name sleep-research --display-name "Sleep research (.venv)"
```

El notebook se abre por defecto en modo revisión de resultados y no vuelve a entrenar al ejecutarlo. Para entrenar desde el notebook, configurar `SOURCE_ROOT` o la variable `SLEEP_RESEARCH_DATA_ROOT` con la raíz que contiene los manifiestos y EDF originales. La variable `RUN_TRAINING` permite iniciar explícitamente otra campaña. Las corridas se identifican por el contenido del dataset, las columnas, los valores y tipos de las variables, el código de todo el paquete, las versiones de los estimadores y los parámetros de búsqueda. Un cambio en constructores de modelos, evaluación o transformaciones temporales invalida automáticamente la reutilización de resultados. Se recomienda un `--output-root` nuevo para conservar por separado campañas con protocolos distintos.
