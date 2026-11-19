# Ajuste de LightGBM y mejora de BiLSTM

## Objetivo y solicitud

Ejecutar la solicitud «haz una calibracion de hiperarametos para lgbm y una mejor configuracion de bilstm». Se trata de búsqueda de hiperparámetros, no de calibración de probabilidades. Comparar ambas familias con LightGBM 0,7822 ± 0,0161 y BiLSTM 0,7687 ± 0,0211, con cinco folds fijos por sujeto, MPS para la red y tracking completo en MLflow.

Rama de trabajo: `feature/tune-lightgbm-and-refine-bilstm`, basada en el PR #5 todavía abierto. Las campañas anteriores se conservan como referencias congeladas.

## Estado y evidencia

- Dataset de desarrollo disponible y verificado: 62 sujetos, 122 registros, 152.167 filas con señales y 152.052 objetivos de cinco clases. CSV `configs/splits/sc_subjects_v1.csv`; 16 sujetos de test excluidos.
- `lstm_research.load_reference` valida hashes, folds y correspondencia con OOF históricas. `fit_sequence` ya implementa MPS, holdout interno, pérdida balanceada, refit fijo, checkpoint/RNG y MLflow.
- LightGBM actual usa 300 iteraciones, 15 hojas, learning rate 0,05, sin búsqueda específica en la campaña original. Sus folds de 300 iteraciones duraron aproximadamente 100 s; se medirá el coste del nuevo espacio antes de lanzar sus cinco folds.
- La BiLSTM actual usa 145 variables, escalado de entrenamiento, una capa de 64 unidades, dropout 0,3 y pérdida con pesos inversos. Las nuevas opciones probarán representación y normalización retrospectiva, regularización y pesos moderados.
- Código inspeccionado: `lstm_research.py`, `lstm_data.py`, `models/lstm.py`, `models/classical.py`, `features/research.py`, `research.py`, pruebas y configuraciones previas. No se requieren nuevas dependencias.

## Alcance y protocolo

Incluido: doce candidatos LightGBM; seis candidatos BiLSTM; selección interna por sujeto y F1 macro; cinco folds externos por familia; refit final de cada familia; comparación, bootstrap emparejado, notebook e informe en español, pruebas y PR.

Fuera de alcance: ensambles, CNN sobre señales crudas, calibración de probabilidades, evaluación del test, cambio de etiquetas/recorte y sustitución automática de aliases existentes.

Cada fold externo reserva el 20 % de los sujetos restantes para selección interna. Escaladores/pesos se ajustan con entrenamiento interno. Se elige configuración y número de iteraciones/épocas; se reentrena desde cero con todo el entrenamiento externo, sin consultar el fold externo para early stopping. Las configuraciones se congelan antes de ver métricas externas.

### LightGBM

- Doce configuraciones explícitas en YAML: árboles máximos, learning rate, hojas, muestras mínimas por hoja, bins, L1/L2, fracción de variables, bagging y exponente de pesos de clase.
- Métrica personalizada F1 macro sin ponderar, con cinco clases. Desactivar el log-loss por defecto como criterio de selección; early stopping interno con paciencia 40 y mejora mínima 0,0001.
- Conservar las 435 variables del campeón y el preprocesamiento de imputación/escalado ajustado solo en entrenamiento. Ejecutar en CPU con cuatro threads, configurables; el benchmark previo con dos threads midió 8,2 s para 30 iteraciones. Con cuatro threads, el mismo benchmark tardó 4,9 s y produjo el mismo F1 interno.
- Guardar cada candidato terminado para reanudar entre candidatos. Refit externo y final con el número de iteraciones seleccionado internamente.

### BiLSTM

- Ventana centrada de 15 segmentos, posición objetivo 7; hidden/cell reiniciados entre ventanas, MPS obligatorio y float32.
- Seis candidatos fijados en YAML: referencia de 145 variables; 145 normalizadas por registro; 435 con contexto; 435 normalizadas por registro; capacidad de 128 unidades y dropout mayor; pérdida sin ponderación. Las variantes usan exponentes de peso 1, 0,5 o 0 y learning rates explícitos.
- La normalización por registro utiliza solo características de esa noche, sin etiquetas; es retrospectiva. El escalado de entrenamiento posterior se ajusta únicamente con sujetos de entrenamiento.
- Extender `LSTMConfig` con `weight_power`, por defecto 1, preservando la fórmula anterior. Exponente 0 equivale a pérdida sin ponderación; 0,5 atenúa el balanceo inverso.
- Máximo 50 épocas, paciencia 7, AdamW, clipping 1 y batch 256. Selección por F1 macro interno; comparar todos los candidatos dentro de cada fold y realizar un único refit externo del elegido.

Presupuesto científico: LightGBM 60 candidatos internos + cinco refits externos + doce candidatos internos finales + un refit final = 78 entrenamientos. BiLSTM 30 internos + cinco refits externos + seis internos finales + un refit final = 42 entrenamientos. Total: 120, más preflight/benchmark separados. Se permite ejecutar el proceso CPU y el proceso MPS a la vez, con una sola red activa en GPU.

## Datos, identidades y artefactos

- Construir todas las vistas antes de excluir M. Mantener las mismas claves y etiquetas OOF, ventanas por registro/hueco y padding de la campaña previa.
- `raw145`, `night145`, `context435`, `night435`: marcos de características con columnas explícitas, valores finitos, hashes y el mismo índice cronológico.
- Identidad ligada a fuente, dataset, splits, vistas/valores, espacio de búsqueda, versiones y dispositivo. Checkpoints y OOF con checksum; solo reutilizar resultados terminados compatibles.
- Experimento `sleep-cassette-lgbm-bilstm-refinement-v1`, en SQLite existente `var/research/sc-w30-v2/mlflow.db`, y artefactos nuevos en `var/research/sc-w30-refinement-v1/`.
- Raíces y locks independientes por familia permiten CPU/MPS concurrentes. Cada familia conserva un run padre con hijos para folds, candidatos internos y refits; callbacks/historias conservan curvas.
- Candidatos finales recargables: pipeline joblib para LightGBM y checkpoint con escalador/ventanas para BiLSTM. Registrar contrato de entradas; conservar aliases históricos.
- Comparación de F1 macro CV ± desviación, accuracy, kappa, F1 por estadio/sujeto y bootstrap de 2.000 remuestreos emparejados de sujetos. Los IC serán condicionales a modelos OOF ya ajustados; selección de familias exploratoria.

## Archivos

| Ruta | Acción y responsabilidad |
| --- | --- |
| `notebooks/06_lgbm_bilstm_tuning.ipynb` | Crear primero con plantilla; revisión reproducible, espacio real, curvas, leaderboard y ganador. |
| `configs/research/refinement.yaml` | Crear: doce configuraciones LGBM y seis BiLSTM. |
| `src/sleep_staging/refinement_data.py` | Crear: vistas normalizadas/contexto, pesos de clase y F1 interno eficiente. |
| `src/sleep_staging/booster_worker.py` | Crear: entrenamiento/predicción LightGBM en proceso CPU sin importar PyTorch; caché de entrada verificada. |
| `src/sleep_staging/refinement.py` | Crear: configuración, entrenamiento LGBM, adaptación BiLSTM, selección interna, refits, MLflow y comparación. |
| `src/sleep_staging/lstm_research.py` | Modificar de forma mínima: exponente de pesos con valor por defecto compatible. |
| `scripts/run_refinement.py` | Crear: preflight/benchmark por familia, campaña y consolidación. |
| `tests/unit/test_refinement.py` | Crear: independencia de etiquetas/noches, correspondencia de vistas, pesos, métrica y configuración. |
| `tests/integration/test_refinement_training.py` | Crear: datos reales, entrenamiento LGBM con selección interna, pérdida MPS moderada y recuperación/recarga. |
| `docs/refinement-results.md`, `docs/assets/refinement/` | Crear al completar: cifras reales, configuraciones finales, figuras y límites. |
| `README.md`, este plan | Actualizar con comandos y evidencia de cierre. |

## Secuencia

1. Crear rama y notebook; escribir pruebas relevantes y observar fallo antes de implementar.
2. Implementar vistas, pesos y callback F1; verificar independencia de etiquetas y preservación de objetivos.
3. Implementar búsqueda LGBM y reusar trainer BiLSTM extendido; probar sobre datos reales y validar MPS sin fallback.
4. Medir el coste y congelar YAML y código antes de ejecutar folds externos. Registrar fuente/configuración.
5. Ejecutar familias con cinco folds; reanudar candidatos/checkpoints compatibles si se interrumpen. Completar ambos refits finales.
6. Auditar separación, escaladores, métricas, hashes, contratos de candidatos e idempotencia; consolidar comparación con referencias.
7. Ejecutar notebook, inspeccionar figuras, redactar resultado real, validar suite y publicar PR con autorrevisión y GO/NO-GO.

## Validación

- `PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m pytest tests/unit/test_refinement.py tests/integration/test_refinement_training.py -q -ra`.
- `SLEEP_STAGING_ROOT=/Users/adrianalarcon/Documents/uniandes/maia_despliegue_tesis PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m pytest tests -q -ra`.
- `.venv/bin/python -m ruff check src tests scripts`.
- `.venv/bin/python -m mypy src/sleep_staging scripts`.
- `.venv/bin/python -m ruff format --check src/sleep_staging/refinement_data.py src/sleep_staging/refinement.py src/sleep_staging/booster_worker.py src/sleep_staging/lstm_research.py scripts/run_refinement.py tests/unit/test_refinement.py tests/integration/test_refinement_training.py`.
- Notebook ejecutado con `RUN_TRAINING=False`, cero errores, figuras legibles; `git diff --check`.

## Hallazgo de compatibilidad durante la implementación

La prueba inicial detectó un fallo nativo/hang de OpenMP al combinar LightGBM 4.7.0 y PyTorch 2.14.1 en el mismo proceso de macOS. Se aislaron entrenamiento y predicción LightGBM en un worker que rechaza importar Torch; el proceso MPS no carga LightGBM. Se comparten entradas inmutables con checksum y se mantienen MLflow y los mismos parámetros. No se cambian bibliotecas globales ni se habilitan opciones inseguras de OpenMP. Las pruebas verifican PID distinto, ausencia de Torch y parámetros reales del estimador.

Fuente primaria: https://github.com/lightgbm-org/LightGBM/issues/6595 y FAQ oficial de LightGBM sobre conflictos OpenMP.

## Riesgos y mitigaciones

- Sobreajuste a pocos sujetos: selección interna agrupada, regularización y test preservado; registrar que la selección entre campañas sigue siendo exploratoria.
- Normalización retrospectiva: nunca usar etiquetas y declarar dependencia de la noche completa; no presentar como inferencia causal.
- Duración de LightGBM: búsqueda acotada, bins/muestreo y parada interna, benchmark antes de evaluar folds externos, reanudación entre candidatos.
- Fuente nueva invalida identidades anteriores: mantener artefactos históricos intactos y tratar sus predicciones como referencias congeladas verificadas.
- Disco/MPS: batches por índice y checkpoints acotados, sin copiar EDF; una red en GPU a la vez.

## Resultado de la ejecución

Se recuperaron 99 entrenamientos anteriores al reinicio y se completaron los 21 pendientes. LightGBM alcanzó F1 macro 0,7893 y BiLSTM 0,7800. Los candidatos finales y las limitaciones se describen en el [informe de resultados](../docs/refinement-results.md).

## Criterios de aceptación

- [x] Doce candidatos LGBM y seis BiLSTM seleccionados internamente en cinco folds cada uno.
- [x] 120 entrenamientos científicos completados y trazables en MLflow, con diez OOF externas y dos candidatos finales.
- [x] Misma cohorte y objetivos, sin sujetos de test ni mezcla de ventanas/particiones.
- [x] MPS real validado para las variantes neuronales; métricas LGBM/NN independientes de etiquetas externas durante selección.
- [x] Checkpoints, métricas recomputadas, candidatos recargados e idempotencia verificados.
- [x] Notebook, informe, ganador global, mejores configuraciones finales y limitaciones entregados.
- [x] 66 pruebas correctas, revisión de tipos y formato de los cambios completados.
