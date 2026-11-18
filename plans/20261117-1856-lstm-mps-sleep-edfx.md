# Plan de experimentación LSTM en MPS para Sleep-EDFx

Estado: implementación y campaña experimental completadas. Fecha: 17 de noviembre de 2026. Los resultados y la auditoría están en [el informe](../docs/lstm-mps-results.md) y [el notebook ejecutado](../notebook/05_lstm_mps.ipynb).

## Objetivo

Implementar y evaluar LSTM en PyTorch con entrenamiento e inferencia en **MPS**, usando las características multicanal ya disponibles y las cinco particiones congeladas por sujeto. Comparar F1 macro, accuracy y kappa con los resultados de desarrollo de la campaña anterior, sin consultar el test reservado. Conservar configuraciones, selección interna, curvas, checkpoints y predicciones en MLflow.

El experimento será válido aunque ninguna LSTM supere a LightGBM: el resultado debe quedar documentado con la misma trazabilidad.

## Solicitud y decisiones resueltas

- Solicitud: «podemos probar una lstm?»; precisión posterior: «haz un plan para correr esto, usa mps».
- Autorización posterior: «implement en otra rama, corre los experimentos, guardalos en mlflow, compara con nuestro campeon actual».
- Responsable/issue: None; solicitud directa del usuario.
- Plan: `plans/20261117-1856-lstm-mps-sleep-edfx.md`.
- La primera campaña será sobre **variables por segmento**, no sobre EEG crudo. Esto reutiliza la extracción existente y permite estudiar el contexto aprendido con un coste acotado.
- Se aplicará la metodología notebook-first de `ml-dl-experimentation`, con cinco folds agrupados por sujeto en lugar de KFold aleatorio por segmento.

## Estado actual y evidencia comprobada

| Elemento | Estado verificado |
| --- | --- |
| Código base | Rama `experiment/literature-features-catboost-20261115`; PR #4 abierto, todavía sin fusionar al preparar este plan. |
| Entorno | Python 3.12.0, PyTorch 2.14.1, 36 GiB de RAM; MPS compilado y disponible. |
| Fallback MPS | `PYTORCH_ENABLE_MPS_FALLBACK` no estaba definido en el proceso de inspección. La ejecución deberá fijarlo explícitamente a `0`. |
| Dataset | `var/research/sc-w30-v2/datasets/40e760aff566477eafd8/features.parquet`, localizado desde `campaign.json`. |
| SHA-256 del dataset | `44b6a7b7d7e4b2a80144fe2becff66627c09e52ee605c264169b331192483f16`. |
| Cohorte | 62 sujetos, 122 registros, 152.167 filas con características; 152.052 segmentos supervisados para las cinco clases. |
| Entradas | 145 variables multicanal por segmento; el modelo con contexto actual recibe 435 variables. |
| Particiones | `configs/splits/sc_subjects_v1.csv`: cinco folds de desarrollo y 16 sujetos reservados para test. |
| Referencia principal | LightGBM con contexto: F1 macro CV 0,7822 ± 0,0161; accuracy 0,8167; kappa 0,7539. |
| Red existente | `CNNClassifier` en `src/sleep_staging/models/cnn.py`: segmentos independientes, un EEG, validación interna y soporte MPS. No implementa LSTM. |

Los módulos revisados son `research.py`, `research_data.py`, `features/research.py`, `research_tracking.py`, `models/cnn.py`, `evaluation.py`, `provenance.py`, `config.py`, `tests/conftest.py`, `pyproject.toml` y `Makefile`. El extra `dl` ya declara PyTorch; no se prevé añadir dependencias.

## Hallazgos y consecuencias

1. `research_views()` calcula el contexto antes de excluir filas sin target. La LSTM debe construir sus ventanas sobre la cronología completa disponible y usar como objetivos exactamente las mismas 152.052 filas supervisadas. Borrar primero esas filas crearía vecindades falsas alrededor de M.
2. `research.py` y su sincronizador están orientados a modelos clásicos y a un ensamble concreto. La nueva campaña tendrá un runner y artefactos propios; reutilizará evaluación, hashes y separación interna por sujeto sin ampliar el contrato del CLI histórico.
3. El entrenamiento requiere escalado de las variables. El escalador se ajustará solo con los sujetos de entrenamiento correspondientes, nunca con el fold externo ni con el conjunto completo antes de dividirlo.
4. La referencia LightGBM utiliza variables suavizadas y normalizadas por noche. La comparación con LSTM será de soluciones sobre los mismos objetivos y folds; no una ablación pura de arquitectura con entradas y normalización idénticas.
5. MPS disponible no demuestra por sí solo que ambas variantes LSTM funcionen correctamente. Antes de entrenar se comprobarán forward, backward, actualización de pesos y recarga de un checkpoint en ese dispositivo.

## Alcance

### Incluido

- Dataset de ventanas con límites por registro y discontinuidades, preparación por fold y procesamiento por lotes.
- Control neuronal sin contexto, LSTM con ventana pasada y BiLSTM retrospectiva.
- Dos valores de learning rate por variante, selección interna y cinco folds externos comunes.
- Notebook ejecutado, comparación con la referencia guardada, métricas por estadio/sujeto, incertidumbre y documentación en español.
- MLflow local con todas las corridas técnicas y experimentales identificadas por su propósito.
- Checkpoints reanudables, verificación de integridad y candidato neuronal independiente del candidato previo.

### Fuera de esta campaña

- CNN-LSTM sobre señales crudas, Transformers, preentrenamiento y búsqueda extensa de arquitecturas.
- Cambiar etiquetas, añadir sujetos, modificar el recorte de vigilia o entrenar con el test reservado.
- Presentar el sistema como inferencia clínica, causal o en tiempo real.
- Sustituir automáticamente el alias del candidato LightGBM o desplegar un servicio.

## Matriz de experimentos

Se fijarán los siguientes experimentos antes de evaluar los folds externos:

| Nombre | Ventana de entrada | Predicción | Hipótesis |
| --- | --- | --- | --- |
| `lstm_single_epoch` | Un segmento de 30 s, 145 variables. | Estadio de ese segmento. | Control con la misma familia y sin contexto temporal. |
| `lstm_past15` | Hasta 15 segmentos: 14 anteriores y el actual, 7,5 min en total. | Estadio del último segmento. | Aprender dependencias respecto al pasado mejora el control. |
| `bilstm_center15` | Hasta 15 segmentos: siete anteriores, el actual y siete posteriores. | Estadio del segmento central. | El contexto futuro aporta valor en procesamiento retrospectivo. |

Las dos últimas variantes cambian dirección y posición del objetivo; no se atribuirá su diferencia exclusivamente a la bidireccionalidad. `lstm_past15` restringe la ventana de entrada al pasado; el filtrado previo de señales y el recorte con anotaciones impiden calificar todo el pipeline como causal.

Configuración inicial común:

- Entrada `float32` con forma `[batch, longitud, 145]` y orden de variables congelado.
- Escalado robusto por variable ajustado en entrenamiento, percentiles 5/95 y mediana; escala mínima positiva y clipping fijo a [-10, 10]. Registrar sus parámetros. No ajustar normalizadores por noche de validación.
- Proyección lineal de 145 a 64 unidades, activación ReLU, LSTM de una capa y 64 unidades ocultas.
- `batch_first=True`, `proj_size=0`; head con dropout 0,3 y salida de cinco logits. El dropout se aplica explícitamente al head: con una sola capa no se confiará en el parámetro de dropout entre capas de `nn.LSTM`.
- Para `lstm_past15` se usa la salida del objetivo final. Para `bilstm_center15` se usa `output[:, 7, :]` con ambas direcciones, no el último timestep de salida como supuesto resumen bidireccional.
- Hidden/cell state reiniciados en cada ventana; no transportar estado entre muestras, batches, sujetos o registros.
- Pérdida CrossEntropy con pesos inversos de frecuencia calculados sobre los targets del entrenamiento correspondiente. Registrar pesos y soporte de las cinco clases; abortar si falta una clase en entrenamiento o validación interna.
- AdamW, weight decay `1e-4`, clipping de gradiente a norma 1, batch inicial 256.
- Dos candidatos internos: learning rate `1e-3` y `3e-4`; el resto de parámetros permanece fijo.
- Máximo 50 épocas de optimización, early stopping por F1 macro interno con paciencia 7 y mejora mínima `1e-4`. Sin scheduler en esta primera campaña; registrarlo explícitamente.
- Semilla base 42, con derivaciones registradas por fold/candidato. Registrar estados RNG de Python, NumPy, PyTorch y MPS para reanudación; no prometer igualdad bit a bit entre ejecuciones o versiones de MPS.

No ampliar el espacio de búsqueda tras inspeccionar los resultados externos y seguir presentándolo como el mismo protocolo. Una ampliación se registrará como otra campaña exploratoria.

## Contrato de secuencias y controles de fuga

1. Resolver el dataset desde `var/research/sc-w30-v2/campaign.json`; verificar su SHA y el SHA del CSV de splits. La ruta física no será una constante del código nuevo.
2. Verificar antes del entrenamiento: ningún sujeto de test presente, un fold por sujeto, cinco folds, claves `epoch_id` únicas, orden de clases W/N1/N2/N3/REM y exactamente los mismos objetivos que la referencia OOF.
3. Identificar las 145 columnas de señal mediante el catálogo de canales existente. Rechazar columnas de target, fold, identificadores y probabilidades como entrada del modelo.
4. Ordenar por `recording_id` y `onset_s`; iniciar un bloque nuevo cuando el siguiente onset no sea exactamente 30 s posterior. No cruzar noche ni huecos. Las filas M disponibles pueden aportar características de contexto, pero su etiqueta no se introduce en la entrada y no se predicen como sexta clase.
5. Mantener todos los objetivos de los extremos: replicar el primer/último vector válido del mismo bloque donde falte contexto. Guardar una máscara y un resumen del padding para auditoría. La política será idéntica entre entrenamiento, validación y predicción.
6. Ajustar el escalador con filas de sujetos del entrenamiento interno; al refit externo, volver a ajustarlo con todo el entrenamiento externo. Ninguna ventana mezcla sujetos, por lo que tampoco mezcla particiones.
7. Guardar matriz base y un índice de vecinos; construir tensores solo al obtener cada batch. Evitar materializar en memoria todo el tensor de ventanas superpuestas.
8. Cada fold externo producirá una y solo una predicción por objetivo, con probabilidades en el orden fijo de cinco clases. El conjunto OOF completo conservará las 152.052 claves de referencia.

El recorte de vigilia de 30 min sigue siendo dependiente de anotaciones. La BiLSTM centrada es retrospectiva. Estas condiciones deben aparecer junto a los resultados, también si una red supera al modelo previo.

## Validación, búsqueda y presupuesto

Para cada una de las tres variantes y cada fold externo:

1. Excluir los sujetos del fold externo.
2. Reservar el 20 % de los sujetos restantes mediante `grouped_inner_split`, con una semilla reproducible por fold. Compartir esa partición entre los dos candidatos internos.
3. Entrenar ambos learning rates; ajustar escalador y pesos de clase solo con entrenamiento interno. Evaluar F1 macro, accuracy y kappa internos en cada época; elegir checkpoint y configuración por F1 macro. Desempatar por menor número de épocas y luego por orden de configuración.
4. Rehacer escalador, pesos y modelo desde cero con todos los sujetos del entrenamiento externo. Entrenar el número de épocas seleccionado, sin consultar el fold externo para early stopping.
5. Predecir el fold externo una vez al terminar el refit. Guardar métricas, probabilidades y auditoría del conjunto de sujetos utilizado.

Presupuesto: **30 entrenamientos internos + 15 refits externos**. Después de CV, el mejor modelo neuronal podrá evaluar dos candidatos internos sobre desarrollo y tener un refit final: hasta **48 entrenamientos**, además de las comprobaciones técnicas. Los entrenamientos técnicos usarán datos reales de desarrollo y se excluirán del leaderboard científico.

Antes de ejecutar la campaña completa, medir una época del entrenamiento interno de cada variante con `torch.mps.synchronize()` al delimitar la medición. Registrar throughput, tiempo de validación y memoria. Estimar el coste con esos tiempos; no prometer una duración sin medirla. Ejecutar una sola corrida en GPU a la vez. Si se interrumpe, conservar checkpoints y declarar la comparación incompleta hasta contar con sus cinco folds.

## Uso obligatorio de MPS

- Configuración y CLI exigirán `device: mps`; no usar `auto` ni pasar silenciosamente a CPU.
- Establecer `PYTORCH_ENABLE_MPS_FALLBACK=0` antes de importar PyTorch. Rechazar una configuración del entorno que permita fallback.
- Comprobar `torch.backends.mps.is_built()` e `is_available()` y abortar con un diagnóstico si fallan.
- Probar un batch real de cada arquitectura con forward, backward y un paso del optimizador; comprobar pérdida/gradientes finitos, parámetros y tensores en MPS y actualización de pesos.
- Empezar con precisión float32, sin AMP. Targets `int64`; nunca enviar arrays float64 a MPS.
- Normalización, lectura de Parquet, DataLoader, cálculo de métricas y escritura de artefactos pueden usar CPU. Las operaciones de la red durante entrenamiento y predicción se ejecutarán en MPS.
- Construcción vectorizada de batches, sin workers ni pinning orientado a CUDA. La matriz de variables escaladas se carga una vez en MPS; el índice de vecinos selecciona únicamente las ventanas del batch. Esto evita transferencias de ventanas completas desde CPU y no materializa todas las ventanas superpuestas.
- Ante falta de memoria, detener la corrida y registrar el fallo. Permitir reiniciar desde cero con batch 128 o 64, creando otra identidad y dejando registrado el cambio. No cambiarlo a mitad de una corrida sin trazabilidad ni desactivar límites del allocator.
- Registrar dispositivo solicitado/real, versión macOS/PyTorch, tamaño de batch, precisión, tiempo y memoria MPS cuando la API esté disponible.

## MLflow, artefactos y comparación

Usar el tracking existente `sqlite:///var/research/sc-w30-v2/mlflow.db` y crear un experimento distinto: `sleep-cassette-lstm-mps-v1`. Resolver la URI SQLite a una ruta absoluta antes de inicializar el cliente. Los artefactos nuevos se guardarán en `var/research/sc-w30-lstm-mps-v1/`, con `artifact_location` absoluto bajo su subcarpeta `mlflow-artifacts/`. La UI existente en el puerto 5002 podrá mostrar ambos experimentos.

- Run padre de campaña con configuración completa, hashes de dataset/splits/código/lockfile, entorno, métricas elegidas y `test_evaluated=false`.
- Runs por variante y fold, con hijos para candidatos internos y refit externo; propósito explícito para separar validación técnica de evaluación científica.
- Métricas por época: pérdida, F1 macro, accuracy y kappa internos, learning rate y tiempo. Nunca registrar F1 de entrenamiento como resultado OOF.
- Artefactos: lista de sujetos por partición, columnas, escalador, pesos de clase, curvas, número de parámetros, mejor época, checkpoints y auditoría del dispositivo.
- Checkpoints de entrenamiento reanudables: modelo, optimizador, época, mejor score/estado, contador de paciencia, RNG e identidad. Conservar best y last por separado. Una reanudación incompatible debe rechazarse.
- Identidad ligada a SHA del dataset y splits, configuración y forma de ventanas, padding, escalado, política de pérdida/selección, código, versiones y dispositivo. Verificar checksums antes de reutilizar folds u OOF; no tomar una corrida fallida como resultado terminado.
- OOF por modelo: `epoch_id`, sujeto, registro, onset, fold, etiqueta real/predicha y cinco probabilidades; verificar unicidad y cobertura exacta.
- Leaderboard con F1 macro medio ± desviación de los cinco folds, accuracy, kappa y F1 por estadio. Mostrar también métricas OOF agrupadas y por sujeto, diferenciando sus agregaciones.
- Comparación emparejada con LightGBM sobre las mismas claves OOF; bootstrap de 2.000 remuestreos de sujetos completos, manteniendo juntas sus noches. El intervalo será condicional a los modelos OOF y no cubrirá toda la incertidumbre del entrenamiento.
- La referencia anterior se importa desde sus artefactos verificados y se etiqueta como referencia histórica del mismo protocolo. No es necesario reentrenar los ocho boosters para esta prueba.
- Guardar el mejor candidato neuronal reentrenado en desarrollo, aunque no supere al modelo clásico, en un contrato propio de secuencias. Incluir `state_dict`, arquitectura, escalador, clases, columnas, ventana y metadatos de procedencia.
- Recargar ese candidato y comprobar igualdad de predicciones dentro de la tolerancia numérica y probabilidades normalizadas. Una promoción a MLflow Model Registry requiere wrapper/contrato de secuencias explícito; no asumir que `predict_features` acepta ventanas ni cambiar el alias previo.

Carpetas nuevas previstas: `runs/<identidad>/fold-<n>/`, `candidate/`, `figures/`, `leaderboard.csv`, `campaign.json`, `tracking-audit.json`, `paired-bootstrap.json` y `validation.json`. Datos, pesos y bases permanecen ignorados por Git; el notebook, las figuras finales y el informe sí serán versionables.

## Archivos y responsabilidades

| Ruta | Acción prevista | Responsabilidad |
| --- | --- | --- |
| `notebook/05_lstm_mps.ipynb` | Crear primero en la implementación | Definir hipótesis/métricas, mostrar cohortes, configuraciones, curvas, leaderboard, ganador y limitaciones. Por defecto, modo revisión; entrenamiento explícito. |
| `configs/research/lstm-mps.yaml` | Crear | Tres variantes, dos learning rates, semillas, ventanas, escalado, MPS obligatorio y límites de entrenamiento. |
| `src/sleep_staging/lstm_data.py` | Crear | Índices de bloques/vecinos, targets elegibles, padding, escalado y Dataset de ventanas por lotes. |
| `src/sleep_staging/models/lstm.py` | Crear | Red LSTM/BiLSTM, head según posición objetivo y comprobaciones MPS. |
| `src/sleep_staging/lstm_research.py` | Crear | Configuración validada, selección interna, entrenamiento/refit, checksums, reanudación, MLflow, OOF, candidato y comparación. |
| `scripts/run_lstm_research.py` | Crear | CLI con config, campaña de referencia, output root, modos preflight/benchmark/run y reanudación. |
| `tests/unit/test_lstm_data.py` | Crear | Cronología, extremos, M, huecos, exclusión de columnas, cobertura y ausencia de contaminación del escalador. |
| `tests/unit/test_lstm_validation.py` | Crear | Particiones anidadas, selección sin etiquetas externas, identidad/checksums y reanudación incompatible. |
| `tests/integration/test_lstm_mps.py` | Crear | Mini entrenamiento con datos reales en MPS, ambos sentidos, gradientes, recarga, probabilidades y artefactos MLflow. |
| `docs/lstm-mps-results.md` | Crear tras entrenar | Protocolo ejecutado, configuraciones, métricas reales, incertidumbre, ganador y siguientes pasos. |
| `docs/assets/lstm-mps/` | Crear tras entrenar | Curvas y figuras estáticas exportadas desde el notebook. |
| `README.md` | Actualizar al completar | Comandos de ejecución y enlace al notebook/informe. |
| Este plan | Actualizar | Checklist, decisiones de ejecución y validaciones reales. |

No se prevén cambios en esquemas persistentes, dependencias ni en `ExperimentConfig`: el nuevo runner tendrá un contrato propio de investigación.

## Pasos de implementación

1. Crear una rama nueva antes de tocar código. Si PR #4 sigue abierto, partir de su HEAD y preparar un PR apilado contra su rama; si ya está fusionado, partir de `origin/main` actualizado. Evitar añadir la nueva campaña al PR anterior.
2. Generar `notebook/05_lstm_mps.ipynb` con el script de la skill de experimentación, modo `pytorch`, clasificación, target `task_5`, métricas `f1_macro`, `accuracy`, `kappa`. Sustituir el KFold de la plantilla por los splits congelados y su fallback de dispositivo por MPS obligatorio antes de ejecutar celdas.
3. Implementar configuración, índice de secuencias y validaciones de datos; probar límites/padding y escalado antes del entrenamiento.
4. Implementar red, trainer MPS y preflight con datos reales; probar ambas arquitecturas y reload.
5. Incorporar selección interna, refit externo, persistencia y MLflow; comprobar una interrupción y reanudación sin duplicar resultados terminados.
6. Medir el coste por época, congelar la configuración de campaña y ejecutar las tres variantes con los cinco folds externos.
7. Consolidar OOF, validar cobertura y métricas, comparar con LightGBM y reentrenar el mejor candidato neuronal en desarrollo.
8. Ejecutar el notebook completo en modo revisión, exportar figuras y redactar el resultado real, incluida una posible ausencia de mejora.
9. Ejecutar pruebas, lint, tipos y formato de los archivos afectados; revisar diff y actualizar el plan. Crear el PR de implementación y publicar revisión/veredicto conforme a `feature-pr-review-gate`; no fusionar automáticamente.

## Pruebas previstas

- Ventanas invariantes al orden inicial de filas; ninguna cruza registro, sujeto, fold o hueco de 30 s.
- Filas M disponibles como contexto sin target M; cambiar etiquetas no modifica las entradas.
- Padding de extremos conserva todos los objetivos y respeta la posición final/central.
- Cambiar valores de un sujeto de validación no cambia el escalador ni las entradas de sujetos de entrenamiento.
- Escalador y pesos recalculados al refit; separación de sujetos en ambas capas de validación.
- Selección interna independiente del target externo; el número de épocas de refit se fija antes de predecir ese fold.
- MPS no disponible o fallback permitido produce fallo claro del runner. En CI sin MPS, las pruebas GPU pueden omitirse explícitamente; la aceptación local de esta campaña exige ejecutarlas en MPS sin omisiones.
- Forward/backward/optimizer en MPS con datos reales, ambas direcciones, pérdida finita, gradientes finitos y parámetros actualizados.
- Salida BiLSTM tomada del segmento objetivo; estado recurrente reiniciado entre ventanas.
- Checkpoint recargado conserva predicciones y probabilidades; checksum alterado o cambio de configuración invalida reanudación.
- Repetir una campaña terminada reutiliza resultados verificados sin duplicar corridas científicas completas.
- OOF con una predicción por target y clase/probabilidades alineadas; comparación con referencia rechaza segmentos diferentes.

## Comandos de ejecución y validación

Los comandos del runner LSTM y los tests LSTM siguientes son **contratos propuestos**: sus archivos deben implementarse antes de ejecutarlos. Los runners `uv`, Ruff, mypy y pytest ya existen en el proyecto.

```bash
uv sync --locked --group dev --group notebook --extra dl --extra boosting

# Ejecutar desde la raíz del repositorio, sin SLEEP_STAGING_ROOT heredado.
env -u SLEEP_STAGING_ROOT PYTORCH_ENABLE_MPS_FALLBACK=0 \
  uv run --locked --extra dl --extra boosting python scripts/run_lstm_research.py \
  --config configs/research/lstm-mps.yaml \
  --reference-campaign var/research/sc-w30-v2/campaign.json \
  --output-root var/research/sc-w30-lstm-mps-v1 --mode preflight

env -u SLEEP_STAGING_ROOT PYTORCH_ENABLE_MPS_FALLBACK=0 \
  uv run --locked --extra dl --extra boosting python scripts/run_lstm_research.py \
  --config configs/research/lstm-mps.yaml \
  --reference-campaign var/research/sc-w30-v2/campaign.json \
  --output-root var/research/sc-w30-lstm-mps-v1 --mode benchmark

env -u SLEEP_STAGING_ROOT PYTORCH_ENABLE_MPS_FALLBACK=0 \
  uv run --locked --extra dl --extra boosting python scripts/run_lstm_research.py \
  --config configs/research/lstm-mps.yaml \
  --reference-campaign var/research/sc-w30-v2/campaign.json \
  --output-root var/research/sc-w30-lstm-mps-v1 --mode run --resume

uv run --locked --extra dl --extra boosting ruff format --check \
  src/sleep_staging/lstm_data.py src/sleep_staging/lstm_research.py \
  src/sleep_staging/models/lstm.py scripts/run_lstm_research.py \
  tests/unit/test_lstm_data.py tests/unit/test_lstm_validation.py \
  tests/integration/test_lstm_mps.py
uv run --locked --extra dl --extra boosting ruff check src tests scripts
uv run --locked --extra dl --extra boosting mypy src/sleep_staging scripts

env -u SLEEP_STAGING_ROOT PYTORCH_ENABLE_MPS_FALLBACK=0 \
  uv run --locked --extra dl --extra boosting pytest \
  tests/unit/test_lstm_data.py tests/unit/test_lstm_validation.py \
  tests/integration/test_lstm_mps.py -q -ra

# La suite histórica requiere los manifiestos/EDF reales del proyecto de origen.
SLEEP_STAGING_ROOT=/Users/adrianalarcon/Documents/uniandes/maia_despliegue_tesis \
  PYTORCH_ENABLE_MPS_FALLBACK=0 \
  uv run --locked --extra dl --extra boosting pytest tests -q -ra
git diff --check
```

Las nuevas pruebas LSTM resolverán su dataset desde la campaña de referencia o una fixture explícita; no usarán `project_root()` para redirigir el código nuevo al proyecto externo. La variable de entorno del último comando solo permite a las pruebas históricas encontrar sus EDF reales.

Para el notebook, registrar un kernel del entorno, verificar `RUN_TRAINING=False` y ejecutarlo con nbclient desde la raíz. Comprobar cero errores, cinco folds por comparación y figuras legibles. No declarar que `make check` completo pasa si persisten diferencias de formato previas ajenas a este cambio.

## Riesgos y mitigaciones

| Riesgo | Mitigación |
| --- | --- |
| Sobreajuste: muchas filas correlacionadas, solo 62 sujetos. | Arquitectura compacta, regularización y selección por sujetos; confirmar posteriormente en test. |
| Mejora aparente por ventanas, cohortes o normalización diferentes. | Mismas claves/folds, control de un segmento y descripción explícita de las diferencias frente al booster. |
| Fuga temporal o estado compartido. | Ventanas dentro de bloques, estado reiniciado y pruebas de invariancia y límites. |
| Operadores o precisión incompatibles con MPS. | Preflight forward/backward, float32 y fallo explícito; no activar fallback CPU. |
| Memoria o duración excesivas. | Ventanas por batch, una corrida GPU a la vez, benchmark previo y checkpoints; nueva identidad si cambia el batch. |
| Confusión entre curva interna y rendimiento externo. | Tags de propósito, nombres de métricas separados y OOF una vez terminado el refit. |
| Checkpoints de otra configuración o campaña. | Identidad completa y checksums; separar directorios y rechazar reanudación incompatible. |
| PR #4 todavía abierto. | Rama y PR apilados durante su dependencia; rebase sobre main cuando se fusione, conservando evidencia de la campaña. |

## Preguntas abiertas

None. La dirección de la primera prueba, dataset, métrica, dispositivo, particiones, referencias y entregables quedan definidos. La duración será un resultado del benchmark, no un bloqueo para implementar el preflight.

## Criterios de aceptación

- [x] Preflight real aprobado para LSTM y BiLSTM en MPS sin fallback CPU.
- [x] Tres variantes evaluadas con cinco folds externos; ninguna comparación parcial presentada como completa.
- [x] Treinta entrenamientos internos, quince decisiones de selección y quince refits externos trazables; refit final neuronal separado de la evaluación OOF.
- [x] Los mismos 152.052 objetivos y cinco clases de referencia, sin sujetos de test y sin ventanas que crucen límites.
- [x] MLflow contiene configuración, entorno, curvas, auditoría de sujetos/dispositivo, checkpoints, OOF y estados de corridas interrumpidas.
- [x] Leaderboard con F1 macro, accuracy, kappa, desviación entre folds y resultados por estadio/sujeto; referencia LightGBM identificada.
- [x] Candidato neuronal recargable y contrato de secuencias documentado; candidato previo conservado si sigue ganando.
- [x] Notebook ejecutado e informe en español con ganador, límites y próximos pasos; ausencia de mejora declarada si ocurre.
- [x] Pruebas relevantes, Ruff, mypy y formato de archivos afectados correctos; validaciones reales registradas en el PR.

### Evidencia de cierre

- Rama: `feature/lstm-mps-sleep-edfx-experiments`, dependiente del PR #4 abierto.
- Campaña `cec1b268c47cf3922935`, run padre `0b66f79bb9c24dbfa20ab038d80bcbd9`, experimento MLflow 2.
- Tres modelos, 15 folds externos, 48 entrenamientos científicos y 611 épocas; 73 runs terminados incluyendo contenedores y comprobaciones técnicas.
- Mejor red: BiLSTM centrada, F1 macro 0,7687 ± 0,0211. Campeón global conservado: LightGBM, 0,7822 ± 0,0161.
- Auditoría independiente: sujetos/escaladores, recomputación de métricas, cobertura OOF, checksums, recarga en MPS y conservación del alias anterior correctos.
- Idempotencia: 73 runs antes y después, campaña idéntica y sin reentrenamiento.
- Suite completa: 59 pruebas aprobadas, sin omisiones; dos warnings de dependencias/piloto histórico. Ruff correcto, mypy sin errores en 34 archivos y siete archivos Python nuevos con formato correcto.
- Notebook: siete celdas de código ejecutadas, tres figuras inspeccionadas y cero errores.
- Ajuste de implementación respecto al diseño inicial: batches por índice sobre la matriz base en MPS, evitando transferir cada ventana completa desde CPU. Se mantuvieron intactas las reglas de ventanas y de separación por sujeto.

## Definición de terminado

Implementación, cinco folds completos por variante, artefactos verificables y notebook/informe revisados. Declarar por separado el mejor modelo neuronal y el ganador global. La victoria sobre LightGBM no es requisito para considerar concluido un experimento válido; la trazabilidad y la comparación honesta sí lo son.

## Referencias técnicas

- [PyTorch: nn.LSTM](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html): formas de entrada/salida, bidireccionalidad y dropout entre capas.
- [PyTorch: backend MPS](https://docs.pytorch.org/docs/stable/notes/mps.html): disponibilidad y ejecución en dispositivo; float32 para esta campaña.
- [PyTorch: variables de entorno MPS](https://docs.pytorch.org/docs/stable/mps_environment_variables.html): fallback CPU y controles de memoria.
- [DeepSleepNet, Supratak et al.](https://arxiv.org/abs/1703.04046): antecedente de aprendizaje temporal para estadificación. La LSTM sobre variables diseñada aquí es una adaptación propia, no una reproducción de su arquitectura ni de sus métricas.

La documentación técnica se contrastó con Context7 al preparar el plan; la compatibilidad concreta debe comprobarse sobre PyTorch 2.14.1 instalado mediante el preflight.
