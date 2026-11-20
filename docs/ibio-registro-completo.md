# Estadificación del sueño a partir de un EDF completo

La campaña de noviembre de 2026 implementa las tres comparaciones de la propuesta IBIO. La plataforma procesa una polisomnografía sin hipnograma, divide el registro en épocas de 30 segundos y devuelve un estadio y sus probabilidades. La clasificación para la plataforma incluye movimiento (M). La comparación con trabajos de estadificación utiliza cinco clases. [Informe de resultados en PDF](../output/pdf/resultados-ibio-registro-completo.pdf) · [Propuesta de experimentos](../output/pdf/revision-propuesta-ibio.pdf)

## Usar el modelo congelado

Las predicciones por época, los pesos por pliegue y las cuatro corridas originales de MLflow se pueden recuperar desde GitHub. [Descarga y restauración de la evidencia](../output/evidencia/README.md)

Desde la raíz del proyecto, los siguientes comandos instalan las dependencias y levantan la interfaz en el puerto 8765. La opción `cpu` basta para puntuar un registro. En otro equipo puedes descomprimir el paquete y asignar a `RUN` el directorio de la campaña extraída. [Abrir la plataforma local](http://127.0.0.1:8765) · [Paquete de modelos y resultados](../output/ibio-modelos-y-resultados.zip)

```bash
uv sync --locked --group dev --extra dl --extra boosting --extra cloud --extra usleep
npm ci --prefix frontend
npm --prefix frontend run build
RUN=var/research/ibio-fullnight/development/681a52f21d273bf190ae
.venv/bin/python scripts/serve_fullnight.py --run "$RUN" --device cpu
```

Para obtener directamente un CSV desde un solo EEG, usa la ruta de una PSG en lugar del marcador.

```bash
.venv/bin/python scripts/serve_fullnight.py --run "$RUN" --device cpu \
  --edf /ruta/noche-PSG.edf --profile eeg_pz_oz --output /tmp/hipnograma.csv
```

Para puntuar cinco estadios con la mejor mezcla global, usa el perfil de ambos EEG y EOG. Los demás perfiles aplican la mezcla de ondas y el LightGBM correspondiente. En un EDF de 2.650 épocas se reprodujeron las etiquetas de ambas mezclas guardadas en MPS, con una diferencia máxima de probabilidad de 0,00000018 al ejecutar en CPU. [Comparación época por época](../output/evidencia/resumen/ibio/five-stage-edf-parity.json)

```bash
.venv/bin/python scripts/serve_fullnight.py --run "$RUN" --task five \
  --edf /ruta/noche-PSG.edf --profile eeg_fpz_cz+eeg_pz_oz+eog --output /tmp/cinco-estadios.csv
```

El contrato admite EDF continuo de 16 bits, duración de 30 segundos a 48 horas y hasta 128 MiB. Reconoce los encabezados de Sleep-EDFx `EEG Fpz-Cz`, `EEG Pz-Oz`, `EOG horizontal` y `EMG submental`. Los EEG y EOG son de 100 Hz. El EMG admite la envolvente nativa a 1 Hz de Cassette o la señal de 100 Hz de Telemetry. La opción `auto` usa los canales reconocidos. Hay 12 perfiles explícitos que combinan tres elecciones de EEG con cuatro elecciones de EOG y EMG. Una época sin un EEG elegido finito y con variación conserva su posición en el CSV, con `eligible=false` y predicción vacía. [Contrato y límites](../src/sleep_staging/serving/edf.py)

La interfaz local ejecuta el modelo **directo de seis clases**, elegido en desarrollo con los cuatro canales. La mezcla de cinco clases se evaluó en el experimento de investigación. La API y el trabajador AWS admiten el nuevo formato con un archivo de pesos versionado. La prueba de esta entrega se hizo en local. [Publicación de pesos opcionales](../scripts/publish_cloud_inputs.py)

## Datos y protocolo

| Muestra | Personas | Registros | Uso |
| --- | ---: | ---: | --- |
| Sleep Cassette, desarrollo | 62 | 122 | Cinco pliegues agrupados por participante y reajuste final |
| Sleep Cassette, prueba | 16 | 31 | Evaluación del modelo congelado |
| Sleep Telemetry | 22 | 44 | Una noche con placebo y otra con temazepam por persona |

Las dos cohortes pertenecen a **Sleep-EDF Expanded 1.0.0**, la base identificada en la actividad. Cambian la adquisición y la población. Telemetry ofrece una prueba de transferencia dentro de la base. Las condiciones provienen de la hoja oficial `ST-subjects.xls` (SHA-256 `b377133e03897559e3e9fa8ef468f4505becaecebe1bc0c988d39a9158267758`). [PhysioNet](https://physionet.org/content/sleep-edfx/1.0.0/) · [Partición Cassette](../configs/splits/sc_subjects_v1.csv) · [Asignación de condiciones por registro](../configs/splits/st_conditions_v1.csv)

**Ventana completa.** Se crean las entradas a partir de la señal adquirida. Se predicen todas las épocas completas con EEG utilizable. La evaluación posterior usa la etiqueta del experto. W, N1, N2, N3 y REM integran el desenlace primario, con los estadios originales 3 y 4 agrupados como N3. La tarea de seis clases incorpora M. Las épocas sin clasificar y los fragmentos de menos de 30 segundos quedan fuera de la puntuación. La normalización por registro y el contexto centrado corresponden a inferencia retrospectiva.

**Ventana histórica W30.** Usa las mismas predicciones y conserva el intervalo entre la primera y la última época de sueño con 30 minutos de vigilia de margen a cada lado. En desarrollo hubo 331.086 épocas con cinco estadios en el registro completo y 152.052 en W30. La proporción de vigilia cambió de 68,8 % a 32,1 %. Las anotaciones intervienen en este análisis de sensibilidad después de la predicción.

**Presupuesto.** Para las CNN se fijaron dos pasadas por toda la señal, 24 unidades de ancho, contexto de 15 épocas, semilla base 42 y pérdida ponderada según las frecuencias del conjunto de entrenamiento. El perfil se muestrea en entrenamiento conservando al menos un EEG. Las redes calculan una predicción determinista con el perfil solicitado en evaluación. Se ajustó un LightGBM de 48 iteraciones por perfil y pliegue. El espectrograma utiliza ventanas de Hann de dos segundos a 50 Hz. Los pesos de las dos mezclas se fijaron de antemano. [Configuración y versiones ejecutadas](../output/evidencia/resumen/ibio/protocol.json) · [Pesos congelados y sujetos](../output/evidencia/resumen/ibio/freeze.json)

## Resultados en los mismos registros

F1 macro acumulado sobre las épocas elegibles, con ambos EEG y EOG. La mezcla doble usa LightGBM y ondas en partes iguales. La triple usa 50 % LightGBM, 25 % ondas y 25 % espectrograma.

| Muestra | LightGBM | Ondas | Espectrograma | Doble | Triple | F1 N1 doble → triple | Exactitud triple |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cassette desarrollo | 0,735 | 0,612 | 0,649 | 0,740 | 0,752 | 0,416 → 0,438 | 90,3 % |
| Cassette prueba | 0,710 | 0,673 | 0,732 | 0,740 | 0,752 | 0,406 → 0,423 | 89,3 % |
| Telemetry placebo | 0,592 | 0,500 | 0,682 | 0,642 | 0,688 | 0,404 → 0,475 | 71,9 % |
| Telemetry temazepam | 0,623 | 0,511 | 0,703 | 0,676 | 0,706 | 0,389 → 0,447 | 76,3 % |
| Telemetry, ambas noches | 0,608 | 0,506 | 0,693 | 0,660 | 0,697 | 0,397 → 0,463 | 74,1 % |

La mejora macro de la mezcla triple frente a la doble fue de **1,23 puntos** en Cassette prueba (IC 95 % de 0,65 a 1,82) y **3,74 puntos** en el total de Telemetry (2,42 a 4,86). El IC para la mejora en N1 incluyó cero en Cassette (1,70 puntos, de −0,02 a 3,51) y quedó por encima de cero en Telemetry (6,55 puntos, de 3,98 a 9,12). Los intervalos usan 1.000 remuestras pareadas por persona, manteniendo sus noches juntas. Una sola semilla limita las conclusiones sobre variabilidad de entrenamiento. Las diferencias entre condiciones son descriptivas. [Intervalos de prueba](../output/evidencia/resumen/ibio/reserved/paired-subject-bootstrap.json)

**Señales.** En Cassette reservado, la mezcla doble pasó de 0,546 con Pz-Oz solo a 0,705 al agregar EOG. En la misma muestra, Fpz-Cz solo obtuvo 0,671 y los cuatro canales, 0,746. La tabla de perfiles contiene la cobertura y la diferencia frente al montaje completo de cada modelo. Los 12 perfiles pasaron también una carga HTTP con un EDF real de 2.650 épocas. [Resultados por perfil](../output/evidencia/resumen/ibio/tables/profiles.csv) · [Verificación de carga](../output/evidencia/resumen/ibio/edf-upload-check.json)

**Movimiento.** En desarrollo, 115 épocas de 28 personas contenían M. La media de F1 macro en seis clases fue 0,550 con el cabezal directo y 0,546 con el jerárquico. En Cassette reservado el directo detectó 5/13 M (sensibilidad 38,5 %, precisión 2,1 %, F1 0,040, 7,52 falsas alarmas por noche). En placebo detectó 4/34 y en temazepam 0/49. La plataforma devuelve M como salida experimental. [Soporte y falsas alarmas por noche](../output/evidencia/resumen/ibio/tables/movement-by-night.csv) · [Matrices de ambas tareas](../output/evidencia/resumen/ibio/reserved/confusion.json)

**N1.** En Cassette prueba, la mezcla triple acertó el 52,5 % de las épocas en transiciones y el 92,6 % en intervalos estables, considerando los cinco estadios. De las 2.760 épocas N1 estables, llamó W a 294 y N2 a 673. De las 2.156 próximas a un cambio, llamó W a 311 y N2 a 555. Las tres épocas N1 sin dos vecinas evaluables se contabilizan aparte. La rama espectral sola consiguió un F1 para N1 de 0,438 en Cassette y 0,475 en Telemetry. [Errores por contexto](../output/evidencia/resumen/ibio/tables/n1-errors.csv) · [Soportes, precisión, sensibilidad, F1 y kappa](../output/evidencia/resumen/ibio/tables/stages.csv)

## Comparación con trabajos publicados

Se reentrenó **U-Sleep de Braindecode 2.0.0**, desde cero, en los mismos cinco pliegues de Cassette, con Fpz-Cz y EOG, los mismos datos normalizados a 50 Hz y las mismas ventanas de evaluación. Usó 12 niveles, cuatro ventanas centradas en clases de 35 épocas por registro y por pasada, dos pasadas y una semilla. Con ese presupuesto alcanzó F1 macro **0,455** en Cassette prueba y **0,267** en Telemetry, frente a 0,729 y 0,647 para la mezcla doble con las mismas señales. Su F1 de N1 fue cero en las dos pruebas. Comparar el potencial de ambas arquitecturas requeriría igualar el presupuesto de ajuste y las muestras vistas. El estudio publicado de U-Sleep entrenó con múltiples estudios y señales a 128 Hz. [Artículo original](https://doi.org/10.1038/s41746-021-00440-5) · [Implementación de referencia](https://braindecode.org/stable/generated/braindecode.models.USleep.html) · [Evaluación del reajuste local](../output/evidencia/resumen/ibio/usleep/2f029ad064736e5f1bd9/reserved/metrics.csv)

DeepSleepNet informó 82,0 % de exactitud y 0,769 de F1 macro en 41.950 épocas, con Fpz-Cz, cinco estadios, recorte de vigilia y 20 pliegues. Nuestra mezcla de dos ramas con **Fpz-Cz y W30** alcanzó 73,5 % de exactitud y 0,656 de F1 macro en 43.417 épocas de Cassette reservado. Los participantes, el entrenamiento y la partición difieren entre trabajos. El **0,8009** del proyecto era la media de pliegues de una campaña anterior con tres canales entrenada en W30. Para la nueva mezcla triple, la media de desarrollo en registro completo fue **0,7544 ± 0,0229**. [DeepSleepNet, tablas III y IV](https://arxiv.org/pdf/1703.04046) · [Campaña histórica](../output/evidencia/resumen/tcn/leaderboard-all.csv)

## Reproducir las corridas y revisar la evidencia

La raíz de datos debe contener los manifiestos de Sleep Cassette, Sleep Telemetry y el inventario completo. Los EDF se verifican por SHA-256 al construir las cachés. La prueba corta usa diez participantes de desarrollo. En Linux con una GPU NVIDIA se sustituye `mps` por `cuda`. Se ejecutaron el piloto en CPU y MPS, la campaña principal y U-Sleep en MPS y la inferencia de la interfaz en CPU. El equipo utilizado carece de CUDA.

```bash
export SLEEP_RESEARCH_DATA_ROOT=/ruta/al/proyecto/con/data
.venv/bin/python scripts/run_fullnight.py --device cpu --smoke
# La última línea del entrenamiento contiene el directorio de esta reproducción.
RUN=$(.venv/bin/python scripts/run_fullnight.py --device mps --epochs 2 --trees 48 | tail -n 1)
.venv/bin/python scripts/run_fullnight.py --device mps --evaluate-frozen "$RUN"
U_RUN=$(.venv/bin/python scripts/run_usleep_reference.py --device mps --run "$RUN" --source-root "$SLEEP_RESEARCH_DATA_ROOT" | tail -n 1)
.venv/bin/python scripts/run_usleep_reference.py --device mps --run "$RUN" --source-root "$SLEEP_RESEARCH_DATA_ROOT" --evaluate-frozen
.venv/bin/python scripts/sync_fullnight_tracking.py --run "$RUN"
.venv/bin/python scripts/sync_fullnight_tracking.py --run "$U_RUN"
.venv/bin/python scripts/report_fullnight.py --run "$RUN" --bundle "$RUN/modelos-y-resultados.zip"
```

Los identificadores dependen del código, partición, versiones, datos y dispositivo. La ruta de ejemplo identifica la campaña entregada. Una reproducción desde una revisión posterior genera su propio identificador. El evaluador devuelve el resultado guardado si la prueba ya terminó y verifica la integridad de los archivos. El archivo ZIP conserva las fuentes exactas del entrenamiento medido. Los registros de MLflow para desarrollo y prueba se guardan en SQLite local. [Fuentes del entrenamiento](../output/evidencia/resumen/ibio/training-source.zip) · [Métricas completas de desarrollo](../output/evidencia/resumen/ibio/metrics.csv) · [Métricas completas de prueba](../output/evidencia/resumen/ibio/reserved/metrics.csv) · [Identificadores de MLflow](../output/evidencia/resumen/ibio/mlflow-runs.json)

La siguiente pregunta del artículo es si el espectrograma también mejora N1 frente a varios anotadores. El plan propone ISRUC para aprender del desacuerdo y Dreem para contrastar con un consenso independiente. La eficacia en tiempo de revisión requiere una prueba cronometrada con lectores. [Plan de investigación](../plans/sleep-staging-research-plan.md)
