# Contratos de datos

Solo se admiten Sleep Cassette y Sleep Telemetry reales de Sleep-EDFx 1.0.0. Los archivos EDF brutos son inmutables y cada manifiesto conserva ruta, URL de descarga, SHA-256 publicado, tamaño y estado.

`recording_id` identifica una noche por el prefijo de seis caracteres del archivo. `subject_id` incluye la cohorte y el identificador del sujeto; todas sus noches se asignan juntas. PSG e hipnograma se emparejan por ese prefijo, con independencia del sufijo del anotador.

El índice de épocas conserva `epoch_id`, sujeto, registro, cohorte, `onset_s`, `onset_sample_eeg`, duración, anotación original, etiquetas `task_5`/`task_6`, motivo de exclusión, fila del array y disponibilidad/calidad por canal. Una época válida tiene 30 segundos y 3000 muestras EEG a 100 Hz.

R se mapea a REM y estadios 3/4 a N3. M solo es objetivo supervisado en `task_6`; `?` queda sin etiqueta. Huecos, solapamientos, anotaciones desalineadas y finales incompletos se reportan. Los intervalos excluidos conservan su posición temporal.

Canales canónicos: `eeg_fpz_cz`, `eeg_pz_oz`, `eog`, `emg`. Se mantienen sus frecuencias nativas y amplitudes en microvoltios; no se vuelve a referenciar el EEG bipolar. El EMG Cassette a 1 Hz se identifica como `emg_envelope`; el EMG Telemetry a 100 Hz, como `emg_raw`. Sus columnas de features tienen prefijos diferentes.

Las señales preparadas son arrays NumPy float32 por registro y canal; su primer eje corresponde a `array_row`. Las características deterministas se almacenan en Parquet, con `epoch_id` y columnas `<canal>__<representación>__<característica>`. Identificadores y etiquetas no entran al modelo. Los atributos Parquet conservan `preprocessing_id` y `feature_set_id`; la inferencia comprueba esa procedencia.

Los fingerprints incluyen entradas, parámetros, código y versiones relevantes. Cada versión tiene otro directorio; `metadata.json` se escribe al completar la preparación. Las salidas individuales usan reemplazo atómico y los pasos que comparten destino usan bloqueos.

La calidad inicial marca segmentos planos y rechaza NaN/Inf y duraciones incompatibles. El tratamiento de saturación y umbrales fisiológicos avanzados queda pendiente de validación con el equipo de investigación; no se elimina M automáticamente como ruido.
