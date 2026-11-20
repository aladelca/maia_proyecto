# Release del híbrido de cinco etapas

El worker sirve la corrida de MLflow `a64415fc2bd649b39c6fd158994b0479` (`stack_five`). La imagen incluye el clasificador reajustado sobre 62 participantes y el checkpoint neuronal del que dependen sus entradas. Ambos se verifican por SHA-256 antes de deserializarlos. El CSV de referencia corresponde al EDF sintético de diez minutos del repositorio.

Para puntuar un EDF en CPU desde la raíz:

```sh
.venv/bin/python -m sleep_staging.serving.stacked_edf noche.edf hipnograma.csv
```

La API publicada admite únicamente EDF con ambos EEG, EOG horizontal y EMG submental en envolvente. Las solicitudes de carga y de predicción envían `format="edf"` y pueden fijar `model="stack_five"`. El ejemplo se ejecuta con el mismo worker y devuelve veinte épocas de demostración. El historial científico y los artefactos originales se conservan en MLflow. [Resultados de la evaluación pareada](../../docs/comparacion-fullnight-lightgbm.md)

Para usar el modelo con la interfaz en este equipo, ejecuta `npm --prefix frontend run build` y `.venv/bin/python scripts/serve_stack.py`. [Abrir el laboratorio local](http://127.0.0.1:8775).
