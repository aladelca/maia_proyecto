# Protocolo experimental local

## Piloto implementado

- Datos: ocho registros reales de ocho sujetos de Sleep Cassette, primera noche, con hasta 240 épocas por registro distribuidas temporalmente.
- Entrada inicial: EEG Fpz-Cz, 100 Hz, épocas de 30 s y filtro Butterworth de orden 4, 0,3–35 Hz, aplicado por registro con fase cero.
- Tarea del piloto: W, N1, N2, N3 y REM. M y `?` no son objetivos de esta tarea; la etiqueta original se conserva.
- Métrica primaria: F1 macro con lista fija de cinco clases. Secundarias: accuracy, kappa, métricas por clase y matrices de confusión.
- Partición: reserva determinista del 20 % de sujetos para test; cinco folds agrupados por sujeto en desarrollo, con semilla 42. Las particiones se reutilizan entre modelos compatibles.
- Modelos: baseline mayoritario, Random Forest de 64 árboles con pesos balanceados, CNN 1D pequeña con normalización de entrenamiento, Adam y validación interna agrupada para early stopping.

El escalador, imputador y pesos se ajustan sobre entrenamiento de cada fold. En CNN, la normalización y los pesos se calculan sobre la parte de entrenamiento interna. La última capa usa pooling con dimensiones compatibles con CPU y MPS. La predicción se realiza en CPU tras trasladar los pesos; no se exige igualdad bit a bit entre dispositivos.

Los runs por fold conservan parámetros, soporte por clase, accuracy, F1 macro, kappa y matriz de confusión. CNN conserva historial, dispositivo, mejor época y checkpoint. El run padre conserva configuración, versiones, hashes, lockfile, predicciones OOF, tabla de comparación y el modelo reentrenado solo sobre desarrollo. El ganador se elige por F1 macro medio; el coste deshace empates.

Las cifras del piloto no constituyen evaluación final del proyecto. La tarea de seis clases está configurada en el contrato, pero el soporte de M en este subconjunto es insuficiente para una comparación fiable con cinco folds. El sistema rechaza entrenamientos con clases ausentes en vez de ocultar esa condición.

## Campañas pendientes

1. Procesar Sleep Cassette completo sin muestreo de épocas y congelar el protocolo científico.
2. Formalizar la evaluación final sobre test reservado y sus intervalos por bootstrap de sujetos. Esta entrega no ejecuta esa evaluación ni incluye aún un comando `evaluate-test`.
3. Comparar canales adicionales sobre una cohorte elegible común, con la cobertura documentada.
4. Validar saturación, artefactos avanzados y criterios de exclusión con los investigadores.
5. Analizar Telemetry por separado y verificar sus condiciones placebo/temazepam antes de usarlas en informes.
6. Añadir boosting y arquitecturas/fusión multimodal según presupuesto y resultados de desarrollo.
7. Elaborar la comparación bibliográfica con cohortes, canales, clases, recortes y particiones compatibles.

La inferencia de esta etapa recibe características o arrays preparados con el contrato del modelo. La plataforma de carga y clasificación de PSG nuevos requiere una integración posterior.
