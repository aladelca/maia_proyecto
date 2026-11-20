# Ejecutar el LightGBM desplegado desde una polisomnografía EDF

## Solicitud y resultado esperado

Una persona sube un EDF, elige **LightGBM · 5 estadios** y obtiene un hipnograma y un CSV con las probabilidades de W, N1, N2, N3 y REM por intervalo de 30 segundos. El backend extrae las mismas variables que usó el modelo durante el entrenamiento. La referencia de implementación es `main` en `20fb50b`.

## Alcance y decisiones

- Usar el LightGBM de producción. La release de referencia es `lgbm-49f74e944ce5457b`, con preprocesamiento `sc-w30-context435-v1`. La ejecución resolverá el modelo y las columnas desde el manifiesto versionado de la release activa.
- Incorporar LightGBM como elección explícita para un EDF. Los clientes actuales que envían un EDF sin modelo seguirán recibiendo FullNightNet. El ejemplo y la carga Parquet seguirán utilizando LightGBM.
- La primera versión cubre el montaje con el que se entrenó esta release. La adaptación a otros montajes, el reentrenamiento y la comparación científica de arquitecturas son trabajos separados. La tabla intermedia de características puede vivir en memoria.

## Evidencia y contrato de entrada

Comprobé el manifiesto en `var/cloud/bundle/manifest.json` y el exportador en `scripts/export_serving_bundle.py`. Cada fila contiene cuatro campos de identificación y elegibilidad (`recording_id`, `epoch_id`, `onset_s`, `eligible`) más las siguientes **145 variables de señal**.

| Canal EDF | Frecuencia esperada | Variables |
|---|---:|---:|
| EEG Fpz-Cz | 100 Hz | 42 |
| EEG Pz-Oz | 100 Hz | 42 |
| EOG horizontal | 100 Hz | 42 |
| EMG submental, envolvente | 1 Hz | 19 |

`serving/contract.py::predictions` construye las **435 entradas** del clasificador al combinar las 145 originales con dos conjuntos de 145 de contexto temporal. El primero usa hasta cuatro intervalos previos y el segundo una ventana ponderada de 15 centrada en el intervalo actual. El contexto se normaliza por registro. Los nombres y su orden vienen de `raw_columns`, `model_columns` y `model.feature_names_in_`.

El entrenamiento en `research_data.py::_record_features` filtraba el registro continuo antes de segmentarlo. Las anotaciones delimitaban la ventana W30, desde 30 minutos antes del primer sueño hasta 30 minutos después del último, y ayudaban a seleccionar las filas. El EDF de desarrollo `SC4001E0-PSG.edf` contiene **2.650 intervalos completos**. El `example.parquet` congelado contiene **841**. Cambiar el conjunto de filas cambia los vecinos y los cuantiles del contexto. Es posible reproducir el ejemplo y puntuar la noche completa, pero cada prueba requiere su propia referencia.

## Tareas de implementación

1. [x] **Compartir la extracción de señal.** Aislar la parte numérica de `src/sleep_staging/research_data.py::_record_features` en `src/sleep_staging/features/`, manteniendo el cálculo de etiquetas y del recorte en investigación. Reutilizar `data/edf.py::read_psg` para unidades y canales y `features/research.py::batch_signal_features` para los descriptores. Reproducir filtros Butterworth de orden 4 de fase cero sobre el canal completo (EEG 0,3–35 Hz y EOG 0,3–15 Hz), envolvente EMG de 1 Hz, tipos numéricos, nombres y lotes de 128 épocas. Si se mueve el código, incluir el módulo compartido en la huella de procedencia de futuras cachés de investigación. Verificar las características históricas antes de conectar el clasificador.
2. [x] **Construir la noche preparada a partir del EDF.** Añadir un adaptador en `src/sleep_staging/serving/` que valide cabecera, tamaño, duración, unidades, duplicados y el montaje anterior. Aceptar canales extra y usar los cuatro requeridos. Identificar por nombre el canal ausente o la frecuencia incompatible. El EMG crudo a 100 Hz produce otras variables y queda fuera de este contrato. Enumerar intervalos completos desde el segundo cero y descartar la cola inferior a 30 s. Formar los identificadores según el convenio existente. Usar la regla de Fpz-Cz plano del extractor histórico. Para construir el contexto, retirar esas filas y conservar los saltos en `onset_s`. Reinsertarlas en el resultado con `eligible=false` y probabilidades nulas. Si todo el registro es plano, devolver cero épocas elegibles. Rechazar señales no finitas con error de entrada.
3. [x] **Ejecutar la release congelada en CPU.** Pasar la tabla válida a `serving/contract.py::predictions`, que ya genera el contexto, comprueba el orden del modelo y reordena las probabilidades por etiqueta. Validar el protocolo del manifiesto antes de predecir. Las filas de una noche se procesan juntas al calcular el contexto. La extracción de señal sí se puede repartir en lotes. Añadir el comando propuesto `scripts/predict_edf_lgbm.py` que lea el paquete local y use este mismo adaptador para escribir un CSV. Contrastar con los artefactos congelados y dejar el comando reproducible en el PR.
4. [x] **Integrar con la API y el worker.** Extender las rutas existentes `/v1/uploads` y `/v1/predictions` y sus equivalentes públicas en `services/api/app.py`. El cliente enviará `format="edf", model="lightgbm"` en ambas solicitudes para esta elección. Un EDF con LightGBM acepta `profile="auto"` o el montaje completo. Validar el modelo, el formato y el perfil antes de firmar o encolar. Los valores desconocidos y las combinaciones incompatibles devuelven HTTP 400. Conservar la forma canónica del evento EDF histórico para solicitudes FullNightNet, incluso si el cliente indica su nombre explícitamente. Para LightGBM, incluir el modelo en el evento de Step Functions y la comparación de idempotencia. Reintentar la misma solicitud recupera el mismo trabajo. Reutilizar la clave con otro modelo devuelve 409.
5. [x] **Entregar el resultado por los canales actuales.** En `serving/worker.py`, reutilizar `load_release()`, la comprobación SHA-256 y las salidas JSON/CSV. Devolver la versión real del modelo, las cinco clases, el perfil resuelto, la cobertura y el identificador del preprocesamiento. Añadir a los metadatos de esta ruta la política de ventana de registro completo para distinguirla de una noche preparada con W30. El EDF y los resultados usan los objetos S3 y el trabajo existentes. La base de datos y el esquema del Parquet histórico no requieren migración.
6. [x] **Exponer la elección en la app.** En `frontend/src/main.tsx`, mostrar el selector de modelo cuando la entrada sea EDF. Con LightGBM mostrar el montaje requerido. Con FullNightNet ofrecer sus perfiles actuales. Reutilizar el panel de hipnograma y la descarga. Un entorno puede disponer de los pesos LightGBM aunque carezca de los pesos neuronales. Publicar una capacidad independiente `edfLgbmEnabled` en la configuración del backend y del frontend mediante `infra/aws/main.tf`, `frontend/scripts/write-public-config.mjs` y la variable pública de Vercel `MAIA_PUBLIC_CONFIG`. Interpretar la ausencia del campo como opción deshabilitada hasta verificar el backend.

## Pruebas y criterios de aceptación

| Prueba | Resultado exigido |
|---|---|
| **Paridad histórica de variables y predicciones** | Leer el EDF de desarrollo SC4001 completo y extraer las 145 columnas. Alinear por `onset_s` con las 841 filas de `var/cloud/bundle/example.parquet`. Contrastar primero las columnas crudas, con nombres, orden, tipos y tolerancia float32 documentada por columna. Para la prueba de predicción, aplicar esos mismos tiempos y la máscara de referencia **antes** de construir el contexto. Exigir las mismas etiquetas y las cinco probabilidades de `expected.parquet` con `rtol=0` y `atol=1e-6`. Las filas seleccionadas por anotaciones solo se usan en esta prueba. |
| **Inferencia desatendida sobre el EDF real completo** | Ejecutar solo con el PSG, sin hipnograma. Para SC4001, esperar 2.650 filas en orden, cada 30 segundos. Comprobar cobertura, cinco probabilidades finitas que sumen uno en cada fila elegible, etiquetas válidas y valores nulos en las excluidas. Comparar la ejecución local y la del worker sobre el mismo intervalo completo, alineadas por tiempo. |
| **Límites y señal degradada** | Probar una sola época, cola parcial, Fpz-Cz plano, huecos de elegibilidad, canal ausente o duplicado, EMG a 100 Hz y EDF truncado. Separar el límite del Parquet de 3.000 filas del de EDF (128 MiB y hasta 48 horas). Validar al menos un caso superior a 3.000 filas y comprobar la normalización sobre todo el registro. Medir memoria máxima y duración en la imagen del worker con una noche larga admisible frente a 3.072 MB y 300 s. |
| **API, UI y compatibilidad** | Cubrir visitante y usuario autenticado, claves idempotentes, versiones S3, errores legibles y selector de modelo. Verificar el ejemplo y Parquet con LightGBM y EDF por omisión con FullNightNet. Ejecutar una predicción desde la UI con el EDF sintético de 10 minutos y confirmar 20 filas y cinco columnas de probabilidad. |

Extender `tests/unit/test_research_features.py`, `tests/unit/test_serving.py`, `tests/unit/test_cloud_api.py` y `frontend/tests/local-edf.spec.ts`. Añadir la paridad EDF–Parquet como prueba de integración con la release congelada y la fuente real de desarrollo. En CI ordinario usar un EDF sintético generado como fixture. En la validación de release ejecutar también SC4001. Ampliar `scripts/smoke_cloud.py` para seleccionar el modelo y verificar paridad local–AWS manteniendo la prueba actual de seis clases.

## Validación y entrega

Al implementar el comando propuesto, esta será la prueba manual con el EDF sintético. El archivo de ejemplo y el paquete del modelo están disponibles en este espacio de trabajo.

```bash
uv run python scripts/predict_edf_lgbm.py --edf output/demo-sueno-sintetico-10min.edf --bundle var/cloud/bundle --output /tmp/maia-edf-lgbm-demo.csv
```

Ejecutar después las pruebas del repositorio. Para Playwright, iniciar Vite en otro terminal con `npm --prefix frontend run dev -- --port 5173`.

```bash
uv run pytest tests/unit/test_research_features.py tests/unit/test_serving.py tests/unit/test_cloud_api.py tests/unit/test_fullnight_serving.py -q
uv run pytest tests/integration -k edf_lgbm -q
npm --prefix frontend run build
DEMO_ORIGIN=http://127.0.0.1:5173 npm --prefix frontend run test:e2e -- tests/local-edf.spec.ts
```

Entregar la comparación de SC4001 con número de filas, cobertura, diferencias máximas y tiempos de ejecución. Usar una rama desde `main`. Validar primero el adaptador local y después el worker en `dev`. Habilitar el selector al pasar la prueba de paridad. Para revertir, retirar la capacidad pública y desactivar esta combinación en la API.

**Interpretación científica.** La igualdad con el Parquet comprueba que el código reproduce el modelo. La eficacia sobre registros completos es otra pregunta. Antes de atribuir al nuevo flujo el F1 histórico, medir el modelo congelado con registros completos y etiquetas expertas bajo un protocolo declarado. Usar las anotaciones únicamente al puntuar. La aplicabilidad a EMG crudo es una cuestión posterior que requeriría validar una transformación equivalente o entrenar otra release.


## Validación de la rama

- SC4001 produjo 2.650 filas elegibles. Al seleccionar los 841 intervalos históricos antes de calcular el contexto, la diferencia máxima frente a la referencia fue cero en las 145 características y en las cinco probabilidades. Las etiquetas coincidieron en todas las filas. La CLI y el handler también coincidieron en la noche completa con S3 simulado.
- Chrome cargó el EDF sintético en la interfaz compilada, ejecutó el modelo real mediante el worker local y descargó las 20 filas con cinco columnas de probabilidad. Las pruebas de la API también cubren visitantes y usuarios autenticados.
- El contenedor Linux amd64 del worker procesó un EDF sintético de 48 horas, 5.760 filas elegibles, en 7,97 s con dos CPU y un límite de 3.072 MiB. El máximo del cgroup fue 1.215,4 MiB. La medición local empleó S3 simulado y excluye la latencia de red de AWS.
- La activación en el entorno compartido y la prueba de humo contra AWS corresponden al despliegue posterior a la revisión de la rama.
