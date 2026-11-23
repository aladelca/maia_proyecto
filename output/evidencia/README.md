# Resultados y modelos de estadificación del sueño

El [informe experimental](../pdf/resultados-ibio-registro-completo.pdf) y las [tablas del estudio](resumen/ibio/metrics.csv) se pueden consultar directamente en GitHub. El [paquete de inferencia](../ibio-modelos-y-resultados.zip) contiene los pesos finales y permite procesar un EDF con las [instrucciones de ejecución](../../docs/ibio-registro-completo.md).

| Archivo Git LFS | Contenido | Tamaño comprimido |
| --- | --- | ---: |
| [ibio-registro-completo.zip](ibio-registro-completo.zip) | Predicciones por época, árboles y redes por pliegue, refit, U-Sleep, matrices y tablas de la evaluación congelada | 830 MiB |
| [tcn-y-referencias.zip](tcn-y-referencias.zip) | Campaña TCN previa, incluidos los modelos, las predicciones fuera de pliegue y las referencias LightGBM y BiLSTM de los cuadernos 07 a 09 | 456 MiB |
| [pruebas-tecnicas.zip](pruebas-tecnicas.zip) | Corridas cortas sobre datos de desarrollo, incluidas las verificaciones en CPU y MPS | 158 MiB |
| [mlflow-ibio.zip](mlflow-ibio.zip) | Copia portátil de SQLite con los cuatro identificadores originales, parámetros, métricas y los 20 archivos registrados en MLflow | 6,9 MiB |

### Recuperar las predicciones y los modelos por pliegue

Desde la raíz de un clon con Git LFS instalado, estos comandos descargan los archivos, verifican sus SHA-256 y restauran las rutas que usan los scripts y cuadernos. Para recuperar un solo paquete, se puede indicar su nombre en `--include` y en `unzip`.

```bash
git lfs pull --include='output/evidencia/*.zip'
(cd output/evidencia && shasum -a 256 -c SHA256SUMS)
for paquete in output/evidencia/*.zip; do unzip -nq "$paquete"; done
```

### Consultar las corridas originales en MLflow

La copia conserva los identificadores y tiempos originales de desarrollo y prueba. Las ubicaciones de los archivos en SQLite son relativas a la raíz del clon. Tras extraer `mlflow-ibio.zip`, ejecuta desde esa raíz el siguiente comando y abre el MLflow local en `http://127.0.0.1:5003/#/experiments/1`.

```bash
uv run --locked mlflow server --host 127.0.0.1 --port 5003 \
  --backend-store-uri sqlite:///var/research/ibio-fullnight/tracking/mlflow.db
```

### Origen de los datos

El estudio utiliza las polisomnografías y anotaciones públicas de **Sleep-EDF Expanded 1.0.0**, distribuidas bajo Open Data Commons Attribution License 1.0. Los archivos de evidencia incluyen resultados derivados e identificadores públicos de registro. Los EDF originales se obtienen de PhysioNet. [Datos y licencia](https://physionet.org/content/sleep-edfx/1.0.0/) · [Kemp et al., 2000](https://doi.org/10.1109/10.867928) · [Pollard et al., 2026](https://doi.org/10.1038/s44360-026-00096-z)
