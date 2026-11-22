# FullNightNet como entrada del LightGBM publicado

En las noches reservadas de Sleep Cassette, añadir las cinco probabilidades de etapa de FullNightNet al LightGBM elevó el F1 macro de 0,7879 a 0,7970 con el protocolo histórico W30 y de 0,7630 a 0,7806 al puntuar el EDF completo. FullNightNet por sí solo obtuvo puntuaciones inferiores a las de LightGBM. La comparación evalúa los mismos pacientes, canales, etiquetas y épocas dentro de cada columna. [Notebook ejecutado](../notebooks/20261120-1342-fullnight-lightgbm-w30.ipynb)

| Modelo | Desarrollo W30, F1 macro medio ± DE | Prueba W30, F1 macro | Prueba W30, F1 N1 | EDF completo, F1 macro | EDF completo, F1 N1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LightGBM publicado | 0,7893 ± 0,0179 | 0,7879 | 0,5156 | 0,7630 | 0,4530 |
| FullNightNet, decisión entre cinco etapas | 0,6259 ± 0,0337 | 0,6840 | 0,4068 | 0,6981 | 0,4033 |
| LightGBM + cinco probabilidades | 0,7940 ± 0,0165 | **0,7970** | **0,5326** | **0,7806** | **0,4841** |
| LightGBM + cinco probabilidades y movimiento | **0,7941 ± 0,0165** | 0,7961 | 0,5278 | 0,7790 | 0,4773 |

En prueba W30, el híbrido con cinco probabilidades ganó **0,92 puntos de F1 macro** frente a LightGBM (IC 95 % de +0,35 a +1,54) y **1,70 puntos de F1 en N1** (+0,61 a +2,82). En EDF completo ganó **1,76 puntos de F1 macro** (+0,91 a +2,67) y **3,11 puntos en N1** (+1,31 a +5,86). La diferencia incremental de añadir movimiento fue de −0,09 puntos en W30 (−0,33 a +0,18) y −0,16 en EDF completo (−0,48 a +0,21). Los intervalos corresponden a 2.000 remuestreos pareados de personas completas y a los modelos ajustados en esta campaña. [Diferencias pareadas](../output/evidencia/resumen/fullnight-lgbm-w30/paired_intervals.csv)

## Protocolo

- **Desarrollo**. Se conservaron los 62 participantes, 122 noches, 152.052 épocas supervisadas y cinco pliegues del LightGBM histórico. La métrica primaria es la media no ponderada de F1 macro por pliegue. La dispersión usa el mismo divisor de cinco que la campaña original. Se comprobaron la identidad de los datos y la paridad de las predicciones de cada LightGBM histórico.
- **Prueba**. Se aplicó el archivo publicado `lgbm-49f74e944ce5457b` y el checkpoint directo `six_direct.pt` de la campaña `681a52f21d273bf190ae`. En los 16 participantes y 31 noches reservadas puntuamos 43.417 épocas W30 y 83.875 épocas al considerar el EDF completo. La cobertura de FullNightNet fue del 100 % sobre las épocas incluidas por LightGBM. La ventana W30 usa la primera y última época de sueño anotadas, con 30 minutos de vigilia a cada lado.
- **Entradas y decisiones**. Todos los modelos usan los dos EEG, el EOG y la envolvente EMG. Las cinco probabilidades de etapa se renormalizan por su suma antes de incorporarlas a las 435 columnas. La segunda variante añade la probabilidad original de movimiento como columna 441. La decisión nativa de FullNightNet también se puntuó sobre las cinco etapas y cada salida M se conservó como error. En W30, devolvió M en 233 épocas y obtuvo 0,6843 de F1 macro. [Resultados por etapa y decisión](../output/evidencia/resumen/fullnight-lgbm-w30/metrics.csv)
- **Prevención de fuga entre pacientes**. Para cada pliegue externo, cuatro predicciones internas por grupos de pacientes forman los datos de entrenamiento del híbrido. Diez redes internas cubren todas las parejas posibles de pliegues excluidos. Cada red desconoce tanto al paciente predicho como a todos los sujetos del pliegue externo. En el ajuste final, las entradas neuronales de desarrollo provienen de los cinco modelos externos. Los híbridos heredan en cada ajuste el preprocesamiento, pesos de clase, hiperparámetros, semilla y número de iteraciones del LightGBM correspondiente.
- **Uso del EDF completo**. En W30 histórico, el contexto y la normalización tabulares se calculan dentro del recorte. En la simulación de la aplicación, se calculan sobre toda la noche antes de consultar las etiquetas. FullNightNet recibe la señal completa en ambos casos. Como control, evaluar exactamente las mismas 43.417 épocas con contexto del EDF completo produjo 0,7573 con LightGBM y 0,7724 con el híbrido de cinco probabilidades. Las dos rutas de inferencia de la aplicación reprodujeron el 100 % de las decisiones de referencia en las 2.802 épocas comprobadas de SC4011. [Tabla de los cuatro protocolos de puntuación](../output/evidencia/resumen/fullnight-lgbm-w30/metrics.csv)

La mejora se observó en Sleep Cassette y con una semilla por ajuste. Los 16 participantes de prueba ya se habían utilizado para evaluar otros modelos del proyecto. Para un artículo, una cohorte independiente con canales compatibles permitiría comprobar la generalización del híbrido.

## Seguimiento y reproducción

La corrida se guardó en el mismo experimento de las corridas históricas de LightGBM y se replicó en el experimento histórico de FullNightNet. En cada almacén la corrida padre contiene las predicciones por época, los diez ajustes internos, los modelos históricos por pliegue, las recetas, los datos de procedencia, el código, las tablas y las matrices de confusión. Las corridas hijas contienen métricas por etapa y los modelos correspondientes. [Abrir corrida en MLflow local](http://127.0.0.1:5002/#/experiments/3/runs/bb7fa78abcfb41ab87c86ee89755ae87) · [Identificadores en ambos almacenes](../output/evidencia/resumen/fullnight-lgbm-w30/mlflow-runs.json)

Desde la raíz del proyecto, con las campañas históricas en `var/research`, el paquete publicado en `var/cloud/bundle` y el catálogo Sleep Cassette disponible, se reproduce mediante los siguientes comandos. Para usar otra ubicación de los EDF se pasa `--source-root` a ambos comandos de comparación.

```bash
.venv/bin/python scripts/stack_fullnight_folds.py --device auto
.venv/bin/python scripts/compare_fullnight_lgbm.py
.venv/bin/python scripts/compare_fullnight_edf.py
.venv/bin/python scripts/report_fullnight_lgbm.py
```
