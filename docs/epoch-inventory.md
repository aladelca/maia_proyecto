# Inventario de épocas de Sleep Cassette

Generado con `scripts/epoch_inventory.py` a partir del índice de épocas del preprocesamiento completo con recorte de vigilia de 30 minutos (`wake_margin_minutes: 30`) y la partición `configs/splits/sc_subjects_v1.csv`.

## Tabla A. Flujo de inclusión y exclusión de épocas

| Concepto | Desarrollo | Prueba reservada | Total |
| --- | ---: | ---: | ---: |
| Sujetos | 62 | 16 | 78 |
| Registros (noches) | 122 | 31 | 153 |
| Épocas de 30 s en los hipnogramas | 332.534 | 83.890 | 416.424 |
| Excluidas: sin puntuar (?) | 1.333 | 2 | 1.335 |
| Excluidas: recorte de vigilia (±30 min) | 179.034 | 40.458 | 219.492 |
| Excluidas como objetivo: movimiento (M), conservadas como contexto | 115 | 13 | 128 |
| Épocas elegibles, tarea de 5 clases | 152.052 | 43.417 | 195.469 |
| de ellas, con algún canal marcado como plano | 95 | 70 | 165 |

Las épocas de movimiento no son objetivo en la tarea de cinco clases, pero se conservan en la secuencia para calcular las variables de contexto temporal. La marca de canal plano es informativa: no excluye épocas.

## Tabla B. Distribución por clase antes y después del recorte de vigilia

| Estadio | Desarrollo: noche completa | Desarrollo: analizadas | Desarrollo: % analizadas |
| --- | ---: | ---: | ---: |
| W | 227.811 | 48.777 | 32,08 % |
| N1 | 16.603 | 16.603 | 10,92 % |
| N2 | 54.955 | 54.955 | 36,14 % |
| N3 | 10.656 | 10.656 | 7,01 % |
| REM | 21.061 | 21.061 | 13,85 % |
| Total | 331.086 | 152.052 | 100,00 % |

## Tabla C. Épocas analizadas por fold de desarrollo

| Fold | Sujetos | Registros | W | N1 | N2 | N3 | REM | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fold 0 | 13 | 25 | 9.400 | 2.854 | 11.193 | 1.892 | 4.449 | 29.788 |
| Fold 1 | 13 | 26 | 10.822 | 3.511 | 12.015 | 2.325 | 4.477 | 33.150 |
| Fold 2 | 12 | 24 | 12.712 | 2.919 | 11.522 | 1.873 | 4.457 | 33.483 |
| Fold 3 | 12 | 23 | 8.852 | 3.847 | 9.309 | 2.779 | 3.566 | 28.353 |
| Fold 4 | 12 | 24 | 6.991 | 3.472 | 10.916 | 1.787 | 4.112 | 27.278 |
| Total | 62 | 122 | 48.777 | 16.603 | 54.955 | 10.656 | 21.061 | 152.052 |
