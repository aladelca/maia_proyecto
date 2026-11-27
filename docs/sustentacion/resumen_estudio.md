# Resumen de estudio para la sustentación

Proyecto: *Del registro al hipnograma*. Clasificación automática de estadios del sueño con aprendizaje de máquina a partir de señales de polisomnografía.

Formato: 15 minutos de exposición y 5 de preguntas. Presentación diseñada de 22 diapositivas más 4 anexos (A1–A4).

## 1. Su parte: diapositivas 9 a 15 (modelo y resultados, 5:00–9:30)

| # | Diapositiva | Idea central | Cifras que debe decir |
| --- | --- | --- | --- |
| 9 | Qué probamos | Los mayores saltos vinieron de usar más canales y mirar las épocas vecinas | 1 canal → 4 canales: +8 a 9 puntos de F1. Contexto temporal: +4 puntos. LSTM/BiLSTM cerca pero por debajo. Mejor: LightGBM con 435 variables («LightGBM inicial»), F1 0,789 en CV |
| 10 | La solución | Híbrido: lo que mide el especialista + lo que aprende una red | Rama 1: 435 características. Rama 2: FullNightNet, 5 probabilidades. LightGBM combina 440 entradas. Clasificador 4,2 MiB + red 50 KiB, en CPU |
| 11 | Qué usa el modelo | Con 5 de 440 entradas, la red aporta el 22 % de la ganancia | Contexto 40 %, variables de la época 37 %, FullNightNet 22 %. Por canal: Fpz-Cz 34 %, EOG 22 %, Pz-Oz 16 %, EMG 5 %. Variable más usada: probabilidad de vigilia de la red. Medida descriptiva, no causal |
| 12 | Validación | Evaluamos sin fuga de información | Separación por persona; probabilidades fuera de muestra (10 redes internas); ajuste dentro de cada pliegue con 20 % de personas; IC 95 % con 2.000 remuestreos por persona |
| 13 | Resultado | En 16 personas nunca vistas, el modelo supera las metas | Registro completo: F1 macro 0,781, kappa 0,808, exactitud 90,3 % (inflada por la vigilia). W30: F1 0,797. Kappa entre expertos: 0,76 |
| 14 | Comparación | Juntos superan a cada uno por separado | Registro completo: red 0,698 · LightGBM 0,763 · híbrido 0,781. Mejora +1,76 puntos (IC 95 % +0,91 a +2,67, no incluye cero). W30: 0,684 · 0,788 · 0,797 |
| 15 | Por estadio | Mejora en los cinco estadios; N1 sigue siendo el reto | Híbrido (registro completo): W 0,97 · N1 0,48 · N2 0,84 · N3 0,79 · REM 0,81. 27,5 % de N1 se predice como N2. Acuerdo entre expertos en N1: 63 %. Transición: «Carlos les muestra cómo se ve esto en una noche real» |

**Corrección pendiente en las diapositivas 13 y 21:** dicen kappa **0,809**, pero el modelo desplegado (`stack_five`) tiene 0,80847, que redondea a **0,808**. El 0,809 corresponde a la variante con probabilidad de movimiento, que se descartó. El artículo y `output/evidencia/.../metrics.csv` dicen 0,808. Conviene unificarlo antes de la sustentación.

## 2. Las 12 cifras que hay que saber de memoria

| Dato | Valor |
| --- | --- |
| Datos | Sleep-EDF Expanded, Sleep Cassette: 153 noches, 78 personas sanas de 25 a 101 años |
| Partición | 62 personas de desarrollo (5 pliegues) y 16 de prueba (31 noches), separadas por persona antes de entrenar |
| Épocas | 30 s; unas 960 por noche. Desarrollo W30: 152.052. Prueba: 43.417 (W30) y 83.875 (registro completo) |
| Canales | EEG Fpz-Cz, EEG Pz-Oz, EOG horizontal (100 Hz) y envolvente de EMG submentoniano (1 Hz) |
| Variables | 145 por época (42 por EEG y EOG, 19 del EMG) + 290 de contexto = 435; +5 de la red = 440 |
| Línea base | Random Forest con 1 canal: F1 macro 0,654 (CV) |
| LightGBM inicial | CV 0,789 · prueba W30 F1 0,788, kappa 0,768 (evaluación única) · registro completo 0,763 |
| Híbrido (desplegado) | Prueba W30 F1 0,797, kappa 0,778 · registro completo F1 0,781, kappa 0,808, exactitud 90,3 % |
| Mejora del híbrido | +1,76 puntos en registro completo (IC 95 % +0,91 a +2,67); +0,92 en W30; +3,1 puntos en N1 |
| N1 | F1 0,48 en registro completo; expertos coinciden solo 63 % |
| Referencia humana | 82,6 % de acuerdo entre más de 2.500 evaluadores; kappa entre expertos 0,76 |
| Tiempos | Ejemplo público 6,9 s; noche completa SC4011 (49 MiB) 21 s de punta a punta (27-nov-2026). Meta: < 5 min |

## 3. El proyecto completo en diez pasos

1. **Problema:** puntuar una noche a mano exige unas 960 decisiones y horas de trabajo, y los expertos no siempre coinciden (82,6 %, y 63 % en N1).
2. **Objetivo:** clasificar cada época en W, N1, N2, N3 o REM y entregar el hipnograma en una aplicación web. Metas: F1 ≥ 0,70, kappa ≥ 0,70, superar la línea base y procesar una noche en menos de 5 minutos.
3. **Datos:** conjunto público estándar, verificado con SHA-256. Todas las noches de una persona quedan del mismo lado de la partición.
4. **Métrica:** F1 macro, porque la vigilia es el 69 % de la noche y un modelo que respondiera siempre «despierto» acertaría el 69 %. Se reportan dos protocolos: W30 (30 minutos de vigilia alrededor del sueño anotado, como la literatura) y registro completo (lo que procesa la aplicación).
5. **Señales:** filtro Butterworth de 4.º orden en fase cero (EEG 0,3–35 Hz, EOG 0,3–15 Hz). Estadios 3 y 4 se unen en N3.
6. **Variables:** 22 espectrales (Welch, 5 bandas, cocientes δ/β, θ/α, σ/β, δ/θ), 11 de amplitud, 7 de dinámica (Hjorth, entropía de permutación, Petrosian), 2 de calidad. Contexto: media de 4 épocas previas y media triangular centrada de 15. Normalización por noche con mediana y percentiles 5–95.
7. **Modelos:** Dummy, Random Forest, CatBoost, LightGBM, LSTM/BiLSTM y FullNightNet. El mejor individual fue LightGBM; el híbrido LightGBM + FullNightNet fue el mejor en total.
8. **Validación:** pliegues por persona, ajuste anidado, probabilidades fuera de muestra para el híbrido y bootstrap pareado por persona.
9. **Despliegue:** web en Vercel; API Gateway + Lambda (Flask); Step Functions; worker Lambda de 3 GB en CPU con el modelo en la imagen; S3 con enlaces firmados que expiran a los 7 días; CloudWatch; todo con Terraform. CI/CD con GitHub Actions: pruebas, CodeQL, preview de Vercel, despliegue con prueba de paridad (tolerancia 10⁻⁶) y reversión automática.
10. **Limitaciones:** falta una cohorte independiente (los 16 de prueba ya intervinieron en comparaciones), N1 débil, un solo montaje de cuatro canales, sin validación clínica ni calibración, una semilla por ajuste. La herramienta propone un hipnograma para revisión; no diagnostica.

## 4. Glosario rápido

- **F1 macro:** promedio del F1 de los cinco estadios, con el mismo peso para cada uno.
- **Kappa de Cohen:** acuerdo corregido por azar. 0 = azar; 1 = acuerdo perfecto. Entre expertos ronda 0,76.
- **W30:** recorte que conserva el sueño anotado más 30 minutos de vigilia a cada lado. Depende de las anotaciones.
- **Registro completo:** se predicen todas las épocas del EDF; las anotaciones solo se usan para calificar.
- **Fuga de información:** que el modelo vea, al entrenar, datos de la persona con la que se evalúa.
- **OOF (fuera de muestra):** predicciones hechas por un modelo que no vio a esa persona.
- **Bootstrap pareado por persona:** remuestrear personas completas 2.000 veces y comparar los dos modelos sobre las mismas personas para obtener el intervalo de confianza de la diferencia.
- **LightGBM:** gradient boosting de árboles; rápido, liviano y bueno con datos tabulares desbalanceados.
- **FullNightNet:** red convolucional propia. Remuestrea EEG y EOG a 50 Hz, usa una CNN 1D compartida por canal y dos capas densas para el EMG. Tres bloques temporales con dilataciones 1, 2 y 4 cubren 15 épocas. Sale con 6 clases (incluye movimiento) y se renormalizan las 5 de estadio.
- **Ganancia (importancia):** cuánto reduce el error cada división de árbol que usa una variable. Es descriptiva y se reparte entre variables correlacionadas.

## 5. Preguntas que le corresponden (modelos y evaluación)

1. **¿Por qué F1 macro y no exactitud?** Por el desbalance: la vigilia es el 69 % de la noche. La exactitud premia acertar la clase mayoritaria; el F1 macro pesa igual N1 y N3. También reportamos kappa y F1 por estadio.
2. **¿Por qué un híbrido y no solo aprendizaje profundo?** Lo medimos: en el registro completo, la red sola obtuvo 0,698, LightGBM 0,763 y el híbrido 0,781. Las dos fuentes se complementan, y el híbrido corre en CPU con menos de 5 MiB.
3. **¿Cómo evitaron la fuga al combinar modelos?** Las probabilidades de entrenamiento salen de redes que no vieron al participante: 10 redes internas, una por cada par de pliegues excluidos. Hiperparámetros con un 20 % de personas dentro de cada pliegue; preprocesamiento ajustado solo con entrenamiento.
4. **¿La mejora de 1,76 puntos es real o es ruido?** El IC 95 % (+0,91 a +2,67) no incluye cero; 2.000 remuestreos pareados de personas completas. Salvedad: una sola semilla por ajuste.
5. **¿Por qué falla en N1?** Es una transición breve (5 % de la noche) que se parece a la vigilia y a N2: el 27,5 % se predice como N2. Los expertos también coinciden solo en el 63 % de N1.
6. **¿Cómo se comparan con el estado del arte?** En W30, F1 0,797; AttnSleep (2021) reporta 0,781 en Sleep-EDF-78. Pero usamos cuatro canales y otro protocolo, así que la comparación es orientativa (anexo A3). La comparación rigurosa es la interna, con las mismas épocas.
7. **¿Los 16 de prueba no estaban «contaminados»?** Nunca se usaron para entrenar, pero sí intervinieron en comparaciones anteriores. Por eso lo declaramos como limitación y proponemos una cohorte independiente con protocolo definido de antemano. El LightGBM inicial sí se evaluó una sola vez con la configuración congelada.
8. **¿Cómo saben que su evaluación es confiable?** Hicimos dos evaluaciones independientes del LightGBM inicial sobre la prueba (una con el modelo reentrenado localmente y otra con el archivo publicado) y coincidieron en todos los decimales: F1 0,7879.
9. **¿Qué variables pesan más?** La probabilidad de vigilia de FullNightNet, el cociente δ/β de Fpz-Cz y el contexto del EOG. Por canal: Fpz-Cz 34 %, EOG 22 %, Pz-Oz 16 %, EMG 5 %.
10. **¿Por qué no siguieron con la CNN del avance?** Con 8 sujetos obtuvo 0,354 de F1 macro frente a 0,583 de Random Forest. Se exploraron después redes recurrentes y FullNightNet, que modela la noche completa.

Respuesta tipo: el dato, la razón y, si aplica, la limitación. Corta y sin leer.

## 6. Lista de verificación del día

- [ ] Ensayar dos veces con cronómetro (tecla N de la presentación). Meta: terminar en 14:30.
- [ ] **45 minutos antes:** Adrián inicia el laboratorio de AWS Academy. Comprobar que `https://9u8vgivsce.execute-api.us-west-2.amazonaws.com/health/ready` responda `"status":"ready"`. Hoy (27-nov) responde 500 porque el laboratorio está apagado.
- [ ] Calentar el servicio: ejecutar una vez el ejemplo público y la noche SC4011; dejar la pestaña con el resultado abierta.
- [ ] Tener a mano `SC4011E0-PSG.edf` (carpeta `Datos/` del repositorio).
- [ ] Abrir la presentación y la aplicación en pestañas separadas; probar proyector y pantalla completa.
- [ ] Plan B: modo demo de la aplicación y anexo A1. Si algo falla, no depurar en vivo.
