# Guion de la sustentación

Tiempo: 15 minutos de exposición + 5 de preguntas. Presentación: [Sustentación: estadificación del sueño](https://claude.ai/artifact/3JPyQyBaMaDh37GDBS7W4j) (cada diapositiva trae notas de orador).

## Reparto de roles

| Bloque | Diapositivas | Presenta | Tiempo |
| --- | --- | --- | --- |
| Problema y objetivo | 1 Portada · 2 Problema · 3 Objetivo · 4 Datos | John Janer Buila | 3 min 30 s |
| Construcción de la solución | 5 Arquitectura · 6 Variables · 7 Modelos · 8 Resultados en desarrollo | Carlos Adrián Alarcón | 4 min |
| Resultados finales | 9 Prueba final · 10 Por estadio · 11 Hipnograma | John Janer Buila | 2 min 30 s |
| Modelo desplegado y demo | 12 Híbrido · 13 Despliegue · 14 Demo en vivo · 15 Limitaciones | Andrés Bautista | 4 min 15 s |
| Cierre | 16 Conclusiones | John Janer Buila | 30 s |
| **Total** | | | **14 min 45 s** |

Cada transición nombra a quien sigue («Le paso la palabra a…»). Ensayar al menos dos veces completas con cronómetro.

## Antes de la sustentación

- [ ] Pedir a Adrián que **encienda el laboratorio de AWS Academy** antes de la hora asignada. Con el laboratorio apagado, `{api}/health` responde 500 y la página muestra «El servicio aún no confirma la versión del modelo EDF».
- [ ] Procesar un EDF una vez antes de empezar: la primera inferencia tarda unos 48 s mientras carga el modelo; las siguientes, unos 2 s.
- [ ] Tener a mano un EDF de Sleep-EDF (por ejemplo, SC4001E0-PSG.edf) para la demo.
- [ ] Grabar un video corto de la demo como plan B si falla la red o la API.
- [ ] Compartir la presentación con el equipo desde el menú Compartir y descargarla en PowerPoint o PDF como respaldo.
- [ ] Repositorio público y enlace a la aplicación abiertos a la hora asignada por el curso.

## Preguntas probables del jurado

**Datos y preparación**

1. *¿Por qué separar por sujeto y no por época?* Las épocas de una misma persona se parecen mucho; mezclarlas entre entrenamiento y prueba infla las métricas. Separamos por sujeto antes de leer las señales: 62 de desarrollo y 16 de prueba.
2. *¿Por qué recortar la vigilia a ±30 minutos?* Los registros incluyen horas de vigilia antes y después de dormir (69 % de las épocas). Es el criterio habitual en la literatura sobre Sleep-EDF-78, lo que permite comparar. Además evaluamos el EDF completo, sin recorte: el híbrido logra 0,781 de F1 macro y 0,808 de kappa sin usar el hipnograma, y así funciona la aplicación.
3. *¿Cómo manejaron el desbalance?* Pesos de clase inversos a la frecuencia y F1 macro como métrica principal, que pondera igual los cinco estadios.
4. *¿Qué pasó con la clase de movimiento (M)?* Solo 128 épocas; se conserva como contexto pero no como objetivo. La tarea de seis clases queda como trabajo futuro.

**Modelos**

5. *¿Por qué LightGBM y no una red profunda?* Comparamos con las mismas particiones: la BiLSTM ajustada llegó a 0,780 y LightGBM a 0,789. La diferencia no es concluyente (IC del 95 % incluye cero), pero LightGBM es más liviano, interpretable y barato de servir en CPU.
6. *¿Cómo evitaron el sobreajuste al elegir hiperparámetros?* Validación anidada: dentro de cada fold externo, 20 % de los sujetos elige configuración e iteraciones; el fold externo se predice una sola vez. La prueba final (0,788) coincide con la validación (0,789).
7. *¿Las variables de contexto filtran información del futuro o de las etiquetas?* No usan etiquetas. Sí usan épocas posteriores de la misma noche: es un análisis retrospectivo de una noche completa, no en tiempo real, igual que la puntuación clínica.
8. *¿Qué variables pesan más?* El cociente sigma/beta de Fpz-Cz (husos de sueño), la envolvente del EMG (tono muscular, clave para REM) y descriptores del EOG con contexto.
9. *¿Por qué no siguieron con la CNN del avance?* Con 8 sujetos obtuvo 0,354 de F1 macro frente a 0,583 de Random Forest; se priorizaron redes recurrentes sobre características, que modelan la secuencia a menor costo.

**Resultados y aplicabilidad**

10. *¿Cómo se comparan con la literatura?* AttnSleep (2021) reporta 0,751 de F1 macro y 0,74 de kappa en Sleep-EDF-78 con un canal; nosotros 0,788 y 0,768 en prueba con cuatro señales. Es orientativo porque los protocolos difieren.
11. *¿Por qué N1 es tan difícil?* Es un estadio de transición; incluso los expertos coinciden solo 63 % en N1. El modelo lo confunde sobre todo con N2.
12. *¿Sirve para uso clínico?* Como apoyo: proponer un hipnograma inicial que el especialista revise. Ya procesa un EDF sin anotaciones, pero falta validar en pacientes y otros equipos (en Sleep Telemetry el desempeño baja).
13. *¿Cuánto cuesta y cómo escala?* Sin servidores encendidos: Lambda cobra por invocación y escala sola. Un EDF completo tarda unos 2 s con el servicio activo y 48 s en frío.
14. *¿Cómo se actualiza el modelo?* Reentrenamiento bajo demanda en EC2 con registro en MLflow; un candidato solo se promueve si reproduce las predicciones de referencia, y cada versión se puede revertir.
15. *¿Qué es el modelo híbrido y por qué se desplegó?* LightGBM recibe sus 435 variables y las 5 probabilidades de FullNightNet, una red convolucional que lee la señal completa. En prueba mejoró el F1 macro en 0,92 puntos con el recorte W30 (0,797) y en 1,76 puntos en el EDF completo (0,781), con la mayor ganancia en N1.
16. *¿No contaminaron el test al elegir el híbrido?* El resultado preestablecido es el de LightGBM (0,788), evaluado una sola vez. El híbrido se comparó después en las mismas noches, así que lo presentamos como mejora exploratoria que debe confirmarse en otra cohorte. Nuestra evaluación y la del equipo coinciden en todos los decimales (F1 0,7879), lo que confirma la reproducibilidad.
