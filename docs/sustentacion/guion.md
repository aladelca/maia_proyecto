# Guion de la sustentación

Tiempo: 15 minutos de exposición + 5 de preguntas. Presentación: [Sustentación: estadificación del sueño](https://claude.ai/artifact/3JPyQyBaMaDh37GDBS7W4j) (cada diapositiva trae notas de orador).

## Reparto de roles

| Bloque | Diapositivas | Presenta | Tiempo |
| --- | --- | --- | --- |
| Problema y objetivo | 1 Portada · 2 Problema · 3 Objetivo · 4 Datos | John Janer Buila | 3 min 30 s |
| Construcción de la solución | 5 Arquitectura · 6 Variables · 7 Modelos · 8 Resultados en desarrollo | Carlos Adrián Alarcón | 4 min |
| Resultados finales | 9 Prueba final · 10 Por estadio | John Janer Buila | 2 min 15 s |
| Despliegue y demo | 11 Despliegue · 12 Demo en vivo · 13 Limitaciones | Andrés Bautista | 4 min 15 s |
| Cierre | 14 Conclusiones | John Janer Buila | 30 s |
| **Total** | | | **14 min 30 s** |

Cada transición nombra a quien sigue («Le paso la palabra a…»). Ensayar al menos dos veces completas con cronómetro.

## Antes de la sustentación

- [ ] Verificar que la API responda: «Probar ejemplo público» debe terminar sin error (el 19/11 respondía HTTP 500).
- [ ] Abrir la aplicación con anticipación para que el servicio esté activo.
- [ ] Grabar un video corto de la demo como plan B si falla la red o la API.
- [ ] Compartir la presentación con el equipo desde el menú Compartir y descargarla en PowerPoint o PDF como respaldo.
- [ ] Repositorio público y enlace a la aplicación abiertos a la hora asignada por el curso.

## Preguntas probables del jurado

**Datos y preparación**

1. *¿Por qué separar por sujeto y no por época?* Las épocas de una misma persona se parecen mucho; mezclarlas entre entrenamiento y prueba infla las métricas. Separamos por sujeto antes de leer las señales: 62 de desarrollo y 16 de prueba.
2. *¿Por qué recortar la vigilia a ±30 minutos?* Los registros incluyen horas de vigilia antes y después de dormir (69 % de las épocas). Es el criterio habitual en la literatura sobre Sleep-EDF-78, lo que permite comparar. Limitación: usa el hipnograma, así que hoy la app procesa noches preparadas.
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
12. *¿Sirve para uso clínico?* Como apoyo: proponer un hipnograma inicial que el especialista revise. Falta validar en pacientes, otros equipos y registros sin anotaciones.
13. *¿Cuánto cuesta y cómo escala?* Sin servidores encendidos: Lambda cobra por invocación y escala sola; una noche toma 1,83 s y cinco simultáneas terminan en 6,5 s (p95).
14. *¿Cómo se actualiza el modelo?* Reentrenamiento bajo demanda en EC2 con registro en MLflow; un candidato solo se promueve si reproduce las predicciones de referencia, y cada versión se puede revertir.
