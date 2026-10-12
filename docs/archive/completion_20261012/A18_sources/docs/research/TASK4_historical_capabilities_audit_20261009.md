# Auditoría de capacidades históricas

Se recuperaron **763 archivos originales, 401.580.342 bytes**, de quince commits fijados mediante el conector autenticado. Se verificaron commits/árboles Merkle, cada blob y todos los manifiestos originales. Los quince runs del proveedor constan como terminados con éxito. La auditoría local no cargó modelos, no entrenó ni generó predicciones nuevas.

El primer recuento pasó **984 comprobaciones**, incluidos 149 archivos NPZ y los contrastes principales de los puntos 9/10 y las trayectorias completas de 15B. Un segundo programa pasó **2.984 comprobaciones** de particiones, admisión de memoria, aislamiento de copias y 324 filas de reparación. Cero discrepancias en ambos. Son auditorías internas de evidencia guardada, no nuevas réplicas científicas independientes.

| Punto | Evidencia incorporable | Límite que se conserva |
|---|---|---|
| 7: particiones | Nueve particiones internamente disjuntas; unión TRAIN elegible 1.319/1.320 triples | No identifica minibatches efectivamente vistos ni crea holdout históricamente limpio. Los contratos son optativos. |
| 8: métricas/PAD | Correcciones y regresiones históricas documentadas, suite original 186 pases y un skip declarado | No mejora empírica ni resultado de competición; fuentes históricas corregidas y núcleo congelado actual son versiones distintas. No se reemplaza el núcleo DEV04. |
| 9: binding complejo | NP 9,628906%, Transformer 49,355469%; diferencia −39,726563 pp, IC [−41,254103; −38,199022], cinco pares | Ambas recetas insuficientemente competentes. Desarrollo/final documentados, población sintética expuesta. |
| 10: bAbI QA4 | NP 50,40%, Transformer 88,66%; diferencia −38,26 pp, IC [−52,385695; −24,134305], cinco pares | Tarea externa sintética, aprendida desde cero; no transferencia cero disparos ni lenguaje general. |
| 11: memoria | NP seleccionado 38/48, GRU 48/48; requisito 46/48; admisión falsa | Panel principal de cinco pares, contraste delay8 e interferencia aprendida **no ejecutados**. Ninguna cifra se inventa para ellos. |
| 12: aprendizaje continuo | Ocho copias nativas conservan exactamente tensores antiguos mientras cambia el experto nuevo; alias negativo detectado | Copiar pesos aumenta almacenamiento y no demuestra adquisición útil de ocho tareas ni routing competente. |
| 13: símbolos nuevos/distractores | Contratos y censo originales guardados; atajo de presencia resuelve panel positivo de palabra nueva | Nuevo token no equivale a nuevo significado aprendido; transformación puede romper disjunción original. |
| 14: reparación | Recuento categórico independiente de todas las 324 filas, con daño inmediato y fuentes held/removed/changed | Controles de pesos asignados separan reconstrucción asistida/retención/redundancia; no reparación autónoma aprendida. |
| 15: estabilidad | Cinco checkpoints, una escena expuesta, 256 pasos adicionales; crecimiento RMS 4,293–18,431; ganancia perturbación 1,557–6,059 | Finito en el horizonte observado; no convergencia, atracción, memoria útil, divergencia asintótica ni estabilidad universal. |

Para 9/10 los intervalos nominales describen realizaciones de entrenamiento sobre las mismas poblaciones, sin ajuste por multiplicidad. H1 original continúa sin apoyo. Los datos actuales DEV04/GLOB05 siguen siendo desarrollo interno y las 264 composiciones del test original permanecen excluidas.

La recuperación usa `D:/PROJECTS/NP-T4-20261009/<commit8>` para evitar el límite de longitud de rutas de Windows; se conservó la incidencia previa y los archivos parciales iniciales. Inventario completo: `results/research/TASK4_review/recovery_inventory.json`; auditorías: `saved_evidence_audit.json` y `control_recount.json`; referencias y estados del proveedor: `coord/recovery/TASK4_cloud_audit`. Los originales siguen separados de la continuación local y no se importan cierres como pruebas de capacidades.

Se cierra la auditoría acotada de la tarea 4. Continúa la tarea 5 con protocolos prospectivos para scanner causal y percepción visual; la contribución excepcional y la replicación por terceros continúan pendientes.
