# OPT03-D: resultado de desarrollo y auditoría

La fase de desarrollo terminó y se auditó el 8 de octubre de2026. **Ninguna condición pasó el gate congelado.** No se selecciona un ganador ni se abre test final. La tarea3 amplia no está completada: competencia suficiente, atribución con controles de capacidad/supervisión y precisión poblacional útil siguen pendientes. H1 permanece no soportada.

## Ejecución y procedencia

Se ejecutaron doce entrenamientos nuevos: seeds100/101/102 × PAD efectivo cero sí/no × firing0,5/1,0,8192updates, batch64,T16, tying/reinjection activos y escuela0,3. [Run37696760073](https://github.com/Agnuxo1/NeuroPixel/actions/runs/37696760073), job113050279929, source59c0dbdc9107176f4f6759172c4b7de60d9f6a3e: success. Ningún entrenamiento cerrado se repitió. El archivo Git original quedó en ab97203080b861535ff2236204372b73a5affbfb. ZIP4.534.582bytes, SHA256ce0ab91443fc36575d9c7b71f835a4b31dae3df59f4df73f7de10887d42fbcff.

La RAM mínima observada durante la cohorte fue14,2327GiB. Los tiempos registrados de entrenamiento suman14.763,44segundos; no son una medida de coste computacional completo ni de energía física. Todos los brazos compartieron el entorno medido del mismo runner CPU.

### Incidencia de hash del plan

El plan local original tiene hashbf54be3c4c8e30885b330e67e92195622e28c85a49e317358d106699d40008df y141 finales de línea CRLF. El archivo del commit ejecutado tiene hashf1caada1af01d0078877b55800e3e7928e7bc8832993b8c189e0ffae2cdadb67 y finales LF. **Los bytes coinciden exactamente después de la única conversión CRLF→LF; los objetos JSON son iguales.** Los doce resultados registran el hash ejecutado, y las nueve fuentes científicas congeladas coinciden. La conversión se produjo al transmitir texto al commit de ejecución, anterior a los entrenamientos.

Se conservan ambos planes y los fallos iniciales del comparador literal (`analysis.json`, `metadata_audit.json`). No se reescribieron resultados ni criterios. Una auditoría complementaria usa exclusivamente el archivo exacto del commit, tras comprobar las dos identidades conocidas y la equivalencia de presentación. No admite un reemplazo arbitrario del plan. Ver `plan_byte_provenance_check.json` y `scripts/audit_OPT03_executed_plan.py`.

## Resultado cuantitativo

Binding es la media de exactitudes de AGENTE/PACIENTE. La unidad de replicación es la inicialización,n=3; cada resultado por máscara, endpoint o ejemplo no aumenta n. Valores en porcentajes.

| PAD efectivo | Firing train | Binding validación100/101/102 | Media | SD | IC95t,df2 | Gate95%probe/90%validation en tres seeds |
|---|---:|---|---:|---:|---|---|
| Legado | 0,5 | 21,777 /26,709 /22,217 | 23,568 | 2,729 | [16,788;30,348] | No |
| Legado | 1,0 | 17,480 /11,768 /20,947 | 16,732 | 4,635 | [5,217;28,247] | No |
| Cero | 0,5 | 26,855 /26,172 /32,959 | 28,662 | 3,737 | [19,379;37,945] | No |
| Cero | 1,0 | 12,354 /25,244 /41,064 | 26,221 | 14,380 | [−9,502;61,943] | No |

Los binding medios de train-probe son23,763%,16,829%,30,013% y23,079%, respectivamente. Tampoco se aproximan al umbral95%. Los intervalos no se recortan artificialmente a los límites de la proporción. Son intervalos de desarrollo, no confirmación sobre test y no prueba de equivalencia.

| Contraste factorial, puntos porcentuales | Media | IC95pareado,df2 |
|---|---:|---|
| PAD cero−legado, promediando firing | +7,292 | [−11,985;+26,568] |
| Firing1,0−0,5, promediando PAD | −4,639 | [−22,066;+12,789] |
| Interacción PAD×firing | +4,395 | [−27,538;+36,327] |

Todas estas bandas incluyen cero; la incertidumbre sigue siendo grande. El brazo PADcero/firing1 tiene una SD14,38pp entre tres inicializaciones. Una estimación de variabilidad con n=3 no justifica presentar una precisión garantizada para una población de entrenamiento ni una ventaja general.

## Verificación independiente

El archivo original y su recibo Git se recuperaron por commit y SHA256, con12resultados declarados. La auditoría complementaria realizó860checks sin incidencias: hashes e inventario, versión exacta del plan ejecutado, regeneración de ambos datasets, aritmética por rol/endpoint, ausencia de test declarado, ventanas y pérdidas, recursos, checkpoint/optimización8192steps, LR/decay/betas/eps, finitud y29824parámetros nominales por modelo. Re-evaluó los doce checkpoints finales sobre train-probe2048 y validation4096, y recontó sus métricas de aciertos y NLL. Los derivados conservan73.728 predicciones de replay.

El runner no archivó decisiones originales por ejemplo ni pesos anteriores de los endpoints1024/4096. Por ello el recuento prueba concordancia de métricas finales desde los checkpoints finales; **no se afirma equivalencia por ejemplo con predicciones originales ausentes** ni replay de pesos intermedios no preservados. Esa limitación se mantiene junto a los resultados. Esta auditoría no es una replicación externa independiente.

## Interpretación y siguiente acción

Proyectar PAD no resolvió la baja competencia bajo esta receta; subir firing tampoco la resolvió. Los resultados no prueban incapacidad arquitectónica general ni identifican un único cuello de botella. PAD modifica lookup, decoder compartido y lens, y elimina16grados efectivos aunque preserve el conteo nominal. Firing modifica aleatoriedad y dosis esperada, y puede modificar el ajuste entre dinámica de entrenamiento y evaluación. La supervisión escolar permanece igual en los cuatro brazos y no se ha aislado a competencia alta.

Las curvas finales muestran pérdidas escolares pequeñas en muchos brazos, pero binding bajo en los probes; esto no demuestra que el objetivo de respuesta esté resuelto ni que la escuela sea suficiente. Los gradientes sin recortar siguen siendo grandes en varios casos, incluyendo211,93 en el último update de PADcero/firing1/seed101. No se deduce de ese único número una causa; todos los gradientes usados fueron controlados por el clip del protocolo.

El siguiente paso de tarea3 será un diagnóstico nuevo, prospectivamente especificado, sobre estos checkpoints de desarrollo ya existentes: separar sensibilidad a dinámica de evaluación y otros controles de optimización antes de invertir en una cohorte confirmatoria de mayor precisión. No se retoca el gate95%/90%, no se elige una configuración por score de test, no se repiten estos doce entrenamientos y no se rescata H1. La hipótesis de una discrepancia train/eval requiere un experimento; no se presenta como hallazgo demostrado.

Artefactos: `results/research/OPT03_cloud_recovery/37696760073/` y `results/research/OPT03_root_review/37696760073/`. El primer fallo literal y su explicación se conservan junto con `analysis_executed_plan.json` y `checkpoint_audit_executed_plan.json`. La faseOPT03-D queda completada como resultado negativo de desarrollo; el programa completo y tarea3 permanecen abiertos.
