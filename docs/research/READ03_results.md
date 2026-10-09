# READ03: diagnostico causal de lectura congelada

Cohorte completa: 36 ajustes nuevos de cabeza sobre doce estados U16 originales. Primario atencion menos control local: **54.225 pp**, ICt95 exploratorio **[37.575; 70.875] pp**. La atencion supera al control local en este contraste exploratorio condicional.

Este resultado se refiere a tres inicializaciones del cuerpo por condicion, un split de desarrollo expuesto y la receta exacta READ03. Los cuerpos, diccionarios y decoder permanecieron congelados. No modifica H1 original, no prueba una causa unica, originalidad, generalizacion ni utilidad externa.

## Contrastes registrados

| Contraste | Semilla 200 | Semilla 201 | Semilla 202 | Media pp | ICt95 pp |
|---|---:|---:|---:|---:|---|
| query_attention_minus_local_query | 48.218 | 53.003 | 61.455 | 54.225 | [37.575; 70.875] |
| query_attention_minus_uniform_global | 49.739 | 52.831 | 62.640 | 55.070 | [38.337; 71.803] |
| query_attention_minus_parent | 53.792 | 56.201 | 65.399 | 58.464 | [43.247; 73.681] |
| local_query_minus_parent | 5.574 | 3.198 | 3.944 | 4.239 | [1.221; 7.257] |
| uniform_global_minus_parent | 4.053 | 3.371 | 2.759 | 3.394 | [1.786; 5.002] |

Promedio de las cuatro condiciones originales dentro de cada inicializacion; luego contraste pareado entre tres inicializaciones. Intervalos sin ajuste por multiplicidad y sin truncar. Cabezas, mascaras, ejemplos y etapas no se cuentan como replicas independientes. Comparar una cabeza con el padre combina capacidad de lectura y etiquetas adicionales; no es un efecto puro de atencion.

## Competencia conjunta y gates

| Condicion original | Lectura | Validacion 200/201/202 (%) | Probe 200/201/202 (%) | Gate en tres semillas |
|---|---|---|---|---|
| Cuatro consultas / escuela 0 | Atencion por consulta | 99.658 / 92.065 / 91.730 | 99.426 / 93.335 / 92.981 | No pasa |
| Cuatro consultas / escuela 0 | Agregacion uniforme | 50.122 / 4.114 / 3.748 | 52.979 / 5.627 / 5.444 | No pasa |
| Cuatro consultas / escuela 0 | Control local con consulta | 57.745 / 8.984 / 9.552 | 62.952 / 16.162 / 13.171 | No pasa |
| Cuatro consultas / escuela 0,3 | Atencion por consulta | 99.762 / 99.835 / 99.670 | 99.695 / 99.841 / 99.744 | Pasa |
| Cuatro consultas / escuela 0,3 | Agregacion uniforme | 95.380 / 92.279 / 94.202 | 95.776 / 91.785 / 94.580 | No pasa |
| Cuatro consultas / escuela 0,3 | Control local con consulta | 91.315 / 84.558 / 89.056 | 91.675 / 85.339 / 89.331 | No pasa |
| Consulta repetida / escuela 0 | Atencion por consulta | 90.289 / 32.880 / 79.675 | 89.331 / 37.781 / 83.069 | No pasa |
| Consulta repetida / escuela 0 | Agregacion uniforme | 3.473 / 1.312 / 1.752 | 3.735 / 1.440 / 2.295 | No pasa |
| Consulta repetida / escuela 0 | Control local con consulta | 5.273 / 0.500 / 3.650 | 8.447 / 0.769 / 6.421 | No pasa |
| Consulta repetida / escuela 0,3 | Atencion por consulta | 71.112 / 87.097 / 82.477 | 72.546 / 88.123 / 86.523 | No pasa |
| Consulta repetida / escuela 0,3 | Agregacion uniforme | 12.891 / 2.850 / 3.290 | 14.160 / 3.784 / 4.382 | No pasa |
| Consulta repetida / escuela 0,3 | Control local con consulta | 13.617 / 5.823 / 5.475 | 17.432 / 9.131 / 7.886 | No pasa |

El gate registrado requiere probe conjunto >=95% y validacion >=90% en las tres inicializaciones. Su cumplimiento no certifica rendimiento en test nuevo; no se ha abierto el test final ni se declara un ganador confirmatorio.

## Controles y limites

Las tres cabezas tienen 3.168 parametros nominales. La uniforme tiene 1.856 parametros con gradiente de respuesta; sus Q/K no eligen pesos. El control local tiene acceso directo a consulta y parametros activos, pero distinto espacio funcional y aritmetica. No se equipara conteo nominal con capacidad efectiva. La igualdad o ausencia de significacion frente a uniforme no demuestra equivalencia: no se registro un margen de equivalencia.

La nueva cabeza lee estados e identidades visibles, consulta (7,6), excluye PAD/consulta como claves y conserva salida (7,7). Ninguna recibe las asignaciones del generador como caracteristicas. El replay reproduce endpoints 1.024/4.096/8.192 y recontabiliza predicciones/metricas; sigue siendo verificacion interna del proyecto, no replicacion cientifica externa.

Los fallos iniciales de empaquetado del verificador y de rutas Windows quedaron conservados y corregidos sin modificar los 32 archivos del experimento. No se repitieron los entrenamientos de los padres ni se amplio la tolerancia de NLL (1e-4); las decisiones se exigen exactas.

El coste y la energia completos no estan establecidos. El tiempo de actualizaciones excluye otras operaciones; no es un presupuesto total ni una medida fisica de energia. La competencia de una lectura congelada no demuestra convergencia del entrenamiento conjunto ni resuelve la precision entre particiones independientes.

## Evidencia reproducible

Plan SHA256: `d273b8284d9a387b49266dc4ec62cfad76aefaa11c5f82c7ab6a97ba0e431ea0`. Los 32 hashes y los doce checkpoints originales se conservan. Recibos de ambos ejecutores, ZIP originales, anclas Git y proofs por caso se encuentran en `results/research/READ03_review/37831051538/` y `READ03_recovery/37831051538/`.

Generacion local sin entrenamiento: `collect_READ03_replay_receipts.py`, `audit_READ03_archives.py`, `collect_READ03_final_receipts.py`, `summarize_READ03.py` y `render_READ03_report.py`, en ese orden despues de finalizar ambos ejecutores. El analisis rechaza cohortes incompletas y JSON diferentes de sus ZIP originales.

Recuento estadistico independiente del informe: 27 comprobaciones, cero discrepancias. Tablas, figuras PNG/SVG y manifiestos SHA-256 estan en la carpeta `analysis/`.

La receta exacta puede cerrarse dentro de este alcance. La tarea 3 amplia permanece abierta hasta resolver la competencia, optimizacion y precision reproducibles requeridas; las tareas 4-8 no se dan por verificadas.
