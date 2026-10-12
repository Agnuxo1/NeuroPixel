# GLOB05: acceso global activo frente a selección espacial por consulta

Cohorte completa: doce cabezas globales nuevas sobre los doce cuerpos DEV04 congelados. Atención menos agregación global uniforme condicionada: **+6.278 pp**, ICt95 exploratorio **[4.108;8.449] pp**, semianchura **2.171 pp**.

Control global activo menos control local original (descriptivo): **+9.966 pp**, IC **[5.868;14.063] pp**.

El contraste nuevo tiene límite inferior positivo y semianchura ≤5 pp. Esto conserva el fallo de precisión del contraste DEV04 atención–local: semianchura 5,658>5 pp. Son preguntas distintas; no se combinan sus intervalos ni se declara reparado el original.

## Los doce pares

| Política | Inicialización | Atención % | Global activo % | Local % | Diferencia atención–global pp | Global gate |
|---|---:|---:|---:|---:|---:|---|
| 101 | 400 | 99.780 | 93.768 | 75.256 | +6.012 | Pasa |
| 101 | 401 | 99.792 | 94.812 | 81.488 | +4.980 | Pasa |
| 101 | 402 | 99.811 | 93.658 | 89.911 | +6.152 | No pasa |
| 101 | 403 | 99.762 | 94.458 | 88.538 | +5.304 | No pasa |
| 102 | 404 | 99.835 | 92.413 | 77.313 | +7.422 | No pasa |
| 102 | 405 | 99.884 | 91.235 | 70.117 | +8.649 | No pasa |
| 102 | 406 | 99.707 | 91.882 | 88.458 | +7.825 | No pasa |
| 102 | 407 | 99.683 | 94.507 | 88.666 | +5.176 | No pasa |
| 103 | 408 | 99.518 | 86.072 | 67.261 | +13.446 | No pasa |
| 103 | 409 | 99.231 | 96.088 | 92.139 | +3.143 | Pasa |
| 103 | 410 | 99.579 | 93.506 | 89.795 | +6.073 | No pasa |
| 103 | 411 | 99.927 | 98.767 | 92.639 | +1.160 | Pasa |

## Controles y límites

Las cabezas comparten dimensiones Q/K/V/salida, 3.168 parámetros nominales activos, tensores iniciales, cuerpos/diccionarios/decoders, datos, labels, batches, masks y 8.192 updates. La cabeza nueva agrega uniformemente los ocho hechos y la consulta modula canales mediante 1+tanh(q⊙k̄/√4); sus Q/K reciben gradiente. Atención elige pesos distintos por celda y consulta. Los espacios funcionales y operaciones difieren: no se afirma igualdad de capacidad funcional o FLOPs.

La selección espacial condicionada mantiene una ventaja en esta comparación interna frente al acceso global con consulta activa. El control global también mejora frente al local. Estas diferencias no son una mediación causal identificada ni prueban una explicación única: agregación, gating y funciones de lectura difieren. El control global no pasa el gate conjunto en todos los doce cuerpos; se publican los fallos.

El estimador usa diferencias pareadas por cuerpo, cuatro inicializaciones promediadas dentro de cada política y tres medias con igual peso, ICt95 df2 exploratorio, no ajustado y sin truncar. Las políticas comparten un universo de desarrollo expuesto; máscaras, ejemplos, cabezas y etapas no son réplicas independientes. No se abrieron las 264 composiciones del test original ni se reinterpretó H1, que permanece no soportada bajo su receta.

## Reproducibilidad

Plan SHA-256 b26f17cec8a98fda6dbd5aa128d67501fd2273bf07718300c1c865391985e0b2, 58 gobernantes congelados; los 43 DEV04 permanecen intactos. Fuente científica d223ea030328648595f9404d9ba710cae05ae43f; wrapper operativo fuera del freeze e4b82b280ece658ef14ea6d7e215e249edb06af3, run37887762153: los doce jobs finalizaron con éxito. El wrapper se creó tras comprobar que create-ref no había lanzado ningún trabajo; cero ajustes cerrados repetidos.

ZIP y manifiestos originales, anclas Git/Merkle, datasets iguales a los padres, checkpoints, AdamW, tres streams RNG y witnesses completos están conservados. El replay independiente verificó 3.538.944 decisiones exactas, 3.120 comprobaciones, error máximo de NLL media0 y cero incidencias. El recuento exige todos los bytes originales; las tolerancias permanecen iguales.

Recuento estadístico ortogonal con denominadores enteros: 52 comprobaciones, cero discrepancias. Las figuras/tablas y el análisis se vinculan a inputs por SHA-256.

La continuación añadió solamente doce cabezas nuevas. No reinició cuerpos, no repitió las veinticuatro lecturas anteriores ni añadió semillas para estrechar el intervalo. Es control interno prospectivo de desarrollo; no replicación externa, nuevo test final, medición física de energía ni demostración de contribución excepcional. El cierre de la tarea 3 amplia requiere una evaluación separada; este informe no cierra todo el programa.
