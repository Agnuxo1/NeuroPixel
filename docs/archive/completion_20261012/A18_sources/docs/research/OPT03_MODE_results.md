# OPT03-MODE: sensibilidad a la dinámica de evaluación

Diagnóstico de desarrollo especificado después del resultado negativo deOPT03-D, congelado antes de sus nuevas evaluaciones y completado el8deoctubrede2026. Se utilizaron doce checkpoints existentes; **cero entrenamientos nuevos**, sin test ni modificaciones de pesos. El gate original95%probe/90%validation y H1 se mantienen intactos.

Se evaluaron los primeros128 ejemplos de cada rol de ambos datasets de desarrollo,512por panel. Cada checkpoint produjo una evaluación con updates completos y ocho con firing0,5, seeds78000–78007. Se conservan todas las máscaras y resultados. Las ocho máscaras no son ocho nuevas inicializaciones de entrenamiento; las diferencias se promedian por modelo y luego se contrastan en tres seeds100/101/102.

## Resultado

| Contraste sobre subconjunto de validación | Efecto medio en binding |
|---|---:|
| Evaluación0,5−1,0, modelos entrenados con0,5, promedio PAD y seeds | +11,100pp |
| Evaluación0,5−1,0, modelos entrenados con1,0, promedio PAD y seeds | −5,200pp |
| Diferencia de ambos efectos, interacción train-firing×eval-modo | +16,300pp |

La interacción por inicialización es+8,789pp,+25,073pp y+15,039pp. SD8,215pp, IC95t(df2)[−4,107;+36,708]pp. En el subconjunto de train-probe, la interacción media es+15,592pp, IC95[−3,678;+34,862]pp. La pantalla descriptiva preregistrada de≥5pp y tres signos positivos pasa en ambos paneles, pero los intervalos amplios incluyen cero; no se presenta como confirmación estadística general.

| PAD/firing entrenamiento | Binding validación con updates completos, seeds100/101/102 | Media de ocho máscaras0,5, seeds100/101/102 |
|---|---|---|
| Legado/0,5 | 17,969%/27,344%/23,047% | 20,361%/53,711%/31,982% |
| Cero/0,5 | 26,953%/27,734%/33,203% | 35,107%/39,844%/41,846% |
| Legado/1,0 | 14,844%/10,156%/17,188% | 8,398%/6,689%/13,672% |
| Cero/1,0 | 10,156%/23,047%/34,375% | 9,570%/14,844%/25,391% |

Se usa la media de exactitudes de rollouts individuales, **no una selección de máscaras ni un ensemble**. El resultado máximo de esa tabla53,711% sigue lejos del gate90%, y pertenece a un subconjunto/seed; no reemplaza el resultado negativo sobre validation4096 ni acredita binding relacional robusto.

## Auditoría y límites

La auditoría independiente recontó110592decisiones y NLL desde los arrays guardados y realizó1052checks sin incidencias. Verificó hashes derivados, seeds, dimensiones, balance por rol, targets/subconjuntos, finitud, métricas y diferencias, efectos pareados e intervalos con n=3. El runner verificó hashes de los checkpoints y estados antes/después; no creó optimizador, no calculó gradientes y no alteró las fuentes científicas. No existen BatchNorm/Dropout que confundan el flagtraining: su uso en la fuente está en la máscara de firing.

Dentro de este diagnóstico, cambiar la máscara de inferencia altera efectivamente las decisiones y la exactitud de estos modelos. La dirección observada coincide con la dinámica de entrenamiento en los doce modelos, pero no identifica una causa única del aprendizaje insuficiente. Firing también modifica dosis esperada por paso, propagación y ruido; el subconjunto de datos está históricamente expuesto. Con tres inicializaciones no se acredita precisión elevada ni generalización externa.

No se han resuelto competencia, controles de capacidad/supervisión o precisión poblacional de la tarea3 amplia. El siguiente diagnóstico deberá distinguir lectura de la consulta y asignación de roles frente a respuestas que solo aprovechan categorías o posiciones. Será otra fase de desarrollo con hipótesis/transformaciones congeladas antes de nuevos resultados, utilizando los pesos existentes y sin rescatar H1.

Artefactos: `docs/research/OPT03_mode_plan.json`, `results/research/OPT03_modes/` y `results/research/OPT03_modes_review/audit.json`. PlanSHA256198407ac903f34113d29b96d2197423270a462ce3d296aa04a7ebe2a6ba1f1f9. La fase de inferencia se completa con estos resultados condicionados; tarea3 y objetivo completo permanecen abiertos.
