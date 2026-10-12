# VIS07: visión externa, cinco parejas y límites temporales

NeuroPixel obtuvo **94.357%** frente a **90.840%** del CNN, diferencia **+3.517pp**, ICt95 exploratorio **[1.930; 5.103]pp**. Las cinco realizaciones NCA superan90%.

| Semilla | NCA correctas/1797 | CNN correctas/1797 | Diferencia pp |
|---|---:|---:|---:|
| 240 | 1715/1797 (95.437%) | 1614/1797 (89.816%) | +5.620 |
| 241 | 1688/1797 (93.934%) | 1643/1797 (91.430%) | +2.504 |
| 242 | 1690/1797 (94.046%) | 1623/1797 (90.317%) | +3.728 |
| 243 | 1688/1797 (93.934%) | 1642/1797 (91.375%) | +2.560 |
| 244 | 1697/1797 (94.435%) | 1640/1797 (91.263%) | +3.172 |

El intervalo describe variación de cinco entrenamientos sobre la misma población, sin custodia ciega ni replicación independiente. Datos públicos UCI con separación de escritores documentada; grupos de imagen exacta disjuntos; selección exclusivamenteDEV300 antes deTEST. NCA5056/CNN5039parámetros cerca en cuenta nominal, con profundidad, geometría yFLOPs diferentes. No demuestra superioridad universal ni rescataH1 de binding.

## Profundidad y reproducción

| Updates retenidos | Exactitud media | RMS medio de estado |
|---|---:|---:|
| 0 | 22.582% | 0.540034 |
| 4 | 78.219% | 1.83129 |
| 8 | 94.357% | 7.30792 |
| 16 | 53.100% | 150.423 |
| 32 | 16.661% | 138914 |

T8 es el punto entrenado. Extender aT16/T32 degrada fuertemente exactitud y magnitud observada; no hay estabilidad arbitraria ni memoria duradera demostrada. Los endpoints son diagnósticos finitos, no prueba de divergencia infinita.

El replay funcional conserva **98.835 decisiones exactas** y pasa los logits/NLL de las30condiciones base/transformación. **Diez comparaciones continuas deT16/T32 fallan la tolerancia congelada**, aunque sus decisiones coinciden. El recibo fallido se conserva; no se declara replay completo exitoso ni se aumenta la tolerancia. Diagnóstico de caminos en un mismo host separado.

Recuento de archivos guardados: 110comprobaciones, cero discrepancias. Diez modelos nuevos,40épocas/2240updates cadauno, sin pesos piloto ni entrenamientos anteriores repetidos. Wall de entrenamiento+DEV registrado: 447.555s; CPUprocess: 894.837s. No es energía ni coste total de todo el proyecto.

Origen inmutable: run37908785698/source8cbc49a29c3526846712c2b5f415d3b8ec0fdb50, archivea783bb77d8e9a73a1c6f3fb624562ffde242ebd7, planacb20b8edfeea6c287531ee14924a19b258de6377ed93c24da49667263f3bd3e. Test original264excluido; publicación de todo resultado autorizada por el usuario.
