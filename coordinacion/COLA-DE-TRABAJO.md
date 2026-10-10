# Cola de trabajo compartida

Actualizada: 2026-09-28 23:18 Europe/Madrid.

| ID | Prioridad | Estado | Responsable | Modelo/ruta | Esfuerzo | Reto | Recursos | Entregable / criterio de aceptacion |
|---|---:|---|---|---|---|---|---|---|
| OPS-001 | 0 | REVISION | Codex | determinista | — | global | ninguno | Mantener protocolo y resolver conflictos de cola |
| FIL-015-AUDIT | 2 | HECHA / PILOTO MODELO PENDIENTE | Codex | principal + herramientas deterministas; fallback local | medio | filament | liberada; cero GPU | Test 60% con vecino <=2dias; bloques purgados 0%; cinco manifiestos intercalados y pares listos en work/temporal-context-20260930 |
| FIL-CV-CODEX | 1 | IMPLEMENTADA Y VERIFICADA / PILOTO GPU PENDIENTE | Codex implementa; Claude integra a su agenda | principal + herramientas deterministas; fallback local JEV no disponible | alto | filament | liberada; cero GPU | Cinco manifiestos en work/cv-protocol-20260930; --split-manifest integrado; test externo heldout sin calibracion; controles de datos PASS |
| SOIL-001 | 1 | HECHA (20:13, EMD CV 39,195 vs media 85,393) | Claude | modelo general | medio | soil | GPU, hasta 6 GiB | Completar 6 folds; `result.json`; comparar EMD con curva media |
| FIL-001 | 2 | HECHA (20:40, val PQ 0,3753) | Claude | modelo rapido | bajo | filament | GPU, hasta 14 GiB | Terminar 6000 it o documentar corte; mejor PQ y checkpoint |
| SOIL-002 | 3 | EN COLA / SIN RUNTIME | Codex | modelo general | medio | soil | sin recurso reservado | Auditar split por telefono; no enviar si no supera de forma robusta la referencia 60,80 |
| FIL-002 | 4 | HECHA (learned+4k PQ 0,4114; bilinear+4k PQ 0,346; FSR+4k PQ 0,3583) | Claude | modelo general | alto | filament | GPU local liberada a FIL-005 | Ninguna continuacion corta supera 0,4263; learned es la unica que mantiene tendencia positiva |
| FIL-003 | 5 | EN COLA / SIN RUNTIME | Codex | modelo general | medio | filament | sin recurso reservado | CV por pliegues y calibracion de posproceso sin ajustar sobre un unico split |
| FIL-004 | 5 | HECHA / RESERVA (Fran 08:20: FSR en reserva para otros retos; mejor c=24 0,367, satura pronto, barato: ~3 min/1000 it) | Claude | modelo general | alto | filament | GPU exclusiva | FSR con mas iteraciones y modelo mayor (a igual tiempo que ms_learned: ~3x iteraciones); comparar con ms_learned 0,406 |
| FIL-005 | 5 | HECHA (PQ 0,4243; no supera 0,4263) | Claude | modelo general | alto | filament | liberada | Consenso largo no mejora al mejor modelo sin consenso |
| FIL-006 | 5 | CERRADA A 1k / NO REANUDAR | Codex | modelo general | alto | filament | ningun runtime nuevo | 1k: PQ 0,3761, Dice 0,6586, TP 812, FP 862, FN 423, 11.055 s; no supera 0,4263 |
| FIL-ENS-001 | 5 | HECHA / PUBLICO 0,35 | Claude | determinista | medio | filament | liberada | Val PQ 0,4293; submission generado y enviado; mejora publica 0,34 -> 0,35 |
| FIL-007 | 6 | HECHA / NEGATIVA (PQ 0,3799) | Claude | modelo general | alto | filament | liberada | Canales limbo+sato+DoG empeoran frente a control 0,4061; no promocionar |
| FIL-012 | 1 | HECHA parcial (Claude 30-09) | Claude | — | — | filament | — | Agenda Codex punto 1: clip de gradiente en TODOS los modulos (bug: `ms` sin clip), paso rechazado si no finito, contador `skipped`, parada >50. FIL-011c relanzado |
| FIL-013 | 2 | CODIGO LISTO, sin entrenar | Claude | modelo general | alto | filament | GPU tras FIL-011c | Agenda punto 2: `fil.split_blocks` (5 bloques temporales + hueco 5 d) y `train_fil.py --block N`. Hallazgo: test mas cercano en el tiempo a train (mediana 1,5 d) que nuestro val aleatorio (2,0 d) -> el split por bloques es PESIMISTA; usar como control de robustez. Falta calibrar posproceso fuera del bloque |
| FIL-014 | 3 | PENDIENTE (tras 013) | Claude | modelo general | alto | filament | GPU | Agenda puntos 3-4: objetivos de consenso (individual / suave / peso por acuerdo) y filamentos pequeños (resolucion fina, muestreo); mismo split por bloques, semillas y presupuesto |
| RES-001 | 1 | ACTIVA (Fran 01-10: dia de investigacion) | Claude (6 subagentes Sonnet) | modelo general | alto | filament | web + CPU | Investigacion bibliografica con base cientifica (arXiv, ADS, MICCAI...) -> informes en `docs/research/01..06_*.md` -> sintesis priorizada -> pruebas para ejecutar de noche en la GPU. Reglas: solo papers abiertos y verificados; hipotesis falsables con criterio PQ y ruido de semilla ±0,005 |
| FIL-011 | 1 | LISTA — PRIMERA TAREA 2026-09-29 (Fran) | Claude | modelo general | medio | filament | GPU exclusiva ~6,5 h | Consenso suave desde cero hasta 40.000 it, validacion cada 2.000 (`kaggle/filament/next_fil011.sh`): ¿sigue mejorando o se estanca? Referencias: cons 8k 0,4141; cons_cont 24k 0,4243; learned_cont 24k 0,4263 |
| DOC-002 | 2 | LISTA (29-09) | Claude | modelo general | medio | filament | CPU | Publicar en Kaggle (cuaderno publico del concurso) el codigo NeuroPixel de filamentos enlazando el GitHub (DEC-009, regla 3.6.b); decidir con Fran si ahora o tras FIL-011 |
| FIL-008 | 5 | HECHA / EMPATE NEGATIVO (PQ 0,4009) | Claude | modelo general | alto | filament | liberada | SDO real no supera control 0,4061; no promocionar v1 |
| FIL-010 | 6 | PROPUESTA (Fran 28-09) | por asignar (JEV) | pendiente | pendiente | filament | red + GPU | Mas datos: miles de imagenes GONG H-alfa publicas (mismo instrumento, 2010-hoy) + sus canales SDO; etiquetas por autoentrenamiento (pseudoetiquetas de alta confianza del mejor modelo) y/o catalogo HEK de filamentos; aceptar solo si mejora val PQ |
| DOC-001 | 4 | ACTIVA (prototipo) | Claude | modelo general | medio | filament | CPU | Informe 4 pag + GitHub con estetica cientifica (Fran 28-09): GIF escaner por canales (`docs/visual/make_scanner_gif.py`, hecho), lienzo en marcha con colores solares reales y diccionario con nombres astronomicos, desglose por filtros; informe con cifras medidas (techo humano 0,35, ablaciones) |
| FIL-009 | 9 | PROPUESTA (Fran 28-09, final de agenda) | por asignar | pendiente | pendiente | filament | GPU | Caracteristicas previas de un coloreador de IA (como DINOv2) como canales de la retina |
| NCA-001 | 6 | HECHA EXTERNA / REPLICA FUERTE EN COLA | Codex | modelo general | xhigh | bateria | sin recurso reservado | Run2: acc 0,9717 vs profesor 0,9625; 3,926x; repetir con profesor fuerte antes de promocionar |
| NCA-002 | 7 | HECHA EXTERNA / VALIDADA | Codex | agente principal | alto (JEV 0,91) | bateria | Lightning T4 cerrada; cero GPU local/Kaggle | A 64 pasos: 916,08 -> 47,42 MiB (-94,82 %); reconstruccion 5,96e-7; gradiente relativo 3,82e-8 |
| FIL-SEED-001 | 7 | HECHA / S2 PQ 0,4100; ENS4 0,4260 | Claude | modelo general | alto | filament | liberada | Mejor version sigue ENS2, val 0,4293 y publico 0,35; no enviar ENS4 |
| BIO-001 | 1 | ENTREGAS COMPLETADAS / PUNTUACION PENDIENTE (29-09 23:03) | Claude: E entregado; Codex: F/G entregados y recibos; JEV verifica | agente principal; pruebas deterministas | alto | Biohub | Sin GPU local; notebooks F/G COMPLETE | C 56674288 y D 56678501 COMPLETE 0,948. E 56687119, F 56687338 y G 56687538 PENDING. F/G enviados22:51 y23:03 tras auditorias;0 cupos hoy. JEV remoto provenance=jev autoriza apuestas exploratorias(1,0), no mejora demostrada. Estado en work/last-two-20260929/checkpoint.md. Solo vigilar puntuaciones hasta cierre, luego cierre factual; no nuevos experimentos ni reenviar |
| CLOUD-001 | 8 | PAUSADA / SIN RUNTIMES NUEVOS | Codex | determinista + modelo general | alto | global | ninguno | Colab/Lightning quedan en cola; Kaggle solo ejecucion final, submission y entrega |

Regla: una tarea `ACTIVA` debe tener reserva en `RECURSOS.md`. Si no actualiza evidencia durante
90 minutos, pasa a `BLOQUEADA/POR_CONFIRMAR`; nunca se mata automaticamente el proceso.

Toda fila nueva incorpora, tras la decision de JEV: `responsable`, `modelo`, `esfuerzo` y motivo
breve. Hasta entonces su responsable es `JEV por asignar` y no se inicia.

Decision JEV vigente: 2026-09-27 19:34, `provenance=jev`, modelo `jev-1.13.0`.
Politica: `single_gpu_queue`. La pausa fue levantada por Fran; seguir la agenda, una sola carga GPU larga.

Decision JEV de contencion RAM: 2026-09-28 14:45, `provenance=jev`, modelo `jev-1.13.0`.
Terminar solo FIL-005 y quedar en reposo (confianza 0,97). No iniciar nada con menos de 8 GiB
libres (confianza 0,58). El resto permanece en cola hasta una nueva priorizacion.

Actualizacion JEV 15:43 tras la reanudacion autorizada: `provenance=jev`, jev-1.13.0. Prioriza
FIL-007 (0,96) y exige cache+humo+piloto 1k antes de 8k (0,87). La guardia de 8 GiB permanece.

Decision JEV de computo externo: 2026-09-28 10:07, `provenance=jev`, modelo `jev-1.13.0`.
Lightning CPU para FIL-003 (0,97) y SOIL-002 (0,98); Colab GPU para la replica fuerte NCA-001
(0,95). El orden emitido tuvo confianza 0,34 y se mantiene provisional, no como decision firme.

