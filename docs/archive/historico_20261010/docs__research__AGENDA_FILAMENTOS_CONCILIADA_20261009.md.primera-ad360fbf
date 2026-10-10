# Agenda de filamentos conciliada (punto 20, 2026-10-09)

Base: estado de `kaggle/filament/` y `kaggle/HISTORIAL.md` en la carpeta compartida, más `coordinacion/COLA-DE-TRABAJO.md`. Verificado directamente: (1) `runs_cv0_base_s1.log` llega a la iteración 4000 sin línea RESULT; (2) `runs_ms_learned_cons_40k.log:27` tiene PQ 0,4242; (3) los `runs_cv0_*` usan `work/cv-protocol-20260930/fold-0.json`.

Referencia de ruido: ±0,005 en val PQ (TABLON 2026-09-28). En el protocolo estricto (fold 0, test externo) solo hay una semilla por variante, así que ninguna diferencia menor de ~0,01 se puede llamar mejora.

## Tabla de estado

| Punto de la agenda | Estado | Evidencia | Riesgo | Siguiente acción |
|---|---|---|---|---|
| Semillas de la base (s1, s2), fold 0 | **Pendiente** | `runs_cv0_base_s1.log` termina en it 4000 sin RESULT; `runs_cv0_base_s2.log` solo tiene esperas de gpuq | Sin ruido medido: no se sabe si cons (0,3953) o pil (0,3972) superan a base (0,385) | Relanzar s1 y s2 con un ticket cada vez y Δ emparejado con bootstrap |
| Semillas en val aleatorio (8k, ms_learned) | Ejecutado | HISTORIAL §25-28; TABLON 2026-09-28: s0 0,4061, s1 0,4001, s2 0,4100 | Solo válido para val aleatorio | No repetir |
| FIL-SEED-001 / ENS4 | Ejecutado | COLA; `runs_ens3.log:26` | Bajo. ENS4 0,4260, no enviado | No repetir |
| Validación en 5 folds | **Pendiente** | Solo fold 0 ejecutado. Manifiestos `fold-0..4.json` listos | Medio: un fold no basta para decidir | Tras cerrar las semillas de fold 0 |
| Consenso suave (cons), fold 0 | Ejecutado | `runs_cv0_cons.log` RESULT; HISTORIAL §31: test 0,3953 frente a base 0,385 | Dentro del ruido sin semillas | No decidir con una semilla |
| Consenso ponderado (`--consensus-w`) | **Pendiente** | Flag en `train_fil.py:122`; encolado en `night_fil_1001b.sh:12`; `runs_cv0_consw.log` sin resultado | Medio | Ejecutar en fold 0 con el mismo presupuesto |
| Esqueleto (`--skel-w`) | **Pendiente** | Flag en `train_fil.py:119`; `night_fil_1001.sh:11` | Bajo | Ejecutar en fold 0 |
| Auxiliar (`--aux-w`) | **Pendiente** | Flag en `train_fil.py:120-121`; `night_fil_1001.sh:12` | Bajo | Ejecutar en fold 0 |
| Filamentos pequeños (`--small-frac`, FIL-014) | **Pendiente** | Flag en `train_fil.py:123`; `night_fil_1001b.sh:15` | Bajo | Ejecutar en fold 0 |
| EMA, U-Net, SDO2, PIL (protocolo estricto) | Ejecutado (fold 0, una semilla) | HISTORIAL §32-35: ema 0,3945; unet 0,3575; sdo2 0,394; pil 0,3972 | Todas empatan con cons dentro del ruido. U-Net queda por debajo | No repetir sin semillas |
| Contexto temporal (vecinos ≤2 d) | Análisis hecho, sin entrenar | `work/temporal-context-20260930/summary.json`: 60 % del test con vecino ≤2 d | Medio: el contexto no entra como entrada del modelo | Diseñar entrada temporal y control sin fuga |
| Split por bloques temporales (FIL-013) | Código listo, sin entrenar | `fil.py:73` (`split_blocks`); `train_fil.py:125` (`--block`) | Medio: el test queda más cerca en el tiempo que la validación aleatoria | Entrenar `--block 0` en fold 0 |
| Calibración de posproceso fuera de bloque | **Pendiente** | TABLON 2026-09-30 20:30 | Medio: sin esto los números por bloques no son comparables | Calibrar sobre el bloque de validación |
| Geometría FIL-007 y FIL-008 | **Pendiente de repetir** | TABLON 2026-10-01 21:15: error de geometría (radio del disco 454 px, no 505); FIL-007 y FIL-008 hay que repetirlos | Medio: los números antiguos del filtro y del SDO no son válidos | Repetir con geometría corregida; `cv0_sdo2` ya usa la corrección (0,394) |
| Mejoras de datos: SDO | Ejecutado con geometría corregida | `cv0_sdo2` (HISTORIAL §33) | Bajo | — |
| Mejoras de datos: GONG + pseudoetiquetas (FIL-010) | Propuesta sin ejecutar | COLA: "PROPUESTA", responsable "por asignar" | Alto: sesgo de autoentrenamiento | Decisión de Fran antes de ejecutar |
| Publicación Kaggle y GitHub (DOC-002, DOC-001) | **No verificable** | COLA: "LISTA (29-09)" y "ACTIVA"; no hay recibo de publicación en el repositorio | Medio: la COLA puede estar desfasada | Confirmar con Fran o con el concurso |
| Consenso largo 40k (FIL-011) | Ejecutado | HISTORIAL §29: PQ 0,4242 en `runs_ms_learned_cons_40k.log:27`; meseta | Bajo | No repetir |

## Contradicciones con `COLA-DE-TRABAJO.md`

1. **FIL-CV-CODEX** aparece como "piloto GPU pendiente", pero los `cv0_*` ya usan `work/cv-protocol-20260930/fold-0.json`.
2. **FIL-011** aparece como "LISTA, primera tarea", pero ya terminó con PQ 0,4242.
3. **FIL-003** aparece como "en cola, sin runtime", pero hay runs del protocolo de fold 0 en la RTX 3090 local.

La COLA está fechada el 2026-09-28 y el tablón llega al 2026-10-09. Hay que actualizarla antes de reservar nada nuevo.

## Qué no se ha verificado

- Estado actual de la cola de gpuq para filamentos (no consultado en esta fase).
- Publicación DOC-002 en Kaggle.
- Resultados de skel, aux, small, consw y de los folds 1-4.
