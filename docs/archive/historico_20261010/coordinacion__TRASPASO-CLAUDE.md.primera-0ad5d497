# Traspaso de Claude — NeuroPixel (2026-09-29 19:15)

Leer al entrar en un chat nuevo, después de `CLAUDE.md` y `coordinacion/README.md`.

## Forma de trabajar (acordada con Fran)

- **Equipo:** Fran (dirección, poco tiempo), Codex (ChatGPT) y Claude. Fran no quiere consultas ni peticiones de
  permiso: autorización máxima para ambos agentes. Solo se le **informa de avances**, y él da feedback.
- **Dudas y reparto:** a JEV. Antes de cada fase, `router.py plan ... --phase <fase> --class <clase>`
  (`~/.claude/skills/enrutador-jev/scripts/`), exigiendo `remote_decision=true`. Una respuesta local o un error no son
  decisión de JEV: se registra el bloqueo.
- **Chat asíncrono con Codex:** `coordinacion/TABLON.md` (hechos y reservas), `THINKTANK.md` (propuestas y respuestas),
  `COLA-DE-TRABAJO.md` (IDs, responsable, estado), `RECURSOS.md` (reservas), `DECISIONES.md` (DEC-xxx). Formato de
  entrada del README. Hora real con `date` (no estimarla). Reclamar el ID y reservar recursos **antes** de ejecutar.
- **GPU:** una sola carga larga a la vez, siempre por la cola `python D:/PROJECTS/.cognition/gpu_queue/gpuq.py run
  --name "<reto>:<ID> <nombre>" --vram <GB> --ram 8 -- <cmd>` con `run_in_background`. Guardia: RAM libre >= 8 GiB.
  Fran puede ceder la 3090 a otros proyectos (29-09 14:05): mirar `gpuq.py status` y el tablón antes de lanzar.
- **Disco:** todo en D: o E:; C: casi lleno.
- **Registro:** cada entrenamiento o envío en `kaggle/HISTORIAL.md` con `python kaggle/log_run.py --comp ... --run ...
  --que ... --debiles ... --cambiar ... --decision ...` (acepta PQ, acierto y EMD de Soil). Checkpoint en
  `D:/PROJECTS/.cognition/pixel-latente/checkpoint.md`.
- **Envíos a Kaggle:** sin preguntar, pero con gate documentado (mejora de validación). Máximo 5 al día.
- **Honestidad:** no absolutos ("imposible"); probabilidades con supuestos; ruido entre semillas ±0,005 en val PQ.
- **Codex (confidencial, no escribir en documentos compartidos):** vigilar con criterios objetivos si se estanca (repite
  corridas sin cambio, sin envíos); reencauzar en positivo con propuestas concretas y JEV. Hasta ahora ha trabajado bien.
- **Otras sesiones de Claude** escriben también en el tablón (Biohub, Neuro3D): no pisar sus reservas.
- **Credenciales:** nunca leer, introducir ni guardar claves (Fran hace los `login`). Lightning ya tiene sesión
  (usuario nautiluskit; solo T4, inviable para el lienzo en bf16; L4 da 403).

## Estado de Solar Filament Segmentation 2026 (cierre 2026-11-15)

- Mejor: ensamble `ms_learned_cont + ms_learned_cons_cont`, val PQ 0,4293 → **público 0,35** (serie 0,31/0,33/0,34/0,35).
- Conclusiones completas: `docs/FILAMENTOS_2026-09-28.md`. Techo humano PQ 0,354; los FN son sobre todo ruido de anotación.
- Código: `kaggle/filament/train_fil.py` (`--ms learned|fsr|bilinear`, `--consensus`, `--filters`, `--sdo`, `--seed`,
  `--init`), `multiscale.py`, `predict_fil.py`, `ensemble_fil.py`, `sdo_channels.py`, `filters3.py`, diagnósticos
  `diag_fn.py`, `diag_lost.py`, `hyst.py`. Datos SDO: `kaggle/filament/sdo/` (AIA 304 + HMI, 887 fechas) y caches en
  `kaggle/filament/cache/` (`imgs_sdo.npy`, `imgs3.npy`).
- Reglas: datos externos públicos permitidos (2.6); el ganador debe explicar el método (2.8.b); compartir código fuera
  de Kaggle exige publicarlo también en Kaggle (3.6.b). Fran (DEC-009): publicar también en Kaggle; objetivo
  visibilidad y, si se puede, premio. Recomendado: cuaderno con enfoque ahora y código completo cerca del cierre.
- Objetivo de Fran: "apuntar a 0,80 para llegar a 0,60"; hito intermedio val 0,50.

## Siguiente, en orden

1. **FIL-011 (primera tarea, aún NO lanzada):** `bash kaggle/filament/next_fil011.sh` — consenso desde cero a 40.000 it,
   validación cada 2.000 (~6,5 h). Comprobar antes que la 3090 no está cedida a otro proyecto.
2. **DOC-002:** cuaderno público en Kaggle (DEC-009).
3. **FIL-008 v2:** líneas de inversión de polaridad del magnetograma HMI como canal.
4. **FIL-010:** autoentrenamiento con más imágenes GONG públicas (+ canales SDO).
5. **DOC-001:** GIFs (`docs/visual/make_scanner_gif.py`, `make_zoom_gif.py`; elegir fecha con filamento grande) e
   informe de 4 páginas; diccionario con nombres astronómicos y colores solares reales.
6. Reserva: FSR (DEC-008) para otros retos; FIL-009 coloreador como características (final de agenda).

## Otros retos

- Soil: SOIL-002 de Codex (público 98,44 con NeuroPixel; mejor previo DINOv2 60,80). Digits: público 0,99375.
- Biohub (cierra 2026-09-29 23:59 UTC): llevado por Codex y otra sesión de Claude en `D:/PROJECTS/biohub-codex-cognition`.

## Git

Commit local `087f0a8` sin push. El push público del 27-09 (`7da18d1`) queda cubierto por DEC-009 al publicar en Kaggle.
