# Estado del programa de 20 puntos (2026-10-09, 23:45 Madrid)

Plazo: 2026-10-10 09:00 Europe/Madrid. Rama: `integracion/neuropixel-main-20261010` (worktree `D:/PROJECTS/196_NeuroPixel_int`), sobre `origin/main` 6f1c7b7. Commits locales, sin push.

Leyenda: **Hecho** = verificado con evidencia en el repositorio. **Parcial** = avance real con límite declarado. **En cola** = espera recursos. **Bloqueado** = requiere algo que no está en nuestras manos (hardware, permiso, terceros). **Pendiente** = posible, no iniciado.

| # | Punto | Estado | Evidencia | Límite / siguiente paso |
|---|---|---|---|---|
| 1 | Confirmar con Codex tarea, rama, resultados y reservas | Parcial | Entrada NP-PROG-20261009 en `coordinacion/TABLON.md` (22:55) y estado en 23:xx | Codex no ha contestado. Reparto propuesto sin confirmar |
| 2 | Conciliar la rama de integración con main vigente, conservando cambios | **Hecho** | `f42badf3`: merge sin conflictos sobre 6f1c7b7; la carpeta compartida no se toca | Sin push: falta revisión D4 de Fran |
| 3 | Actualizar inventarios, README, manuscrito e índices | Pendiente de Codex | Lista de correcciones enviada por tablón | Codex debe confirmar. No se ha editado el manuscrito |
| 4 | Corregir el recuperador de energía (100 frente a 103) | **Hecho** | `06517b39`: admisión de 103 filas y comprobación de las 2 ausentes; verificación posterior de 105 filas; dry run sin GPU OK | Revisión de Codex pendiente |
| 5 | Medir solo los dos bloques RGB8 y verificar cohorte completa | **En cola** | Ticket gpuq `NeuroPixel:R11-RGB8-bloque2`, espera 8 GiB de RAM (hoy ~5,8 GiB). Análisis listo: `scripts/summarize_RENDER11_energy.py <carpeta>` | Depende de RAM. Sin RAM no se mide |
| 6 | Medir consumo total del equipo | **Bloqueado** | Sin medidor de pared. Contador "Medidor de energía" de Windows no legible sin permisos elevados | Decisión de Fran: medidor de pared o autorizar permisos de contadores |
| 7 | Completar coste de entrenamiento, supervisión, cachés, evaluación, routing, crecimiento, almacenamiento y transferencias | **Parcial** | `docs/research/COSTE_CADENA_20261009.md` (`ad360fbf`): medido, parcial y no medido por componente | Supervisión, cachés, routing, transferencias y energía de entrenamiento: no medidos |
| 8 | Escalabilidad (resolución, pasos, canales, capacidad, datos, batch, expertos, otros dispositivos) | **Pendiente** | Protocolo en `docs/research/PROTOCOLOS_PUNTOS_8_19_20261009.md` (`0979b04a`) | GPU y RAM. Otros dispositivos no disponibles |
| 9 | Baselines fuertes con presupuesto comparable | **Pendiente** | Protocolo; control U-Net de filamentos (una semilla) | Semillas y RAM |
| 10 | Nuevas comparaciones sin repetir cerrados | **Pendiente** | Lista de cerrados en el protocolo | Requiere revisión con Codex |
| 11 | Generalización con datos realmente nuevos | **Bloqueado** | Protocolo | Descarga de datos nuevos: requiere tu permiso |
| 12 | Percepción, símbolos y tiempo (tarea prospectiva con controles) | **Pendiente** | Protocolo con controles de atajos | GPU y RAM |
| 13 | Interpretación causal del scanner | **Pendiente** | Protocolo | GPU y RAM |
| 14 | Estabilidad y memoria más allá de 8 pasos | **Pendiente** | Protocolo; el experimento actual degrada en T16 y T32 | GPU y RAM |
| 15 | Aprendizaje continuo, retención y routing | **Pendiente** | Protocolo con controles | GPU y RAM |
| 16 | Adquisición semántica y reparación autónoma | **Pendiente** | Protocolo | GPU y RAM |
| 17 | Entrega reproducible desde cero y replicación independiente | **Parcial** | `requirements-measured-20261009.txt` y `docs/reproducibility/ENTORNO_MEDIDO_20261009.md` (`fa91ba3b`), marcado como no probado desde cero | Instalación limpia pendiente. Replicación independiente: fuera del plazo |
| 18 | Revisar el manuscrito | **Parcial** | Línea 65 incoherente con línea 83; cifras sin fuente (5.056/5.039, 3.538.944, 52 controles HTTP). Enviado a Codex | Corrección: Codex |
| 19 | Originalidad, predicciones distintivas y utilidad confirmada | **Bloqueado / pendiente** | Protocolo; `docs/research/01_prior_art.md` sin revisar en esta fase | Confirmación externa: fuera de nuestro alcance |
| 20 | Reconciliar la agenda de filamentos | **Hecho** | `docs/research/AGENDA_FILAMENTOS_CONCILIADA_20261009.md` (`ad360fbf`) con verificación directa de tres afirmaciones | Semillas de fold 0 sin terminar; contradicciones en COLA a corregir |

## Decisiones que necesito de Fran

1. **Revisión D4** (dos ficheros con `token=` o `api_key`). Es condición para cualquier push. Sin ella no se publica la rama.
2. **Punto 6:** medidor de pared, o autorización explícita para leer los contadores de energía de Windows (requiere cambiar permisos del sistema).
3. **Punto 11 y 17:** permiso para descargar datos nuevos y para la instalación limpia en D:.

## Qué no se ha hecho y por qué

- Ninguna ejecución de GPU fuera de gpuq.
- Ningún cambio en la carpeta compartida (Codex trabaja en ella).
- Ningún cambio en permisos, configuración del sistema ni procesos ajenos.
- Ningún dato descargado.

## Actualizacion 2026-10-10 (05:20 Madrid)

- **Punto 5: hecho.** `results/research/RENDER11_energy/recovered_20261009B/receipt.json`: 105 filas, 77 controles, 0 repeticiones. `summarize_RENDER11_energy.py` sobre esa carpeta pasa todas las aserciones. Energia activa total 17 521 J y 20 167 llamadas completas (salida del resumen independiente).
- **Punto 7:** la cadena de coste de RENDER11 pasa de parcial a completa para la energia por inferencia. Sigue sin medirse el consumo total del equipo (punto 6).
- **Filamentos (punto 20):** cadena de GPU en curso. `cv0_base_s1_rerun1` heldout 0,3878 frente a 0,3850 de la semilla 0.
