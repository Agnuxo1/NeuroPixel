# Estado de los 20 puntos del programa (2026-10-10, 06:00 Madrid)

Plazo ampliado por Fran +24 h: **2026-10-11 09:00 Europe/Madrid**.

Rama: `integracion/neuropixel-main-20261010` (worktree `D:/PROJECTS/196_NeuroPixel_int`), sobre `origin/main` 6f1c7b7. Commits locales, sin push. El push y el PR esperan a la revisión D4 de Fran.

Leyenda: **Hecho** = verificado con evidencia en el repositorio. **Parcial** = avance real con límite declarado. **Bloqueado** = requiere algo fuera de nuestro alcance. **Pendiente** = posible, no iniciado.

| # | Punto | Estado | Evidencia (commit o ruta) | Límite o siguiente paso |
|---|---|---|---|---|
| 1 | Confirmar tarea, rama y reservas con Codex | Parcial | Presencia y propuesta en `coordinacion/TABLON.md` | Sin respuesta de Codex. Fran pidió no ocuparse de ello |
| 2 | Conciliar la rama de integración con main vigente | **Hecho** | `f42badf3` y siguientes: 14 commits sobre 6f1c7b7, sin conflictos | Sin push: falta la revisión D4 |
| 3 | Inventarios, README, manuscrito e índices | **Hecho** en lo verificado | `6be4025a` (manuscrito), `63cf177d` (README e índice) | Verificadas las cifras sin fuente del manuscrito: 52 controles HTTP, 3.538.944 decisiones, 5.056/5.039 parámetros, todas respaldadas |
| 4 | Corregir el recuperador de energía | **Hecho** | `06517b39`; guard anti-relanzamiento `66362999` | — |
| 5 | Medir las dos filas RGB8 y verificar la cohorte | **Hecho** | `results/research/RENDER11_energy/recovered_20261009B`, 105 filas, 77 controles, 0 repeticiones; commit `87c60227` | — |
| 6 | Consumo total del equipo | **Bloqueado** | Sin medidor de pared. El "Medidor de energía" de Windows no se lee sin permisos elevados | Requiere medidor de pared o un cambio de permisos del sistema, que no hago |
| 7 | Coste de la cadena completa | **Parcial** | `docs/research/COSTE_CADENA_20261009.md` (`a1f05088`) | Energía por inferencia del render: completa. Energía de entrenamiento, supervisión, cachés, routing y transferencias: no medidas |
| 8 | Escalabilidad | **Parcial** | `docs/research/ESCALADO_OBSERVADO_20261009.md` (`fe64f862`) | Resolución (grid 8 a 128) y canales (C16 y C48): medidos. Pasos, batch, expertos, datos y otros dispositivos: no medidos |
| 9 | Baselines fuertes con presupuesto comparable | **Parcial** | VIS07: CNN frente a NeuroPixel, 5 semillas, +3,52 pp (ya existente). Filamentos: baseline en 3 semillas; consenso en curso con preregistro `0796d702` | Veredicto preregistrado: sin apoyo (Δ medio +0,0078; semilla 2: −0,0001). Ver VEREDICTO_CONSENSO_FILAMENTOS_20261010.md |
| 10 | Nuevas comparaciones sin repetir cerrados | **Pendiente** | Lista de cerrados en `docs/research/PROTOCOLOS_PUNTOS_8_19_20261009.md` (`0979b04a`) | Requiere diseño previo y revisión de la lista de cerrados |
| 11 | Generalización con datos realmente nuevos | **Bloqueado** | Protocolo en `0979b04a` | Requiere descargar datos nuevos y código de adaptación que no existe. JEV: no descargar en esta ventana |
| 12 | Percepción, símbolos y tiempo | **Pendiente** | Protocolo con controles de atajos | No hay ejecución en la ventana |
| 13 | Interpretación causal del scanner | **Pendiente** | Protocolo | No hay ejecución en la ventana |
| 14 | Estabilidad y memoria más allá de 8 pasos | **Pendiente** | Protocolo. Hecho previo: el experimento actual degrada en T16 y T32 | No hay ejecución en la ventana |
| 15 | Aprendizaje continuo, retención y routing | **Pendiente** | Protocolo con controles | No hay ejecución en la ventana |
| 16 | Adquisición semántica y reparación autónoma | **Pendiente** | Protocolo | No hay ejecución en la ventana |
| 17 | Entrega reproducible y replicación independiente | **Parcial** | `requirements-measured-20261009.txt` y `docs/reproducibility/ENTORNO_MEDIDO_20261009.md` (`fa91ba3b`), marcado como no probado desde cero | Instalación limpia: JEV la pospone a después de la cadena de GPU. Replicación independiente: fuera del plazo |
| 18 | Revisión del manuscrito | **Parcial** | `6be4025a`: tres estados obsoletos corregidos (R10 incompleto, RENDER11 100/105, energía) | Revisión científica externa pendiente |
| 19 | Originalidad, predicciones distintivas y utilidad externa | **Parcial** | `docs/research/01_prior_art.md` (14 fuentes primarias, sección 7). Predicciones distintivas: sección 5 | Originalidad no establecida. Confirmación externa de utilidad: fuera de alcance |
| 20 | Agenda de filamentos | **Parcial** | `ad360fbf` y `d8c65192`: agenda conciliada y ruido corregido. Baseline en 3 semillas (0,3850; 0,3878; 0,4015) | Consenso en 3 semillas: sin apoyo (ver veredicto). `aux` y `small` con una semilla. FIL-010 sin ejecutar |

## Decisiones tomadas por JEV en esta ventana

- Estrategia de las horas de GPU: consenso semillas 1 y 2 antes que `aux` y `small` (confianza 0,94).
- Instalación limpia: después de la cadena de GPU (confianza 0,52).
- Estrategia del programa: `bloqueos_primero` (confianza 0,8), de la primera consulta.

## Lo que sigue sin hacerse y por qué

- Push y PR: esperan la revisión D4 de Fran sobre los dos ficheros con `token=` o `api_key`. No leo sus valores.
- Cambios de permisos del sistema, descargas de datos nuevos y credenciales: no se hacen.
