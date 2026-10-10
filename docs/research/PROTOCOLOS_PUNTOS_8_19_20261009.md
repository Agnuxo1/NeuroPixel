# Protocolos preregistrados, puntos 8 a 19 del programa (2026-10-09)

Propósito: fijar hipótesis, controles, métricas, presupuestos y criterios de decisión antes de ejecutar nada. Cada punto indica su estado real y el bloqueo, si lo hay. Ningún punto de este documento tiene resultados todavía. Cualquier ejecución posterior se registra aquí con fecha y con su recibo.

Reglas comunes:
- Una hipótesis que no pueda fallar no cuenta como evidencia.
- Presupuesto emparejado en parámetros, FLOPs, pasos, datos y supervisión. Si no se puede emparejar, se declara.
- Una semilla no basta para decidir. Mínimo 5 semillas o intervalos bootstrap emparejados, según el caso.
- Ejecución solo por gpuq, con tickets pequeños y el guard RAM8 respetado.
- No se repiten estudios cerrados para mejorar su resultado (lista en la sección "Cerrados").

## Cerrados (no repetir sin hipótesis nueva)

- Comparación original H1: NeuroPixel 12,39 % frente a Transformer relativo 49,40 % en binding (`docs/research/05_results.md`). H1 no soportada.
- Precisión DEV04: objetivo original no alcanzado. No ampliar a posteriori.
- OPT03-D, MODE, ROLE, QTRAIN y U16: cerrados.
- Filamentos: FIL-005, FIL-006, FIL-007, FIL-008 y ENS2 cerrados sin promoción (ver `AGENDA_FILAMENTOS_CONCILIADA_20261009.md`).

## Punto 8. Escalabilidad

- **Hipótesis.** La energía y el tiempo por inferencia escalan de forma predecible con resolución, pasos, canales y tamaño de lote. Se espera escalado aproximadamente lineal en píxeles para el lienzo y no lineal en pasos sin el bloque reversible.
- **Datos.** R10 cubre 7 geometrías en una RTX 3090. Esto no prueba escalabilidad general.
- **Controles.** Mismo driver, misma GPU, misma receta congelada (`docs/research/RENDER11_energy_plan.json`).
- **Métrica.** J por llamada residente y s por llamada, con intervalo bootstrap.
- **Criterio.** Se declara la forma observada de la curva en el rango medido. No se extrapola a otros dispositivos.
- **Estado.** Pendiente. Bloqueo: RAM y GPU compartida. Otros dispositivos: no disponibles.

## Punto 9. Baselines fuertes con presupuesto comparable

- **Hipótesis.** A presupuesto emparejado (parámetros, FLOPs, pasos y datos), el modelo NeuroPixel no supera a un baseline CNN o U-Net bien entrenado.
- **Datos existentes.** VIS07 (CNN, 5 semillas); control U-Net en filamentos (99 750 parámetros, PQ 0,3575 en fold 0). En el control de filamentos, el lienzo supera a la U-Net en ~0,03-0,04, a igualdad de datos e iteraciones. Esto es una observación con una semilla.
- **Controles.** Misma supervisión, misma selección de checkpoint en validación, mismo número de actualizaciones.
- **Criterio.** Diferencia emparejada con IC95 excluyendo cero, con 5 semillas.
- **Estado.** Pendiente de semillas. Bloqueo: RAM para entrenar.

## Punto 10. Nuevas comparaciones sin repetir estudios cerrados

- **Hipótesis.** Un diseño nuevo, con un control que no esté en la lista de cerrados, produce un efecto medible frente al estado actual.
- **Estado.** Ninguna comparación nueva se ha diseñado todavía con control aprobado. Se propone solo tras revisar la lista de cerrados con Codex.

## Punto 11. Generalización a tareas, vocabularios y generadores nuevos

- **Hipótesis.** El rendimiento se mantiene en datos que no se usaron para ajustar el sistema.
- **Datos.** El corpus de visión es público (UCI) y ya se usó en VIS07. Un corpus realmente nuevo requiere descarga.
- **Bloqueo.** La descarga de datos nuevos requiere permiso explícito de Fran, según la política de descargas. No se descarga nada sin su OK.
- **Estado.** Pendiente de permiso.

## Punto 12. Percepción, símbolos y tiempo en una tarea prospectiva

- **Hipótesis.** La tarea conjunta exige combinar percepción, binding simbólico y memoria temporal, y el modelo supera controles que descartan atajos.
- **Controles.** Permutación de etiquetas; memorización de tripletas (baseline de tabla); modelo sin tiempo; modelo sin percepción (símbolos dados).
- **Criterio.** El modelo supera a todos los controles con IC95 excluyendo cero.
- **Estado.** Diseño pendiente. Ejecución bloqueada por RAM y GPU.

## Punto 13. Interpretación causal del scanner aprendido

- **Hipótesis.** Intervenir un estado, canal o posición concreta cambia la respuesta futura de forma específica, y no con los controles equivalentes.
- **Base existente.** Intervenciones construidas, controles inactivos y censo de matrices (`docs/research/`). El display presente puede ocultar diferencias que cambian una respuesta futura.
- **Criterio.** Efecto de la intervención frente al control equivalente, con IC.
- **Estado.** Pendiente de ejecutar. La interpretación actual no identifica circuitos causales.

## Punto 14. Estabilidad y memoria útil a horizontes mayores

- **Hecho previo.** El experimento actual pierde precisión y aumenta magnitudes en T16 y T32, con 8 pasos entrenados.
- **Hipótesis.** Un currículo de horizonte o un bloque reversible mantiene la precisión a T16 y T32 frente al control sin currículo.
- **Criterio.** Precisión a T16 y T32 frente al control, con 5 semillas.
- **Estado.** Pendiente. Bloqueo: RAM y GPU.

## Punto 15. Aprendizaje continuo, retención y routing entre expertos

- **Hipótesis.** Aprender tareas en secuencia con routing conserva las anteriores mejor que copiar pesos o que un modelo de capacidad fija.
- **Controles.** Modelo de capacidad fija; copia de pesos sin routing; fine-tuning secuencial sin regularización.
- **Métricas.** Olvido (caída en tareas anteriores), precisión media final, parámetros activos y supervisión contada.
- **Estado.** Pendiente. Copiar pesos aislados no acredita adquisición ni routing.

## Punto 16. Adquisición semántica y reparación autónoma aprendida

- **Hipótesis.** El modelo adquiere un concepto nuevo con pocos ejemplos y se repara tras un daño sin reentrenar la capa dañada.
- **Controles.** Distractores equilibrados; daño de magnitud equivalente en un control sin el mecanismo de reparación.
- **Estado.** Pendiente. Los contratos de presencia y los pesos asignados no acreditan aprendizaje conceptual ni reparación aprendida.

## Punto 17. Reproducibilidad desde cero y replicación independiente

- **Parte 1, reproducibilidad.** Instalación limpia en un entorno nuevo en D:, con el `requirements-measured-20261009.txt`, y ejecución de los tests y de las comprobaciones de hashes. **Pendiente**: la descarga del entorno es un paso de instalación del proyecto; se hará cuando la RAM y el disco lo permitan.
- **Parte 2, replicación independiente.** Requiere investigadores ajenos a esta ejecución. No puede completarse dentro del plazo ni desde este equipo. Lo que sí se puede preparar es el paquete para que ellos lo ejecuten.
- **Estado.** Parte 1 pendiente. Parte 2 fuera de alcance del plazo.

## Punto 18. Revisión del manuscrito

- **Hallazgos ya verificados (origin/main 6f1c7b7).** Línea 65 incoherente con línea 83 (R10 incompleto frente a completo); cifras sin fuente localizada (5.056/5.039, 3.538.944, 52 controles HTTP).
- **Criterio.** Cada cifra del manuscrito tiene una fuente primaria en `results/` o `docs/research/`. Las que no la tengan se corrigen o se retiran.
- **Estado.** Hallazgos pasados a Codex por el tablón. No edito el manuscrito.

## Punto 19. Originalidad, predicciones distintivas y utilidad confirmada

- **Originalidad.** Requiere comparación con antecedentes primarios. `docs/research/01_prior_art.md` existe, pero no se ha revisado en esta fase. No se afirma originalidad.
- **Predicciones distintivas.** Una predicción que distinga el mecanismo de un baseline. Ninguna está formulada todavía con un control aprobado.
- **Utilidad confirmada externamente.** Fuera de alcance: requiere terceros.
- **Estado.** Pendiente. El manuscrito no puede afirmar un descubrimiento excepcional con la evidencia actual.
