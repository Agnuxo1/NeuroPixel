# ExecPlan: completar NeuroPixel según la guía ampliada 2026-10-12b

Documento vivo y autosuficiente. Un agente sin memoria previa debe poder continuar leyendo solo cuatro cosas: este archivo, la salida de `plan_status.py status`, `git log` y las líneas de la guía que indique `focus_section_lines`. Se actualiza en cada parada. El estado de las tareas vive en `tasks.json` y solo se modifica con `plan_status.py`. `CHECKLIST.md` es una vista generada: no se edita a mano.

## Estado actual y siguiente acción

Máximo 12 líneas. Reescribir en cada parada, antes de cualquier comando largo y antes de terminar la sesión.

- Actualizado (UTC): 2026-10-12T11:38:44+00:00
- Tarea en curso o siguiente: G0, Congelar inventario y restaurar trazabilidad; intento 1; presupuesto 30 minutos.
- Criterio de cierre: cero discrepancias no explicadas en los originales y un mapa de afirmación → protocolo → fuente → ejecución → datos → recuento. Las dependencias faltantes se identifican, sin inventar archivos ni valores. [guía §5]
- Hecho en esta tarea: P0 completa, 4 tareas cerradas; 63 pruebas OK.
- Falta en esta tarea: Congelar inventario y verificar originales y trazabilidad.
- Siguiente comando exacto: python docs/execplan/plan_status.py section 5
- Trabajos en segundo plano: ninguno propio salvo lo anotado en Progreso.
- Dudas o riesgos abiertos: RAM >=8 GiB antes de carga neural; procesos ajenos intactos.

## Propósito

Ejecutar de principio a fin la guía `D:\PROJECTS\NeuroPixel_Guia_Completa_20261012_ampliada.md` (solo lectura): de G0 a G22, las secciones §18A–§18N y la entrega del §19. Cada punto se cierra con su evidencia, aunque el resultado sea negativo. El éxito no es «demostrar» una hipótesis, sino cerrar cada protocolo con honestidad y llevar la mejor afirmación al nivel de evidencia más alto que los datos permitan.

## Reglas de operación

### Fuentes de verdad, por prioridad

1. Recibos, datos y salidas de comandos.
2. La guía ampliada (sección indicada).
3. `tasks.json`.
4. Este ExecPlan.
5. Tu memoria de la conversación, que es la menos fiable. Si contradice a los archivos, ganan los archivos.

### Reanudación

Aplicar al empezar cada sesión, tras una compactación de contexto o en cuanto no recuerdes por qué haces algo:

1. `python docs/execplan/plan_status.py status`. Si devuelve `problems`, corrígelos primero, sin editar `tasks.json` a mano (hay copia en `tasks.json.bak`).
2. Leer «Estado actual y siguiente acción» y las entradas de Progreso de la tarea en curso. No releer el resto del plan salvo que lo necesites.
3. `git log --oneline -8` y `git status --short`. Si hay cambios sin commit, decidir si son trabajo válido de la tarea en curso: commit o anotación.
4. `python D:/PROJECTS/.cognition/gpu_queue/gpuq.py status` si hay tickets propios, y las últimas 40 líneas de `coordinacion/TABLON.md`.
5. Leer solo las líneas de la guía de `focus_section_lines`.
6. Antes de repetir un paso, comprobar si su salida ya existe y es válida (idempotencia). No rehacer lo hecho.

### Bucle de trabajo: una tarea cada vez

1. `plan_status.py next` da la tarea elegible. Si no hay ninguna, ir a «Bloqueos».
2. Leer su sección de la guía y su criterio de cierre. Escribir en Progreso los sub-pasos previstos, sin marcar, y en «Estado actual» el presupuesto estimado.
3. Si la tarea no cabe en una sesión, tiene más de 6 sub-pasos o mezcla CPU, GPU y terceros, dividirla **antes** de empezar: `plan_status.py add --id <id>.1 ... --blocks <id> --before <id>`. Si ya estaba en curso: `pause` y después `add`.
4. `plan_status.py start <id>`.
5. Ejecutar. Tras cada sub-paso con resultado:
   - marcarlo en Progreso con hora UTC y ruta de la salida;
   - actualizar «Siguiente comando exacto»;
   - hacer un commit pequeño si hay cambios.
6. Verificar el criterio de cierre ejecutando comandos y guardando su salida en `docs/execplan/evidence/<id>_<qué>.txt|json`, o usar los archivos de resultados que exige la guía. Sin evidencia ejecutada no hay cierre.
7. Commit del trabajo. Luego `plan_status.py close <id> --status done|negative --evidence <rutas> --commit <sha> --note "<resultado en una línea>"`. Por último, commit `plan: close <id>` con `tasks.json`, `CHECKLIST.md` y este plan.
8. Condensar en una sola línea las entradas de Progreso de la tarea cerrada. Si eran largas, moverlas a `docs/execplan/archive/PROGRESS_ARCHIVE.md`: se mueven, nunca se borran.
9. Volver al paso 1. No pedir permiso para continuar.

### Contexto

- No leer la guía entera ni archivos grandes. Usar rangos de líneas (`plan_status.py section <n>`), `Select-String`, `Get-Content -TotalCount` y `-Tail`.
- Las salidas largas van a archivo; de ellas se lee solo el resumen o las últimas líneas.
- Mantener este plan por debajo de unos 25 KB moviendo detalle al archivo histórico.
- «Escribir antes de olvidar»: actualizar «Estado actual» antes de comandos largos, antes de compactar y al cerrar cada sub-paso.
- Si el contexto supera aproximadamente el 70 %, terminar el sub-paso, actualizar el plan, hacer commit y continuar tras compactar, o en una sesión nueva con el prompt de reanudación.

### Bloqueos, GPU y terceros

- GPU solo mediante la cola. Antes, `python D:/PROJECTS/.cognition/gpu_queue/gpuq.py status`. Después, `gpuq.py run --name "NeuroPixel:<id>" --vram <GB> --ram <GB> -- <comando>` en segundo plano.
  - Mientras espera: `plan_status.py block <id> --kind resource --note "ticket <nombre> en cola desde <hora>"` y seguir con la siguiente tarea elegible.
  - Al terminar el ticket: `reopen`, `start`, verificar y cerrar.
  - Nunca lanzar GPU fuera de la cola ni detener procesos ajenos.
  - Respetar el guard de 8 GiB de RAM; no rebajarlo.
- Para un acto de terceros o una acción externa (publicar, preregistrar, depositar, contactar, gastar, push público):
  1. preparar el paquete completo;
  2. añadirlo a «Pendiente del propietario»;
  3. `block <id> --kind external --evidence <paquete> --note "<qué falta exactamente>"`.
- Cuando la guía acepta como cierre un «paquete terminado con dependencia externa explícita» (por ejemplo §17):
  1. cerrar la tarea como `done` con el paquete como evidencia;
  2. crear `<id>.ext` bloqueada externamente para la parte de terceros.

  Las tareas siguientes dependen de `<id>`, no de `<id>.ext`.
- Si no queda ninguna tarea elegible:
  - con algún `blocked_resource`: comprobar la cola cada 30 minutos sin inventar trabajo;
  - si todo lo restante es `blocked_external`: entrega final parcial (§19) y fin de sesión.

### Reglas científicas innegociables (de la guía)

- No reabrir estudios cerrados. No añadir semillas, épocas, tolerancias ni endpoints a un estudio cerrado. Una continuación lleva identificador, hipótesis y protocolo nuevos.
- Los negativos se cierran como `negative` y se publican. Un resultado no significativo no prueba equivalencia.
- Fijar criterios antes de mirar los datos. Sellar los modelos antes de abrir TEST. Custodia con `commitments` (§18F).
- No convertir una implementación terminada en una capacidad demostrada. No declarar un nivel L que `claims_ledger` rechace.
- Conservar originales. Recuperación antes de tocar `neuropixel/model.py` (§6.2).
- Código, informes y manuscrito en inglés. Plan y registros, en español.
- Discos D: y E:; nada pesado en C:. Rama y checkout propios; no editar archivos de otros agentes.

### Anti-bucle y calidad

- Cada intento nuevo de una tarea declara en Progreso qué cambia respecto del anterior: hipótesis, parámetro, código o datos. Prohibido repetir una ejecución idéntica, salvo para medir el no determinismo.
- Tres intentos con el mismo enfoque sin cumplir el criterio obligan a una de tres salidas: dividir la tarea, cambiar de enfoque tras consultar a JEV o bloquear con un diagnóstico.
- Si una tarea supera el doble de su presupuesto, parar, registrar en Sorpresas y replanificar.
- No declarar nada «inviable» sin evidencia ejecutada. No dar el programa por terminado mientras `plan_status.py status` no muestre `plan_complete: true`.

### Coordinación y comunicación

- Al empezar, reclamar la tarea en `coordinacion/TABLON.md`. Al cerrar cada fase, añadir allí una línea con el resultado. Leer `COLA-DE-TRABAJO.md` antes de lanzar algo pesado: no duplicar lo que otro agente ya ejecuta.
- Dudas técnicas o de reparto: JEV. Ejecutar `python C:/Users/Windows-500GB/.claude/skills/enrutador-jev/scripts/router.py jev --state <estado.json> --questions <preguntas.json>`; las plantillas están en `references/jev-plantillas.md` de esa skill. Registrar la decisión y su `provenance` en el Registro de decisiones. JEV aconseja; no sustituye a las pruebas.
- No hacer preguntas rutinarias al propietario. Al final de cada sesión, escribir 3–6 líneas en `SESSION_LOG.md`: tareas cerradas, resultados, bloqueos y siguiente acción.

### Definición de terminado

Sin tareas `pending` ni `in_progress`. `claims_ledger` válido. Cada bloqueo externo tiene su paquete. Entrega del §19 escrita, con la puerta alcanzada (§18N) y el nivel L de cada afirmación.

## Orientación

Rellenar en P0 y mantener al día:

- Checkout: D:/PROJECTS/NeuroPixel-Completion-20261012; fuente viva D:/PROJECTS/196_NeuroPixel.
- Rama: research/completion-20261012.
- Commit base: d8721a59d36d127642a5664c0c41e6091296ba21.
- Python del kit: E:/Proyecto-concursos/NeuroPixel-env-completion/Scripts/python.exe (Python 3.13.7); NumPy 2.2.6, SciPy 1.15.1, ModernGL 5.12.0, glcontext 3.1.0.
- Python GPU histórico (`$GpuPython`): pendiente inventario §4.3.
- Recursos medidos: RAM libre 4.92 GiB, CPU 20 hilos / 10 núcleos (34,2 %); disco D: 347,29 / E: 215,64 GiB libres; gpuq con trabajo compartido y cola ajenos.
- Rutas de resultados: results/completion/; evidencia del plan docs/execplan/evidence/.
- Evaluador designado: no designado; custodia/replicación se prepararán y bloquearán si necesitan propietario.
- Procesos ajenos que no se tocan: docs/execplan/evidence/P0.1_resources_initial.json; sin interrupciones.

## Progreso

Obligatorio. Un hito por parada, con hora UTC. Las tareas a medias se dividen en «hecho» y «falta».









- [x] (2026-10-12T11:36:59+00:00) P0.1: done; commit f59c41a8b67c19395b87c48659384c6e467f8dd6; Lecturas, recursos reales y reclamacion verificados; sin carga neural.; detalle en archive/PROGRESS_ARCHIVE.md.
- [x] (2026-10-12T11:36:59+00:00) P0.2: done; commit 92d5f3ae283d8a6554e87d2c2e8868369482495a; Checkout y rama aislados; main remoto coincide con base fijada.; detalle en archive/PROGRESS_ARCHIVE.md.



- [x] (2026-10-12T11:38:41+00:00) P0.3: done; commit 2a6199cb8558bb5dd6eabaee07b90eae95ff2533; 26 archivos autenticados; entorno E:; 38+25 pruebas aprobadas.; detalle en archive/PROGRESS_ARCHIVE.md.


- [x] (2026-10-12T11:38:44+00:00) P0.4: done; commit 2ec302d252d8e1bcf40636537209d5790a0011fe; Sistema persistente validado sin problemas, checklist y AGENTS instalados.; detalle en archive/PROGRESS_ARCHIVE.md.

- [ ] (2026-10-12T11:38:44+00:00) G0, intento 1: hecho P0 completa, 4 tareas cerradas; 63 pruebas OK.; falta Congelar inventario y verificar originales y trazabilidad.. Siguiente: `python docs/execplan/plan_status.py section 5`.

## Sorpresas y hallazgos

- Arranque: consola Python cp1252 fallo al imprimir simbolos; se fija PYTHONIOENCODING=utf-8. Git interpreto CRLF como whitespace; usar comprobacion con cr-at-eol y corregir linea vacia final del log.

## Registro de decisiones

Formato: fecha · decisión · motivo · fuente (JEV con `provenance`, evidencia o regla de la guía).

- 2026-10-12T11:31:06.027278+00:00 · bootstrap ligero sin carga neural · RAM <8 GiB · JEV connected/provenance=jev, route lightweight_isolated_bootstrap, confianza 1,0; evidence/P0.1_jev.json.

## Pendiente del propietario

Acciones externas preparadas y no ejecutadas.

## Resultados y retrospectiva

Se escribe al cerrar cada fase.

## Cambios a este plan

Formato: fecha · qué cambió · por qué.
