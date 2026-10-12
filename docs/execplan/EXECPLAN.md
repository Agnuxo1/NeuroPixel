# ExecPlan: completar NeuroPixel según la guía ampliada 2026-10-12b

Documento vivo y autosuficiente. Un agente sin memoria previa debe poder continuar leyendo solo cuatro cosas: este archivo, la salida de `plan_status.py status`, `git log` y las líneas de la guía que indique `focus_section_lines`. Se actualiza en cada parada. El estado de las tareas vive en `tasks.json` y solo se modifica con `plan_status.py`. `CHECKLIST.md` es una vista generada: no se edita a mano.

## Estado actual y siguiente acción

Máximo 12 líneas. Reescribir en cada parada, antes de cualquier comando largo y antes de terminar la sesión.

- Actualizado (UTC): 2026-10-12T22:36:47+00:00
- Tarea en curso o siguiente: G20.2, Comparacion directa con Hopfield: memoria y reparacion con el mismo estado; intento 1; presupuesto 90 minutos.
- Criterio de cierre: Plan congelado y fijado por hash antes de ejecutar; Hopfield global, Hopfield local y decodificador explicito evaluados en toda la rejilla registrada; resultados con negativos e informe en ingles.
- Hecho en esta tarea: Plan ANT01 congelado (commit d2700793, sha 38436579); ejecutor escrito
- Falta en esta tarea: Ejecutar ANT01 (CPU numpy), informe en ingles, afirmacion en registro, cierre
- Siguiente comando exacto: python docs/execplan/run_evidence.py --out docs/execplan/evidence/G20.2_run.txt -- <kit python> scripts/run_ANT01.py
- Trabajos en segundo plano: ninguno propio salvo lo anotado en Progreso.
- Dudas o riesgos abiertos: Sin GPU ni Torch; resultados exploratorios

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
- Python GPU histórico: C:/Python313/python.exe, metadatos verificados Torch 2.6.0+cu124/Python 3.13.7; coincide con software del recibo. Carga GPU actual pendiente de RAM >=8 GiB.
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






- [x] (2026-10-12T11:45:41+00:00) G0: done; commit 9bb4645443129a3987b036cb86199a3736a6e9b2; 1797 originales y 44 arboles sin discrepancias; 417 extras y 19 informes mapeados; dependencias explicitas.; detalle en archive/PROGRESS_ARCHIVE.md.


- [x] (2026-10-12T11:47:45+00:00) G1.1: done; commit 6344902cc82f000025c8aadf31c86002a4e727c5; 103 de 105 filas; dos claves derivadas del plan y recibo sin nuevas mediciones.; detalle en archive/PROGRESS_ARCHIVE.md.




- [x] (2026-10-12T11:50:14+00:00) G1.2: done; commit ee7f886c5e80c957ea17cb4e654ec4f752c69ef3; Recovery original commit87c60227 admitida segun §6.2:105 claves,103 originales identicas,cero nuevas mediciones.; detalle en archive/PROGRESS_ARCHIVE.md.





- [x] (2026-10-12T11:55:35+00:00) G1.3: done; commit ba9f3f84e4a779919bddb6119ef9eb99125f2f84; Documentacion local reconciliada, auditoria sin incidencias; publicacion preparada y separada como dependencia externa.; detalle en archive/PROGRESS_ARCHIVE.md.

- [ ] (2026-10-12T11:55:36+00:00) G2, intento 1: hecho G1 local cerrada, 8 tareas cerradas; G1.3.ext bloqueada con paquete; sin GPU ejecutada.; falta Leer §7 y preparar contrato T0; reproduccion historica requiere recursos reales.. Siguiente: `python docs/execplan/plan_status.py section 7`.




- [x] (2026-10-12T12:00:42+00:00) G2.1: done; commit 0183a4ee1221918630f57079d9f4d2e418a81dc7; Fuente T0 instalada con alcance AST verificado; runtime no probado; frozen exacto y limpio.; detalle en archive/PROGRESS_ARCHIVE.md.

- [ ] (2026-10-12T12:00:42+00:00) G2.2, intento 1: hecho G2.1 cerrada; version nueva separada del frozen; RAM reciente2,58GiB<8.; falta Ejecutar admision real antes de importar Torch; bloquear si guard falla y seguir elegible.. Siguiente: `python docs/execplan/plan_status.py start G2.2`.

- [ ] (2026-10-12T12:02:10+00:00) G2.2, intento 2: hecho Guard real rechazo2,039GiB; intento1 no guardo JSON porque el helper esperaba MemoryError. Cambio: capturar RuntimeError exacto del guard, sin modificar umbral.; falta Guardar rechazo estructurado, bloquear y continuar A18.. Siguiente: `python docs/execplan/run_evidence.py --out docs/execplan/evidence/G2.2_admission_v2.txt -- python scripts/completion_resource_admission.py docs/execplan/evidence/G2.2_admission_v2.json`.




- [x] (2026-10-12T12:05:40+00:00) A18.1: done; commit 887aff487fb010e13ae0aeaeefc637fb45b11002; Fuentes actuales congeladas por hash, cobertura completa de lineas; clasificacion y niveles aun pendientes.; detalle en archive/PROGRESS_ARCHIVE.md.




- [x] (2026-10-12T21:38:11+00:00) A18.2.H: done; commit d1b09d0a75d1789301bcac90a1eb611795bf36b8; 305/305 candidatos clasificados; 33 afirmaciones (12 L2, 12 L1, 9 L0); validador valido; estudio05 L0 blocked_external por archivo crudo ausente; 1 discrepancia de redondeo A06.1.; detalle en archive/PROGRESS_ARCHIVE.md.

- [x] (2026-10-12T21:53:05+00:00) A18.2.D: done; commit dd728c279cbaf2d28925d7ec6f553bbbd2a4e847; 234/234 candidatos; 24 afirmaciones (14 L2, 10 L1); validador valido; 2 discrepancias (RMS VIS07 y admision ROLE) registradas.; detalle en archive/PROGRESS_ARCHIVE.md.

- [x] (2026-10-12T21:55:52+00:00) A18.2.G: done; commit f0609c064b2520b0ccde969e43130dab4f55613e; 74/74 candidatos; 6 afirmaciones (4 L2, 2 L1); todas las celdas recontadas; energia de placa separada de pared; 2 discrepancias de redaccion.; detalle en archive/PROGRESS_ARCHIVE.md.

- [x] (2026-10-12T22:03:03+00:00) A18.2.P: done; commit 5324db225181a010b0ebb01915b409e3208b2ae4; 182/182 candidatos P; 10 afirmaciones nuevas incl. T0 pendiente explicito; 7 discrepancias de redaccion; registro integrado 795/795, 73 afirmaciones, validador valido.; detalle en archive/PROGRESS_ARCHIVE.md.
- [x] (2026-10-12T22:03:03+00:00) A18.2: done; commit 5324db225181a010b0ebb01915b409e3208b2ae4; Registro completo: 73 afirmaciones vinculadas a artefactos (L0:10, L1:33, L2:30); 795 candidatos cubiertos; exclusiones editoriales justificadas; sin timestamps ni independencia inventados.; detalle en archive/PROGRESS_ARCHIVE.md.


- [x] (2026-10-12T22:24:17+00:00) A18.3: done; commit 3e808a44012ebd9caac8112f550750457e6d18a7; Validador valido sobre 74 afirmaciones (max L2); verificacion independiente pass_with_fixes con 16 hallazgos corregidos (incl. negativos VIS07 omitidos); 22 correcciones de texto archivadas en 3 rondas.; detalle en archive/PROGRESS_ARCHIVE.md.
- [x] (2026-10-12T22:24:17+00:00) A18: done; commit 3e808a44012ebd9caac8112f550750457e6d18a7; Registro docs/research/claims_ledger.json: 795 candidatos publicos cubiertos, 74 afirmaciones (L0:10, L1:35, L2:29), validador valido; textos corregidos para no superar su nivel; L0 del estudio 05 pendiente del propietario.; detalle en archive/PROGRESS_ARCHIVE.md.


- [x] (2026-10-12T22:38:34+00:00) G3: done; commit 70f937dd61344e746108df5775f32e510b31cf5a; Unidad: familia estructural (renombrado de entidades y relaciones); NEWREL train/DEV sin fugas; selector DEV fijo; registro de exposicion con 8 comprobaciones; sin custodia ciega; TEST para el evaluador (G17.1).; detalle en archive/PROGRESS_ARCHIVE.md.
- [x] (2026-10-12T22:38:34+00:00) G20.2: done; commit 4a68e382c66c3bd398534f1eb7ca079020f8a5dc; ANT01: plan congelado antes de ejecutar; Hopfield global domina al local con mismo estado y el decodificador explicito domina todo; PR1 24/24, PR2 60/60; afirmacion ANT01-01 L2.; detalle en archive/PROGRESS_ARCHIVE.md.

## Sorpresas y hallazgos

- Arranque: consola Python cp1252 fallo al imprimir simbolos; se fija PYTHONIOENCODING=utf-8. Git interpreto CRLF como whitespace; usar comprobacion con cr-at-eol y corregir linea vacia final del log.
- A18.2.H (2026-10-12, Claude): los recibos de 04, A06.1 y conciliacion no estaban en el arbol publicado pero si, intactos, en D:/PROJECTS/NeuroPixel-Validation-20261006 y -20261007-continuation; se importaron autenticados (seleccion 04 a62300cf, original.zip 61cbc148 y closure_receipt A06.1). El archivo crudo del estudio 05 no existe en ningun sitio local ni remoto. Discrepancia: informe A06.1 linea 38 da -0,272 pero el recibo da -0,27145 (redondea a -0,271).
- A18.2.D (2026-10-12, Claude): varias cifras resumen de los informes (medias, IC, sumas) no se guardan en recibos; se recalculan desde los recibos por semilla (VIS07, MODE, U16, QTRAIN). Dos valores no se reproducen: columna RMS de VIS07 (recibo de 128 filas da 0,543...134.843) y admision 8,6749 GiB de ROLE (ausente).
- A18.2.P (2026-10-13, Claude): el registro completo cubre 795/795 candidatos con 73 afirmaciones (L0:10, L1:33, L2:30). Textos publicos con discrepancias pendientes de correccion archivada en A18.3: disponibilidad sobredeclarada (README 96, manuscrito 73, indice 17, REPRODUCE 11), REPRODUCE 17 dice 100/105, README/manuscrito 12.25x (recibo 12.24), RENDER08 etiqueta, RENDER09 frase, A06.1 -0,272, VIS07 RMS, ROLE 8,6749, figuras precientificas del README.
- A18.3 (2026-10-13, Claude): el verificador independiente encontro que VIS07 tenia dos condiciones de test registradas sin publicar: desplazamiento de una columna NeuroPixel 55,826% frente a CNN 85,120% (-29,293 pp) y media intensidad 88,737% frente a 67,056% (+21,681 pp). Registradas como D-VIS07-05 y publicadas en README, manuscrito e informe.

## Registro de decisiones

- G2: JEV connected/provenance=jev, preservar frozen y parche acotado (1,0), bloquear carga bajo8GiB y continuar metadatos (0,98); G2_jev.txt.
- A18: JEV connected/provenance=jev, congelar fuentes completas y clasificar por artefactos reales (1,0 en ambas); A18_jev.txt.

- G0: JEV connected/provenance=jev, bounded_readonly_audit (1,0); evidencia G0_jev.txt.
- G1: JEV connected/provenance=jev: inspect CPU (0,98), admitir recovery exacta (0,92), reconciliar localmente (1,0); G1_jev.txt.
- G1.2: §6.2 permite admitir recovery completa autentica sin repetir; fuentes y 103 originales verificadas, commit87c60227; energy_recovery_admission.json.

Formato: fecha · decisión · motivo · fuente (JEV con `provenance`, evidencia o regla de la guía).

- 2026-10-12T11:31:06.027278+00:00 · bootstrap ligero sin carga neural · RAM <8 GiB · JEV connected/provenance=jev, route lightweight_isolated_bootstrap, confianza 1,0; evidence/P0.1_jev.json.
- 2026-10-12 · A18.2.H: importar recibos historicos autenticados a rutas documentadas (0,98), estudio05 en L0 blocked_external sin sustituir por resumen (0,96), auditorias sin protocolo congelado en L1 (1,0) · JEV provenance=jev · docs/execplan/evidence/A18.2.H_jev.txt.
- 2026-10-13 · A18.3: subcomando verificable `supersede` para evidencia cerrada que debe evolucionar (0,97) y registrar como L2 los negativos omitidos de VIS07 con correcciones de texto (1,0) · JEV provenance=jev · docs/execplan/evidence/A18.3_jev.txt.

## Pendiente del propietario

Acciones externas preparadas y no ejecutadas.

- G1.3.ext: revisar y publicar el paquete completo results/research/COMPLETION20261012/G1_owner_publication_package.json (78 archivos; recuento del manifiesto). Version publicable anclada al commit ba9f3f84 y verificada en G1_owner_publication_locator.json; la rama actual contiene el parche T0 separado. No push ni publicacion ejecutados.
- A18.2.H.ext: recuperar del ejecutor Linux (commit 0a56621; workspace /workspace/scratch/61d6e3e6e688) el archivo crudo del estudio 05: 05_experiment/ (215 archivos), 05_analysis.json (11fd91ee...), 05_independent_check.json (adef6b11...), salidas historicas 05_historical*. Paquete: results/research/COMPLETION20261012/study05_raw_evidence_recovery_package.json. Sin contacto ni accion externa.
- Publicacion: los textos publicos corregidos en A18.3 (commits 51afbdea y 78ebf8cb; registro docs/research/claims_ledger.json) son posteriores al paquete G1.3.ext (ba9f3f84). Al publicar, usar la version mas reciente de la rama research/completion-20261012 tras su revision. Sin push ejecutado.
- G3: aprobar la descarga de los archivos publicos UCI optdigits (optdigits.tra y optdigits.tes, ~0,6 MB, archive.ics.uci.edu, CC BY 4.0) para reverificar la unicidad y disjuncion de imagenes (fuga L1.4); hoy solo lo afirma el manuscrito.
- G17.1.ext: designar un evaluador independiente (no autor) y aprobar la ruta de sellado de tiempo externo (p. ej. OpenTimestamps o deposito publico fechado) para cualquier TEST nuevo. Paquete: results/research/COMPLETION20261012/G17_evaluator_package.json. Nadie contactado.
- M3.ext: aprobar descargas e instalacion para reconstruir Lean/Comparator (elan + lean4 v4.34.1 ~0,3 GB, openai/math fd4aeeb2, cache Mathlib de varios GB, comparator, landrun, lean4export) en E:/Proyecto-concursos/NeuroPixel-lean; script preparado scripts/formal/m3_comparator_rebuild.sh. La distro WSL Ubuntu vive en C: (no apta para carga pesada).

## Resultados y retrospectiva

- Sesion1: P0, G0 y G1 local cerradas; subpasos G2.1 y A18.1 cerrados.10 tareas cerradas de48 (37 originales preservadas,11 divisiones/dependencias agregadas).
- Verificacion:63 pruebas;1797 blobs y44 arboles de origen; recovery105/103 sin nuevas mediciones; auditoria editorial posterior sin incidencias.
- G2 no cerrado: solo parche AST, Torch y backend pendientes. Guard ejecutado rechaza RAM; ultima medida 1.43GiB.
- A18 no cerrado:25 fuentes y795 unidades conservadas; niveles aun no asignados. Siguiente grupoH305 unidades; despuesD234/G74/P182.
- Sin cargas propias activas, sin procesos ajenos detenidos, sin acciones externas ni push. plan_complete:false; §19 final pendiente.

Se escribe al cerrar cada fase.

## Cambios a este plan

- 2026-10-12T12:11:47+00:00 · 37 tareas originales conservadas y criterios intactos; G2 separado por CPU/Torch/GPU y A18 por inventario, semantica y auditoria; publicacionG1 separada externamente.48 tareas totales; evidencia de integridad en session1_integrity_resources.json.

Formato: fecha · qué cambió · por qué.
- 2026-10-13 · plan_status rechazo cerrar A18.2.P porque regenerar D/G cambio evidencia cerrada; se restauraron los bytes de cierre y la biblioteca de revision es ahora idempotente · integridad del ledger.
- 2026-10-13 · plan_status.py gana `supersede` (snapshot con hash de cierre identico, traza en la tarea); evidencia de A18.2/A18.2.P redirigida a docs/archive/completion_20261012/ledger_snapshots/A18.2/ · el registro canonico debe evolucionar con las fases nuevas. Regla nueva: cerrar tareas con snapshots, nunca con archivos que evolucionan.
