# Prompt de incorporacion para Claude

Estas colaborando con Codex y Fran en `D:\PROJECTS\196_NeuroPixel`. El objetivo es ganar concursos
de Kaggle usando NeuroPixel y las tecnologias verificables de los repositorios locales y GitHub de
Fran, sin duplicar trabajo ni saturar el equipo.

Antes de actuar:

1. Lee `CLAUDE.md`, `coordinacion/README.md`, `coordinacion/TABLON.md`,
   `coordinacion/COLA-DE-TRABAJO.md`, `coordinacion/RECURSOS.md`,
   `coordinacion/THINKTANK.md` y `coordinacion/DECISIONES.md`.
2. Lee `kaggle/<reto>/COORDINACION.md` y `kaggle/HISTORIAL.md` para el concurso afectado.
3. Escribe en el tablon una entrada firmada con fecha y hora confirmando que has leido el estado.
   Reclama cualquier proceso preexistente que hayas iniciado; si no es tuyo, no lo modifiques.
4. No inicies una tarea sustancial hasta que JEV haya indicado tarea, agente, modelo y nivel de
   esfuerzo mediante una respuesta con `status=connected` y `provenance=jev`. Si JEV falla, registra
   el bloqueo; no atribuyas a JEV una respuesta local. Fran puede ordenar una contingencia.
   Usa el enrutador instalado con `router.py plan` antes de cada fase sustancial; exige
   `remote_decision=true`. Para decisiones tipadas usa el bridge v2 y exige `provenance=jev`.
5. Antes de ejecutar, reclama el ID en `COLA-DE-TRABAJO.md` y reserva GPU/CPU/RAM en `RECURSOS.md`.
   Tras los procesos actuales, solo se permite una carga GPU larga a la vez. El otro agente trabaja
   en codigo, datos, analisis, baselines, documentacion o pruebas sin GPU.
6. Guarda codigo, datasets, entornos, caches, temporales, checkpoints y modelos en D: o E:. No uses
   C: para trabajo pesado ni descargues alli: tiene muy poco espacio.
7. Conserva cambios ajenos. No borres, reinicies, mates procesos, sobrescribas artefactos, hagas
   push ni envies a Kaggle fuera de una tarea activa y su gate documentado.
8. Cada experimento debe registrar configuracion, semilla, hardware, tiempo, metrica, baseline,
   artefactos y criterio de exito. Al acabar, actualiza cola, ficha del reto, historial y libera la
   reserva.
9. Usa el Markdown compartido como chat asincrono con el formato definido en `README.md`. Guarda
   hechos y decisiones, no deliberacion privada ni secretos. Al acercarse a 45.000 palabras, archiva
   el original y crea un resumen factual antes de superar 50.000.

Estado inicial que debes verificar, no asumir: Soil CV figuraba en 4/6 folds; la continuacion de
filamentos en 4000/6000; la RTX 3090 estaba al 99 %, 19,3 GiB y 82-83 C; quedaban 4,5 GiB de RAM.
No interrumpas esos procesos. Tu primera respuesta debe quedar en `coordinacion/TABLON.md` e incluir:

- confirmacion de lectura;
- procesos que reclamas como tuyos;
- correcciones al inventario o al protocolo;
- modelos disponibles para ti y equivalencia con los niveles de esfuerzo solicitados por JEV;
- propuesta de la primera sincronizacion, sin iniciar trabajo nuevo hasta la decision valida de JEV.

