# Sistema de colaboracion NeuroPixel

Objetivo: coordinar a Fran, Codex, Claude y JEV para desarrollar NeuroPixel y competir en Kaggle
sin duplicar trabajo ni saturar el PC.

## Fuentes de verdad

- `TABLON.md`: fotografia breve del momento, presencia, bloqueos y avisos.
- `COLA-DE-TRABAJO.md`: tareas, responsables, dependencias y criterios de aceptacion.
- `RECURSOS.md`: reservas de GPU, CPU, RAM y computo externo.
- `THINKTANK.md`: propuestas y debate tecnico antes de decidir.
- `DECISIONES.md`: decisiones aceptadas, evidencia y revision JEV.
- `kaggle/<reto>/COORDINACION.md`: estado y cola especificos de cada concurso.
- `kaggle/HISTORIAL.md`: entrenamientos y envios ya realizados.

Los resultados experimentales siguen viviendo en JSON, CSV, modelos y logs. Los Markdown los
indexan y resumen; no reemplazan la evidencia.

## Flujo

1. Leer tablon, recursos y ficha del reto.
2. Crear o reclamar un ID de tarea. Estados: `PROPUESTA`, `LISTA`, `ACTIVA`, `BLOQUEADA`,
   `REVISION`, `HECHA`, `CANCELADA`.
3. Pedir a JEV el reparto: agente, modelo, nivel de esfuerzo, orden y presupuesto de recursos.
4. Reservar recursos con inicio, limite y hora esperada de liberacion.
5. Ejecutar una sola hipotesis verificable y conservar configuracion, semilla, metrica y artefactos.
6. Actualizar la ficha del reto y el historial. Liberar la reserva.
7. Para una decision sustancial, registrar alternativas y pedir revision JEV. Solo vale como JEV
   si la respuesta confirma `provenance=jev`.

JEV es el planificador preferido. Fran conserva la direccion final, pero ha delegado la operacion
cotidiana: Codex y Claude avanzan sin pedirle decisiones ni permisos rutinarios, le informan de
avances y aplican su feedback. Ambos conservan la responsabilidad de verificar resultados y
seguridad.

Si JEV no esta conectado, se ejecuta primero su procedimiento de recuperacion. Si sigue fallando,
se usa un fallback local conservador, marcado como tal y registrado; el trabajo seguro, reversible
y ya autorizado no queda paralizado. Las confirmaciones tecnicamente obligatorias de una plataforma
y las salvaguardas ante gastos, publicaciones irreversibles, credenciales o acciones destructivas
siguen vigentes.

## Formato de mensajes

Cada entrada compartida usa:

```text
### YYYY-MM-DD HH:MM Europe/Madrid — AGENTE — TEMA/ID
Tipo: HECHO | PROPUESTA | PREGUNTA | DECISION | BLOQUEO
Mensaje: ...
Evidencia/artefacto: ...
Respuesta solicitada: ...
```

No se guardan razonamientos privados, credenciales, tokens, datos personales ni salidas enormes.

## Rotacion de documentos

Al llegar a 45.000 palabras se prepara la rotacion, sin esperar a rebasar 50.000:

1. Crear `coordinacion/archivo/<NOMBRE>-AAAA-MM-DD.md` con el documento completo.
2. Sustituir el documento activo por un resumen factual de 1.000-3.000 palabras.
3. Preservar decisiones, cifras, incertidumbre, enlaces a artefactos, tareas abiertas y responsables.
4. Registrar la rotacion en `DECISIONES.md`. Nunca borrar el original.

