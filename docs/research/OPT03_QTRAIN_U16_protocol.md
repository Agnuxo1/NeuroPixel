# OPT03-QTRAIN-U16: presupuesto adicional con estados completos

**Protocolo prospectivo congelado tras preflight; ninguna actualización nueva ejecutada al congelar.** Se prepara después del cierre negativo de QTRAIN. No modifica sus resultados ni H1. La recomendación JEV tiene provenance=jev y confianza0,27; es asesoramiento limitado, no evidencia de que la intervención funcionará.

## Pregunta y controles

¿Duplicar la dosis de entrenamiento, de8.192 a16.384updates, con arquitectura, LR, supervisión, datos y estados de optimización conservados, permite alcanzar competencia conjunta estable? Las doce pérdidas seguían descendiendo al cerrar QTRAIN y una inicialización mostró aprendizaje parcial; esto justifica medir más presupuesto, no asumir éxito ni una causa única.

Se continuarán **los doce padres**, semillas200/201/202 × paired/repeated × escuela0/0,3. Ninguna semilla favorable se selecciona. Se restaura el modelo, AdamW y sus momentos/contador, RNG global de firing y los generadores independientes de contexto y asignación de consulta. Primera actualización nueva8.193; última16.384. Los primeros8.192updates y los resultados cerrados permanecen inmutables y no se repiten. Son doce extensiones de las mismas tres inicializaciones por condición, no doce inicializaciones nuevas ni n=6.

El presupuesto interviene conjuntamente sobre updates, exposición adicional a ejemplos y cómputo. No identifica un efecto puro del algoritmo de optimización. Continúan los mismos streams deterministas:16draws de contexto con reemplazo,64ejemplos y16consultas por rol en cada batch. Backbone29.824parámetros, T16, firing0,5, PAD legado, tying/reinjection, AdamW LR0,001/betas0,9–0,999/eps1e−8/decay0,0001/clip1,0 se conservan.

## Endpoints y análisis previsto

Nuevos endpoints12.288 y16.384: decisiones originales/queryflip, modo denso y promedio de ocho rollouts individuales correspondientes al entrenamiento79000–79007. Probe2048/validación4096 siguen siendo pools expuestos de desarrollo; cero test final. Los datos, etiquetas, posiciones query7,6/output7,7 y hashes de contenidos deben coincidir con el padre. No se reevalúa el endpoint8.192 cerrado: se reutilizan sus decisiones auditadas.

Criterio de competencia **sin cambios**: joint-query probe≥95% y validación≥90% a16.384, en las tres semillas de una condición, con integridad de fuentes, datos, recursos y estados. No seleccionar sobre12.288, máscaras o una semilla. Si ninguna condición pasa, ningún ganador ni test final. Si alguna pasa, solo habilita diseñar la siguiente fase prospectiva de precisión; no cierra por sí sola la tarea3.

Familia exploratoria 2×2×2 sobre presupuesto8.192/16.384, consultas y escuela. Unidad n=3inicializaciones, siempre pareada. Contraste continuo primario: cambio16.384−8.192 promediado sobre las cuatro condiciones dentro de cada seed. Conservar los tres valores/media/SD/ICt95df2. Secundarios: efectos de consultas y escuela promediados sobre los otros factores; interacciones presupuesto×consultas, presupuesto×escuela y consultas×escuela promediadas sobre el tercer factor; triple interacción como cambio de la interacción consultas×escuela entre presupuestos. Conservar también todas las celdas y cambios por condición, sin selección. IC sin ajuste y sin truncar, exclusivamente exploratorios.

Las máscaras, endpoints, ejemplos y etapas no aumentan n. La variación de inicialización y la precisión poblacional siguen requiriendo otra fase con nuevas inicializaciones/particiones independientes. No mezclar esta receta con A06.1 ni reinterpretar H1.

## Admisión, recuperación y reproducibilidad

Antes de cualquier actualización: verificar plan LF/hash independiente/Gitblob, todas las fuentes gobernantes, padre12/12 mediante los archivos SHA-256 de `OPT03_QTRAIN_U16_parent_inventory.json`, checkpointstep8.192 y contador AdamW8.192. Comprobar equivalencia exacta de tensores, optimizer y tres streams tras restaurar; preservar parent bytes. RAM disponible≥8GiB real antes de entrenar y controles periódicos; CPU2/versión científica fijada. Registrar diferencias de entorno y prohibir cambios no previstos de software, arquitectura o receta.

Persistir checkpoints completos cada128updates, endpoints antes/después de evaluar y cachés de decisiones con identidad modelo/datos/máscaras. Archivar además un parcial inmutable cada1.024updates nuevos y cada extensión completa antes de continuar; comprobar que archivar no cambia ningún estado de entrenamiento. Reanudar solo desde un checkpoint propio íntegro; resolver evaluaciones pendientes sin repetir el prefijo durable recuperado. Conservar parciales y distinguirlos de completados. Un intento posterior debe recuperar y omitir extensiones ya cerradas, nunca empezar los doce padres de nuevo. Un fallo de VM entre archivos remotos puede perder trabajo local no recuperable; ese coste no se considera cero ni se cuenta como otra réplica.

Fijar Python3.12.14 (versión de los padres, distinta de la3.12.15 del verificador), Torch2.6.0CPU, NumPy2.2.6, SciPy1.15.1 y psutil6.1.1, CPU2 y flags deterministas. Se permiten diferencias del host/kernel del ejecutor público si se registran; no se permiten cambios de software/dispositivo/determinismo. La igualdad exacta exigida es la restauración inicial de tensores/AdamW/RNG y la equivalencia en fixtures sobre un mismo entorno, no una promesa de identidad bit a bit de toda una trayectoria futura entre hosts físicos.

Preflight requerido: fixtures de restauración/continuación comparadas bit a bit contra una trayectoria interrumpida equivalente; rechazo de fuentes/padres/pasos/optimizer/RNG alterados y RAM insuficiente; muestreo/roles/contextos y evaluación que conserve streams. Los fixtures no cuentan como experimentos científicos. Comprobar también admisión de los doce padres sin nuevos updates ni inferencia.

Ejecución únicamente en recursos legales y sin coste autorizado: public CPU estándar, sin Actionsartifact/cache de pago; archivos y recibos inmutables en rama aislada. El límite de ejecución debe admitir archivo de cada caso y recuperación de incompletos. La verificación posterior requiere recuento independiente de decisiones/contrastes y replay de checkpoints. Otro ejecutor del proyecto no es replicación externa independiente.

**Preflight antes del freeze:** run37770211221/source9d0bd82530d361778e62c517cd05941b5fb99e63, completed/success; cinco contratos de transporte, siete de continuidad y doce padres reales restaurados exactamente, cero updates/inference científicos. Primer preflight37769442809 falló por Python3.12.15 frente al padre3.12.14 y se conserva. El manifiesto LF fija los hashes de todas las fuentes; el workflow, que incorpora su hash, queda ligado en un manifiesto externo de ejecución para evitar una dependencia circular. Congelar no significa ejecutar ni cerrar científicamente la fase.
