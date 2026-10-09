# Fallo observable y vías alternativas de NeuroPixel

Fecha: 8 de octubre de 2026. Explicación solicitada por el usuario; no es una nueva receta ni autoriza modificar los entrenamientos congelados activos.

El bloqueo científico es competencia insuficiente e inestable en asignación de valores a roles condicionada por la pregunta. No se ha demostrado una incapacidad arquitectónica ni localizado una causa neuronal única. La obtención de resultados y la ejecución están funcionando: el antiguo HTTP401 ya se resolvió. El mínimo de RAM sigue vigente y la fase activa usa un ejecutor admitido; aumentar hardware no identifica ni resuelve por sí mismo el mecanismo.

## Evidencia cerrada

- ROLE, modelos PAD legado/firing0,5: selección de uno de los dos nouns candidatos63,574%, cambio de predicción al cambiar consulta18,506%, acierto conjunto7,503%. Son métricas distintas del mismo diagnóstico, no factores de una descomposición multiplicativa.
- QTRAIN, cuatro consultas y escuela0,3: acierto conjunto en validación74,6765%,2,4658%,7,5317% en tres inicializaciones. Media28,2247%; IC95 exploratorio[-71,9062;128,3556]%. Falla el gate de todas las semillas. No mezclar esta receta ni métrica con los resultados anteriores.
- La respuesta nominal del caso favorable alcanza86,865%; acertar original y consulta opuesta sobre el mismo contexto exige74,6765%. No equivale a superioridad al Transformer histórico.
- La escuela actual reconstruye los tokens de las celdas ocupadas en estados intermedios. Su pérdida no es una supervisión explícita de correspondencias rol/valor ni de respuestas contrafactuales. Una pérdida escolar baja no demuestra resolución de binding.
- Cambiar PAD o firing no produjo competencia suficiente en las condiciones probadas. Las pérdidas de respuesta aún descendían a8192updates. Más presupuesto es una hipótesis en evaluación, no una solución garantizada.

## Localización funcional y límites

Ruta del cálculo: tokens y consulta -> estados locales -> comunicación recurrente -> representación en celda de salida -> diccionario de lectura -> respuesta. El fallo observable está en convertir consulta y pares rol/valor en la respuesta correcta y estable. Las auditorías de estados/datasets/predicciones descartan numerosos errores de integridad comprobados; no descartan todo defecto lógico o de entrenamiento. No hay datos suficientes para atribuir el fallo exclusivamente a transporte, representación, decoder, gradientes, capacidad o supervisión.

## Vías prospectivas, todavía no ejecutadas

| Vía | Experimento que discrimina causas | Qué permitiría concluir y límite |
|---|---|---|
| Descomponer la tarea | Reutilizar controles simbólicos ya cerrados (03_controls.md,51152aciertos/51152) y añadir solo la nueva escalera de microtareas: un par, dos pares, consulta opuesta y posiciones aleatorias | Distinguir ambigüedad del dato y dificultades de consulta/posición/composición; los controles nuevos no sustituyen el test original |
| Cambiar lo que se enseña | Supervisión explícita de rol/valor o distilación de un docente competente frente a reconstrucción actual; controles de etiquetas/capacidad/datos/coste | Comprobar si un objetivo más informativo facilita aprender; mejora sería con supervisión adicional, no efecto gratuito de arquitectura |
| Observar e intervenir en la ruta | Medir estados a lo largo de recurrencia; probes separados y sustitución controlada de estados de consulta/roles con controles de posición y norma | Ubicar información disponible y comprobar influencia causal; un probe decodificable por sí solo no demuestra que el modelo utilice la información |
| Cambiar comunicación y memoria | Control con recurrencia con compuertas, consulta persistente o atención global; slots explícitos si los controles lo justifican | Poner a prueba transporte/interferencia; es otra arquitectura, necesita controles de parámetros, supervisión y cómputo, no rescata H1 |
| Estabilizar optimización | Solo tras el diagnóstico: comparación prospectiva de tasa de aprendizaje/programación, recurrencia y capacidad con presupuesto controlado | Separar elecciones de entrenamiento sin búsquedas retrospectivas sobre test; ampliar semillas después para estimar estabilidad y precisión |

## Orden de decisión

Terminar las doce extensiones U16 y su auditoría sin modificar receta ni seleccionar resultados parciales. Luego usar diagnóstico de tarea y ruta para elegir la intervención mínima que responda a una causa concreta. Antes de atribuir ventajas a componentes, demostrar competencia estable y aumentar replicación conforme a un análisis prospectivo de precisión. Mantener H1 no soportada, no abrir test final ni declarar resuelta la tarea3 por acabar una fase.

A la revisión de esta nota, siete extensiones estaban recuperadas y seis tenían auditoría completa y replay fijados; los dos jobs seguían activos. Recuperado no equivale a auditado, ni verificación interna a replicación externa.

Fuentes locales: OPT03_ROLE_results.md, OPT03_QTRAIN_results.md, OPT03_D_results.md y OPT03_QTRAIN_U16_protocol.md. Código leído: ResearchNCA.forward y update_once de qtrain_budget.py.

Bibliografía primaria para mecanismos alternativos, sin evidencia de éxito en NeuroPixel:
- Growing Neural Cellular Automata: https://distill.pub/2020/growing-ca/ . Dinámica local y entrenamiento recurrente; no establece que resuelva binding simbólico.
- Object-Centric Learning with Slot Attention: https://arxiv.org/abs/2006.15055 . Representaciones separadas por slots; adaptación a roles es una propuesta, no un resultado del artículo ni de NeuroPixel.
- Grokking: https://arxiv.org/abs/2201.02177 . Generalización tardía observada en ciertas tareas; no permite diagnosticar grokking en estas curvas.

JEV consultado: exit0/statusconnected/provenancejev, recomendación evidence_first. Es asesoramiento, no validación científica.
