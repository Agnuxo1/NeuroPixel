# A06.1: auditoría y atribución de componentes

Esta es la receta local exacta, separada del estudio de la rama cloud. Ejecución original: 7 de octubre de 2026; recuperación y auditoría: 8 de octubre de 2026. La auditoría no entrena modelos, selecciona configuraciones ni reabre H1. **Investigación A06.1 cerrada con límites explícitos:** el inventario, las decisiones y los contrastes se verificaron; los efectos generales y la competencia arquitectónica siguen sin establecerse. El recibo de cierre conserva los controles aprobados y la incidencia de precisión numérica.

## Identidad y completitud

- Commit ejecutado: `a009f2a9b520a3c7333d092cd3d4f0026b17c335`.
- Manifiesto congelado: `4bc25ed4acce0c95f41d0a595289f854e0330fe237aee1210a8787378cdf6b01`.
- [Ejecución GitHub](https://github.com/Agnuxo1/NeuroPixel/actions/runs/37588312570), job `112683516968`, artefacto `11475707064`.
- ZIP original: `61cbc1484697b4004099e865a8fde1839a847d5bc40db1320fe475f44b57bf0f`, 7.267.197 bytes, 251 entradas. Se conserva sin alteración.
- Los 250 archivos incluidos en el manifiesto archivado concuerdan en bytes y SHA256. Los hashes del protocolo, implementación y fuentes históricas registrados concuerdan con los archivos locales.
- Inventario exacto del núcleo: 26 entrenamientos y 34 evaluaciones. Crecimiento: seis trayectorias, 18 etapas y ocho ajustes de gate. Todas las condiciones se conservan.
- Auditoría independiente: 4.362 comprobaciones, cero incidencias; 139.264 predicciones finales del núcleo, 53.248 de validación y 98.304 predicciones de los cuatro routers en ocho bancos recontadas.
- El log de regresión original contiene 142 marcas de aprobación y dos de skip, con paso exitoso; conteo reconstruido del log, sin archivo JUnit. No se volvió a ejecutar la suite cerrada.
- RAM mínima registrada en entrenamiento: 14,2755 GiB; se cumplió el mínimo de 8 GiB. No se ha iniciado entrenamiento local.

## Método y alcance

Binding es la media equiponderada de las exactitudes de AGENTE y PACIENTE. Se preservan exactitud global, por rol y por tema, NLL, curvas, checkpoints, estados de expertos y decisiones de crecimiento. El factorial tiene tying sí/no, escuela 0/0,3 y reinyección sí/no, con ocho combinaciones por inicialización. Se comparan T1/T4/T16, truncamiento de inferencia sobre los mismos pesos, entrenamiento con daño y extensión de 1.024 a 8.192 updates. Crecimiento usa A/B/C, bancos fijos, slots duplicados, novedad y resonancia; routers histórico, aleatorio, mezcla uniforme y gate aprendido.

La unidad de replicación es la inicialización: n=2, semillas 20/21 y un split. Las celdas factoriales, temas y ejemplos no aumentan n. Los intervalos t95 pareados tienen df=1; su supuesto de distribución no se puede evaluar con dos muestras. No se recortan los intervalos a los límites físicos del efecto. Los bootstrap por ejemplo archivados son condicionales al checkpoint y no sustituyen incertidumbre entre entrenamientos. Todas las conclusiones son exploratorias; el test y el contexto científico ya tienen exposición previa.

## Aportación cuantificada

Valores en puntos porcentuales; signo positivo favorece el primer término. Se conserva cada valor por semilla y cada interacción en `audit_saved.json`.

| Componente/contraste | Efecto medio | IC95 pareado, df1 |
|---|---:|---:|
| Diccionario/readout compartido, efecto factorial | +2,014 | [−10,394; +14,423] |
| Escuela 0,3 frente a 0, efecto factorial | +0,439 | [−13,675; +14,554] |
| Reinyección, efecto factorial | −1,135 | [−11,062; +8,791] |
| Entrenamiento T1 frente a T16 | +0,146 | [−14,123; +14,416] |
| Entrenamiento T4 frente a T16 | +2,832 | [−2,752; +8,416] |
| Mismos pesos T16: evaluar T1 frente a T16 | +0,732 | [−16,639; +18,104] |
| Mismos pesos T16: evaluar T4 frente a T16 | +0,293 | [−9,013; +9,599] |
| Entrenar con daño: efecto en evaluación limpia | +0,732 | [−38,974; +40,439] |
| Entrenar con daño: efecto bajo lesión | +1,392 | [−37,385; +40,168] |
| Interacción entrenamiento con daño × lesión | +0,659 | [−0,272; +1,590] |
| 8.192 frente a 1.024 updates, escuela 0 | +17,310 | [+0,868; +33,751] |
| 8.192 frente a 1.024 updates, escuela 0,3 | +15,283 | [−48,620; +79,186] |
| Banco fijo: router histórico frente a aleatorio | +16,178 | [−2,434; +34,791] |
| Banco fijo: router histórico frente a mezcla uniforme | +17,546 | [+3,069; +32,022] |
| Banco fijo: router histórico frente a gate aprendido | −0,684 | [−3,579; +2,212] |
| Novedad frente a banco fijo, router histórico | −0,163 | [−12,158; +11,832] |
| Resonancia frente a banco fijo, router histórico | −12,207 | [−52,328; +27,914] |

Los tres efectos principales y las interacciones factoriales se calcularon también mediante un contraste independiente de las ocho celdas, sin llamar al analizador experimental para obtenerlos. Sus medias coinciden con el análisis congelado; las pequeñas diferencias de los extremos t se deben a precisión del valor crítico.

## Interpretación de los seis componentes

1. **Diccionario compartido:** efecto medio positivo pequeño con incertidumbre amplia. La ablación del tying también altera número de parámetros; no aísla un beneficio a capacidad exactamente igual. No queda demostrado un beneficio general.
2. **Escuela:** resultado promedio cercano a cero y dependiente de la semilla y del resto de componentes. No se establece beneficio robusto ni equivalencia.
3. **Reinyección:** efecto promedio negativo; el intervalo no permite concluir perjuicio general ni ausencia de beneficio. No rescata H1.
4. **Recurrencia:** T4 entrenado supera T16 en estas dos semillas, pero ambos siguen con competencia reducida y el intervalo incluye cero. Entrenar a otra profundidad y truncar los mismos pesos son intervenciones diferentes; se informan por separado.
5. **Daño:** los efectos medios limpios y bajo lesión son positivos pequeños e inciertos. No se demuestra reparación autónoma ni recuperación de memoria útil.
6. **Crecimiento/routing:** el banco fijo da 23,242 % y 28,060 % de binding con router histórico, media 25,651 %. El gate aprendido da 24,154 % y 28,516 %, media 26,335 %. Los bancos de novedad conservan tres expertos y no son byte-idénticos al fijo en A06.1. Resonancia retiene uno y dos expertos; las diferencias mezclan capacidad retenida y política de actualización. El control de slots duplicados conserva capacidad nominal pero cambia diversidad. Ninguno acredita crecimiento adaptativo competente o superioridad general del routing.

## Optimización, multiplicidad y límites

Los cuatro endpoints a 8.192 updates alcanzan 27,832 %, 34,082 %, 24,951 % y 18,115 % de binding, según semilla/escuela. Los 26 probes están bajo el umbral de suficiencia del protocolo: `budget_limited`. La mayor exposición mejora los cuatro contrastes de presupuesto; no resuelve competencia ni separa por sí sola optimización, representación y dinámica.

Como sensibilidad retrospectiva separada, se aplicó Holm a los 40 contrastes principales y de routing rederivados. Ninguno queda por debajo de 0,05 tras esa corrección. La familia se definió durante la auditoría, por lo que no es una familia confirmatoria prerregistrada. Ni los intervalos que incluyen cero demuestran equivalencia, ni los que excluyen cero sin ajuste prueban un beneficio general.

Las comparaciones usan updates iguales, no necesariamente FLOPs o coste total iguales; escuela y profundidad añaden trabajo. Los gates aprenden con etiquetas adicionales sobre expertos congelados. Los temas tienen señales léxicas; selección de tema no equivale a razonamiento relacional. Datos sintéticos, un split, dos inicializaciones, baja competencia y ausencia de replicación externa impiden extrapolar a inteligencia general o impacto científico excepcional.

H1 permanece **no soportada** con el resultado del punto 5: −37,00 pp frente al Transformer, IC95 [−42,03; −31,97]. A06.1 permite cerrar el inventario de ablaciones con un resultado científicamente limitado; no resuelve la atribución causal de una arquitectura de alta competencia.

## Reproducibilidad

Originales: `results/research/06_A06_1_recovery/original.zip` y `archive/`. Derivados: `audit_saved.json`, `audit_replay.json`, `ci_receipt.json` y `multiplicity_sensitivity.json`. Scripts de auditoría separados: `scripts/audit_A06_1_saved.py` y `scripts/replay_A06_1_checkpoints.py`. Los scripts y la receta originales no se modificaron.

La inferencia reproducida usa CPU y dos threads, sin actualizaciones de pesos. El recibo de replay conserva coincidencia de etiquetas/predicciones/índices, diferencias numéricas máximas, comprobación de tensores finitos, oráculos y experto final. La evaluación de energía física sigue sin medida; no se convierte tiempo o consumo nominal en joules observados.

**Resultado de replay y precisión:** 34/34 evaluaciones del núcleo y 8/8 bancos de routing reproducen exactamente decisiones y selecciones; 44 checkpoints y 448 tensores son finitos. Los oráculos y las métricas del experto final se reprodujeron. La tolerancia inicial de 0,0001 falla en cinco comparaciones continuas de cuatro checkpoints largos: máximo error absoluto de NLL 0,000413895 y de confianza 0,000118315. Se conserva `audit_replay.json` con estado `issues_found`; no se elevó retrospectivamente la tolerancia. En los ocho diagnósticos de esos cuatro checkpoints con MKLDNN activado/desactivado no cambió ninguna predicción, y el máximo cambio absoluto en NLL media fue 0,000000446. La hipótesis de redondeo/plataforma es plausible, pero el cambio de ese backend no aisló la causa. La reproducción numérica entre Linux y Windows no es bit a bit; las decisiones primarias y los recuentos archivados sí concuerdan.

Los 204 controles de intervalos Wilson y bootstrap condicional archivados también pasan sin incidencias. Los efectos factoriales no constituyen una partición única del rendimiento que deba sumar 100 %; con interacciones, esa asignación necesitaría una convención adicional.

Fuentes metodológicas primarias: [NIST, diseños factoriales completos](https://www.itl.nist.gov/div898/handbook/pri/section3/pri3331.htm) y [NIST, intervalos para la media](https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm). La metodología aplicada sigue el protocolo congelado, con las sensibilidades posteriores identificadas.
