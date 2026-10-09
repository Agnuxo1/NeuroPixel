# Evaluación de la investigación acotada de optimización

La investigación de la tarea 3 queda cerrada con resultados mixtos. Esto registra respuestas verificadas y una meta fallida; no declara cumplidas todas las metas positivas ni terminado el proyecto.

| Requisito | Evidencia | Alcance y resultado |
|---|---|---|
| Competencia fuera del desarrollo inicial | DEV04: 12 cuerpos nuevos, tres particiones y cuatro inicializaciones por partición; 24 cabezas | Atención supera los umbrales registrados en los doce casos. Es validación interna, no prueba final ni comparación nueva con Transformer. |
| Capacidad y supervisión | Cabezas atención/local/global con 3168 parámetros activos, mismas etiquetas, presupuestos y secuencias; cuerpos congelados compartidos | Controla esos factores específicos; no iguala espacios funcionales ni FLOPs. |
| Acceso a información global | GLOB05: doce cabezas nuevas con mezcla espacial uniforme y compuerta dependiente de consulta | Atención menos control global: +6.278483 pp, IC t95 [4.107922, 8.449044], semiancho 2.170561 pp. El control global no cumple todos los umbrales de competencia. |
| Incertidumbre de DEV04 | Contraste original registrado, tres medias de políticas, df=2 | +16.243998 pp, IC [10.586385, 21.901612]; semiancho 5.657614 > 5: **meta de precisión fallida**. GLOB05 no repara este fallo. |
| Límites de optimización | Recetas OPT03/MODE/ROLE/QTRAIN/U16 y curvas DEV04 conservadas | Diagnósticos bajo presupuestos finitos. No prueba de convergencia global ni incapacidad universal. No se repitieron entrenamientos cerrados. |
| Integridad y reproducción interna | DEV04 9,732,096 decisiones; GLOB05 3,538,944 decisiones reproducidas exactamente | Procedimientos internos y fuentes fijadas; no replicación por un tercero independiente. |

Los intervalos son exploratorios, sin ajuste por multiplicidad. Las medias de políticas pueden compartir semántica; no representan tres poblaciones externas independientes. No se ha evaluado el test original de 264 composiciones. H1 original sigue sin apoyo. No se establece superioridad general de NCA, mecanismo biológico único, percepción general ni calidad Nobel.

Referencias de evidencia: `DEV04_results.md`, `GLOB05_results.md`, planes originales inmutables, recibos de cierre en `results/research/DEV04_review/37845945011/closure_receipt.json` y `results/research/GLOB05_review/37887762153/closure_receipt.json`. JEV conectado, procedencia verificada, aconseja este cierre acotado; ese consejo no sustituye evidencia experimental. Continúa la tarea 4: auditoría de afirmaciones históricas antes de aceptarlas.
