# NeuroPixel: trabajo realizado y trabajo pendiente

Auditoría de continuidad del 7 de octubre de 2026. Cerrar una investigación no equivale a demostrar una ventaja arquitectónica. La calidad de un descubrimiento excepcional depende de resultados sólidos, originales, reproducidos e importantes; completar este programa no garantiza un Premio Nobel.

## Lo realizado

1. **Prior art y límites de contribución:** catorce trabajos primarios revisados, mecanismos especificados y antecedentes reconocidos. Prioridad mundial no establecida.
2. **Hipótesis falsables y protocolo prospectivo:** criterios y comparaciones congelados antes de los resultados.
3. **Controles elementales:** controles exactos resolvieron 51.152/51.152 entradas; se identificaron atajos del generador.
4. **Baselines comparables:** ocho pilotos con NeuroPixel, NCA, ConvGRU y Transformer relativo. No se estableció ventaja de NeuroPixel.
5. **Replicación e incertidumbre:** 28 entrenamientos completos, cero fallos y dos auditorías previamente verificadas. Resultado principal: NP 12,39 %, Transformer 49,39 %, diferencia −37,00 pp, IC95 pareado [−42,03; −31,97] pp. H1 no soportada. Todos los probes quedaron limitados por presupuesto; esto no demuestra incapacidad arquitectónica general. El cierre se reconstruyó por continuidad; los datos crudos históricos del punto 5 no están disponibles en esta copia.
6. **Preparación y ejecución del punto 6 A06.1:** ocho hashes de implementación coinciden con el manifiesto local congelado. Inventarios verificados: 26 entrenamientos, 34 evaluaciones, seis trayectorias, 18 etapas y ocho ajustes de gate. GitHub Actions 37588312570, commit a009f2a9b520a3c7333d092cd3d4f0026b17c335, declara éxito en regresión, núcleo, crecimiento, análisis y archivo. Esto verifica el estado del trabajo remoto, todavía no una auditoría de sus resultados crudos.
7. **Trabajo adicional encontrado en otra rama:** el ledger cloud del commit 3434635b9074d9e834604331b6dc27565eaf713c declara investigaciones 6–15 cerradas y 16 abierta. Se recuperaron informes 6–15. Incluyen gobernanza de splits, correcciones de métricas/PAD, binding complejo, generalización, memoria/interferencia, aprendizaje continuo, contratos de vocabulario/distractores, controles de reparación y dinámica finita. Las conclusiones positivas amplias siguen sin establecerse. No se han incorporado automáticamente estos cierres al ledger local.

## Lo que falta

1. **Cerrar exactamente A06.1:** recuperar el artefacto `neuropixel-item6`, ID 11475707064, de la ejecución 37588312570; verificar hashes, checkpoints, curvas, predicciones, completitud, gates de acceso a evaluación, recontar métricas y contrastes pareados, y redactar la conclusión de los seis componentes. La descarga devuelve HTTP401 incluso al intentar el mecanismo de autenticación disponible; no repetir entrenamientos exitosos por este problema de acceso.
2. **Conciliar las dos continuaciones:** el manifiesto cloud tiene SHA256 20f4e9b4385c95c5f206ac26702bca96ac4232930ea343fd718c4de5bec6048a; A06.1 local tiene 4bc25ed4acce0c95f41d0a595289f854e0330fe237aee1210a8787378cdf6b01. Los inventarios del núcleo son iguales, pero la implementación y descripción de crecimiento no son idénticas. No sustituir resultados entre recetas sin auditoría explícita.
3. **Resolver limitaciones científicas que persisten aunque una investigación esté cerrada:** optimización suficiente; competencia absoluta en binding; mayor precisión estadística que dos semillas; controles de capacidad, retención y supervisión del gate; evaluación nueva realmente no expuesta; generalización a generadores y vocabularios distintos; memoria útil, reparación autónoma y estabilidad más allá de horizontes finitos. Definir cualquier estudio nuevo prospectivamente, sin reinterpretar H1 ni seleccionar sobre test.
4. **Después del cierre local del 6, conciliar y verificar secuencialmente 7–15:** reutilizar informes y experimentos existentes; no repetirlos por defecto. Los controles de implementación o testigos construidos no demuestran por sí solos aprendizaje o competencia.
5. **16:** fidelidad causal del scanner y análisis del espacio nulo.
6. **17–18:** generalización visual y combinación de percepción, símbolos y tiempo.
7. **19–21:** coste computacional completo, energía física medida y cómputo realmente omitido.
8. **22–23:** escalado multidimensional y validación de optimizaciones/variantes.
9. **24, opcional:** factibilidad fotónica, con evidencia física separada de simulación y GPU.
10. **25–27:** paquete reproducible externo, replicación independiente por otros investigadores y manuscrito con límites y resultados negativos.
11. **28–29:** principios explicativos con predicciones nuevas y utilidad/impacto independientes. Esta es la distancia científica principal hacia un descubrimiento excepcional.
12. **30:** considerar reconocimiento externo solo si la evidencia lo justifica; no tratar una nominación como prueba de calidad.

## Resultados exploratorios cloud: referencia separada

El informe cloud 06_results.md declara tying −0,244 pp, escuela +0,928 pp y reinyección −0,122 pp como efectos factoriales medios; todos sus intervalos incluyen cero. T1−T16 = −1,050 pp y T4−T16 = −0,171 pp para entrenamientos separados. Entrenar con daño aporta +0,195 pp en evaluación limpia y +1,050 pp bajo lesión; intervalos también incluyen cero. Scanner frente a routing aleatorio, en el mismo banco, +14,616 pp, IC95 sin ajuste por multiplicidad [13,375;15,857], pero binding absoluto ≈22,88 %. Novedad reprodujo el banco fijo; resonancia quedó −11,784 pp respecto al scanner del banco fijo, con intervalo amplio que incluye cero. Son resultados de esa receta cloud, exploratorios con n=2, no el cierre local de A06.1 ni replicación externa.

## Evidencia y siguiente acción

- Rama local conservada: research/scientific-validation-2026-10-07-continuation; HEAD a009f2a. main y copias históricas no modificados.
- [Ejecución exacta A06.1](https://github.com/Agnuxo1/NeuroPixel/actions/runs/37588312570).
- Informes y ledgers remotos guardados bajo `coord/recovery/status-audit-20261007/`, separados del protocolo y resultados locales.
- JEV consultado con status=connected, provenance=jev, exit_code=0: recuperar y auditar la ejecución exacta antes de cerrar; no duplicar entrenamiento ni sustituir recetas.
- RAM observada: 7,19 GiB; mínimo requerido 8 GiB. No se inició ningún entrenamiento ni se alteraron cargas ajenas.
- Siguiente acción dependiente: recuperar el artefacto A06.1 mediante acceso GitHub válido y auditarlo. El punto 7 no se abre en esta continuación.


## Actualización de 8 de octubre de 2026
El bloqueo HTTP401 de esta instantánea histórica se resolvió mediante la conexión GitHub. A06.1 recuperada, auditada y cerrada como investigación exploratoria limitada. Ver06_A06_1_results.md y results/research/06_A06_1_recovery/closure_receipt.json. No se han demostrado beneficios generales ni competencia suficiente. Se conserva la incidencia numérica de replay. Conciliación de recetas en curso.
