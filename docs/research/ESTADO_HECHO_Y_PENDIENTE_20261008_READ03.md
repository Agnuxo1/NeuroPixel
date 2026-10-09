# NeuroPixel: realizado y pendiente tras READ03

Estado del 8 de octubre de 2026. Se distingue el cierre de una receta de la validación del programa científico completo.

## Realizado

- Los hitos originales 1–5 están cerrados dentro de su alcance. H1 original no está soportada: NP 12,393%, Transformer relativo 49,395%, diferencia −37,002 pp, IC95 [−42,030; −31,974]. El presupuesto limitó esos experimentos.
- A06.1 se recuperó y auditó. Se conservaron los efectos negativos, los intervalos imprecisos y cinco discrepancias continuas frente a la tolerancia original. Las decisiones se reproducen; no se amplió la tolerancia retrospectivamente.
- Las dos continuaciones se conciliaron conservando recetas, pesos y semillas separados. Los informes posteriores recuperados todavía requieren verificación científica antes de incorporarlos.
- OPT03-D, MODE, ROLE, QTRAIN y U16 están terminados dentro de sus recetas. U16 continuó los doce estados originales sin repetir sus primeros 8.192 updates; ninguna condición U16 pasó los gates.
- **READ03 está cerrado:** 36 ajustes de cabeza sobre doce estados congelados, 792 comprobaciones de archivos, 11.981 comprobaciones de replay, 10.616.832 decisiones exactas, error máximo de NLL media 0 y 27 recuentos estadísticos independientes sin discrepancias. Ambos ejecutores terminaron correctamente. Código, ZIP originales, checkpoints, tablas, figuras, hashes e incidencias están conservados.
- **La lectura de atención aporta un efecto medido en este diagnóstico:** atención menos control local +54,225 pp, ICt95 exploratorio [37,575; 70,875], promediando las cuatro condiciones dentro de cada semilla. Atención menos uniforme +55,070 pp, IC [38,337; 71,803]. La capacidad nominal coincide, pero la uniforme tiene menos parámetros activos; los límites funcionales y de cómputo están declarados.
- **Competencia alta en desarrollo:** cuatro consultas, escuela 0,3 y atención alcanzaron exactitud conjunta media 99,756%, con 99,762/99,835/99,670% en las tres semillas. Sus probes también superaron 95%. Es la única condición que pasó el gate completo. No es una comparación nueva y justa contra Transformer ni evaluación de test final.
- Los fallos de empaquetado, rutas Windows y cuota de API se resolvieron preservando sus evidencias. No se reiniciaron entrenamientos cerrados ni se modificaron los 32 archivos del experimento después de congelarlos. Los ejecutores y el observador de READ03 han terminado.

## Pendiente, en orden

1. **Completar la tarea 3 amplia:** demostrar reproducción y precisión útiles en particiones e inicializaciones nuevas; establecer controles de capacidad/supervisión y límites de optimización de la receta que se pretenda promover. READ03 usa tres inicializaciones por condición y un único split de desarrollo expuesto. Su resultado positivo no cubre estas exigencias generales.
2. **Tarea 4:** verificar los trabajos existentes sobre separación definitiva de desarrollo/validación/test, generalización, memoria, aprendizaje continuo, reparación y estabilidad. Auditar fuentes, datos, predicciones, checkpoints y controles antes de incorporar conclusiones.
3. **Tarea 5:** completar scanner causal, generalización visual y percepción/símbolos/tiempo con intervenciones y protocolos propios. Los pesos de atención o el binding sintético no prueban por sí solos esas capacidades.
4. **Tarea 6:** medir coste completo, energía real, cómputo omitido y escalabilidad. Evaluar fotónica solo si su hipótesis y coste completo lo justifican.
5. **Tarea 7:** preparar reproducibilidad externa, obtener replicación independiente y redactar el manuscrito acorde con las evidencias y sus límites.
6. **Tarea 8:** demostrar originalidad, predicciones nuevas y utilidad importante confirmada por otros. Todavía no está establecida; completar experimentos internos no garantiza un descubrimiento excepcional ni un Nobel.

La siguiente intervención debe registrarse prospectivamente. Se conservarán READ03 y H1 originales sin reinterpretarlos, y no se repetirá ningún entrenamiento cerrado.

Informe completo: `READ03_results.md`. Estado operativo y referencias: `programme_20261008.json`. Cierre exacto: `results/research/READ03_review/37831051538/closure_receipt.json`.
