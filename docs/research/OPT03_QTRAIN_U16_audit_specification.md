# Auditoría U16: criterio antes de la cohorte completa

No modifica el plan a4093bb8d559404b9f68587b753f8cea180a1e75d33363a04c1f2c7ebcb22dc0 ni sus19fuentes. Se prepara con el primer caso completo ya auditado y los otros once pendientes; no se usan efectos de la cohorte incompleta.

El recuento exige archivos/manifiestos/parentSHA y baseline8.192 inmutables, digest de admisión igual al preflight real, contador/prefijo nuevo8.193–16.384,64ventanas nuevas, roles balanceados, contextstreams iguales dentro de seed, datos/targets/contrafactuales correctos, recursos≥8GiB y decisiones/NLL finitos con hashes exactos. Métricas desde arrays: tolerancia absoluta1e−12. Gate95%probe/90%validation, tres seeds y endpoint16.384, sin cambios.

Replay separado exige checkpoint real, AdamW/momentos/contador/receta y tresRNG, digest y restauración exactos, regeneración de datos y cada decisión original/queryflip exacta. Tolerancia NLLmedia1e−4 definida antes de concluir; no se subirá tras un fallo. Inference fork conserva los estados de entrenamiento y no cambia archivos. CPU2/RAM8, ninguna nueva actualización ni test; segundo ejecutor interno no es replicación externa.

Siete contrastes prescritos se rederivan mediante pesos factoriales ortogonales. Con factores−1/+1, un efecto principal es2β, una interacción doble4β y la triple8β; coincide con diferencias de niveles y promedios del protocolo. Cuatro fixtures analíticos comprueban respuesta constante, polinomio de siete efectos conocido, cambio presupuesto frente a diferencia pareada directa y rechazo de celdas/valores incompletos. n=3inicializaciones, no n=6etapas ni ocho máscaras como nuevas réplicas. Los ICt95df2 son exploratorios, sin ajuste ni truncamiento.

No selección, efectos ni IC de cohorte hasta12casos y auditorías completas. Artefactos de pruebas en results/research/OPT03_QTRAIN_U16_contrast_contracts/receipt.json. Toda discrepancia se preserva con su alcance; ninguna modifica retrospectivamente H1.


## Recibos finales y estado terminal

El colector scripts/collect_U16_final_receipts.py requiere fuente exacta y completed/success de ambas ejecuciones y de los jobs declarados. Fija los dos recibos por commit/blob/SHA256, exige12casos distintos y correctos, ninguno parcial ni fallido, cero actualizaciones en replay, ningún test, ninguna repetición del prefijo y ninguna inicialización nueva. Un job activo guarda solo una observación de no disponibilidad; no se reinicia y no genera recibo final.

Dieciocho contratos sintéticos verificaron rechazos de casos faltantes/duplicados, receta alterada, test, entrenamiento en replay, falsa independencia externa, replay fallido o en otra ruta y estados activos/fallidos. La consulta real de8deoctubre15:37UTC confirmó ambas ejecuciones activas y rechazó el cierre. Estos contratos no son resultados de modelos ni recibos de cohorte reales.

Una vez completos los12casos: auditar todos los arrays; recopilar los12replays; ejecutar colector final; generar tablas/figuras y recuento estadístico independiente con summarize_OPT03_QTRAIN_U16.py. Este generador exige auditoría completa, ambos recibos terminales y cada replay fijado; no debe producir efectos, intervalos o selección de cohorte parcial. Ningún paso cierra por sí solo la tarea3 amplia.
