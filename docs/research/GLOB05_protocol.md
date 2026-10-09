# GLOB05: control global activo condicionado por consulta

Diseño prospectivo dentro de la tarea 3 amplia, posterior al cierre de DEV04. **Protocolo final admitido tras 17 contratos sintéticos y los doce padres originales; el plan JSON fijará el freeze antes de ajustar.**

DEV04 conserva +16,244 pp atención–local, ICt95 [10,586;21,902] pp, competencia en los doce candidatos y precisión fallida: semianchura 5,658>5 pp. Este estudio no modifica esa conclusión, su tolerancia ni su cohorte. No añade semillas de cuerpo, no extiende los presupuestos cerrados ni pretende rescatar el intervalo de DEV04.

## Pregunta distinta

¿La selección espacial de atención condicionada por consulta aporta ventaja frente al acceso global a los mismos estados e identidades mediante agregación uniforme y condicionamiento activo por consulta?

El control local anterior tiene menor acceso global. La uniforme READ03 calculaba Q/K que no recibían gradiente útil de respuesta. GLOB05 añade un control con acceso global, consulta visible y todas las familias Q/K/V/salida activas. Es una comparación de funciones de lectura específicas, no equivalencia de capacidad funcional o FLOPs.

## Intervención y ecuación

Conservar los doce cuerpos DEV04 a 16.384 updates, diccionario y decoder congelados. Entrenar una cabeza nueva por cuerpo, `gated_uniform_global`. Dimensiones C48/Cid16/d16, cuatro grupos de anchura cuatro. Q, K, V y salida tienen exactamente las mismas dimensiones, biases y 3.168 parámetros nominales que la lectura de atención. Al inicializar con la misma semilla se exigen tensores iniciales idénticos. Las cuatro familias deben recibir gradiente; no se cuentan parámetros de relleno.

Para las ocho celdas visibles fuera de consulta/PAD, calcular K/V de `[estado;identidad]`, medias uniformes `k̄` y `v̄` y consulta `q`. Por componente y grupo: `g=1+tanh(q⊙k̄/√4)`, `contexto=v̄⊙g`. Proyectar a Δestado y sumar al estado local de salida, como en atención. Los pesos sobre celdas son uniformes e independientes de la consulta; la consulta modula los canales después de agregar. No se introducen etiquetas, asignaciones del generador o pertenencias como características.

La ecuación retiene globalidad y condicionamiento activo, pero limita la selección espacial. La comparación mezcla formas funcionales y aritmética distintas. No atribuir un efecto a una propiedad arquitectónica universal ni a un mecanismo biológico.

## Cohorte, datos y presupuesto fijos

Doce cabezas nuevas sobre los doce cuerpos existentes, particiones 101/102/103 e inicializaciones 400–411. No son doce inicializaciones nuevas de cuerpo. Reutilizar originales DEV04, predicciones y pesos cerrados de atención/local; no ajustar otra vez esas lecturas. Todas las fuentes/datasets/checkpoints originales se verifican por hash y se conservan.

Nueva cabeza: semillas 700–711, train 4.096 ejemplos en grupos de cuatro consultas, 8.192 updates, AdamW LR0,001/betas0,9–0,999/eps1e-8/decay0,0001/clip1. Mismos streams privados 89004/89005 de batches y máscaras, caches de train 94000–94007 y evaluación 95000–95007; firing0,5. Los witnesses de batches/máscaras deben coincidir con los de atención original. Endpoints 1.024/4.096/8.192. Regenerar las mismas tres colecciones head_train/probe/validation y exigir sus bytes/tensores originales; 264 composiciones del test original excluidas y su muestreo bloqueado.

Software exacto DEV04, CPU dos threads, RAM disponible ≥8 GiB en las ventanas y cargas, como máximo dos ejecutores científicos. Nuevas ramas de evidencia y nuevo plan; 43 gobernantes DEV04 intactos. No extraer conclusiones antes de completar/auditar/reproducir los doce controles y sus recibos terminales.

## Análisis previsto

Primario: atención original menos control GLOB05 en exactitud conjunta de validación al endpoint 8.192. Diferencias pareadas por cuerpo, cuatro inicializaciones promediadas dentro de cada política y tres medias con igual peso; ICt95 exploratorio df2 sin truncar ni ajustar. Informar todo el intervalo y los doce pares. Apoyo específico a ventaja: límite inferior >0; precisión descriptiva del nuevo contraste: semianchura ≤5 pp. Su cumplimiento no cambia la precisión fallida del contraste atención–local original ni constituye replicación externa.

Descriptivos: control GLOB05 frente a local original, gates probe95%/validación90% en todos los cuerpos, roles, pérdidas y evolución por endpoint. No probar equivalencia por ausencia de significación. Pesos, máscaras, endpoints, ejemplos y las tres lecturas por cuerpo no son réplicas independientes; las políticas comparten un universo de desarrollo expuesto. Preservar negativos y no ampliar la cohorte para obtener un resultado favorable.

## Admisión y condición de lanzamiento

Inventario de doce padres con anclas/hashes originales y proofs de DEV04; tests de 3.168 parámetros activos, igualdad de inicialización, gradientes Q/K/V/salida, uniformidad/invariancia espacial, consulta/PAD excluidos, freeze, resume exacto, streams compartidos y rechazo de padres/planes alterados. Controlador, transporte, replay independiente y análisis con guard de cohorte completa; preflight real con cero ajustes científicos; entonces freeze por hashes antes del primer ajuste. La preflight 37884990868/fuente 06561e38750085f66829fe1df2b2840de61da3c3 verificó 17 contratos y los doce padres, sin ajustes científicos. El lanzamiento exige todavía plan/manifest/ram y fuentes congelados por hash.
