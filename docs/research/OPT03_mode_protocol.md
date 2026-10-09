# OPT03-MODE: diagnóstico de dinámica de evaluación

Registro prospectivo local NP-OPT03-MODE-20261008-D1. Pregunta: ¿cambia el binding de los modelos OPT03-D al evaluar con máscaras0,5 en lugar de updates completos, y depende ese cambio del firing usado al entrenar?

Es un diagnóstico de desarrollo formulado **después del resultado negativo de OPT03-D**. Se congelan código, datos y contrastes antes de ejecutar las nuevas evaluaciones. No se entrenan modelos, no se abren datos de test, no se modifica H1 ni el gate95%/90% de la fase anterior.

Todos los doce checkpoints finales se conservan. Para cada dataset guardado de probe/validation se seleccionan los primeros128 ejemplos de cada rol,512en total, sin usar aciertos ni pérdidas para seleccionar. Es un subconjunto de desarrollo ya expuesto, no una nueva prueba final. Se mantiene T16 y batch de inferencia256.

Cada modelo se evalúa con updates completos1,0 y con firing0,5 en ocho rollouts individuales. Seeds78000–78007, comunes entre modelos y paneles de igual tamaño. La media es la exactitud esperada de un rollout individual bajo esas ocho máscaras; no es la predicción de un ensemble ni permite elegir la máscara favorable. Las ocho realizaciones no aumentan el número de inicializaciones independientes.

La intervención usa `model.train()` únicamente para activar la máscara de ResearchNCA dentro de `torch.inference_mode`, con firing cambiado solo en inferencia. El modelo no tiene BatchNorm/Dropout, y la fuente congelada usa el flagtraining solo para la máscara. Se verifican hash del archivo de checkpoint y estado antes/después; no se crea optimizador ni se calcula gradiente. Se preservan todos los logits derivados necesarios, decisiones, etiquetas, seeds, métricas y entornos.

Contraste primario diagnóstico: por inicialización se promedia el efecto evaluación0,5−1,0 sobre los dos niveles PAD para cada grupo de firingtrain; luego se calcula interacción(effecttrain0,5−effecttrain1). Informar los tres valores, media, SD e ICt95df2; conservar efectos simples por PAD/modelo y rango de máscaras. Pantalla de consistencia descriptiva: media de interacción≥5pp y los tres valores positivos. No convierte el diagnóstico en confirmatorio, no cambia el gate de competencia y un intervalo amplio que incluye cero no demuestra ausencia o equivalencia.

Firing0,5 y1 alteran dosis esperada y ruido. La evaluación anterior activaba todas las celdas, por lo que el grupo entrenado con0,5 también experimentaba un cambio de dinámica al evaluarse. El diagnóstico solo compara estas intervenciones concretas; no separa automáticamente ruido, dosis, horizonte y generalización, ni identifica por sí solo la causa de baja competencia. El uso de un subconjunto tampoco prueba competencia sobre validation4096.

Recursos: CPU dos threads, RAM disponible≥8GiB al admitir el diagnóstico y entre checkpoints, sin GPU ni entrenamiento. Se persiste un resultado completo por checkpoint; los resultados completos con el mismo plan/entorno se reutilizan. Una evaluación pendiente puede recuperarse conservando incidentes. Originales y derivados están separados en D:.
