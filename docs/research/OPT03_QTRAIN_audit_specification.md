# Auditoría independiente de QTRAIN

Preparada mientras el run37728384380/job113151663191 seguía entrenando, antes de inspeccionar resultados completos. No modifica plan20610e354288b48eeb7316367f61da5173fb2a124a01effa93e7d40e61bf3fb5, fuentes, sampling, presupuesto ni gates.

Cada ZIP por caso se valida por recibo/blob Git y SHA256/bytes, fijando el commit de archivo y la fuente científica6a5af5b68aaa9173edf46658dd8f9d16845fbad6. Los originales se preservan separados; el manifiesto por caso se verifica antes de unificar datos de la cohorte. Casos parciales y errores nunca se etiquetan como completos.

El auditor independiente de control verifica por un parser clave–valor propio:16contextos muestreados,64ejemplos,16por rol, selección de query paired/repeated, filler labels, indices/contextos exactos y ausencia de cambios fuera del píxel de consulta. No presume que16draws sean16contextos únicos. Los digests de contextos por ventana deben coincidir entre todos los brazos de una misma seed; las comparaciones se completan al recuperar los cuatro brazos, sin inventar un counterpart pendiente.

Se verifica estado/config/hash del plan,29.824parámetros nominales reportados, RAM≥8GiB,64ventanas de128updates hasta8192, finitud de pérdidas/normas/tiempo, counts de roles por ventana y aritmética respuesta+school_weight×escuela dentro del margen de redondeo float32 escalado. Escuela0 requiere pérdida escolar0. Optimizador y tensores se verificarán desde los checkpoints, no únicamente de su nombre o contadores declarados.

Decisiones de original y queryflip: dense en1024/4096/8192 y ocho máscaras79000–79007 en8192. Se comprueban datos/roles/golds exactos, target del contrafactual por el decoder nativo, máscaras, forma/finitud/NLL, control de escenas no modificadas y recuento de binding nominal, joint-query, global/por rol. Se aplicará el gatecongelado95%jointprobe/90%jointvalidation; no se confundirá con accuracy global o binding de una sola consulta.

Cada peso de endpoint se conserva para replay de inferencia original/queryflip sobre sus datasets exactos, en modo correspondiente, sin gradientes ni optimizer. Se verificará identidad lógica de modelo/dataset/seeds y finitud de estados/opt, número real de steps e hyperparams. Se preservarán diferencias de plataforma; la política prevista es igualdad exacta de decisiones y tolerancia absoluta1e-4deNLLmedia, sin relajación retrospectiva. Para la aritmética sobre arrays archivados se usa tolerancia1e-12. El replay es verificación interna de ejecución, no replicación independiente de otro investigador.

Los arrays almacenados contienen también hashes de estado y contenido del dataset. El replay comparará esos identificadores con los tensors reales del checkpoint y datos regenerados; no aceptará el hash como prueba de sí mismo. El código del auditor no usa el evaluador del estudio para recontar aciertos, joint correctness ni efectos.

Los seis contratos construidos pasaron: controles decontexto/roles válidos; targets o contenido corruptos rechazados; estructura dequeries/tamaños falsos rechazados; exactitud conjunta perfecta versusmodelo que ignoraquery; desacuerdo de controlesconRNGcomún; NLL olabelsCFinválidos. Estos tests verifican el auditor, no un resultado aprendido.

Un recuento parcial no seleccionará condición ni informará efectos con seeds incompletas. Al completar los doce casos se rederivan diferencias paired−repeated, school e interacción dentro de cada seed, luego tres diferencias/SD/ICt95df2. Las máscaras, endpoints, roles o ejemplos no aumentan n. Selección y futuro test requieren la totalidad de los gates y auditorías. La tarea3 amplia y el objetivo completo permanecen abiertos.
