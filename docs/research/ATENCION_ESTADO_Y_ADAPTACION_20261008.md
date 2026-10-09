# AtenciÃƒÂ³n: estado comprobado y adaptaciÃƒÂ³n posible

8 de octubre de 2026. Respuesta a la consulta del usuario, sin modificar recetas activas.

## QuÃƒÂ© existe

El nÃƒÂºcleo ResearchNCA entrenado en A06.1/QTRAIN/U16 utiliza percepciÃƒÂ³n local3x3, actualizaciones recurrentes compartidas y lectura de una sola celda. No contiene atenciÃƒÂ³n query/key/value. El RelativeTransformer de comparaciÃƒÂ³n sÃƒÂ­ implementa atenciÃƒÂ³n multihead y sesgo relativo2D. El scanner decodifica estados; el gate de crecimiento mezcla expertos. Ninguno equivale a selecciÃƒÂ³n de celdas mediante atenciÃƒÂ³n condicionada por consulta.

Esta conclusiÃƒÂ³n se refiere al nÃƒÂºcleo y recetas locales auditadas. Los cierres externos recuperados todavÃƒÂ­a pendientes de verificaciÃƒÂ³n no se incorporan como implementaciones validadas.

## AdaptaciÃƒÂ³n prospectiva sugerida, no implementada ni evaluada

El mÃ³dulo nuevo obtiene la consulta de la identidad visible en7,6; obtiene claves y valores de estados recurrentes mÃ¡s identidades visibles; pondera esos valores por compatibilidad con la consulta; combina el contexto con el estado de salida7,7 y traduce mediante el mismo diccionario. Excluye como claves la celda de consulta y las vacÃ­as. Implementa cuatro cabezas, dimensiÃ³n total16 y control de pooling uniforme. No modifica los estados recurrentes ni las lentes histÃ³ricas. En este lienzo de hechos visibles no hay motivo automÃƒÂ¡tico para copiar la mÃƒÂ¡scara temporal autoregresiva de un GPT. La elecciÃƒÂ³n de fuentes de claves/valores, canales, cabezas, posiciones, celdas vacÃƒÂ­as y normalizaciÃƒÂ³n debe registrarse antes de entrenar.

La lectura global introduce una nueva ruta de comunicaciÃƒÂ³n y parÃƒÂ¡metros. Deben compararse: nÃƒÂºcleo original, atenciÃƒÂ³n condicionada, pooling uniforme y control de capacidad; incluir atenciÃƒÂ³n sin recurrencia para saber si el nuevo mÃƒÂ³dulo resuelve todo y vuelve prescindible al nÃƒÂºcleo. Ajustar y registrar parÃƒÂ¡metros efectivos, supervisiÃƒÂ³n, exposiciÃƒÂ³n y cÃƒÂ³mputo; igualdad de parÃƒÂ¡metros/updates no iguala todas las capacidades o costes. Mantener semillas completas, consultas opuestas y posiciones variables, sin supervisiÃƒÂ³n extra oculta ni selecciÃƒÂ³n de mÃƒÂ¡scaras.

No se ha demostrado que atenciÃƒÂ³n sea la causa ÃƒÂºnica o la soluciÃƒÂ³n. El Transformer histÃƒÂ³rico tambiÃƒÂ©n quedÃƒÂ³ limitado por optimizaciÃƒÂ³n. Una variante exitosa serÃƒÂ­a evidencia de la nueva receta, nunca reparaciÃƒÂ³n retrospectiva de H1.

Mantener U16 congelada hasta sus12casos/auditorÃƒÂ­as; cualquier variante con atenciÃƒÂ³n necesita otra identidad y protocolo. Controles simbÃƒÂ³licos histÃƒÂ³ricos ya cerrados se reutilizan, no se repiten.

CÃƒÂ³digo comprobado: neuropixel/research/models.py:ResearchNCA y RelativeSelfAttention; neuropixel/model.py:NeuroPixel; neuropixel/research/growth.py:fit_gate; neuropixel/scanner.py:lens.

Referencias primarias: https://arxiv.org/abs/1706.03762 y https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf .

## Preflight terminado, sin resultados de aprendizaje

CÃ³digo separado: neuropixel/research/query_attention.py. AÃ±ade3168parÃ¡metros nominales; hÃ­brido32992frente a29824delnÃºcleo. El control uniforme computa las mismas proyecciones, pero Q/K no influyen en sus pesos; su capacidad efectiva y gradientes difieren y esto debe declararse.

Siete contratos pasaron en CPU2/Python3.12.14/Torch2.6.0: cÃ¡lculo manual de atenciÃ³n escalada y cambio de consulta; exclusiÃ³n de padding/consulta; caso sin hechos finito y contribuciÃ³n cero; promedio uniforme independiente de consulta; gradientes finitos e inputs intactos; estados/frames/lens/local_logits exactamente iguales al nÃºcleo original y decoder compartido; rechazo de entradas incorrectas/no finitas. Los contratos no entrenan modelos cientÃ­ficos ni evalÃºan datasets.

Run37798763696/job113385053310 completed/success; source92435df7e429a1db97514ad71ad16b619771b019; archivoGit104250802afb04a2ab495b77c228848a7243022a. RAM admitida14.5766GiB. La ejecuciÃ³n local fue rechazada con2.1201GiB antes de importarTorch. No se rebajÃ³ el mÃ­nimo8GiB.

Recibos fijados por commit/blob/SHA256 en results/research/query_attention_preflight_review/37798763696/. El ejecutor verificÃ³ tambiÃ©n las19fuentes U16 congeladas sin cambios. Preflight demuestra contratos del mÃ³dulo, no competencia, mejora, originalidad o replicaciÃ³n externa. Ninguna actualizaciÃ³n cientÃ­fica nueva ni acceso al test.

## Control activo de capacidad y acceso a consulta

Implementado aparte en neuropixel/research/query_attention_controls.py. Su MLP local48â†’32â†’48 aporta3152parÃ¡metros y una compuerta dinÃ¡mica de16canales condicionada por la identidad visible de consulta aporta16mÃ¡s: total3168, hÃ­brido32992. No agrega informaciÃ³n global en la cabeza. Todas sus familias de parÃ¡metros reciben gradientes en el contrato construido; no son parÃ¡metros de relleno. Igual nÃºmero nominal no iguala funciÃ³n efectiva, FLOPs, inicializaciÃ³n de cabezas o dificultad de optimizaciÃ³n.

Cuatro pruebas locales pasaron: conteos exactos de local/atenciÃ³n y T1; cÃ¡lculo manual de residual/compuerta; cada parÃ¡metro adicional con gradiente finito no nulo en fixture; estados, frames, lens y logits locales exactamente iguales al nÃºcleo con los mismos pesos, y decoder compartido. RAM admitida8.7868GiB. Python3.13.7/Windows/Torch2.6.0+cu124, tensoresCPU; no es el ejecutorLinux fijado de entrenamiento. Recibo y metadatos en results/research/query_attention_preflight/capacity_receipt.json y capacity_environment.json. El preflight cloud anterior sigue siendo7pruebas de su fuente original; no se le atribuyen estas4pruebas.

Se ha preparado la ampliaciÃ³n del workflow para futuros contratos, pero no se ha despachado otra ejecuciÃ³n. El controlT1 usa una actualizaciÃ³n local y el mismo mÃ³dulo de atenciÃ³n; puede distinguir comunicaciÃ³n global con una codificaciÃ³n local de la necesidad de repetir la regla16veces. No se denomina ausencia total de procesamiento.

La comparaciÃ³n cientÃ­fica deberÃ¡ fijar mÃ©tricas, semillas, actualizaciones, splits, etiquetas, costes y efectos prospectivamente. Los controles permiten separar selecciÃ³n dependiente de consulta, acceso directo a consulta, ampliaciÃ³n nominal de capacidad y repeticiÃ³n de la regla. No permiten atribuir automÃ¡ticamente todo beneficio a una causa Ãºnica. No se ha entrenado ninguna de estas variantes.

## Diferencias de cÃ³mputo que deberÃ¡n declararse

Inventario analÃ­tico parcial por ejemplo de8x8, contando MACs de convoluciones, lineales y productos de atenciÃ³n (sin costes completos de entrenamiento):

| Variante | ParÃ¡metros nominales | MACs de forward sin escuela |
|---|---:|---:|
| OriginalT16 | 29824 | 28198192 |
| AtenciÃ³nT16 | 32992 | 28332336 |
| PoolinguniformeT16, implementaciÃ³n actual | 32992 | 28332336 |
| ControlquerylocalT16 | 32992 | 28202592 |
| AtenciÃ³nT1 | 32992 | 1943856 |

El mÃ³dulo de atenciÃ³n aÃ±ade134144MACs frente a3072delMLPlocal, ademÃ¡s de operaciones y trÃ¡fico de memoria distintos. El poolinguniforme actual calcula tambiÃ©n Q/K/scores antes de sustituir sus pesos; igualdad de MACs nominales no garantiza tiempos iguales. La frecuencia de firing no divide estos MACs: la mÃ¡scara se aplica despuÃ©s de calcular delta.

La escuelaT16 aÃ±ade339968MACs de lectura espacial de forward; faltan su backpropagation y otros costes. El inventario excluye explÃ­citamente energÃ­a, tiempos, memoria, embedding, no linealidades, softmax, gradientes, optimizador, datos, evaluaciÃ³n y archivo. No cierra la tarea6 ni demuestra eficiencia. Sirve para evitar llamar igualdad de cÃ³mputo a igualdad de parÃ¡metros/updates en el futuro estudio. Evidencia: results/research/query_attention_preparation/partial_forward_cost_inventory.json.

Se conserva un primer inventario analítico que omitía1328MACs por ejemplo: los wrappers calculan el readout original para local_logits y después calculan el nuevo readout. El inventario corregido cuenta ambos; no cambia resultados ni recetas. La incidencia y el archivo inicial quedan en query_attention_preparation/forward_inventory_correction.json. El generador reproducible es scripts/inventory_query_attention_forward_cost.py.
