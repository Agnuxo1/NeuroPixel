# RENDER08: medir ejecución por renderizado con equivalencia previa

Hipótesis del usuario: ejecutar la dinámica de neuronas/píxeles mediante renderizado puede ofrecer ventajas en tareas concretas y facilitar observación en tiempo real. No se presupone una ventaja, originalidad histórica ni equivalencia biológica.

El prototipo implementa el núcleo nativo sin alterar sus 43/58 archivos congelados: texturas RGBA32F, percepción depthwise, ReLU local y update residual en tres pases por paso. Retina dilatada, seed, lectura y diccionario también son pases de renderizado. CPU/Mesa passed105checks; extensión retina/decoder y GPU RTX3090 passed136checks. Son pruebas de ingeniería con pesos asignados, no entrenamiento ni competencia aprendida.

Antes de cualquier timing, todos los siete workloads deben conservar estados/logits dentro de1e-4+1e-5|referencia| y todas las decisiones categóricas exactamente. Si alguno falla, no se admite ningún contraste de velocidad completo. No ampliar tolerancias ni seleccionar las geometrías favorables.

Seis workloads usan C16/c_id8/hidden32/vocab11 y C48/c_id16/hidden128/vocab35, cadauno con rejillas8,32,128 cuadradas. Un séptimo es imagen RGB8×8→retina width8→C16→scanner11clases. Pesos y campos sintéticos fijados por seed85200+índice, f2 no nulo; eval/fire1,16updates. No implican una tarea aprendida ni precisión de clasificación. Cada operación produce el estado completo y el scanner completo, igual para ambos backends. Entradas/weights residentes; compilación, preparación, exportación y transferencias se reportan separadas. No llamar a esta latencia coste total de todo el proyecto.

Comparadores: CUDA PyTorch2.6 eager y CUDA Graph del mismo operador nativo, TF32off/determinismo/cuDNNbenchmarkoff. CUDA Graph reduce sobrecoste de lanzamiento Python; es un control relevante para no favorecer al renderer por un baseline evitablemente lento. Warmup Tensor3 y prueba completa render previa; se conservará cualquier indisponibilidad de captura por caso, sin ocultarla.

Siete bloques técnicos porcaso ybackend, tres inferencias porbloque; orden alternado completo hacia delante/reversa. GPUtimerQuery paraGL, CUDAEvents paraTensor, timerhost y sincronización explícita paraambos. Publicar todas las filas, medianas/rangos y comparar con el mejor comparador CUDA disponible porcaso; no IC poblacional ni siete réplicas científicas desde estos bloques técnicos. Una tarjeta/driver/máquina no establece superioridad universal ni resultados de otros dispositivos.

NVMLtotalEnergyConsumption mide julios agregados de la GPU, incluidas actividad de escritorio/idle y contexto; no CPU/RAM/PSU/energía de pared. Se guardan energía bruta, estimación de idle en bracket fijo0.25s y resta firmada sin truncar; discrepancias/ruido quedan visibles. No usarTDP×tiempo ni energía de runner virtual. Estado y fuentes guardados porSHA256. La energía completa de sistema sigue requiriendo una medición distinta.

FIFOcompartida, bootstrapRAM9GiB para mantener guardreal≥8 antes/después deimportar/cada caso/bloque, VRAM4GiB solicitados,CPU2. Sin entrenamiento ni cargas ajenas interrumpidas. Estudio acotado dentro del deadline13:49:31UTC.

Antecedentes primarios: Mordvintsev etal.2020 implementaron NCA conWebGL/GLSL (https://distill.pub/2020/growing-ca/); la ejecución render por sí sola no es originalidad histórica. CUDA Graphs: https://pytorch.org/docs/2.6/notes/cuda.html#cuda-graphs . Las cifras de CHIMERA2025 son afirmaciones de otro manuscrito, no evidencia para este benchmark.
