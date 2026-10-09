# SCN06: intervención causal y compresión del scanner

Nueva receta prospectiva: el núcleo y scanner congelados de DEV04 permanecen byte a byte iguales. Se adopta la pregunta del borrador histórico 16, cuyo plan final/auditor nunca se ejecutaron; no se importa su versión distinta del núcleo. Fuente y plan se fijan antes de la ejecución.

Modelo nativo float64, V4/c_id3/C4/hidden8, rejilla1×1, evaluación, dos pasos, input PAD. Todos los pesos asignados explícitamente. Diccionario no-PAD I3, read=[I3|0]. Estado instalado tras paso1: A=(1,0,0,0), B=(1,0,0,2). Regla positiva agrega ReLU(s3) a s1; negativa tiene update cero. Predicción: scanner actual idéntico en ambos estados, siguiente respuesta distinta solo con acoplamiento. Se guardan todos los frames, lentes pre-update, logits/distribución resumida y pesos; auditor NumPy deriva la trayectoria independientemente.

Controles adicionales: vectores (log .6, log .3, log .1, 0) y (log .6, log .1, log .3, 0) dan el mismo top-token/confianza y distintos logits; sentinela PAD−10000 se contrasta con logits no-PAD ordinarios y extremos inferiores. No se cambia la convención histórica.

Un censo separado de cinco matrices read/diccionario autenticadas de 15B calcula rango, espacio nulo y proyectores en float64 sin forward aprendido ni deserialización neuronal. Tolerancia de rango=epsilon×max(dimensiones)×sigma_max; se publica espectro completo y sensibilidad de umbral. Los pesos son los originales de cinco inicializaciones, no cinco nuevas réplicas.

No hay optimización, selección, nueva evaluación del test original ni prueba de causalidad semántica aprendida. Los controles asignados muestran posibilidades y límites del instrumento, no circuitos descubiertos. CPU2 threads, Python3.12.14/Torch2.6.0+cpu/NumPy2.2.6/SciPy1.15.1/psutil6.1.1/Pillow11.3; admisión/fin RAM≥8GiB; un worker científico; máximo global2. Guardar fallos originales y no repetir experimento cerrado.
