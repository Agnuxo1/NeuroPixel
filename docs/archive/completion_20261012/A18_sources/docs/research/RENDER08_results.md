# RENDER08: todos los workloads y comparadores

| Workload | CUDA eager ms | CUDA Graph ms | Render vector ms | Mejor CUDA / render |
|---|---:|---:|---:|---:|
| C16_grid8 | 4.3157 | 2.3780 | 11.7644 | 0.2021 |
| C16_grid32 | 3.7682 | 2.4694 | 12.7516 | 0.1937 |
| C16_grid128 | 10.9388 | 10.5937 | 33.2468 | 0.3186 |
| C48_grid8 | 4.0530 | 2.8595 | 35.0129 | 0.0817 |
| C48_grid32 | 3.9564 | 3.9657 | 44.1713 | 0.0896 |
| C48_grid128 | 4.1841 | 4.0195 | 43.9263 | 0.0915 |
| RGB8_retina_C16 | 4.7212 | 2.8151 | 13.8339 | 0.2035 |

Milisegundos medianos dehost por16updates+scanner completo, inputsresidentes, unGPU RTX3090, FP32, todas las decisiones y gates pasados. Cociente>1 indica ventaja deeste renderer sobreel mejorCUDA medido; ≤1 no. Siete bloques técnicos no son siete réplicas científicas. Todos los casos se presentan, sin promedio que oculte los negativos.

El renderer escalar es más lento en los siete casos.

Los bytes de inputs, paridad, todas las filas y costes depreparación/compilación están conservados. Energía de cada inferencia corta **desconocida**: ceros/staleness del contador no equivalen a energía cero. FuenteNVML mide sóloGPU, no pared/CPU/RAM. Nuevas mediciones largas se registran separadas.
