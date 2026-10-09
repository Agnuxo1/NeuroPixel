# RENDER09: todos los workloads y comparadores

| Workload | CUDA eager ms | CUDA Graph ms | Render vector ms | Render escalar ms | Mejor CUDA / render |
|---|---:|---:|---:|---:|---:|
| C16_grid8 | 3.9252 | 2.4069 | 2.3169 | 11.6511 | 1.0389 |
| C16_grid32 | 4.2909 | 2.4884 | 2.5381 | 12.8596 | 0.9804 |
| C16_grid128 | 10.2256 | 10.7174 | 6.6585 | 32.9164 | 1.5357 |
| C48_grid8 | 4.0699 | 3.2982 | 4.0005 | 40.0137 | 0.8245 |
| C48_grid32 | 4.8460 | 3.9967 | 4.0835 | 44.6626 | 0.9788 |
| C48_grid128 | 4.3203 | 9.7945 | 5.6004 | 74.9438 | 0.7714 |
| RGB8_retina_C16 | 5.3879 | 2.8597 | 2.6677 | 13.8908 | 1.0720 |

Milisegundos medianos dehost por16updates+scanner completo, inputsresidentes, unGPU RTX3090, FP32, todas las decisiones y gates pasados. Cociente>1 indica ventaja deeste renderer sobreel mejorCUDA medido; ≤1 no. Siete bloques técnicos no son siete réplicas científicas. Todos los casos se presentan, sin promedio que oculte los negativos.

La vectorización mejora frente al escalar y obtiene un punto estimado1.536× enC16/grid128; otras configuraciones no superanCUDA y las mejoras pequeñas pueden depender delruido. No se establece ventaja general, entrenamientográfico ni eficiencia deenergía.

Los bytes de inputs, paridad, todas las filas y costes depreparación/compilación están conservados. Energía de cada inferencia corta **desconocida**: ceros/staleness del contador no equivalen a energía cero. FuenteNVML mide sóloGPU, no pared/CPU/RAM. Nuevas mediciones largas se registran separadas.
