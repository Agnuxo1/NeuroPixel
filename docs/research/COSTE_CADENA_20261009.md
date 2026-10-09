# Coste de la cadena NeuroPixel (punto 7, 2026-10-09)

Estado: inventario de lo medido frente a lo no medido. Las cifras de energía de RENDER11 son **parciales** (103 de 105 filas; faltan RGB8_retina_C16, bloque 2, vector_render y render), en espera de la medición encolada en gpuq.

## Corrección previa

Una lectura inicial sobre la carpeta compartida dio por inexistente el recibo de RENDER11 y concluyó que la energía por inferencia no estaba medida. Era un error: la carpeta compartida está en `a036359b`, anterior a `6f1c7b7`. En `origin/main` el recibo existe (`results/research/RENDER11_energy/original_20261009A/receipt.json`) y mide energía sostenida del dispositivo GPU completo por inferencia residente.

## Tabla de componentes

| Componente | Estado | Cifra | Fuente |
|---|---|---|---|
| Entrenamiento NP 30k (GPU), 3 semillas | Medido (tiempo) | 4778,3 / 4780,1 / 4782,7 s | `runs/gpu30k_np_s{0,1,2}/result.json` |
| Entrenamiento NP 30k con lente | Medido (tiempo) | 4866,6 s | `runs/gpu30k_np_lens_s0/result.json` |
| Entrenamiento TF 30k (GPU), 3 semillas | Medido (tiempo) | 1575,3 a 1579,1 s | `runs/gpu30k_tf_s{0,1,2}/result.json` |
| Entrenamiento NP cartográfico y zero-shot 20k | Medido (tiempo) | 4415,8 a 4751,4 s | `runs/zs20k_*/result.json` |
| Entrenamiento NP retrieval y cartográfico | Medido (tiempo) | 3995 a 4205,5 s | `runs/cart_np_*`, `runs/ret_np_*/result.json` |
| CNN baseline (cartográfico y retrieval 57k) | Medido (tiempo) | 157,2 s; 149,1 s | `runs/cart_cnn_s0`, `runs/ret_cnn57k_s0/result.json` |
| NP en CPU (v1, v2_8k, v3_lens) | Medido (tiempo) | 852,3 / 3883,0 / 2637,0 s | `runs/np_v*/result.json` |
| TF en CPU (v1) | Medido (tiempo) | 317,1 s | `runs/tf_v1/result.json` |
| VIS07 NCA, 10 ajustes train + DEV | Medido (tiempo y CPU) | 413,318 s de pared; 826,410 s de CPU | `docs/research/COST_AND_LIMITS_20261009.md:11` |
| VIS07 CNN, 10 ajustes train + DEV | Medido (tiempo y CPU) | 34,237 s de pared; 68,427 s de CPU | `docs/research/COST_AND_LIMITS_20261009.md:11` |
| **RENDER11, energía sostenida por inferencia** | **Medido, parcial (103/105)** | Mediana por llamada: vector_render 0,358 J; render 0,376 J; cuda_graph 0,405 J; cuda_eager 0,419 J; scalar_render 1,555 J | `results/research/RENDER11_energy/original_20261009A/receipt.json` (`gpu_gross_j_per_call`) |
| RENDER11, rango de energía por llamada | Medido, parcial | Entre 0,28 y 12,6 J según caso y método | mismo recibo |
| RENDER11, línea base inactiva | Medido, con dispersión | `idle_before_w` entre 18 y 113 W: la GPU no estuvo en reposo homogéneo | mismo recibo |
| RENDER11, RAM disponible en admisión | Medido | Mínimo 8,03 GiB (guard RAM8 = 8 GiB) | mismo recibo |
| RENDER09 y RENDER08 (render y cuda_graph) | Medido (energía bruta) | 27,795 J (método render, 49 filas); 187,681 J y 10,814 J | `results/research/RENDER09_benchmark/...`, `RENDER08_benchmark/...` |
| RENDER10 (vector_render y render) | Medido | 11,907 J y 1,851 J | `results/research/RENDER10_benchmark/original_20261009A/receipt.json` |
| Compilación y shaders (RENDER09) | Medido (preparación) | 0,688 s en C16_grid8 | `RENDER09_benchmark/original_20261009A/receipt.json` |
| Almacenamiento de checkpoints VIS07 | Medido | 104 861 B por archivo (payload tensor 20 224 B) | `results/research/SPRINT_review/publication_storage_cost.json` |
| Almacenamiento del repositorio publicado | Medido | 1 436 304 202 B lógicos; 1 409 959 791 B únicos | `results/research/SPRINT_review/publication_storage_cost.json` |
| Kaggle (tiempo parcial) | Medido, sin desglose | 13 697 s y 1 573 s | `kaggle/HISTORIAL.md:118` y `:130` |
| Energía del entrenamiento en GPU | **No medido** | — | — |
| Consumo del equipo completo (CPU, RAM, PSU, pared) | **No medido** | — | Sin medidor de pared. El contador "Medidor de energía" de Windows no es legible sin permisos elevados |
| Etiquetas y supervisión humana | **No medido** | — | — |
| Generación y preprocesado de datos | **No medido** | — | Solo código |
| Cachés | **No medido** | — | — |
| Selección DEV separada | **No medido** | — | Incluido en los tiempos de entrenamiento |
| Evaluación durante el entrenamiento | **No medido** (tiempo separado) | — | Campo `eval_every` en `runs/*/result.json` |
| Routing y crecimiento de expertos | **No medido** | — | Solo cualitativo: `COST_AND_LIMITS_20261009.md:17` |
| Transferencias host-GPU y readbacks | **No medido** | — | Excluidos en `COST_AND_LIMITS_20261009.md:9` |
| Coste económico o cuota de GPU | **No medido** | — | — |

## Lo que dice el recibo parcial y sus límites

- Son 103 de 105 filas. Las dos que faltan son las del bloque 2 de RGB8 (vector_render y render). **No se debe sacar conclusión comparativa de la cohorte hasta completarla y verificarla** (105 filas, 77 controles, las 103 originales intactas).
- Alcance: dispositivo GPU completo, incluido el tráfico de fondo y el inactivo. Excluye CPU, RAM, PSU y pared.
- La línea base inactiva varía entre 18 y 113 W, lo que indica que la GPU no estuvo en reposo homogéneo entre filas. Eso limita la comparación entre métodos y hay que tratarlo como incertidumbre.
- El recibo declara que la atribución de energía en bloques cortos **no está validada** por la resolución del contador (`short_block_energy_attribution`). Las cifras por llamada son observaciones crudas, no atribuciones.

## Cifras que no se pueden dar hoy

- Energía total del equipo por inferencia o por entrenamiento: no medido.
- Energía por inferencia del modelo de aplicación completo: no medido. RENDER11 mide el render de la cadena visual, no el modelo entero.
- Coste económico: no medido.
