# Escalado observado en RENDER11 (punto 8, 2026-10-10)

Fuente: `results/research/RENDER11_energy/recovered_20261009B/independent_summary.json` (cohorte completa, 105 filas, 77 controles). Métrica: mediana de `gpu_gross_j_per_call` sobre tres bloques activos por caso y método. Energía bruta del dispositivo GPU completo (NVML); excluye CPU, RAM, PSU y pared.

## Qué se ha medido

Dos ejes del punto 8 tienen datos: **resolución** (grid 8, 32 y 128 en C16) y **canales** (C16 frente a C48 a grid 8, 32 y 128). No hay datos de pasos, batch, expertos, datos ni otros dispositivos.

## Tabla (mediana, J por llamada)

| Caso | cuda_eager | cuda_graph | scalar_render | vector_render | render |
|---|---|---|---|---|---|
| C16_grid8 | 0,419 | 0,319 | 0,658 | 0,307 | 0,306 |
| C16_grid32 | 0,378 | 0,448 | 0,614 | 0,327 | 0,295 |
| C16_grid128 | 0,641 | 0,825 | 1,555 | 0,485 | 0,519 |
| C48_grid8 | 0,371 | 0,328 | 2,145 | 0,352 | 0,333 |
| C48_grid32 | 0,380 | 0,383 | 2,449 | 0,403 | 0,384 |
| C48_grid128 | 1,818 | 1,901 | 12,019 | 2,554 | 3,546 |
| RGB8_retina_C16 | 0,357 | 0,316 | 0,497 | 0,317 | 0,284 |

## Lectura

1. **Resolución (C16).** Pasar de grid 32 a grid 128 multiplica por 16 los píxeles, pero la energía por llamada de `render` solo sube 1,76× (0,295 → 0,519 J) y la de `vector_render`, 1,48× (0,327 → 0,485 J). El coste es sublineal en este rango.
2. **Canales (C48 frente a C16).** A grid 8 y 32 el cambio es pequeño (1,1–1,3× en `vector_render` y `render`). A grid 128 se dispara: `vector_render` 5,3×, `render` 6,8× y `scalar_render` 7,7×, para 3× más canales. A esta resolución el coste crece más que linealmente con los canales.
3. **Métodos.** Con C16 el render de OpenGL y el vector son los más baratos. `scalar_render` es el peor en todos los casos (hasta 12 J por llamada en C48 grid 128). Con C48 grid 8 y 32 `cuda_eager` y `cuda_graph` son comparables al render.
4. **RGB8_retina_C16** (el caso de las dos filas recuperadas) queda en el rango de C16_grid8 y C16_grid32: `vector_render` 0,317 J y `render` 0,284 J.

## Límites

- n = 3 bloques por caso y método. Las medianas son indicativas.
- La línea base de inactividad varió entre 18 y 113 W. Las cifras brutas incluyen esa variación; la tabla del resumen da el rango ajustado por línea base (`baseline_low_median` y `baseline_high_median`), que hay que citar junto a la mediana bruta.
- Los casos C48 grid 128 usan más memoria y podrían estar limitados por ancho de banda. No hay medida de eso.
- No se puede extrapolar a otras resoluciones, a otros canales ni a otras GPU con estos datos.

## Pendiente del punto 8

Pasos (horizonte), batch, número de expertos, tamaño de datos y otros dispositivos no tienen medidas. Otros dispositivos no están disponibles en este equipo.
