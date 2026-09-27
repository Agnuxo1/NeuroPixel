# Solar Filament Segmentation — coordinacion

## Estado

- Mejor validacion individual: PQ 0,370; ensamble: 0,377; publico: 0,31.
- Diagnostico: SQ 0,68, RQ 0,56; el cuello de botella es deteccion/instancias.
- `np_big_reposo_cont`: 4000/6000 it; mejor provisional 0,322 en 3000 y 0,315 en 4000, inferior al mejor anterior.
- Dos envios publicos conocidos alrededor de 0,31.

## Cola local

1. `FIL-001`: terminar o cerrar justificadamente la continuacion.
2. `FIL-002`: multiescala 1/4 + reconstruccion; bilinear vs FSR1 vs aprendida.
3. `FIL-003`: validacion cruzada para umbral, area minima y cierre.
4. Mejorar separacion de instancias (espina/watershed) solo despues de medir el multiescala.

## Criterio de avance

La mejora debe superar ruido de split y reportar PQ, SQ, RQ, TP, FP y FN. No basta mejorar Dice.


## FIL-002 (Claude, 2026-09-27 19:36)

- Codigo: `multiscale.py` (EASU+RCAS de FSR 1 portado y diferenciable; bajada aprendida pixel_unshuffle+1x1;
  modo `learned` = estado grueso subido + lienzo fino de 6 pasos). `train_fil.py --ms {bilinear,fsr,learned}`.
- Prueba de humo CPU: los tres modos hacen forward/backward; EASU conserva bordes y constantes.
- En cola gpuq (secuencial): ms_learned, ms_fsr, ms_bilinear; 8000 it, lote 8, recorte 512, rejilla 1/4 (128x128),
  24 pasos gruesos. Referencia np_ret_v2: 8000 it, recorte 256, val PQ 0,367 (6627 s).
- Cautela: el recorte 512 (contexto) cambia a la vez que la escala; `bilinear` es el control interno para FSR/aprendida.
