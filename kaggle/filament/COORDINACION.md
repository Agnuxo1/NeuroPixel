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

## FIL-001 cerrada (Claude, 2026-09-27 20:40)

- np_big_reposo_cont 6000 it: validacion cruda 0,3088 -> 0,3335; con posproceso (0,85/120/2) **PQ 0,3753**,
  Dice 0,636, TP 723 FP 655 FN 512. Nuevo mejor individual, pero dentro del ruido del split. Historial entrada 8.
- FIL-002 ms_learned arranco en gpuq a las 20:39 (91.072 params).
- 20:48: ms_learned 1000/8000, PQ 0,2984, Dice 0,6156, TP 670, FP 1023, FN 565. Es
  un punto temprano, no una conclusion; GPU 77-80 C. La cola viva no muestra todavia los controles
  FSR/bilinear, aunque siguen documentados como plan.
- 21:59: entrenamiento ms_learned alcanzo 8000/8000. Mejor provisional PQ 0,3758 en 7000 y final
  0,3750, esencialmente igual a FIL-001 0,3753; no demuestra ganancia sobre el ruido. GPU ya ociosa,
  pero aun no existe result.json y el proceso conserva RAM, por lo que FIL-002 no se marca cerrada.

## Saturacion — revision Codex 2026-09-28 06:11

- Referencia vigente: ms_learned_cont PQ 0,4263; ensamble con big_reposo baja a 0,3973.
- ms_fsr_small (c=24, 16k) cerrado: PQ 0,3670. Mejora a otros FSR, pero no alcanza learned.
- ms_fsr_plus4k completo 4000/4000: mejor crudo provisional 0,3212; falta resultado posprocesado.
- Cola viva: ms_learned_big a continuacion. RAM libre 4,4 GiB: no añadir cargas; dejar que gpuq
  aplique su gate antes del siguiente inicio.
- 06:49: ms_fsr_plus4k cerrado con PQ 0,3583; continuar FSR no cambia la conclusion. ms_learned_big
  activo (247k parametros, 16 GiB de tope); bilinear_plus4k espera. GPU 100 %, 17,4 GiB, 80 C;
  RAM libre 5,8 GiB. Mantener exclusividad y no añadir trabajos.
- 23:05: envio ms_learned -> publico PQ 0,33 (antes 0,31).
- 2026-09-28 15:30: envio ensamble learned_cont + cons_cont (val 0,4293) -> publico PQ 0,35. En curso FIL-007 (filtros).
