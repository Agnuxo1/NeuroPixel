# Historial de entrenamientos y envíos

Una fila por entrenamiento. Para volver a una versión: `git checkout <commit>` (o la etiqueta).

| # | Fecha | Concurso | Ejecución | Línea | Params | Tiempo | Validación | Kaggle | ¿Más entrenamiento? | Commit / etiqueta |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-09-27 | filament | np_ret | L2 | 58480 | 78 min | PQ 0.2306 · Dice 0.5252 | — | no: el mejor punto fue la validación 1 de 4 (sobreajuste o inestabilidad) | 1d3f0d9 |
| 2 | 2026-09-27 | filament | np_ret_v2 | L2 | 58480 | 110 min | PQ 0.3667 · Dice 0.6296 | PQ 0,31 público (≈ puesto 420-454 de 734) | quizá: cerca de la meseta | 41a8275 / filament-v2 |
| 3 | 2026-09-27 | filament | np_pure_v2 | L1 | 29312 | 116 min | PQ 0.3619 · Dice 0.6265 | — | quizá: cerca de la meseta | 41a8275 |
| 4 | 2026-09-27 | digits | np_pure_cpu | L1 | 15776 | 96 min | acierto 0.9825 | 0,98064 público (≈ puesto 524 de 865) | pocas validaciones para saberlo | 094edad / digits-v1 |
| 5 | 2026-09-27 | filament | np_big_reposo | L2 | 136880 | 170 min | PQ 0.37 · Dice 0.6323 | — | sí: seguía subiendo en la última validación | 094edad |
| 6 | 2026-09-27 | digits | np_pure_gpu_v2 | L1 | 29440 | 23 min | acierto 0.9945 | 0,99375 público | pocas validaciones para saberlo | 2b115d6 / digits-v2 |
| 7 | 2026-09-27 | soil | np_votes | L2 | None | 228 min | EMD CV 39.195 (curva media 85.393, uniforme 100.316) | — | pocas validaciones para saberlo | 2b115d6 |
| 8 | 2026-09-27 | soil | np_votes | L2 | 59000.0 | 26 min | EMD CV 39.195 (curva media 85.393, uniforme 100.316) | 98,44 público (3 de 10 muestras; envío 56616585 con espejo; peor que el previo DINOv2 60,80) | pocas validaciones para saberlo | 2b115d6 / soil-v1 |
| 9 | 2026-09-27 | filament | np_big_reposo_cont | L2 | 136880 | 254 min | PQ 0.3753 · Dice 0.6356 | — | sí: seguía subiendo en la última validación | 7da18d1 |
| 10 | 2026-09-27 | filament | ms_learned | L2 | 91072 | 88 min | PQ 0.4061 · Dice 0.6457 | 0,33 público (envío 56619342; antes 0,31) | quizá: cerca de la meseta | 7da18d1 |
| 11 | 2026-09-27 | filament | ms_fsr | L2 | 62592 | 45 min | PQ 0.3485 · Dice 0.619 | — | no: el mejor punto fue la validación 6 de 8 (sobreajuste o inestabilidad) | 7da18d1 |
| 12 | 2026-09-27 | filament | ms_bilinear | L2 | 62592 | 33 min | PQ 0.3427 · Dice 0.6295 | — | sí: seguía subiendo en la última validación | 7da18d1 |
| 13 | 2026-09-28 | filament | ms_learned_cont | L2 | 91072 | 159 min | PQ 0.4263 · Dice 0.6591 | 0,34 público (envío 56627270; antes 0,33) | quizá: cerca de la meseta | 7da18d1 |
| 14 | 2026-09-28 | filament | ms_fsr_long | L2 | 62592 | 75 min | PQ 0.3555 · Dice 0.6283 | — | no: el mejor punto fue la validación 10 de 12 (sobreajuste o inestabilidad) | 7da18d1 |
| 15 | 2026-09-28 | filament | ms_fsr_big | L2 | 140992 | 76 min | PQ 0.3392 · Dice 0.6129 | — | quizá: cerca de la meseta | 7da18d1 |
| 16 | 2026-09-28 | filament | ms_fsr_small | L2 | 41824 | 45 min | PQ 0.367 · Dice 0.6367 | — | quizá: cerca de la meseta | 7da18d1 |
| 17 | 2026-09-28 | filament | ms_fsr_plus4k | L2 | 62592 | 24 min | PQ 0.3583 · Dice 0.6317 | — | sí: seguía subiendo en la última validación | 7da18d1 |
| 18 | 2026-09-28 | filament | ms_learned_big | L2 | 247104 | 171 min | PQ 0.407 · Dice 0.6557 | — | sí: seguía subiendo en la última validación | 7da18d1 |
| 19 | 2026-09-28 | filament | ms_learned_cons | L2 | 91072 | 92 min | PQ 0.4141 · Dice 0.6529 | — | quizá: cerca de la meseta | 7da18d1 |
| 20 | 2026-09-28 | filament | ms_bilinear_plus4k | L2 | 62592 | 24 min | PQ 0.346 · Dice 0.6238 | — | sí: seguía subiendo en la última validación | 7da18d1 |
| 21 | 2026-09-28 | filament | ms_learned_plus4k | L2 | 91072 | 50 min | PQ 0.4114 · Dice 0.647 | — | sí: seguía subiendo en la última validación | 7da18d1 |
| 22 | 2026-09-28 | filament | ms_learned_cons_cont | L2 | 91072 | 180 min | PQ 0.4243 · Dice 0.6621 | — | quizá: cerca de la meseta | 7da18d1 |
| 23 | 2026-09-28 | filament | ms_learned_f3 | L2 | 91072 | 92 min | PQ 0.3799 · Dice 0.6405 | — | sí: seguía subiendo en la última validación | 7da18d1 |
| 24 | 2026-09-28 | filament | ms_learned_sdo | L2 | 91072 | 89 min | PQ 0.4009 · Dice 0.6393 | — | quizá: cerca de la meseta | 7da18d1 |
| 25 | 2026-09-28 | filament | ms_learned_s1 | L2 | 91072 | 91 min | PQ 0.4001 · Dice 0.6459 | — | sí: seguía subiendo en la última validación | 7da18d1 |
| 26 | 2026-09-28 | filament | ms_learned_s2 | L2 | 91072 | 91 min | PQ 0.41 · Dice 0.6535 | — | quizá: cerca de la meseta | 7da18d1 |
| 27 | 2026-09-28 | filament | ms_learned_s1 | L2 | 91072 | 91 min | PQ 0.4001 · Dice 0.6459 | — | sí: seguía subiendo en la última validación | 7da18d1 |
| 28 | 2026-09-28 | filament | ms_learned_s2 | L2 | 91072 | 91 min | PQ 0.41 · Dice 0.6535 | — | quizá: cerca de la meseta | 7da18d1 |
| 29 | 2026-10-01 | filament | ms_learned_cons_40k | L2 | 91072 | 427 min | PQ 0.4242 · Dice 0.6669 | — | no: el mejor punto fue la validación 18 de 20 (sobreajuste o inestabilidad) | d4bc4a1 |
| 30 | 2026-10-01 | filament | cv0_base | L2 | 91072 | 93 min | PQ 0.4072 · Dice 0.6793 | — | sí: seguía subiendo en la última validación | d4bc4a1 |
| 31 | 2026-10-01 | filament | cv0_cons | L2 | 91072 | 104 min | PQ 0.4074 · Dice 0.6762 | — | sí: seguía subiendo en la última validación | d4bc4a1 |

## 1. filament · np_ret (2026-09-27)

- **Qué se hizo:** Retina + lienzo a 1024; cada píxel dice FONDO/FILAMENTO con el diccionario; recortes de 192 (70 % centrados en filamento), peso 8 al filamento, actualización estocástica del 50 %
- **Configuración:** `{"iters": 8000, "batch": 12, "crop": 192, "steps": 24, "c": 48, "hidden": 128, "no_retina": false, "lr": 0.002, "pos_frac": 0.7, "w_pos": 8.0, "eval_every": 2000, "vram_cap": 10, "name": "np_ret"}`
- **Tiempo:** 78 min en RTX 3090 local
- **Validación local:** PQ 0.2306 · Dice 0.5252 · **Kaggle:** sin enviar
- **¿Mejoraría con más entrenamiento?** no: el mejor punto fue la validación 1 de 4 (sobreajuste o inestabilidad)
- **Puntos débiles:** Tras la iteración 2000 sobrepredice filamentos en la imagen completa (47k falsos positivos): desajuste entre recortes sesgados y la imagen entera, y entre actualización estocástica al entrenar y completa al predecir
- **Cambiar / quitar / mejorar:** Bajar el sesgo de recortes (0,4) y el peso (3), recortes de 256, actualización completa (fire 1,0), validar cada 1000; posprocesado con área mínima alta y cierre (mejor: umbral 0,85, área 120, cierre 2 -> PQ 0,2735)
- **Decisión:** Descartada como envío; sustituida por np_ret_v2
- **Volver atrás:** `git checkout 1d3f0d9` · envío: `—`

## 2. filament · np_ret_v2 (2026-09-27)

- **Qué se hizo:** Retina + lienzo (58k params) a 1024; recortes 256 (40 % centrados en filamento), peso 3, actualización completa, 8000 it, lote 8; posprocesado: umbral 0,75, área mínima 120 (×4 a 2048), cierre 2; componentes conexas como instancias
- **Configuración:** `{"iters": 8000, "batch": 8, "crop": 256, "steps": 24, "c": 48, "hidden": 128, "no_retina": false, "lr": 0.002, "pos_frac": 0.4, "w_pos": 3.0, "eval_every": 1000, "vram_cap": 10, "fire_rate": 1.0, "force_gpu": true, "name": "np_ret_v2"}`
- **Tiempo:** 110 min en RTX 3090 local
- **Validación local:** PQ 0.3667 · Dice 0.6296 · **Kaggle:** PQ 0,31 público (≈ puesto 420-454 de 734)
- **¿Mejoraría con más entrenamiento?** quizá: cerca de la meseta
- **Puntos débiles:** Caída de validación 0,367 a público 0,31 (distinta distribución o agregación de PQ); FP 662 y FN 524 en validación; se predice a 1024 y se reescala a 2048 (bordes finos); no se separan filamentos que se tocan; un solo modelo, sin TTA
- **Cambiar / quitar / mejorar:** Probar TTA (4 vistas), predicción a 2048, lienzo multiescala (contexto global), modelo mayor (~105k) y separar instancias por espina; revisar por qué 47 equipos empatan en 0,55 (posible cuaderno público o truco)
- **Decisión:** Primer envío oficial: NeuroPixel certificado en un reto real con 58k parámetros
- **Volver atrás:** `git checkout 41a8275` · envío: `kaggle/filament/runs/np_ret_v2/submission.csv`

## 3. filament · np_pure_v2 (2026-09-27)

- **Qué se hizo:** Lienzo puro sin retina (29k params), misma receta que np_ret_v2
- **Configuración:** `{"iters": 8000, "batch": 8, "crop": 256, "steps": 24, "c": 48, "hidden": 128, "no_retina": true, "lr": 0.002, "pos_frac": 0.4, "w_pos": 3.0, "eval_every": 1000, "vram_cap": 10, "fire_rate": 1.0, "force_gpu": true, "name": "np_pure_v2"}`
- **Tiempo:** 116 min en RTX 3090 local
- **Validación local:** PQ 0.3619 · Dice 0.6265 · **Kaggle:** sin enviar
- **¿Mejoraría con más entrenamiento?** quizá: cerca de la meseta
- **Puntos débiles:** Algo por debajo de la híbrida (FP 704)
- **Cambiar / quitar / mejorar:** Enviar como comparación L1 si sobra cupo; misma lista de mejoras
- **Decisión:** No enviado; demuestra que el lienzo puro percibe filamentos casi igual que el híbrido (0,362 frente a 0,367)
- **Volver atrás:** `git checkout 41a8275` · envío: `—`

## 4. digits · np_pure_cpu (2026-09-27)

- **Qué se hizo:** Lienzo puro sin retina (c=32, hidden 96), 28x28; cada píxel vota el dígito con el diccionario (voto medio + escuela 0,3); reposo: 12-20 pasos + daño 30 %; 3000 it, lote 64, solo CPU
- **Configuración:** `{"iters": 3000, "batch": 64, "steps": 12, "steps_max": 20, "c": 32, "hidden": 96, "lr": 0.002, "threads": 6, "name": "np_pure_cpu"}`
- **Tiempo:** 96 min en CPU (6 hilos)
- **Validación local:** acierto 0.9825 · **Kaggle:** 0,98064 público (≈ puesto 524 de 865)
- **¿Mejoraría con más entrenamiento?** pocas validaciones para saberlo
- **Puntos débiles:** Por debajo de las CNN habituales (≥99 %); se degrada si piensa más de lo entrenado (32 pasos: 83 %)
- **Cambiar / quitar / mejorar:** Rango de pasos más amplio (8-32), más iteraciones en GPU, aumentos (desplazamientos/rotaciones pequeñas), modelo algo mayor
- **Decisión:** Segunda certificación pública: el lienzo puro percibe (98 %) y se autorrepara (50 % del estado borrado: 97,65 %)
- **Volver atrás:** `git checkout 094edad` · envío: `kaggle/digits/runs/np_pure_cpu/submission.csv`

## 5. filament · np_big_reposo (2026-09-27)

- **Qué se hizo:** Retina + lienzo grande (c=96, hidden 256, ~110k params) con reposo (24-36 pasos + daño 30 %), misma receta de datos que np_ret_v2
- **Configuración:** `{"iters": 8000, "batch": 8, "crop": 256, "steps": 24, "c": 96, "hidden": 256, "no_retina": false, "lr": 0.002, "pos_frac": 0.4, "w_pos": 3.0, "eval_every": 1000, "vram_cap": 14.0, "fire_rate": 1.0, "force_gpu": true, "steps_max": 36, "damage_p": 0.5, "name": "np_big_reposo"}`
- **Tiempo:** 170 min en RTX 3090 local
- **Validación local:** PQ 0.37 · Dice 0.6323 · **Kaggle:** sin enviar
- **¿Mejoraría con más entrenamiento?** sí: seguía subiendo en la última validación
- **Puntos débiles:** Apenas mejora al pequeño (0,370 frente a 0,367): el tamaño no es el cuello de botella; validación oscilante
- **Cambiar / quitar / mejorar:** Priorizar posprocesado y separación de instancias (espina/watershed) y contexto multiescala antes que más tamaño
- **Decisión:** Candidato para combinar; no enviado solo
- **Volver atrás:** `git checkout 094edad` · envío: `—`

## 6. digits · np_pure_gpu_v2 (2026-09-27)

- **Qué se hizo:** Lienzo puro (c=48, hidden 128) en GPU; reposo con 8-32 pasos + daño 30 %; desplazamientos de ±2 px; 8000 it, lote 128; cada píxel vota
- **Configuración:** `{"iters": 8000, "batch": 128, "steps": 8, "steps_max": 32, "c": 48, "hidden": 128, "lr": 0.002, "threads": 6, "name": "np_pure_gpu_v2", "device": "cuda", "aug": 2}`
- **Tiempo:** 23 min en RTX 3090 local
- **Validación local:** acierto 0.9945 · **Kaggle:** 0,99375 público
- **¿Mejoraría con más entrenamiento?** pocas validaciones para saberlo
- **Puntos débiles:** Aún por debajo de las mejores CNN (≥99,6 %)
- **Cambiar / quitar / mejorar:** Más aumentos (rotación/escala), promedio de vistas, más iteraciones
- **Decisión:** Mejora de 0,98064 a 0,99375 corrigiendo el punto débil (rango de pasos): 50 % del estado borrado -> 99,5 %, estable 16-32 pasos
- **Volver atrás:** `git checkout 2b115d6` · envío: `kaggle/digits/runs/np_pure_gpu_v2/submission.csv`

## 7. soil · np_votes (2026-09-27)

- **Qué se hizo:** Retina + lienzo (c=48, hidden 128); cada píxel vota 11 tramos de tamaño; media de votos = curva acumulada; CV 6 pliegues dejando fuera muestras enteras (24 muestras, 127 fotos); SIN espejo en entrenamiento (añadido después)
- **Configuración:** `{"iters": 3000, "segundos": 13697}`
- **Tiempo:** 228 min en RTX 3090 local (compartida con FIL-001)
- **Validación local:** EMD CV 39.195 (curva media 85.393, uniforme 100.316) · **Kaggle:** sin enviar
- **¿Mejoraría con más entrenamiento?** pocas validaciones para saberlo
- **Puntos débiles:** Solo 24 muestras (±ruido alto por pliegue 35-49); CV aleatoria por muestra mezcla teléfonos: el test es iPhone y no se ha medido el cambio de cámara; la CV no incluye el espejo que usa el envío
- **Cambiar / quitar / mejorar:** CV dejando fuera un teléfono; medir el espejo en CV; comparar con textura clásica/CNN pequeña; más iteraciones
- **Decisión:** Supera con claridad la curva media (39,2 frente a 85,4, -54 %) y la uniforme (100,3): pasa el gate de CV. SOIL-002 (Codex) audita split y crea el envío
- **Volver atrás:** `git checkout 2b115d6` · envío: `—`

## 8. soil · np_votes (2026-09-27)

- **Qué se hizo:** Retina + lienzo; cada píxel vota uno de 11 tamaños de grano; la media de votos es la curva acumulada; pérdida EMD logarítmica + escuela; fotos reescaladas a 6 px/mm con la tabla de ppm; rotaciones + espejo; 3000 it
- **Configuración:** `{"iters": 3000, "test_muestras": 10, "test_sin_fotos": [], "segundos": 1573, "best_postproc": {"PQ": null}, "best_val_acc": "EMD CV 39.195 (curva media 85.393)", "params": 59000.0}`
- **Tiempo:** 26 min en RTX 3090 local
- **Validación local:** EMD CV 39.195 (curva media 85.393, uniforme 100.316) · **Kaggle:** 98,44 público (3 de 10 muestras; envío 56616585 con espejo; peor que el previo DINOv2 60,80)
- **¿Mejoraría con más entrenamiento?** pocas validaciones para saberlo
- **Puntos débiles:** No generaliza al test: CV 39,2 (mismo origen) frente a 98,4 en público. Cambio de cámara (entrenamiento Motorola/Samsung, test solo iPhone) y de procedencia de los suelos (test HPC_*); solo 24 muestras
- **Cambiar / quitar / mejorar:** Normalizar color y nitidez por cámara, aumentos de color y desenfoque, rasgos preentrenados como retina (DINOv2 ya en disco), validación que deje fuera un teléfono entero, mezclar con la curva media
- **Decisión:** Fallo registrado; para la clasificación final, seleccionar en Kaggle el mejor envío previo de Fran (60,8). PÚBLICO 98,44 ≈ uniforme: la CV (39,2) no transfiere al iPhone; revisar en SOIL-002
- **Volver atrás:** `git checkout 2b115d6` · envío: `kaggle/soil/runs/np_votes/submission.csv`

## 9. filament · np_big_reposo_cont (2026-09-27)

- **Qué se hizo:** Continuación de np_big_reposo (c=96, hidden 256) 6000 it más con lr 5e-4, misma receta (recorte 256, reposo 24-36 pasos + daño)
- **Configuración:** `{"iters": 6000, "batch": 8, "crop": 256, "steps": 24, "c": 96, "hidden": 256, "no_retina": false, "lr": 0.0005, "pos_frac": 0.4, "w_pos": 3.0, "eval_every": 1000, "vram_cap": 14.0, "fire_rate": 1.0, "force_gpu": true, "steps_max": 36, "damage_p": 0.5, "init": "np_big_reposo", "name": "np_big_reposo_cont"}`
- **Tiempo:** 254 min en RTX 3090 local
- **Validación local:** PQ 0.3753 · Dice 0.6356 · **Kaggle:** sin enviar
- **¿Mejoraría con más entrenamiento?** sí: seguía subiendo en la última validación
- **Puntos débiles:** Mejora mínima sobre el padre (0,375 frente a 0,370) dentro del ruido del split (~±0,03); FP 655 / FN 512: la detección (RQ) sigue siendo el cuello de botella
- **Cambiar / quitar / mejorar:** No seguir alargando esta línea: cambiar de eje (multiescala FIL-002, posproceso por pliegues FIL-003, separación de instancias)
- **Decisión:** Cerrada FIL-001; mejor modelo individual en validación (PQ 0,3753) y candidato para el ensamble; no se envía solo
- **Volver atrás:** `git checkout 7da18d1` · envío: `—`

## 10. filament · ms_learned (2026-09-27)

- **Qué se hizo:** FIL-002: lienzo multiescala. Retina a 1024 -> bajada aprendida (pixel_unshuffle+1x1) a 1/4 -> 24 pasos del lienzo en la rejilla gruesa -> estado subido bilineal + lienzo fino de 6 pasos (regla propia) a resolución completa. Recorte 512, lote 8, 8000 it, semilla 0
- **Configuración:** `{"iters": 8000, "batch": 8, "crop": 512, "steps": 24, "c": 48, "hidden": 128, "no_retina": false, "lr": 0.002, "pos_frac": 0.4, "w_pos": 3.0, "eval_every": 1000, "vram_cap": 10.0, "fire_rate": 1.0, "force_gpu": true, "steps_max": 0, "damage_p": 0.5, "init": null, "ms": "learned", "scale": 4, "steps_fine": 6, "name": "ms_learned"}`
- **Tiempo:** 88 min en RTX 3090 local
- **Validación local:** PQ 0.4061 · Dice 0.6457 · **Kaggle:** 0,33 público (envío 56619342; antes 0,31)
- **¿Mejoraría con más entrenamiento?** quizá: cerca de la meseta
- **Puntos débiles:** Recorte (512) y escala cambian a la vez frente a np_ret_v2; FN 525 sigue alto; una semilla; posproceso ajustado en el mismo split
- **Cambiar / quitar / mejorar:** Más iteraciones (seguía subiendo); comparar con ms_fsr y ms_bilinear; control recorte 512 sin multiescala; CV del posproceso (FIL-003); ensamble con big_reposo_cont
- **Decisión:** Mejor modelo de filamentos: val PQ 0,4061 (crudo 0,375) frente a 0,3753 del mejor previo, con 1/5 del tiempo por iteración; candidato a envío
- **Volver atrás:** `git checkout 7da18d1` · envío: `kaggle/filament/runs/ms_learned/submission.csv`

## 11. filament · ms_fsr (2026-09-27)

- **Qué se hizo:** FIL-002: como ms_learned pero la salida gruesa (probabilidad a 1/4) se sube con FSR 1 portado (EASU x4 + RCAS), sin lienzo fino; 8000 it, recorte 512, semilla 0
- **Configuración:** `{"iters": 8000, "batch": 8, "crop": 512, "steps": 24, "c": 48, "hidden": 128, "no_retina": false, "lr": 0.002, "pos_frac": 0.4, "w_pos": 3.0, "eval_every": 1000, "vram_cap": 10.0, "fire_rate": 1.0, "force_gpu": true, "steps_max": 0, "damage_p": 0.5, "init": null, "ms": "fsr", "scale": 4, "steps_fine": 6, "name": "ms_fsr"}`
- **Tiempo:** 45 min en RTX 3090 local
- **Validación local:** PQ 0.3485 · Dice 0.619 · **Kaggle:** sin enviar
- **¿Mejoraría con más entrenamiento?** no: el mejor punto fue la validación 6 de 8 (sobreajuste o inestabilidad)
- **Puntos débiles:** FN 599: sin información fina la segmentación de filamentos finos se pierde; EASU está pensado para x2-x3, no x4
- **Cambiar / quitar / mejorar:** Descartar FSR sin parámetros como reconstrucción; si acaso FSR como inicialización del lienzo fino
- **Decisión:** Peor que ms_learned (0,3485 frente a 0,4061): la superresolución aprendida con la imagen a resolución completa es la que aporta; FSR no
- **Volver atrás:** `git checkout 7da18d1` · envío: `—`

## 12. filament · ms_bilinear (2026-09-27)

- **Qué se hizo:** FIL-002 control: como ms_fsr pero la probabilidad a 1/4 se sube con interpolación bilineal; 8000 it, recorte 512, semilla 0
- **Configuración:** `{"iters": 8000, "batch": 8, "crop": 512, "steps": 24, "c": 48, "hidden": 128, "no_retina": false, "lr": 0.002, "pos_frac": 0.4, "w_pos": 3.0, "eval_every": 1000, "vram_cap": 10.0, "fire_rate": 1.0, "force_gpu": true, "steps_max": 0, "damage_p": 0.5, "init": null, "ms": "bilinear", "scale": 4, "steps_fine": 6, "name": "ms_bilinear"}`
- **Tiempo:** 33 min en RTX 3090 local
- **Validación local:** PQ 0.3427 · Dice 0.6295 · **Kaggle:** sin enviar
- **¿Mejoraría con más entrenamiento?** sí: seguía subiendo en la última validación
- **Puntos débiles:** FN 604: sin detalle fino se pierden filamentos; peor que la línea base a resolución completa (0,367)
- **Cambiar / quitar / mejorar:** Nada: es el control
- **Decisión:** Control: bilineal 0,3427 < FSR 0,3485 < learned 0,4061. El recorte 512 y la escala 1/4 solos no mejoran; la ganancia viene del lienzo fino aprendido
- **Volver atrás:** `git checkout 7da18d1` · envío: `—`

## 13. filament · ms_learned_cont (2026-09-28)

- **Qué se hizo:** Continuación de ms_learned 16000 it, lr 1e-3 (OneCycle), recorte 512
- **Configuración:** `{"iters": 16000, "batch": 8, "crop": 512, "steps": 24, "c": 48, "hidden": 128, "no_retina": false, "lr": 0.001, "pos_frac": 0.4, "w_pos": 3.0, "eval_every": 2000, "vram_cap": 12.0, "fire_rate": 1.0, "force_gpu": true, "steps_max": 0, "damage_p": 0.5, "init": "ms_learned", "ms": "learned", "scale": 4, "steps_fine": 6, "name": "ms_learned_cont"}`
- **Tiempo:** 159 min en RTX 3090 local
- **Validación local:** PQ 0.4263 · Dice 0.6591 · **Kaggle:** 0,34 público (envío 56627270; antes 0,33)
- **¿Mejoraría con más entrenamiento?** quizá: cerca de la meseta
- **Puntos débiles:** Crudo oscila 0,346-0,394; FN ~500 persiste
- **Cambiar / quitar / mejorar:** Ensamblar con big_reposo_cont; comparar con +4k (SAT-001b)
- **Decisión:** Nuevo mejor: val PQ 0,4263 (antes 0,4061); la línea aprendida sigue mejorando con más iteraciones (tortuga). Ensamble con big_reposo_cont peor en val (0,3973): enviado solo
- **Volver atrás:** `git checkout 7da18d1` · envío: `kaggle/filament/runs/ms_learned_cont/submission.csv`

## 14. filament · ms_fsr_long (2026-09-28)

- **Qué se hizo:** FIL-004: FSR 24000 it (mismo tiempo que ms_learned), c=48
- **Configuración:** `{"iters": 24000, "batch": 8, "crop": 512, "steps": 24, "c": 48, "hidden": 128, "no_retina": false, "lr": 0.002, "pos_frac": 0.4, "w_pos": 3.0, "eval_every": 2000, "vram_cap": 12.0, "fire_rate": 1.0, "force_gpu": true, "steps_max": 0, "damage_p": 0.5, "init": null, "ms": "fsr", "scale": 4, "steps_fine": 6, "name": "ms_fsr_long"}`
- **Tiempo:** 75 min en RTX 3090 local
- **Validación local:** PQ 0.3555 · Dice 0.6283 · **Kaggle:** sin enviar
- **¿Mejoraría con más entrenamiento?** no: el mejor punto fue la validación 10 de 12 (sobreajuste o inestabilidad)
- **Puntos débiles:** Crudo se estanca en ~0,31 desde 14k it; FN 580
- **Cambiar / quitar / mejorar:** FSR puro no escala con iteraciones; probar FSR como guía del lienzo fino
- **Decisión:** 0,3555 frente a 0,3485 a 8k: +0,007 con 3x iteraciones; satura pronto (liebre)
- **Volver atrás:** `git checkout 7da18d1` · envío: `—`

## 15. filament · ms_fsr_big (2026-09-28)

- **Qué se hizo:** FIL-004: FSR con c=96, hidden 256, 16000 it
- **Configuración:** `{"iters": 16000, "batch": 8, "crop": 512, "steps": 24, "c": 96, "hidden": 256, "no_retina": false, "lr": 0.002, "pos_frac": 0.4, "w_pos": 3.0, "eval_every": 2000, "vram_cap": 12.0, "fire_rate": 1.0, "force_gpu": true, "steps_max": 0, "damage_p": 0.5, "init": null, "ms": "fsr", "scale": 4, "steps_fine": 6, "name": "ms_fsr_big"}`
- **Tiempo:** 76 min en RTX 3090 local
- **Validación local:** PQ 0.3392 · Dice 0.6129 · **Kaggle:** sin enviar
- **¿Mejoraría con más entrenamiento?** quizá: cerca de la meseta
- **Puntos débiles:** Peor que el pequeño (FN 618); más capacidad en la rejilla gruesa no ayuda
- **Cambiar / quitar / mejorar:** El cuello de botella es el detalle fino, no la capacidad gruesa
- **Decisión:** 0,3392 < 0,3485 (c=48, 8k): el tamaño no mejora FSR
- **Volver atrás:** `git checkout 7da18d1` · envío: `—`

## 16. filament · ms_fsr_small (2026-09-28)

- **Qué se hizo:** SAT-001: FSR c=24, hidden 64, 16000 it
- **Configuración:** `{"iters": 16000, "batch": 8, "crop": 512, "steps": 24, "c": 24, "hidden": 64, "no_retina": false, "lr": 0.002, "pos_frac": 0.4, "w_pos": 3.0, "eval_every": 2000, "vram_cap": 12.0, "fire_rate": 1.0, "force_gpu": true, "steps_max": 0, "damage_p": 0.5, "init": null, "ms": "fsr", "scale": 4, "steps_fine": 6, "name": "ms_fsr_small"}`
- **Tiempo:** 45 min en RTX 3090 local
- **Validación local:** PQ 0.367 · Dice 0.6367 · **Kaggle:** sin enviar
- **¿Mejoraría con más entrenamiento?** quizá: cerca de la meseta
- **Puntos débiles:** FN 572
- **Cambiar / quitar / mejorar:** Probar FSR c=24 más largo y FSR como guía del lienzo fino
- **Decisión:** Mejor FSR: 0,367 (c=24) > 0,3555 (c=48, 24k) > 0,3392 (c=96): en FSR menos capacidad gruesa generaliza mejor
- **Volver atrás:** `git checkout 7da18d1` · envío: `—`

## 17. filament · ms_fsr_plus4k (2026-09-28)

- **Qué se hizo:** SAT-001b: ms_fsr (8k) + 4000 it lr 5e-4
- **Configuración:** `{"iters": 4000, "batch": 8, "crop": 512, "steps": 24, "c": 48, "hidden": 128, "no_retina": false, "lr": 0.0005, "pos_frac": 0.4, "w_pos": 3.0, "eval_every": 1000, "vram_cap": 12.0, "fire_rate": 1.0, "force_gpu": true, "steps_max": 0, "damage_p": 0.5, "init": "ms_fsr", "ms": "fsr", "scale": 4, "steps_fine": 6, "name": "ms_fsr_plus4k"}`
- **Tiempo:** 24 min en RTX 3090 local
- **Validación local:** PQ 0.3583 · Dice 0.6317 · **Kaggle:** sin enviar
- **¿Mejoraría con más entrenamiento?** sí: seguía subiendo en la última validación
- **Puntos débiles:** Crudo plano 0,316-0,321
- **Cambiar / quitar / mejorar:** —
- **Decisión:** +0,010 (0,3485 -> 0,3583): FSR casi saturado a 8k
- **Volver atrás:** `git checkout 7da18d1` · envío: `—`

## 18. filament · ms_learned_big (2026-09-28)

- **Qué se hizo:** SAT-001: ms_learned con c=96, hidden 256, 8000 it (2x tiempo que c=48)
- **Configuración:** `{"iters": 8000, "batch": 8, "crop": 512, "steps": 24, "c": 96, "hidden": 256, "no_retina": false, "lr": 0.002, "pos_frac": 0.4, "w_pos": 3.0, "eval_every": 2000, "vram_cap": 16.0, "fire_rate": 1.0, "force_gpu": true, "steps_max": 0, "damage_p": 0.5, "init": null, "ms": "learned", "scale": 4, "steps_fine": 6, "name": "ms_learned_big"}`
- **Tiempo:** 171 min en RTX 3090 local
- **Validación local:** PQ 0.407 · Dice 0.6557 · **Kaggle:** sin enviar
- **¿Mejoraría con más entrenamiento?** sí: seguía subiendo en la última validación
- **Puntos débiles:** Mismo PQ que el pequeño con el doble de tiempo; FN 518
- **Cambiar / quitar / mejorar:** No crecer el tamaño; invertir en iteraciones (tortuga) y en separación de instancias
- **Decisión:** 0,407 frente a 0,4061 (c=48, 8k): el tamaño no aporta; las iteraciones sí (0,4263 con +16k)
- **Volver atrás:** `git checkout 7da18d1` · envío: `—`

## 19. filament · ms_learned_cons (2026-09-28)

- **Qué se hizo:** FIL-005: ms_learned (8k, recorte 512) con objetivo de CONSENSO suave (media de anotadores por imagen; muestreo uniforme por imagen)
- **Configuración:** `{"iters": 8000, "batch": 8, "crop": 512, "steps": 24, "c": 48, "hidden": 128, "no_retina": false, "lr": 0.002, "pos_frac": 0.4, "w_pos": 3.0, "eval_every": 1000, "vram_cap": 12.0, "fire_rate": 1.0, "force_gpu": true, "steps_max": 0, "damage_p": 0.5, "init": null, "ms": "learned", "scale": 4, "steps_fine": 6, "consensus": true, "name": "ms_learned_cons"}`
- **Tiempo:** 92 min en RTX 3090 local
- **Validación local:** PQ 0.4141 · Dice 0.6529 · **Kaggle:** sin enviar
- **¿Mejoraría con más entrenamiento?** quizá: cerca de la meseta
- **Puntos débiles:** Mejora pequeña (+0,008) que puede ser ruido de semilla; FN 513
- **Cambiar / quitar / mejorar:** Esperar varianza de semillas (Lightning); si es real, combinar consenso + continuación larga
- **Decisión:** 0,4141 frente a 0,4061 (mismo presupuesto): pendiente de la varianza entre semillas
- **Volver atrás:** `git checkout 7da18d1` · envío: `—`

## 20. filament · ms_bilinear_plus4k (2026-09-28)

- **Qué se hizo:** SAT-001b: ms_bilinear (8k) + 4000 it lr 5e-4
- **Configuración:** `{"iters": 4000, "batch": 8, "crop": 512, "steps": 24, "c": 48, "hidden": 128, "no_retina": false, "lr": 0.0005, "pos_frac": 0.4, "w_pos": 3.0, "eval_every": 1000, "vram_cap": 12.0, "fire_rate": 1.0, "force_gpu": true, "steps_max": 0, "damage_p": 0.5, "init": "ms_bilinear", "ms": "bilinear", "scale": 4, "steps_fine": 6, "name": "ms_bilinear_plus4k"}`
- **Tiempo:** 24 min en RTX 3090 local
- **Validación local:** PQ 0.346 · Dice 0.6238 · **Kaggle:** sin enviar
- **¿Mejoraría con más entrenamiento?** sí: seguía subiendo en la última validación
- **Puntos débiles:** Crudo 0,29-0,31 plano
- **Cambiar / quitar / mejorar:** —
- **Decisión:** 0,3427 -> 0,346: saturado
- **Volver atrás:** `git checkout 7da18d1` · envío: `—`

## 21. filament · ms_learned_plus4k (2026-09-28)

- **Qué se hizo:** SAT-001b: ms_learned (8k) + 4000 it lr 5e-4
- **Configuración:** `{"iters": 4000, "batch": 8, "crop": 512, "steps": 24, "c": 48, "hidden": 128, "no_retina": false, "lr": 0.0005, "pos_frac": 0.4, "w_pos": 3.0, "eval_every": 1000, "vram_cap": 12.0, "fire_rate": 1.0, "force_gpu": true, "steps_max": 0, "damage_p": 0.5, "init": "ms_learned", "ms": "learned", "scale": 4, "steps_fine": 6, "consensus": false, "seed": 0, "name": "ms_learned_plus4k"}`
- **Tiempo:** 50 min en RTX 3090 local
- **Validación local:** PQ 0.4114 · Dice 0.647 · **Kaggle:** sin enviar
- **¿Mejoraría con más entrenamiento?** sí: seguía subiendo en la última validación
- **Puntos débiles:** Con lr bajo solo +0,005; con +16k y lr 1e-3 llegó a 0,4263
- **Cambiar / quitar / mejorar:** Para seguir mejorando hace falta presupuesto largo con lr más alto
- **Decisión:** 0,4061 -> 0,4114 (+4k), 0,4263 (+16k): sigue aprendiendo pero lento
- **Volver atrás:** `git checkout 7da18d1` · envío: `—`

## 22. filament · ms_learned_cons_cont (2026-09-28)

- **Qué se hizo:** FIL-005: ms_learned_cons + 16000 it lr 1e-3 (consenso suave)
- **Configuración:** `{"iters": 16000, "batch": 8, "crop": 512, "steps": 24, "c": 48, "hidden": 128, "no_retina": false, "lr": 0.001, "pos_frac": 0.4, "w_pos": 3.0, "eval_every": 2000, "vram_cap": 12.0, "fire_rate": 1.0, "force_gpu": true, "steps_max": 0, "damage_p": 0.5, "init": "ms_learned_cons", "ms": "learned", "scale": 4, "steps_fine": 6, "consensus": true, "seed": 0, "name": "ms_learned_cons_cont"}`
- **Tiempo:** 180 min en RTX 3090 local
- **Validación local:** PQ 0.4243 · Dice 0.6621 · **Kaggle:** sin enviar
- **¿Mejoraría con más entrenamiento?** quizá: cerca de la meseta
- **Puntos débiles:** Empata con learned_cont sin consenso (0,4243 frente a 0,4263)
- **Cambiar / quitar / mejorar:** Ensamblar ambos (modelos distintos, errores quizá no correlados); medir varianza de semillas
- **Decisión:** El consenso no aporta tras entrenamiento largo; candidato solo para ensamble
- **Volver atrás:** `git checkout 7da18d1` · envío: `—`

## 23. filament · ms_learned_f3 (2026-09-28)

- **Qué se hizo:** FIL-007: ms_learned (8k) con retina sobre [limbo corregido, Sato, DoG] en lugar de 3 copias del gris (idea de Fran: filtros de otros campos)
- **Configuración:** `{"iters": 8000, "batch": 8, "crop": 512, "steps": 24, "c": 48, "hidden": 128, "no_retina": false, "lr": 0.002, "pos_frac": 0.4, "w_pos": 3.0, "eval_every": 1000, "vram_cap": 12.0, "fire_rate": 1.0, "force_gpu": true, "steps_max": 0, "damage_p": 0.5, "init": null, "ms": "learned", "scale": 4, "steps_fine": 6, "consensus": false, "seed": 0, "filters": true, "name": "ms_learned_f3"}`
- **Tiempo:** 92 min en RTX 3090 local
- **Validación local:** PQ 0.3799 · Dice 0.6405 · **Kaggle:** sin enviar
- **¿Mejoraría con más entrenamiento?** sí: seguía subiendo en la última validación
- **Puntos débiles:** FN 561 (peor); el filtro por píxel no ayuda a una red que ya aprende sus filtros; Sato/DoG normalizados por imagen pueden amplificar ruido
- **Cambiar / quitar / mejorar:** Descartar como sustituto; como mucho añadir un canal extra conservando el gris
- **Decisión:** PEOR: 0,3799 frente a 0,4061 sin filtros. La retina ya aprende lo que dan los filtros clásicos
- **Volver atrás:** `git checkout 7da18d1` · envío: `—`

## 24. filament · ms_learned_sdo (2026-09-28)

- **Qué se hizo:** FIL-008: ms_learned (8k) con retina sobre canales REALES coetaneos [Halfa GONG, He II 30,4 nm SDO/AIA, magnetograma SDO/HMI] (idea de Fran)
- **Configuración:** `{"iters": 8000, "batch": 8, "crop": 512, "steps": 24, "c": 48, "hidden": 128, "no_retina": false, "lr": 0.002, "pos_frac": 0.4, "w_pos": 3.0, "eval_every": 1000, "vram_cap": 12.0, "fire_rate": 1.0, "force_gpu": true, "steps_max": 0, "damage_p": 0.5, "init": null, "ms": "learned", "scale": 4, "steps_fine": 6, "consensus": false, "seed": 0, "filters": false, "sdo": true, "name": "ms_learned_sdo"}`
- **Tiempo:** 89 min en RTX 3090 local
- **Validación local:** PQ 0.4009 · Dice 0.6393 · **Kaggle:** sin enviar
- **¿Mejoraría con más entrenamiento?** quizá: cerca de la meseta
- **Puntos débiles:** Empate: 0,4009 frente a 0,4061 (dentro del ruido probable de semilla); menos FN a mitad de curva pero más FP; 5,5 % canales ausentes
- **Cambiar / quitar / mejorar:** Medir la varianza de semillas antes de decidir; v2: dar las líneas de inversión de polaridad (PIL) del magnetograma ya calculadas; entrenamiento largo
- **Decisión:** Sin mejora medible a 8k; no se descarta hasta conocer la varianza de semillas
- **Volver atrás:** `git checkout 7da18d1` · envío: `—`

## 25. filament · ms_learned_s1 (2026-09-28)

- **Qué se hizo:** FIL-SEED: ms_learned idéntico (8k, recorte 512) con semilla 1
- **Configuración:** `{"iters": 8000, "batch": 8, "crop": 512, "steps": 24, "c": 48, "hidden": 128, "no_retina": false, "lr": 0.002, "pos_frac": 0.4, "w_pos": 3.0, "eval_every": 1000, "vram_cap": 12.0, "fire_rate": 1.0, "force_gpu": true, "steps_max": 0, "damage_p": 0.5, "init": null, "ms": "learned", "scale": 4, "steps_fine": 6, "consensus": false, "seed": 1, "filters": false, "sdo": false, "name": "ms_learned_s1"}`
- **Tiempo:** 91 min en RTX 3090 local
- **Validación local:** PQ 0.4001 · Dice 0.6459 · **Kaggle:** sin enviar
- **¿Mejoraría con más entrenamiento?** sí: seguía subiendo en la última validación
- **Puntos débiles:** —
- **Cambiar / quitar / mejorar:** —
- **Decisión:** Varianza entre semillas: s0 0,4061 / s1 0,4001 / s2 0,4100 -> media 0,4054, desviación ~0,005
- **Volver atrás:** `git checkout 7da18d1` · envío: `—`

## 26. filament · ms_learned_s2 (2026-09-28)

- **Qué se hizo:** FIL-SEED: ms_learned idéntico (8k, recorte 512) con semilla 2
- **Configuración:** `{"iters": 8000, "batch": 8, "crop": 512, "steps": 24, "c": 48, "hidden": 128, "no_retina": false, "lr": 0.002, "pos_frac": 0.4, "w_pos": 3.0, "eval_every": 1000, "vram_cap": 12.0, "fire_rate": 1.0, "force_gpu": true, "steps_max": 0, "damage_p": 0.5, "init": null, "ms": "learned", "scale": 4, "steps_fine": 6, "consensus": false, "seed": 2, "filters": false, "sdo": false, "name": "ms_learned_s2"}`
- **Tiempo:** 91 min en RTX 3090 local
- **Validación local:** PQ 0.41 · Dice 0.6535 · **Kaggle:** sin enviar
- **¿Mejoraría con más entrenamiento?** quizá: cerca de la meseta
- **Puntos débiles:** —
- **Cambiar / quitar / mejorar:** —
- **Decisión:** Varianza entre semillas: s0 0,4061 / s1 0,4001 / s2 0,4100 -> media 0,4054, desviación ~0,005
- **Volver atrás:** `git checkout 7da18d1` · envío: `—`

## 27. filament · ms_learned_s1 (2026-09-28)

- **Qué se hizo:** FIL-SEED: ms_learned idéntico (8k) con semilla 1
- **Configuración:** `{"iters": 8000, "batch": 8, "crop": 512, "steps": 24, "c": 48, "hidden": 128, "no_retina": false, "lr": 0.002, "pos_frac": 0.4, "w_pos": 3.0, "eval_every": 1000, "vram_cap": 12.0, "fire_rate": 1.0, "force_gpu": true, "steps_max": 0, "damage_p": 0.5, "init": null, "ms": "learned", "scale": 4, "steps_fine": 6, "consensus": false, "seed": 1, "filters": false, "sdo": false, "name": "ms_learned_s1"}`
- **Tiempo:** 91 min en RTX 3090 local
- **Validación local:** PQ 0.4001 · Dice 0.6459 · **Kaggle:** sin enviar
- **¿Mejoraría con más entrenamiento?** sí: seguía subiendo en la última validación
- **Puntos débiles:** —
- **Cambiar / quitar / mejorar:** —
- **Decisión:** Varianza de semillas: s0 0,4061 / s1 0,4001 / s2 0,410 (media 0,405, desv. ~0,005)
- **Volver atrás:** `git checkout 7da18d1` · envío: `—`

## 28. filament · ms_learned_s2 (2026-09-28)

- **Qué se hizo:** FIL-SEED: ms_learned idéntico (8k) con semilla 2
- **Configuración:** `{"iters": 8000, "batch": 8, "crop": 512, "steps": 24, "c": 48, "hidden": 128, "no_retina": false, "lr": 0.002, "pos_frac": 0.4, "w_pos": 3.0, "eval_every": 1000, "vram_cap": 12.0, "fire_rate": 1.0, "force_gpu": true, "steps_max": 0, "damage_p": 0.5, "init": null, "ms": "learned", "scale": 4, "steps_fine": 6, "consensus": false, "seed": 2, "filters": false, "sdo": false, "name": "ms_learned_s2"}`
- **Tiempo:** 91 min en RTX 3090 local
- **Validación local:** PQ 0.41 · Dice 0.6535 · **Kaggle:** sin enviar
- **¿Mejoraría con más entrenamiento?** quizá: cerca de la meseta
- **Puntos débiles:** —
- **Cambiar / quitar / mejorar:** —
- **Decisión:** Con desv. ~0,005: filtros (0,380) peor de verdad; SDO (0,401) y consenso (0,414) dentro de ~2 desv.: no concluyentes
- **Volver atrás:** `git checkout 7da18d1` · envío: `—`

## 29. filament · ms_learned_cons_40k (2026-10-01)

- **Qué se hizo:** FIL-011c: consenso suave desde cero a 40.000 it (lr 1e-3, clip en todos los módulos, guarda anti-divergencia, --lowmem), val cada 2.000
- **Configuración:** `{"iters": 40000, "batch": 8, "crop": 512, "steps": 24, "c": 48, "hidden": 128, "no_retina": false, "lr": 0.001, "pos_frac": 0.4, "w_pos": 3.0, "eval_every": 2000, "vram_cap": 12.0, "fire_rate": 1.0, "force_gpu": true, "steps_max": 0, "damage_p": 0.5, "init": null, "ms": "learned", "scale": 4, "steps_fine": 6, "consensus": true, "seed": 0, "lowmem": true, "block": -1, "split_manifest": null, "filters": false, "sdo": false, "name": "ms_learned_cons_40k"}`
- **Tiempo:** 427 min en RTX 3090 local
- **Validación local:** PQ 0.4242 · Dice 0.6669 · **Kaggle:** sin enviar
- **¿Mejoraría con más entrenamiento?** no: el mejor punto fue la validación 18 de 20 (sobreajuste o inestabilidad)
- **Puntos débiles:** Sin mejora sobre 24k: crudo ~0,40 desde 34k; posproceso 0,4242 = learned_cont 24k (0,4263) y cons_cont (0,4243). Misma split aleatoria, optimista
- **Cambiar / quitar / mejorar:** No alargar más una sola trayectoria; mejorar datos/objetivo/resolución fina; promediar semillas
- **Decisión:** Estable (0 pasos rechazados) pero meseta: el rendimiento de un modelo individual satura ~0,424-0,426 a 24-40k it
- **Volver atrás:** `git checkout d4bc4a1` · envío: `—`

## 30. filament · cv0_base (2026-10-01)

- **Qué se hizo:** Piloto del protocolo estricto de Codex (fold 0): train/calibración/test externo separados por bloques temporales, base ms_learned 8k, lr 1e-3, anotación individual, --lowmem
- **Configuración:** `{"iters": 8000, "batch": 8, "crop": 512, "steps": 24, "c": 48, "hidden": 128, "no_retina": false, "lr": 0.001, "pos_frac": 0.4, "w_pos": 3.0, "eval_every": 2000, "vram_cap": 12.0, "fire_rate": 1.0, "force_gpu": true, "steps_max": 0, "damage_p": 0.5, "init": null, "ms": "learned", "scale": 4, "steps_fine": 6, "consensus": false, "seed": 0, "consensus_w": 0.0, "small_frac": 0.0, "lowmem": true, "block": -1, "split_manifest": "work\\cv-protocol-20260930\\fold-0.json", "filters": false, "sdo": false, "name": "cv0_base"}`
- **Tiempo:** 93 min en RTX 3090 local
- **Validación local:** PQ 0.4072 · Dice 0.6793 · **Kaggle:** sin enviar
- **¿Mejoraría con más entrenamiento?** sí: seguía subiendo en la última validación
- **Puntos débiles:** Una sola semilla y un solo fold; el bloque externo es 142 imágenes (1908 GT)
- **Cambiar / quitar / mejorar:** Comparar con cons / consw / small en el mismo fold; repetir semillas si algo mejora
- **Decisión:** Referencia estricta: calibración PQ 0,4072; test EXTERNO PQ 0,385 (TP 1082, FP 786, FN 826). Es el número honesto del lienzo base
- **Volver atrás:** `git checkout d4bc4a1` · envío: `—`

## 31. filament · cv0_cons (2026-10-01)

- **Qué se hizo:** Protocolo estricto fold 0, ms_learned 8k lr 1e-3, objetivo de consenso suave (media de anotadores por imagen), --lowmem
- **Configuración:** `{"iters": 8000, "batch": 8, "crop": 512, "steps": 24, "c": 48, "hidden": 128, "no_retina": false, "lr": 0.001, "pos_frac": 0.4, "w_pos": 3.0, "eval_every": 2000, "vram_cap": 12.0, "fire_rate": 1.0, "force_gpu": true, "steps_max": 0, "damage_p": 0.5, "init": null, "ms": "learned", "scale": 4, "steps_fine": 6, "consensus": true, "seed": 0, "consensus_w": 0.0, "small_frac": 0.0, "lowmem": true, "block": -1, "split_manifest": "work\\cv-protocol-20260930\\fold-0.json", "filters": false, "sdo": false, "name": "cv0_cons"}`
- **Tiempo:** 104 min en RTX 3090 local
- **Validación local:** PQ 0.4074 · Dice 0.6762 · **Kaggle:** sin enviar
- **¿Mejoraría con más entrenamiento?** sí: seguía subiendo en la última validación
- **Puntos débiles:** Una semilla; IC de PQ ±0,026 por imagen: la diferencia con cv0_base (+0,010) no es distinguible del ruido sin bootstrap emparejado
- **Cambiar / quitar / mejorar:** Calcular Δ emparejado con bootstrap por imagen frente a cv0_base; repetir semillas 1-2 si el IC no excluye 0
- **Decisión:** Calibración 0,4074 (base 0,4072); TEST EXTERNO 0,3953 (TP 1134, FP 811, FN 774) frente a 0,385 de la base: +0,010, dirección favorable al consenso, sin confirmar
- **Volver atrás:** `git checkout d4bc4a1` · envío: `—`
