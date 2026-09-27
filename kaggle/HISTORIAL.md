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
| 8 | 2026-09-27 | soil | np_votes | L2 | 59000.0 | 26 min | EMD CV 39.195 (curva media 85.393, uniforme 100.316) | EMD 98,44 público (peor que el mejor previo de Fran, 60,8) | pocas validaciones para saberlo | 2b115d6 / soil-v1 |

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
- **Validación local:** EMD CV 39.195 (curva media 85.393, uniforme 100.316) · **Kaggle:** EMD 98,44 público (peor que el mejor previo de Fran, 60,8)
- **¿Mejoraría con más entrenamiento?** pocas validaciones para saberlo
- **Puntos débiles:** No generaliza al test: CV 39,2 (mismo origen) frente a 98,4 en público. Cambio de cámara (entrenamiento Motorola/Samsung, test solo iPhone) y de procedencia de los suelos (test HPC_*); solo 24 muestras
- **Cambiar / quitar / mejorar:** Normalizar color y nitidez por cámara, aumentos de color y desenfoque, rasgos preentrenados como retina (DINOv2 ya en disco), validación que deje fuera un teléfono entero, mezclar con la curva media
- **Decisión:** Fallo registrado; para la clasificación final, seleccionar en Kaggle el mejor envío previo de Fran (60,8)
- **Volver atrás:** `git checkout 2b115d6` · envío: `kaggle/soil/runs/np_votes/submission.csv`
