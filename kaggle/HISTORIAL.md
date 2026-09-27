# Historial de entrenamientos y envíos

Una fila por entrenamiento. Para volver a una versión: `git checkout <commit>` (o la etiqueta).

| # | Fecha | Concurso | Ejecución | Línea | Params | Tiempo | Validación | Kaggle | ¿Más entrenamiento? | Commit / etiqueta |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-09-27 | filament | np_ret | L2 | 58480 | 78 min | PQ 0.2306 · Dice 0.5252 | — | no: el mejor punto fue la validación 1 de 4 (sobreajuste o inestabilidad) | 1d3f0d9 |
| 2 | 2026-09-27 | filament | np_ret_v2 | L2 | 58480 | 110 min | PQ 0.3667 · Dice 0.6296 | PQ 0,31 público (≈ puesto 420-454 de 734) | quizá: cerca de la meseta | 41a8275 / filament-v2 |
| 3 | 2026-09-27 | filament | np_pure_v2 | L1 | 29312 | 116 min | PQ 0.3619 · Dice 0.6265 | — | quizá: cerca de la meseta | 41a8275 |
| 4 | 2026-09-27 | digits | np_pure_cpu | L1 | 15776 | 96 min | acierto 0.9825 | 0,98064 público (≈ puesto 524 de 865) | pocas validaciones para saberlo | 094edad / digits-v1 |
| 5 | 2026-09-27 | filament | np_big_reposo | L2 | 136880 | 170 min | PQ 0.37 · Dice 0.6323 | — | sí: seguía subiendo en la última validación | 094edad |

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
