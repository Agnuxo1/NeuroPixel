# Historial de entrenamientos y envíos

Una fila por entrenamiento. Para volver a una versión: `git checkout <commit>` (o la etiqueta).

| # | Fecha | Concurso | Ejecución | Línea | Params | Tiempo | Validación | Kaggle | ¿Más entrenamiento? | Commit / etiqueta |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-09-27 | filament | np_ret | L2 | 58480 | 78 min | PQ 0.2306 · Dice 0.5252 | — | no: el mejor punto fue la validación 1 de 4 (sobreajuste o inestabilidad) | 1d3f0d9 |

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
