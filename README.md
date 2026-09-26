# NeuroPixel

Red neuronal cuyo estado entero vive en un lienzo de píxeles: los tokens entran como
"colores" (diccionario aprendido), la actividad se propaga con reglas locales tipo
autómata celular neuronal y la respuesta se lee traduciendo el color del píxel de salida
con el mismo diccionario. Todo el lienzo es observable paso a paso ("escáner").

Objetivo de esta fase: **entrenar la red completa de principio a fin** (gradientes),
algo que los prototipos OpenGL anteriores (CHIMERA, pixelflow) no permitían.

## Estructura

| Archivo | Qué hace |
|---|---|
| `neuropixel/codec.py` | Fase 0: token ↔ píxel RGB24 (negro = vacío), ida y vuelta por PNG |
| `neuropixel/hrr.py` | Representaciones holográficas reducidas (ligar = convolución circular) |
| `neuropixel/task.py` | Tarea 1: papeles (agente/acción/paciente/lugar) con combinaciones nuevas en test |
| `neuropixel/model.py` | `NeuroPixel` (lienzo local) y `TinyTransformer` (referencia global) |
| `neuropixel/safety.py` | Guardas: CPU por defecto, prioridad baja, RAM mínima, GPU solo si está libre y con tope |
| `scripts/train.py` | Entrenamiento, evaluación en combinaciones no vistas y traza visual `trace.png` |

## Uso

```bash
python -m pytest
python scripts/train.py --model neuropixel --iters 2000
python scripts/train.py --model transformer --iters 2000
python scripts/train.py --device cuda --model neuropixel   # solo si la GPU está libre
```

Resultados en `runs/<nombre>/` (`log.jsonl`, `result.json`, `trace.png`).

## Motor

PyTorch (núcleo entrenable, ya instalado 2.6+cu124). Núcleos propios en SlangPy o
triton-windows solo si el perfilado lo exige. Visor en vivo previsto con
moderngl + glfw + imgui-bundle (~12 MB). Decisión y mediciones en
`D:\PROJECTS\.cognition\pixel-latente\`.
