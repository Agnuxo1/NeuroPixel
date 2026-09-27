# Digit Recognizer — coordinacion

## Estado

- `np_pure_gpu_v2`: 29.440 parametros, validacion 0,9945, publico 0,99375.
- Estable entre 16 y 32 pasos; con 50 % del estado borrado conserva aproximadamente 0,995.
- Ya certifica que el lienzo puro percibe y se autorrepara; no es la prioridad competitiva.

## Cola local

1. Mantener como prueba de regresion y demostracion publica.
2. Solo reabrir para una ablacion necesaria: aumentos, TTA, multiescala o salto temporal.
3. No consumir GPU larga mientras Soil o filamentos tengan una prueba con mayor valor esperado.

