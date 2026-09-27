# NeuroPixel · Fase 3 «lienzo vivo» — conclusiones (noche del 26 al 27-09-2026)

Tarea base: papeles (agente/acción/paciente/lugar) en un lienzo de 8×8, evaluada siempre en
**combinaciones nunca vistas**. Referencias: transformer del mismo tamaño o mayor, GRU y un LLM
local. GPU RTX 3090 compartida entre 6 y 8 trabajos. Datos crudos en `runs/phase3/*.json` e
informe automático en `runs/phase3/INFORME_FASE3.md`, con la gráfica `escala.png`.

## Resultado principal: escala

| Parámetros | Transformer | NeuroPixel |
|---|---|---|
| 5 k | — | 49 % |
| 13–16 k | 54 % | 85 % |
| 30–44 k | 54 % | 98 % |
| 105–163 k | 54 % | 99,5 % |
| 236 k | — | 99,7 % |
| 811 k | 54 % | — |

- El transformer **no mejora con el tamaño** (60× más parámetros, el mismo 54 %). Memoriza
  qué sustantivos van juntos y falla sistemáticamente al intercambiar agente y paciente.
- NeuroPixel **mejora con el tamaño** hasta ~99,7 % y generaliza igual que entrena.
- Con el 50 % del estado borrado: NeuroPixel 79–94 % frente a 25–33 % del transformer, y la
  diferencia crece con el tamaño.

## Capacidades medidas

| # | Capacidad | Resultado | Lectura |
|---|---|---|---|
| 1 | Estado de reposo (pasos variables + daño al entrenar) | 99,6 %; estable de 8 a 64 pasos (96 % a 64); con la mitad del estado borrado se repara hasta el 99,4 % | Sin reposo: 67 % a 32 pasos y 23 % a 64. **Resuelto** |
| 2 | Palabra nueva con 5 ejemplos, solo el diccionario | Hasta 98,6 % de palabra nueva conservando el 92,6 % de lo viejo; con repaso, lo viejo no baja (99,4 %) y lo nuevo se queda en 74 % | Hay equilibrio entre aprender y no olvidar. Con varios intentos, a escala: 66–92 % (transformer 50–61 %) |
| 3 | Memoria persistente (hechos de uno en uno, pregunta tras N fotogramas vacíos) | 99 % / 97 % / 86 % / 70 % / 32 % con retrasos de 0 / 8 / 12 / 16 / 32 | GRU: 77 % casi constante. **NeuroPixel es más preciso y la GRU retiene más tiempo** → falta una capa de consolidación |
| 4 | Crecimiento automático | Por resonancia absoluta, **falló** (umbral mal planteado). **Por novedad ("palabras que no sé leer", tipo ART), funciona:** crea exactamente 3 lienzos para 3 temas y reutiliza el suyo en el repaso. Acierto final 83 % / 96 % / 99 % (media 93 %) frente al 66 % de un solo lienzo que olvida | Margen estrecho en el repaso (novedad 0,094 frente al umbral 0,10); el escáner falla el lienzo en el 24 % de las preguntas del tema 0 |
| 5 | Composición lejana (papel y palabra separados, 12×12) | Transformer 56 % (48 k y 320 k) < NeuroPixel 30 k 66 % < con reposo 71 % < 108 k con reposo 74 % | **Ventaja real pero sin resolver.** Mejora con el tamaño y el reposo; seguía subiendo |
| 6 | Energía (penalizar actividad) | 99,1 % de acierto con la mitad de las actualizaciones apagadas (99,9 % sin penalización) | El estado final no vuelve a negro: falta un mecanismo de apagado |
| 7 | Coste frente a un LLM (Qwen2 494M, CPU) | 99,5 % frente a 94 %; 5,7 ms frente a 279 ms; 56 MFLOP frente a 158 GFLOP | ~2.800× menos cálculo. **Nicho "tipo JEV" confirmado** |
| 8 | Imaginación (rellenar un hueco) | Categoría plausible 100 %, variado (4,1 de 5 distintos), exacto como el azar | Imagina, no copia |

## Qué es extrapolable y qué no

**Extrapolable, con cautela:**
- La tendencia de escala: el sesgo local del lienzo se aprovecha más al crecer; el transformer
  pequeño no aprende la ligadura aunque crezca (hasta 811 k).
- La autorreparación y el estado de reposo, que mejoran con el tamaño.
- El coste por respuesta: órdenes de magnitud por debajo de un LLM en tareas acotadas.

**No extrapolable todavía:**
- Tareas sintéticas y pequeñas. No está probado en lenguaje real ni en vocabularios grandes.
- La mayoría de las cifras vienen de **una sola semilla**. Confirmado con 2 semillas: NeuroPixel
  30 k 95,4–98,4 % frente a transformer 44 k 53,8–54,6 %. Toda afirmación publicable necesita 3 o más.
- La composición lejana no está resuelta (74 %).
- La memoria decae más rápido que en una GRU.
- Un transformer con más datos, otra codificación de posición o más entrenamiento podría
  resolver la tarea; aquí se compara con el mismo presupuesto.

## Siguientes pasos recomendados

1. **Semillas:** 3 por prueba clave (escala, lejana, memoria, reposo).
2. **Composición lejana:** más iteraciones y 230 k parámetros con reposo, porque seguía subiendo.
3. **Memoria:** capa de consolidación (canales que decaen lentamente) para retener como la GRU
   sin perder precisión.
4. **Crecimiento por novedad:** repetirlo con 5–10 temas y medir el coste de N lienzos con vmap.
5. **Paper:** escala + autorreparación + coste frente a LLM + palabra nueva, que es el núcleo
   más sólido.
