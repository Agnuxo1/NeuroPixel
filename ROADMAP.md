# NeuroPixel: plan de estudios ("como un niño en la escuela")

Principio: el mismo diccionario de colores une lo que se ve y lo que se dice. Cada curso
reutiliza el lienzo y el diccionario del anterior, y solo avanza si supera su examen.
Los cursos se solapan: al pasar al siguiente se sigue repasando el anterior para no olvidar.

| Curso | Aprende | Datos (públicos, pequeños primero) | Examen para pasar |
|---|---|---|---|
| 0. Papeles | Quién hizo qué a quién (ligar papel y palabra) | Tarea sintética `RoleTask` | Combinaciones nuevas, sin intercambiar agente/paciente |
| 1. Palabras y colores | Nombre ↔ color perceptivo | Encuesta de colores XKCD (954 nombres, CC0) + `GROUNDED_RGB` | Nombrar colores nuevos; de un píxel de cámara a su concepto sin ejemplos |
| 2. Imágenes y palabras | Escena ↔ frase: el lienzo *es* la imagen | CIFAR-10 → Flickr8k/COCO con descripciones | Describir imágenes no vistas; escáner coherente (zona de cielo dice "cielo") |
| 3. Vídeo sintético | Tiempo y causa: qué pasa después | Moving MNIST, pelotas que rebotan, acciones simples | Predecir el siguiente fotograma y nombrar la acción |
| 4. Vídeo real | Mundo real: acciones y objetos en movimiento | UCF101 → fragmentos cortos de Kinetics/Something-Something | Nombrar acciones nuevas; responder preguntas sobre el vídeo |

## Cómo encaja en el lienzo

- **Imagen:** los píxeles de cámara entran en los 3 canales perceptivos; las palabras, por el
  diccionario. Imagen y texto comparten el mismo espacio desde el primer paso.
- **Vídeo:** el estado del lienzo persiste entre fotogramas (memoria); cada fotograma nuevo
  se inyecta y la dinámica local ya es temporal por naturaleza. Señal de aprendizaje sin
  etiquetas: predecir el fotograma siguiente; con etiquetas: decir la acción con el diccionario.
- **Escáner:** en cada curso se puede leer qué palabra "piensa" cada zona mientras mira.

## Límites a vigilar

- Cómputo: la RTX 3090 basta para los cursos 0–3 a baja resolución (64×64–128×128);
  el curso 4 exige recortes cortos y pocas clases al principio.
- El color es una pista (3,5–6,6 bits), no la identidad (17,6 bits para 200k palabras).
- Afirmar solo lo medido: cada curso compara con una referencia fuerte del mismo tamaño.

## Líneas de trabajo (decisión 2026-09-26)

Se evalúan por separado, con las mismas pruebas, para medir hasta dónde llega cada una.

| Línea | Qué es | Estado |
|---|---|---|
| **L1 pura** | Diccionario aprendido + escuela en todos los píxeles, sin colores anclados | Papeles 93–95 % (98 % con escuela) vs transformer 55 %; fotos CIFAR 16–19 % (no percibe) |
| **L2 híbrida** | Retina (módulo de visión) que da a cada píxel un "color" rico + el mismo lienzo | Pendiente |

## Línea futura L3: NeuroPixel como modelo completo multimodal (decisión de Fran, 2026-09-27)

Montar un modelo completo tipo Qwen2.5-0.5B-Instruct (lenguaje + visión) cuyo núcleo sea el
lienzo, porque **las habilidades emergentes solo se pueden observar a esa escala**; no se deducen
de las tareas pequeñas.

Estimaciones de coste (4 × RTX 3090 ≈ 50 TFLOPS efectivos):

- Transformer de 50M como referencia: de 1 a 17 h según los datos (0,5–10B tokens).
- NeuroPixel de 50M con un lienzo por token: ~6 años (inviable).
- NeuroPixel de 50M con 256 tokens por lienzo, predichos en paralelo: ~9 días por cada 1B tokens.
  Es investigación, no un resultado garantizado.
- Visión: DINOv2-small preentrenado (ya en disco) + puente: horas.

Paso previo recomendado: una prueba de ~1 día a escala pequeña para ver si el lienzo modela
lenguaje. Mientras tanto, el prototipo práctico es híbrido: LLM pequeño + DINOv2 + lienzos
NeuroPixel como memoria y razonamiento.
