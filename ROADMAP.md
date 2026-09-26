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
