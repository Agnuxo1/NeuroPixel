# NCA y arquitecturas ligeras: qué probar en NeuroPixel

Fecha: 2026-10-01. Alcance: literatura sobre autómatas celulares neuronales (NCA), refinamiento fino y U-Net ligeras.
Regla de honestidad: solo se citan trabajos cuyo resumen o página se abrió en esta sesión. Los HECHOS salen de esas páginas. Las INTERPRETACIONES son mías y se marcan así. Lo no abierto figura como NO VERIFICADO.

Punto de partida (datos de Fran): PQ 0,406 (aprendida) frente a 0,343 (bilineal). El entrenamiento satura en 0,424-0,426 y c=96 no ayuda. Ruido de semilla ±0,005.

## 1. Tabla de referencias

| Referencia | Idea (hecho) | Coste (hecho) | Aplicabilidad (interpretación) |
|---|---|---|---|
| Growing NCA (Distill 2020) | Regla local entrenada con pool de muestras, daño y L2 por variable en los gradientes | ~8k parámetros, 64-96 pasos | Receta base. El pool importa poco aquí porque la entrada es fija (no hay crecimiento) |
| Self-classifying MNIST (Distill 2020) | Pérdida L2 tras 20 pasos (la entropía cruzada hacía crecer los logits sin límite); ruido 2e-2 en las actualizaciones | <25k parámetros | Aviso: usar L2 o logits acotados en el estado |
| Self-Organising Textures (Distill 2021) | NCA para textura | no abierto en detalle | Poca relevancia |
| Med-NCA (arXiv 2302.03473) | Comunicación global en imagen reducida y luego refinado por parches | 500 veces menor que UNet; mejora 2-3 % Dice | Es nuestro esquema grueso→fino. Valida el diseño, no prueba que gane a un UNet del mismo tamaño |
| M3D-NCA (arXiv 2309.02954) | n niveles de parcheo; la varianza sirve de control de calidad | corre en Raspberry Pi 4 | Pirámide multinivel |
| OctreeNCA (arXiv 2508.06993) | Vecindad en octree para llevar conocimiento global | 90 % menos VRAM que UNet en evaluación | Comunicación global jerárquica barata |
| Frequency-Time Diffusion NCA (arXiv 2401.06291) | Paso en espacio de Fourier que da comunicación global en un solo paso | no medido por mí | Candidato a canal global. Interpretación: exige tamaño fijo |
| MedSegDiffNCA (arXiv 2501.02447) | NCA multinivel que refina estimaciones del nivel bajo | 60-110 veces menos parámetros que UNet, Dice 87,84 % | Confirma el refinado por niveles. La parte de difusión no aplica |
| rNCA (arXiv 2512.13397) | NCA que repara una máscara ya predicha | mejora SegRefiner, DenseCRF y SegFix en miocardio y DRIVE | Sustento para un refinado fino separado |
| NCA: From Cells to Pixels (arXiv 2506.22899) | NCA en rejilla gruesa y decodificador MLP local con coordenadas relativas (LPPN) | ~25-30 % más parámetros | Encaja con nuestro lienzo fino |
| Review Spitznagel y Keuper (arXiv 2604.24990) | Receptivo: 5×5 dilatada y 3×3 combinadas mejoran; longitud T aleatoria; el pool es clave a largo plazo | memoria cuadrática con la resolución | Guía de receta. Los números de CIFAR y emoji no son transferibles |
| ViTCA (arXiv 2211.01233) | Autoatención local dentro del NCA, global por iteración | el trabajo dice coste lineal amortizado | La review la describe como inestable de entrenar. Baja prioridad |
| DiffLogic CA (arXiv 2506.04912) | NCA con puertas lógicas diferenciables, estado discreto | no medido | Sin evidencia en segmentación |
| DEQ (arXiv 1909.01377) | Punto fijo e implicit diff con memoria constante | hasta -88 % de memoria (lenguaje) | Ahorra memoria, no capacidad |
| NCA y DEQ (arXiv 2501.03573) | Un DEQ espacial equivale al límite de un NCA | ensayo con MNIST | Teórico |
| Universal Transformers | Bloque con pesos compartidos iterado | solo conocido por resultado de búsqueda | Sin evidencia visual verificada |
| FADE (arXiv 2207.10392 y 2407.13500) | Núcleos de upsampling a partir de codificador y decodificador, con puerta | "poco sobrecoste" según los autores | Upsampling guiado por la imagen fina |
| CARAFE (arXiv 1905.02188) | Núcleos de reensamblado según contenido, campo amplio | ligero según los autores | Alternativa guiada por contenido |
| DySample (arXiv 2308.15085) | Upsampling por muestreo de puntos | menos parámetros que CARAFE, FADE y SAPA; no usa guía de alta resolución | Barato, pero no usa la imagen fina |
| Deep Guided Filter (arXiv 1803.05619) | Capa de filtro guiado entrenable (salida baja + guía alta) | 10-100 veces más rápida que el filtro clásico | Directa para refinar con la imagen |
| FeatUp (arXiv 2403.10516) | Joint Bilateral Upsampling entrenable | núcleo CUDA propio | Referencia de JBU aprendido |
| UNeXt (arXiv 2203.04967) | U-Net con bloque MLP tokenizado | 72 veces menos parámetros que referencias (no se abrió el número absoluto) | Control posible |
| U-Lite (arXiv 2306.16103) | Convoluciones depthwise y axiales 7×7 | ~878k parámetros | Referencia de "1M" |
| Fast-SCNN (arXiv 1902.04502) | "Learning to downsample" y dos ramas | 68,0 % mIoU a 123,5 fps en Cityscapes | Inspira la entrada multiescala |
| EfficientNet-Lite0 | Sin SE ni swish, para móvil | 4,7M parámetros | Demasiado grande para control a tamaño igual |

NO VERIFICADO: BioNCA (la búsqueda no devolvió ningún artículo con ese nombre). Ali 2024, Yue 2024 y Lemke 2025a solo aparecen citados en la revisión arXiv 2609.24595. No se buscó nada en MIDL. Mobile-UNet solo se localizó como repositorio Keras sin artículo.

## 2. Las 5 mejoras más prometedoras

Protocolo común: 3 semillas, misma receta que la base, 8k iteraciones. Éxito = media de PQ ≥ 0,435, es decir +0,010 sobre 0,425 (2 veces el ruido). Entre 0,430 y 0,435 = inconcluso, repetir con 5 semillas. Menos de 0,430 = refutada. Se cambia una sola cosa por experimento.

**M1. Pérdida en pasos intermedios con longitud aleatoria.**
- Hipótesis: la señal solo llega al final de los 24 pasos gruesos y de los 6 finos, y los pasos tempranos aprenden poco. Con pérdida intermedia el techo sube.
- Cambio: pérdida de salida (rejilla gruesa) en los pasos 8, 16 y 24 con pesos 0,25, 0,25 y 1, más T gruesa muestreada en [16, 28]. Normalizar el gradiente por variable como en Growing NCA.
- Éxito: PQ ≥ 0,435. Falsable: si cae por debajo de 0,430, el problema no es la señal temporal.
- Interpretación: es el cambio más barato. No cuesta parámetros.

**M2. Comunicación global barata.**
- Hipótesis: si el límite es el campo receptivo, añadir contexto global mejora sobre todo los objetos grandes. Se puede comprobar desglosando PQ por tamaño de objeto.
- Cambio (probar A y luego B): A) un nivel extra 1/16 con 8 pasos cuyo estado se sube y se suma a la rejilla 1/4, al estilo Med-NCA y OctreeNCA. B) canal de percepción 5×5 dilatada además de 3×3, como recomienda la revisión.
- Éxito: PQ ≥ 0,435 y la ganancia concentrada en objetos grandes. Si la ganancia aparece solo en objetos pequeños, la hipótesis del campo receptivo se descarta.
- Coste esperado: <10 % de parámetros extra (a medir).

**M3. Upsampling del grueso al fino guiado por la imagen.**
- Hipótesis: el límite de detalle viene de subir el estado grueso con un método ciego a la imagen, y un núcleo dependiente de la imagen fina recupera bordes.
- Cambio: sustituir la interpolación del estado grueso por un núcleo 5×5 por píxel predicho desde (imagen fina, estado grueso), tipo CARAFE o FADE. Variante barata: capa de filtro guiado entrenable (DGF).
- Éxito: PQ ≥ 0,435 y mejora de la métrica de borde si existe. DySample no se prueba primero porque no usa la imagen fina.

**M4. Entrada multiescala con codificación posicional.**
- Hipótesis: promediar la imagen a 1/4 destruye información que el lienzo grueso necesita, y un tallo aprendido la conserva.
- Cambio: tallo con convoluciones con stride (1/2 y 1/4, "learning to downsample" de Fast-SCNN) en lugar de promedio, más 2 canales de coordenadas normalizadas (la positional encoding de DyNCA ayudó a la consistencia).
- Éxito: PQ ≥ 0,435. Si las coordenadas solas dan ≥ +0,005 y el tallo no, se registra por separado.

**M5. Decodificador subcelda tipo LPPN.**
- Hipótesis: un MLP pequeño que recibe el estado interpolado y coordenadas locales renderiza mejor el detalle que 6 pasos de lienzo fino.
- Cambio: sustituir (o preceder) el lienzo fino por un LPPN de ~25 % de parámetros extra. Variante 5b: pérdida de refinamiento ponderada ×3 en los bordes.
- Éxito: PQ ≥ 0,435 con ≤ 30 % de parámetros extra. Si solo iguala con más parámetros, se descarta.

Orden recomendado: M1, M2A, M3, M4, M5. Combinarlas solo después de medirlas por separado.

## 3. Control honesto: U-Net ligera de parámetros iguales

Entrenar una U-Net pequeña de unos 91k parámetros con la misma entrada, salida, datos, pérdida, iteraciones (8k y 40k) y 3 semillas. Variante A: convoluciones normales. Variante B: depthwise separables (estilo MobileUNet / U-Lite), con anchos ajustados al mismo presupuesto. Cota superior opcional: una de ~0,9M parámetros (escala U-Lite).

Por qué: Med-NCA, M3D-NCA y OctreeNCA ganan frente a un UNet que es 500 veces mayor o más pesado en memoria, no frente a uno del mismo tamaño (HECHO sobre lo que dicen sus resúmenes). Sin este control no sabemos si el NCA aporta algo por parámetro. Criterio: si la U-Net de 91k llega a ≥ 0,424, el NCA no gana en PQ por parámetro y su valor estaría en otras propiedades (diccionario, interpretabilidad, robustez). Si queda por debajo de 0,41, se refuerza la línea NCA. Registrar también FLOPs y latencia.

## 4. Descartes

- DEQ y Universal Transformers: ahorran memoria o profundidad, no resuelven detalle fino ni campo receptivo con 24 pasos que ya caben en memoria.
- ViTCA: la revisión la describe como poco estable y exige ajuste fino. Reconsiderar solo si M2 prueba que falta contexto global.
- DiffLogic CA: estado discreto y resultados en generación de patrones, sin evidencia en segmentación.
- Pool y daño aleatorio de Growing NCA: sirven para regeneración, y nuestra entrada es fija. Solo se mantiene la normalización de gradiente.
- Difusión de MedSegDiffNCA, textura y MNIST: tareas distintas.
- EfficientNet-Lite0 (4,7M) y UNeXt como control a tamaño igual: demasiado grandes y con preentrenamiento posible, lo que rompe la comparación con un modelo de 91k.
- DySample como primera opción: no usa la imagen fina, que es nuestra fuente de detalle.
