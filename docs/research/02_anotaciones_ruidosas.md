# Aprendizaje con anotaciones ruidosas y multi-anotador (filamentos solares, PQ)

Fecha: 2026-10-01. Convención: **[H]** = hecho leído en el resumen/página del paper en esta sesión; **[I]** = interpretación mía; **NO VERIFICADO** = no abierto o no encontrado. De los papers solo he leído la página/resumen (no los PDF completos), así que no doy cifras de sus tablas. Ninguna cifra de NeuroPixel se ha medido aquí: son las del encargo y `docs/FILAMENTOS_2026-09-28.md`.

## 0. Idea central (derivación propia, [I])

PQ se calcula contra un anotador al azar. Para un objeto predicho que coincide con ese anotador con probabilidad p y IoU medio q: si coincide, TP sube 1 y FN baja 1 (el denominador TP+FP/2+FN/2 sube 0,5); si no, FP sube 1 (el denominador sube 0,5). En ambos casos el denominador sube 0,5 y el numerador sube p·q de media. Añadir el objeto mejora PQ si y solo si **p > PQ/(2q)**. Con PQ=0,43 y q entre 0,6 y 0,8, el umbral cae entre 0,27 y 0,36. Los 313 filamentos pequeños que los otros anotadores marcan el 31 % de las veces están justo en el umbral. Eso explica que el consenso suave diera solo +0,008: no era un error de entrenamiento, era un caso de beneficio marginal casi nulo.

Supuestos: PQ agrupado por conjunto (si se promedia por anotador-imagen, las imágenes con pocos objetos pesan más) y p independiente entre objetos. Hay que comprobarlo simulando sobre la validación antes de fiarse.

Consecuencia: el objetivo útil es **predecir bien la distribución de anotadores** y decidir por objeto con ese umbral. Estimar la "verdad limpia" latente (estilo Zhang et al.) optimiza otra cosa: recuperaría filamentos pequeños que los humanos no marcan.

## 1. Tabla de métodos

Coste: B=bajo (horas, sin reentrenar), M=medio (1 entrenamiento), A=alto. Aplicabilidad a 91k parámetros y ~700 imágenes: alta/media/baja.

| Referencia (verificada) | Idea | Coste | Aplicabilidad |
|---|---|---|---|
| STAPLE, Warfield-Zou-Wells, TMI 2004 (solo resumen de búsqueda; página oficial inaccesible) | EM: segmentación verdadera latente + sensibilidad/especificidad por anotador | B | Media-baja: con 1-3 anotadores sin identidad estable los parámetros por anotador no se estiman; da una etiqueta "limpia" (ver aviso en 0) |
| Zhang et al., arXiv 2007.15963 (NeurIPS 2020) [H]: dos CNN acopladas, fiabilidad del anotador + distribución de la etiqueta verdadera, anotadores "maximamente poco fiables" con fidelidad a los datos | Matriz de confusión por anotador en cada píxel + regularizador | M | Media: la CNN de anotador solo se usa en entrenamiento. Pero optimiza la verdad latente, no PQ vs anotador. Código: no figura en la página arXiv |
| Kohl et al. Probabilistic U-Net, 1806.05034 [H]; Monteiro et al. Stochastic Segmentation Networks, 2006.06015 [H] (código en GitHub: biomedia-mira/stochastic_segmentation_networks, visto en búsqueda) | Distribución sobre mapas completos (cVAE / normal de bajo rango en logits) y muestreo de hipótesis | M-A | Media: permite estimar p **por objeto** (fracción de muestras donde aparece), no por píxel |
| Ji et al. MR-Net, CVPR 2021 (solo resumen de búsqueda) | Modela acuerdo y desacuerdo multi-anotador con módulo de pericia | M | Baja: necesita identidad de anotador |
| Wu et al. Multi-rater Prism, 2212.00601 [H] | Confianza por anotador y segmentación calibrada, iterativo | M-A | Baja: necesita identidad y más capacidad |
| Wang et al. UMA-Net, 2304.00466 [H] | Incertidumbre de anotación por píxel e imagen; las muestras malas no se descartan | M | Media: la idea de ponderar por píxel cabe en una pérdida |
| Lemay et al., 2202.07550 [H] (MELBA 2022) | Compara STAPLE, media y muestreo aleatorio de anotadores, con SoftSeg o convencional; SoftSeg calibra mejor | B-M | Alta: es una variación de objetivo |
| Gros et al. SoftSeg, 2011.09041 [H] | Sin binarizar, ReLU normalizada, pérdida de regresión; el resumen dice que mejora detección de estructuras pequeñas (ivadomed) | B | Alta |
| Gheibi y Ghazizadeh, 2605.20642 [H] (workshop ICML 2026, clasificación) | Con pocas anotaciones por ejemplo, entregar votos individuales (multipass, muestreo por época) mejora sobre la etiqueta suave | B | Media: es clasificación, no segmentación; pero tenemos 1-3 anotadores |
| Liu et al. ELR, 2007.00151 [H] | Regulariza hacia objetivos formados por las propias predicciones en la fase de aprendizaje temprano | B | Baja (ver sección 3) |
| Han et al. Co-teaching, 1804.06872 [H]; Li et al. DivideMix, 2002.07394 [H] (código en GitHub según la página) | Dos redes se pasan las muestras de pérdida pequeña / mezcla gaussiana + semi-supervisado | M-A | Baja |
| Huang et al. Co-Seg, 2102.00523 [H] | Co-teaching para segmentación con refinado de etiquetas | M-A | Baja |
| Yao et al., 2308.02498 [H]; Tadokoro et al., 2504.14795 [H] (código en GitHub según la página) | Ruido de segmentación con correlación espacial y sesgo (modelo de Markov; distribución discreta correlada) | M-A | Baja-media: pensados para etiquetas únicas, no desacuerdo entre anotadores |
| Kim et al. NSegment+, 2508.10383 [H] | Deformaciones elásticas **solo en la etiqueta** contra ruido implícito (bordes difusos) | B | Alta |
| Tschirschwitz et al., 2309.09742 [H] | Etiquetas repetidas en detección: reducir localización a clasificación y agregar con EM o voto mayoritario | M | Media: útil como agregador a nivel de objeto |
| Kimhi et al., 2406.10891 [H] | Bancos de ruido en instancias; cuestiona la eficacia de los métodos de ruido habituales | n/a | Aviso: los métodos estándar pueden no ayudar |
| Kirillov et al., 1801.00868 [H] | Define PQ | n/a | Base de la derivación |

No verificados: Learning from Crowds (Raykar et al.), Co-Mix (no hallado con ese nombre en la búsqueda). "MaskSup" (2210.00923) existe, pero es aprendizaje supervisado enmascarado, no multi-anotador: no aplica.

## 2. Cuatro recetas

Protocolo común: 3 semillas por brazo (ruido ±0,005, media de 3 ≈ ±0,003), mismo posproceso ajustado en la misma validación (168 anotaciones). Éxito = media del brazo menos media de la base ≥ +0,010 y al menos 2 de 3 semillas por encima de la mejor semilla base. Si es un cambio posterior al modelo, se aplica a las 3 semillas base existentes. En ambos casos, intervalo bootstrap por imagen (1000 remuestreos) con límite inferior > 0.

### R1. Umbral por objeto dependiente del tamaño, con ajuste cruzado (sin entrenar)
- **Hipótesis:** un umbral τ(área) sobre la puntuación del objeto (media de probabilidad en la componente) supera al umbral global, porque el punto óptimo p* = PQ/(2q) difiere entre pequeños (p≈0,31) y grandes (p≈0,9).
- **Cambio:** componentes conectadas del mapa de probabilidad; 3 cubos de área (< 200 px, 200-500, > 500; ajustar a terciles reales); elegir τ por cubo en validación por ajuste cruzado en 2 mitades de imágenes (ajustar en una, evaluar en la otra) para evitar el optimismo (~0,03) que ya tiene el posproceso. Referencia: τ por cubo inicial = p* con q medido en los TP.
- **Éxito:** ΔPQ ≥ +0,010 en la mitad no usada para ajustar, en las 3 semillas.
- **Falsada si:** el posproceso actual ya hace algo equivalente (comprobar antes) o ΔPQ < 0,005.

### R2. Reescoring de objetos con p de acuerdo aprendida (sin reentrenar la red)
- **Hipótesis:** una regresión logística o GBM pequeña sobre rasgos de la componente (área, longitud, media y máximo de probabilidad, contraste local, distancia al limbo) predice p = fracción de anotadores que coinciden (IoU ≥ 0,5) mejor que la probabilidad media; conservar objetos con p ≥ PQ/(2q).
- **Cambio:** etiquetas de objeto desde las anotaciones de validación; validación cruzada por imagen (5 pliegues). Rasgos < 10, regularización fuerte (L2, C=0,1) por tener unos pocos miles de objetos.
- **Éxito:** AUC de p frente a coincidencia ≥ 0,05 por encima de usar solo la probabilidad media, y ΔPQ ≥ +0,010 en los pliegues de reserva.
- **Falsada si:** el AUC no mejora; entonces la probabilidad de la red ya es suficiente y el techo es ruido de anotación.

### R3. Objetivo de regresión suave con muestreo de anotadores (SoftSeg + Lemay + Gheibi)
- **Hipótesis:** muestrear un anotador por imagen y paso, con pérdida de regresión sobre la máscara suave, calibra mejor el mapa que la media fija (el modelo ve 1-3 votos reales, no un promedio) y mejora la detección de pequeños.
- **Cambio:** y_t = máscara de un anotador elegido al azar por imagen y época; pérdida L = MSE(p, y_t) + 0,5·Dice suave(p, ȳ_media) con salida ReLU normalizada o sigmoide (probar ambas); mismo lienzo multiescala aprendido y 16k+ iteraciones (el efecto de las iteraciones es conocido).
- **Éxito:** ΔPQ ≥ +0,010 frente a consenso suave de 24k (0,424), y calibración de píxel (error de calibración esperado frente a fracción de anotadores) mejor que la base.
- **Falsada si:** con BCE el gradiente esperado coincide con el de la media; si el resultado es igual al consenso, la ventaja era solo varianza y se descarta.

### R4. Deformación elástica solo en la etiqueta (NSegment+) con borde tolerante
- **Hipótesis:** en filamentos finos, un desplazamiento de 1-2 px entre anotadores hunde el IoU; entrenar con etiquetas deformadas hace al modelo tolerante a esa variación y sube el SQ y el acierto de IoU ≥ 0,5.
- **Cambio:** por muestra, campo de desplazamiento suave (σ de suavizado 8-12 px, amplitud 1-2 px a resolución nativa; barrido {1, 2, 3}) aplicado solo a la máscara; probabilidad 0,5.
- **Éxito:** ΔPQ ≥ +0,010 y SQ medio de los TP sube; la semilla no debe empeorar en FP.
- **Falsada si:** el SQ baja o PQ cae más de 0,005 con amplitud 1 (el resumen de NSegment+ da ganancias en benchmarks de mIoU sobre objetos grandes, no demuestra nada para filamentos finos).

Orden de ejecución: R1, R2 (horas, sin GPU), luego R3 y R4 en la cola de GPU de uno en uno. R1 y R2 fijan el techo del lado de decisión; si ya lo agotan, R3-R4 no compensan.

## 3. Descartes

- **Zhang et al., Prism, MR-Net (modelos de anotador):** requieren identidad estable de anotador (no consta), añaden parámetros, y optimizan una verdad latente que según la sección 0 no es lo que premia PQ. Reservar solo si se descubre que los IDs existen.
- **ELR, Co-teaching, DivideMix, Co-Seg:** pensados para evitar memorizar etiquetas erróneas con redes de alta capacidad. Con 91k parámetros y ~700 imágenes la memorización es poco probable [I]; además el desacuerdo aquí es de muestreo, no de etiquetas corruptas. Kimhi et al. avisan de que los métodos estándar tienen una eficacia dudosa en ruido de instancias.
- **STAPLE como paso previo de etiqueta:** produce una etiqueta casi binaria "limpia" (del mismo problema que Zhang) y con 1-3 anotadores casi no añade a la media. Probar solo como brazo de control.
- **Probabilistic U-Net / SSN completos:** la idea más principiada para R2 (p por objeto desde muestras), pero coste de entrenamiento y parámetros extra; segunda ola solo si R2 muestra señal.
- **Modelos espaciales de Markov (Yao, Tadokoro):** diseñados para una etiqueta ruidosa, no para varios anotadores.
- **MaskSup:** no es multi-anotador.
