# Estructuras delgadas, objetos pequeños e instancias: qué probar en NeuroPixel (filamentos H-alfa)

Fecha: 2026-10-01. Fuentes: resúmenes/páginas abiertas en esta sesión (arXiv, CVF, bioRxiv/PMC). «Hecho» = dicho por el resumen del paper; «Interpretación» = mía. Lo no abierto figura como NO VERIFICADO.

**Contexto del proyecto (de `docs/FILAMENTOS_2026-09-28.md`):** val PQ 0,426-0,429; ruido de semilla ±0,005; el posproceso ajustado en validación es optimista (~+0,03); los 313 FN pequeños son en gran parte ruido de anotación (los demás anotadores los marcan 31 % de las veces frente a 90 %); la histéresis no mejoró. Consecuencia: **toda receta que solo suba la recall de pequeños arriesga subir los FP espurios (145)**; por eso las recetas están condicionadas al consenso y se juzgan con PQ y RQ, no con recall.

## 1. Tabla de referencias

| Referencia | Idea (hecho) | Coste | Aplicabilidad al lienzo celular de 91k sin atención |
|---|---|---|---|
| clDice, arXiv 2003.07311 (CVPR 2021) | Dice sobre esqueletos; soft-clDice diferenciable | Medio-alto (esqueleto blando iterativo, interpretación) | Posible, pero caro; Skeleton Recall lo mejora en coste |
| Skeleton Recall, 2404.03010 (ECCV 2024) | Recall del esqueleto del GT con operaciones en CPU; reduce >90 % la sobrecarga frente a pérdidas previas (afirmación del paper) | Bajo | **Alta**: esqueleto precalculado, un término |
| Betti matching, 2211.15272; versión 3D eficiente 2407.04683 | Emparejamiento de códigos de barras de homología persistente | Alto (CPU, interpretación) | Baja: nuestros fallos son fragmentos, no ciclos |
| TopoLoss (Hu et al.), 1906.05404 | Pérdida diferenciable que iguala números de Betti | Alto (interpretación) | Baja |
| SCNP, 2603.18671 (CVPR 2026) | Penaliza logits con su vecino peor clasificado; mejora topología; válida en instancias | Bajo-medio (según el paper, «eficiente») | Media; candidata de reserva, no probada aquí |
| cbDice, 2407.01517 (MICCAI 2024) | clDice + frontera + radio del esqueleto para equilibrar anchos | Medio | Media-baja: filamentos de ancho casi constante (interpretación) |
| Boundary loss, 1812.07032 (MIDL 2019/MedIA 2021) | Integral sobre la interfaz con mapa de distancia precalculado; para desbalance fuerte | Bajo (precalculo CPU) | Alta técnicamente; pero el retraso FP no es de desbalance |
| Signed distance, 1912.03849 (AAAI 2020) | Predecir SDM; contornos más suaves, menor Hausdorff | Bajo | Media |
| Hausdorff loss, 1904.10030 | Aproxima HD con transformada de distancia/erosión/convolución | Bajo-medio | Baja: PQ penaliza IoU, no outliers |
| Omnipose (PMC9636021, Nat. Methods 2022) | Campo de distancia + flujo = gradiente; Cellpose falla en alargadas por «múltiples sumideros» | Medio (cabeza + posproceso) | **Alta** como idea; sin atención |
| Cellpose (bioRxiv 2020.02.02.931238) | Flujos hacia el centro por difusión | Medio | Baja para alargadas (ver Omnipose) |
| Deep Watershed, 1611.08303 | Mapa de energía con cuencas por instancia | Medio | Alta, mismo patrón que Omnipose |
| HoVer-Net, 1812.06499 | Distancias horizontal/vertical al centro de masa para separar tocándose | Medio | Media: separa contactos, no unifica trozos |
| StarDist, 1806.03535 | Polígonos estrella-convexos por píxel | Medio | Baja: filamentos curvos/ramificados (interpretación) |
| CenterMask, 2004.04446 / Mask R-CNN en filamentos, 1912.02743 | Instancias por centro / propuestas | Alto | Baja: requiere FPN/cabezas grandes |
| Discriminative loss, 1708.02551 | Embeddings pull/push | Medio + agrupamiento | Baja |
| Batra et al. (CVPR 2019, openaccess.thecvf.com) | Orientación + segmentación conjuntas mejoran conectividad en carreteras (+9 %/+7,5 % topología en SpaceNet/DeepGlobe) | Medio | Media: cabeza de orientación sobre el eje |
| Blob loss, 2205.08209; Instance-awareness, 2604.24276 | Pérdida por instancia: las pequeñas pesan igual que las grandes; mejora F1 (MS +5 %) y PQ (0,38→0,40 en BraTS-METS) | Bajo | **Alta** |
| Kisantal et al., 1902.07296 | Sobremuestrear imágenes con pequeños + copiar-pegar; +9,7 % rel. en instancias pequeñas COCO | Bajo | Media (copiar-pegar físicamente dudoso en H-alfa) |
| SAHI, 2202.06934 | Inferencia por teselas con solape; +AP en objetos pequeños | Bajo (solo inferencia) | **Alta**, sin reentrenar |
| Focal loss, 1708.02002; Tversky, 1706.05721; Unified Focal, 2102.04525 | Reponderar fáciles/difíciles; sesgo precisión-recall | Mínimo | Alta pero beneficio esperado bajo (ver descartes) |
| Lovász-Softmax, 1705.08790 | Sustituto convexo del IoU/Jaccard | Bajo | **Alta** |
| SoftPQ, 2505.12155; Abbas & Swoboda, 2106.03188 | Métrica PQ con emparejamiento blando; sustituto suave de PQ vía corte multiway | Alto | SoftPQ: NO VERIFICADO como pérdida; Abbas: solver pesado |
| Med-NCA, 2302.03473 | Autómata celular: fase global a baja resolución y parcheada fina; 500× menor que UNet | n/a | Es nuestra familia; apoya inferencia por teselas (invariancia de escala, según el paper) |
| EdgeAttNet, 2509.02964 | Atención guiada por bordes para barbas de filamentos (MAGFILO) | Medio | Baja (atención); cifras NO VERIFICADO |

## 2. Cinco recetas, por coste/beneficio

**R1. Teselas con solape y escala 1,5× en inferencia (SAHI) + TTA D4.** Coste: cero entrenamiento.
- Hipótesis: el lienzo fino aprendido ve más detalle de los pequeños si su tamaño relativo aumenta; el consenso de 8 vistas reduce FP «trozos». Falsable: si no mejora, la invarianza de escala del lienzo es débil.
- Cambio: teselas 768 px (a 1024) con solape 25 %, reescaladas a 1024 (×1,33); fusión por media de probabilidades con ventana cosenoidal; flips/rot90 promediados antes del posproceso.
- Éxito (val, mismo posproceso congelado): PQ ≥ +0,008 (> 1,5σ de semilla), RQ no baja, FN pequeños (área <200 px a 1024) −10 % sin subir espurios >+10.

**R2. Skeleton Recall en el lienzo fino.** Coste bajo.
- Hipótesis: los FP «trozos» son cortes de conectividad; recompensar el esqueleto GT los reduce.
- Cambio: S = esqueleto(GT_consenso) dilatado 1 px (precalculado en CPU); L = L_base + λ·(1 − Σ p·S / Σ S), λ = 0,3 (valor mío; el del paper NO VERIFICADO), activo tras 2k it. Aplicar sobre el objetivo de consenso suave: esqueleto de umbral 0,5.
- Éxito: 3 semillas a 8k it; PQ medio ≥ +0,01 sobre 0,406; FP «trozos» 219 → ≤185; espurios ≤ 160.

**R3. Pérdida por instancia con peso por tamaño y compuerta de consenso (blob loss simplificada).** Coste bajo.
- Hipótesis: el Dice global deja que los grandes dominen; igualar instancias sube FN pequeños detectados. La compuerta evita premiar ruido de anotación.
- Cambio: por componente conexo i del GT: w_i = clip((A_med/A_i)^0,5; 1; 4)·c_i, con c_i = fracción de anotadores que lo marcan (≥0,5 para activar); L_inst = Σ w_i·(1 − IoU_blando_i en su caja dilatada 8 px)/Σ w_i, peso 0,5. Sobremuestrear ×2 las imágenes con pequeños de consenso alto (Kisantal, sin copiar-pegar).
- Éxito: FN pequeños con consenso ≥0,5 −15 %; RQ +0,01; PQ ≥ +0,008; si solo sube la recall y baja PQ, la hipótesis queda refutada.

**R4. Cabeza de distancia al eje + semillas/cuencas (Omnipose / Deep Watershed).** Coste medio (posproceso nuevo).
- Hipótesis: una cresta continua de distancia unifica trozos y separa fusiones mejor que el umbral actual.
- Cambio: canal extra D = min(EDT_interior / 4 px, 1) (a 1024), pérdida L1 enmascarada en el GT dilatado 3 px, peso 0,5. Posproceso: semillas = comp. conexos de D̂ > 0,5 en la máscara; cuenca sobre −D̂; eliminar instancias con área < percentil 10.
- Éxito: PQ ≥ +0,01; FP «trozos» −20 %; SQ no baja más de 0,005. Descartar si la fragmentación solo se traslada a «mal delimitados».

**R5. Lovász hinge por imagen + IoU blando por instancia (optimización directa de IoU>0,5).** Coste bajo.
- Hipótesis: parte de los FP «mal delimitados» son IoU 0,4-0,5; optimizar IoU empuja esos casos al emparejamiento. (Interpretación: no hay evidencia en la literatura abierta de que un sustituto de PQ supere a Dice+Lovász aquí.)
- Cambio: L = 0,5·Dice + 0,3·Lovász_hinge + 0,2·(1 − IoU_blando medio por instancia GT), calentamiento 2k it; exigir la combinación con R3 solo si R5 solo no mejora.
- Éxito: nº de coincidencias con IoU en [0,4;0,5] −25 %; PQ ≥ +0,008.

**Protocolo común:** 3 semillas, 8k it, posproceso fijado antes; ganancia = media − 1σ > 0. Registrar en `kaggle/HISTORIAL.md` con `log_run.py`. Probar R1 primero (sin GPU de entrenamiento), luego R2 y R3 en paralelo en la cola GPU; R4 y R5 después.

## 3. Descartes (con motivo)

- **clDice blando, Betti matching, TopoLoss:** coste de esqueleto iterativo/homología persistente; fallos de topología nuestros son fragmentos, no huecos. Skeleton Recall cubre el objetivo más barato.
- **StarDist, Cellpose (centros), CenterMask/Mask R-CNN, discriminative loss:** formas curvas/alargadas contradicen centro único (Omnipose documenta «múltiples sumideros» y errores >15 % en alargadas); las cabezas de propuestas no caben en 91k.
- **Tversky/focal/Unified Focal como único cambio:** solo mueve el punto de operación recall-precisión, que el posproceso ya ajusta (la histéresis no mejoró); riesgo de subir espurios. Reserva: Focal-Tversky α=0,3/β=0,7 si R3 funciona (el exponente focal NO VERIFICADO).
- **Hausdorff, cbDice, boundary loss:** optimizan outliers o anchos variables; aquí la métrica es IoU>0,5 sobre filamentos de ancho casi constante. Boundary/SDM quedan como reserva de R5 si fallan los «mal delimitados».
- **Super-resolución de entrada genérica:** ya medido: FSR satura (0,349-0,367) y el lienzo aprendido da la ganancia.
- **SoftPQ, Abbas y Swoboda:** SoftPQ es métrica (diferenciabilidad no confirmada); el solver combinatorio no cabe.
- **EdgeAttNet, atención:** requiere atención (fuera de restricción) y la ganancia en MAGFILO NO VERIFICADO.
- **Filtros clásicos de borde/cresta:** ya medidos, −5σ.
