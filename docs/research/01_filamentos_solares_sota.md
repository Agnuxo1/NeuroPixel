# Estado del arte en segmentacion de filamentos solares (H-alfa) - informe 01

Fecha: 2026-10-01. Alcance: solo se citan fuentes cuya pagina/resumen se abrio en esta sesion (WebFetch o resultado de busqueda). No se consulto ADS directamente (no verificado en esta sesion). Marca **[R]** = leido el resumen/pagina; **[T]** = solo titulo en resultado de busqueda; **NO VERIFICADO** = no abierto.

**Aviso metodologico clave (hecho):** los papers cientificos reportan IoU/Dice por pixel o precision/recall de deteccion, NO Panoptic Quality (PQ). Sus cifras no son comparables con nuestro 0,43. Las unicas cifras PQ salen de repositorios de participantes del propio concurso (README autodeclarados, sin auditar).

## 1. Tabla de metodos

| Referencia | Dataset | Receta clave | Cifra (hecho del paper/README) | Aplicabilidad |
|---|---|---|---|---|
| Diercke et al., A&A 686, A213 (2024), arXiv:2402.15407 [R] | ChroTel, GONG, KSO | YOLOv5 -> cajas; mascara por umbral 1 sigma dentro de la caja; U-Net a 1024x1024 sobre ~16.759 imagenes GONG; sin aumentos ni perdida declarados | U-Net: exactitud 92,4 %, precision 49,8 %, recall 93,2 %. Con filtro de area 100 px la precision sube a ~75 % y baja el recall | Media: confirma que los filamentos pequenos son el punto debil y que el filtro de area cambia precision/recall |
| Zhu, Lin, Wang, Liu, Yang, Sol. Phys. 294 (2019), arXiv:1909.06580 [R] | BBSO (60 pares corregidos) | U-Net | Sin cifras en el resumen | Baja (historica) |
| Ahmadzadeh et al., arXiv:1912.02743 (2019) [R] | BBSO 5 anos + HEK | Mask R-CNN | Sin cifras en el resumen; "compite" con el modulo FFT de NASA | Baja |
| Guo et al., Sol. Phys. 297, 104 (2022) [R via busqueda] | ~10.000 filamentos | CondInst (instancias directas) con backbone ResNet-C/D/v2 | Cifras NO VERIFICADO | Media: ataca instancia directamente, pero modelo grande |
| Zheng, Hao et al., ApJ (2024), DOI 10.3847/1538-4357/ad2be9 [R] | CHASE, 120 imagenes | U-Net 2048x2048, **focal loss gamma=4, w=2** | Test: precision 0,79, recall 0,83, IoU 0,67 | Media: perdida focal para desbalance |
| Zhu et al., Flat U-Net, ApJ, arXiv:2502.07259 [R] | BBSO ~1.000 imagenes 2022-23 (mascaras automaticas filtradas), 512x512 | Canales constantes (no se duplican), atencion SCA/CSA, BCE, Adam 1e-3 | **0,26 M params**: DSC 0,79, recall 0,69 frente a U-Net 28,95 M: DSC 0,69, recall 0,53 | **Alta en idea**: modelo diminuto igual o mejor que el grande; ya estamos en 91k |
| Solomon et al., EdgeAttNet, arXiv:2509.02964 [R] | MAGFiLO, 1.439 imagenes (1.295/45/99) | U-Net + atencion guiada por mapa de bordes, BCE+Dice, 50 epocas, sin aumentos | 22,7 M params; mIoU pareado 0,6451 vs U-Net 0,5724 | Media: la idea del mapa de bordes es transferible, la atencion no (91k) |
| Hu et al., MORDEN, arXiv:2607.24525 (jul 2026) [R] | MHAS 120 + AHAS 775 (CHASE) | Focal (alfa 0,66, gamma 2), AdamW 2e-4, 512-1024 px, kernel 5, InstanceNorm, pooling piramidal adaptativo; aumentos: rotacion +-45 (p 0,9), gamma 0,5-1,5 (p 0,9), escala 0,5-1; **DenseCRF** + **DBSCAN eps=45, minPts=140** para unir fragmentos | IoU 0,696 (MHAS) / 0,812 (AHAS); ablacion: quitar el pooling piramidal -0,018 IoU | **Alta**: posproceso de union de fragmentos y aumentos fotometricos |
| VisionX (gopi470, GitHub) [R] | Concurso 2026 | Cascada YOLO11s + CLAHE + U-Net (4 canales: gris, CLAHE, semillas, transformada de distancia); 0,30 BCE+0,40 Dice+0,30 Boundary; EMA/SWA; TTA 6 pasos; cierre de huecos por esqueleto | Sin cifras en el README | Media: canales de entrada extra y perdida de borde |
| Hisernberg/solar (GitHub) [R] | Concurso 2026 | U-Net EfficientNet-B3, teselas 512 nativas con **75 % de sobremuestreo centrado en filamentos**, **cada lectura de anotador como muestra separada**, BCE+Dice, EMA, TTA dihedral x8, posproceso histeresis -> reunir fragmentos -> area minima -> rellenar huecos -> dilatar, **ajuste por ascenso de coordenadas del PQ** sobre mapas out-of-fold | Sin cifras | **Alta**: es la receta mas cercana a nuestro problema |
| srivatsav-kannan/solar-seg (GitHub) [R] | Concurso 2026 (707 imagenes, 1.154 registros, 8.199 poligonos) | U-Net ancho 24, **Dice por lote**, 3.000 pasos, TTA 4 flips; validacion por bloques de 27 dias con embargo de 3 | Holdout PQ 0,346 (IC 95 % 0,329-0,363); LB 0,30 -> 0,31. Inferencia nativa en teselas: solo 0,29 (peor) | **Alta**: referencia real de PQ y de validacion honesta |
| ameyypawar/filament-seg (GitHub) [R] | Concurso 2026 | Baseline sin aprendizaje (umbral + componentes) | PQ local 0,1245, LB 0,08; RQ<=0,20; **PQ entre anotadores 0,343**; filamento mediano = 1.228 px (0,03 % de la imagen), 41 % < 1.000 px | **Alta** (diagnostico): coincide con nuestro 0,35 humano |
| sebastiansalutare (GitHub) [R] | Concurso 2026 | Mask R-CNN 1024 px, sin aumentos | Val PQ 0,150; con suelo de confianza 0,10: 0,165 | Baja, pero el suelo de confianza por instancia es util |
| Birch et al., arXiv:2605.22233 [R] | SDO/AIA 304 | YOLOv5 con imagen compuesta de 3 canales (gris, corona realzada, disco retirado) | mAP@50 0,749, recall 78 % | Baja (otra modalidad, limbo) |
| Chalmers & Ahmadzadeh, arXiv:2509.18214 [R]; MAGFiLO (Ahmadzadeh et al., Sci. Data 2024, Harvard Dataverse DOI 10.7910/DVN/J6JNVK) [R/T] | GONG 2011-2022, >10.000 filamentos | Clasificacion de quiralidad (no segmentacion) | kappa 0,66 en etiquetas de quiralidad | Solo contexto: confirma alta subjetividad humana |

Notas: Hisernberg afirma que MAGFiLO 1.0 publico contiene las etiquetas del test del concurso (afirmacion de un README, NO VERIFICADO). Un titulo de busqueda adicional, "Automatic detection of solar filament oscillations I" (arXiv:2607.01095), no es de segmentacion y no se abrio.

## 2. Cinco ideas mas prometedoras

**Idea 1. Posproceso optimizado para PQ sobre mapas out-of-fold (union de fragmentos + area minima + suelo de confianza).**
- Hipotesis: con nuestra prediccion actual, un barrido de (umbral de histeresis alto/bajo, area minima, radio de union de fragmentos) maximiza PQ por encima del posproceso fijo actual. Predice que descartar instancias de confianza baja sube RQ mas de lo que pierde (el suelo de confianza dio +0,015 en Mask R-CNN; fragmentacion se penaliza tres veces).
- Cambio: tras el lienzo, histeresis -> union de componentes a distancia <= eps (probar 15-45 px, como DBSCAN de MORDEN) -> area minima -> rellenar huecos; ajuste por ascenso de coordenadas sobre validacion cruzada.
- Exito: PQ val >= 0,45 (+0,02) con la misma red, en al menos 2 de 3 particiones por bloques temporales. Falsado si <+0,005.

**Idea 2. Etiquetas blandas por consenso de anotadores (o cada lectura como muestra).**
- Hipotesis: como el PQ se mide contra cada anotador por separado y la concordancia humana es 0,35, entrenar con la media de anotadores (diana blanda) o con cada lectura como muestra calibra mejor la confianza en filamentos dudosos que una mascara unica (medoide/union).
- Cambio: diana = fraccion de anotadores que marcan el pixel (BCE blanda) en lugar de mascara binaria; umbral final ajustado como en idea 1.
- Exito: PQ val +0,01 o mas frente a la mascara binaria actual; ademas recall de filamentos pequenos (<1.000 px) sube sin bajar la precision.

**Idea 3. Perdida Dice por lote (o BCE+Dice+borde / focal) en vez de la actual.**
- Hipotesis: Dice por lote mejora la recuperacion de objetos pequenos con fondo dominante (0,03 % de area). srivatsav reporta que "supero ampliamente" al baseline; focal (gamma 2-4) la usan Zheng y MORDEN.
- Cambio: sustituir la perdida por 0,3 BCE + 0,4 Dice(lote) + 0,3 perdida de borde; variante focal gamma=2, alfa=0,66.
- Exito: PQ val >= +0,01 con 3 semillas; falsado si la variacion entre semillas es mayor que la ganancia.

**Idea 4. Canales de entrada fisicos: aplanado del limbo, contraste local (CLAHE) y razon radial.**
- Hipotesis: el oscurecimiento del limbo y los bordes del disco generan falsos positivos y ocultan filamentos tenues; dar al lienzo canales normalizados (contraste, intensidad, razon radial; VisionX y Hisernberg lo hacen) eleva recall y precision.
- Cambio: preproceso con deteccion de disco + ajuste radial de fondo; entrada de 3 canales; ablacion canal a canal.
- Exito: PQ val +0,015; la ganancia debe concentrarse en filamentos cerca del limbo (medir por anillos radiales).

**Idea 5. Muestreo centrado en filamentos + TTA dihedral + EMA.**
- Hipotesis: oversampling 75 % de teselas con filamento (Hisernberg) y TTA de 4-8 simetrias con EMA reducen la varianza y aumentan el recall sin aumentar los falsos positivos.
- Cambio: sampler de teselas con p=0,75 centradas en filamento; EMA de pesos; promedio de 8 simetrias dihedrales antes del umbral. Aumentos fotometricos tipo MORDEN (gamma 0,5-1,5).
- Exito: PQ val +0,01 (TTA solo ya dio mejora en srivatsav con 4 flips). Cuidado: srivatsav midio que inferencia nativa por teselas bajo a 0,29; comprobar que el TTA no se mezcla con ese cambio de resolucion.

Prioridad sugerida (coste/beneficio): 1 > 2 > 3 > 5 > 4. Las ideas 1 y 2 no cambian la red y se pueden probar con pesos ya entrenados.

## 3. Ideas descartadas

- **Atencion pesada / backbones preentrenados (EdgeAttNet 22,7 M, MORDEN, EfficientNet-B3):** rompen la linea de 91k parametros de Fran. Flat U-Net sugiere que hace falta menos, no mas. Solo se conserva la idea del mapa de bordes como cabeza auxiliar (opcional, no priorizada).
- **Cascada YOLO + U-Net semi-supervisada con archivo GONG (Diercke):** requiere datos externos y no se sabe si las reglas del concurso lo permiten (NO VERIFICADO); su precision del U-Net era del 49,8 %.
- **Entrenar con MAGFiLO publico:** posible filtracion de etiquetas de test segun un README (NO VERIFICADO) y riesgo reglamentario; no usar sin confirmar con los organizadores.
- **Mask R-CNN / CondInst:** PQ 0,15 en el unico dato disponible; CondInst sin cifras verificadas.
- **DenseCRF:** exige 1K-2K de resolucion y 10 iteraciones por imagen; ganancia en bordes (barbas), no en conteo, y el PQ con IoU>0,5 depende mas de contar bien que de pulir bordes.
- **Deteccion en SDO/AIA 304 (Birch) y deteccion de oscilaciones o quiralidad:** otra modalidad o tarea.
- **Inferencia nativa por teselas como mejora automatica:** srivatsav midio 0,29 frente a 0,33 de referencia; no aplicar sin ablacion.

## Limitaciones de este informe
Solo se leyeron resumenes o paginas parciales de varios papers (Springer y MDPI devolvieron redireccion o 403; Sci. Data, Springer 2026 y A&A HTML no se abrieron). No hay cifras PQ publicadas en revistas. Las cifras de GitHub son autodeclaradas. La interpretacion (ideas, hipotesis y prioridades) es mia, no de los papers.
