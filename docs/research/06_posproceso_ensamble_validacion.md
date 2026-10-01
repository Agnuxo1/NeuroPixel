# 06. Posproceso para PQ, ensambles, TTA, EMA/SWA y validación con pocas imágenes

Fecha: 2026-10-01. Alcance: literatura + una medición local sobre `val_probs.npz`.
Convención: **[ABIERTO]** = leí título/resumen/contenido en esta sesión; **[SNIPPET]** = solo vi un resumen de búsqueda; **NO VERIFICADO** = no pude abrirlo o no existe con ese nombre. Hecho = lo que dice la fuente; *Interpretación* = mi inferencia para NeuroPixel.

## 0. Medición local (hechos, calculados hoy)

Script reproducible: `docs/research/06_bootstrap_pq.py` (ejecutar desde `kaggle/filament`). M1 = `ens_ms_learned_cont`, M2 = `+ms_learned_cons_cont`, M4 = los 4 modelos (según el nombre de carpeta, sin sufijo `_tta`; no comprobé si usaron TTA). 168 anotaciones sobre 106 imágenes de validación; PQ agregado igual que `fil.pq_counts`; 2000 remuestreos **por imagen**.

- PQ con IC95: M1 0,4263 [0,400; 0,450]; M2 0,4293 [0,402; 0,454]; M4 0,4264 [0,400; 0,449]. Semiancho = 0,026 (sd 0,013). El bootstrap ingenuo por anotación da sd 0,0114: subestima ~11 %.
- Comparación emparejada: M2-M1 = +0,0030 [-0,006; +0,013], P(Δ>0)=0,74. M4-M1 = +0,0001 [-0,009; +0,010]. M4-M2 = -0,0029 [-0,012; +0,007]. **Ni el +0,003 del ensamble ni su "empeora a veces" se distinguen de cero.** El error típico emparejado es ~0,005, así que con esta validación solo es detectable una diferencia de **~0,010 o más** (coherente con el ±0,005 entre semillas de `FILAMENTOS_2026-09-28.md`).
- Umbral global agotado: la mejor configuración de `ensemble_fil.py` siempre cae en `thr=0,6`, el borde inferior de su rejilla (0,6-0,85). Barrido sobre M2 (min_area 120, sin cierre): thr 0,3 → 0,4238; 0,4 → 0,4242; 0,5 → 0,4281; 0,55 → 0,4285; 0,6 → 0,4293; 0,7 → 0,4228. Es una meseta de ±0,003, dentro del ruido. Con M2: TP 727, FP 365, FN 508.
- Optimismo del ajuste en rejilla: elegir la configuración en la mitad de las imágenes y evaluar en la otra (100 particiones) da +0,005 de media, pero con sd 0,028 entre particiones: el PQ de val depende más de *qué* imágenes caen que de la configuración. El rango entero de la rejilla es 0,399-0,429.

## 1. Tabla de referencias

| Referencia | Idea | Coste | Aplicabilidad a NeuroPixel |
|---|---|---|---|
| Kirillov et al., *Panoptic Segmentation*, arXiv 1801.00868 [ABIERTO] | PQ = SQ x RQ con emparejado IoU>0,5 | 0 | Define la métrica; con emparejado >0,5 un objeto pequeño y dudoso cuesta 0,5 FP. |
| Lipton, Elkan, Narayanaswamy, *Thresholding Classifiers to Maximize F1*, arXiv 1402.1892 [ABIERTO] | Con probabilidades calibradas el umbral óptimo es F1*/2; el óptimo depende de la métrica | 0 | *Interpretación*: con F1 de objetos ~0,5 el umbral de objeto óptimo queda bajo, y a nivel de píxel el umbral 0,6-0,85 actual no es el de objeto. |
| Huang et al., *Mask Scoring R-CNN*, arXiv 1903.00241 [ABIERTO] | La puntuación de clase correlaciona mal con la calidad de máscara; se aprende un score de IoU | 0 (heurística) / medio (aprendido) | Idea: puntuar cada componente con algo más que "hay filamento" (media de prob., área, forma). |
| StarDist: `optimize_thresholds` (`prob_thresh`, `nms_thresh`) [SNIPPET de documentación de terceros] | Ajustar umbral de objeto en validación para maximizar F1 | 0 | Precedente de práctica estándar; el artículo original NO VERIFICADO. |
| Caicedo et al., *Nucleus segmentation across imaging experiments: DSB 2018*, Nat. Methods [ABIERTO, PMC] | 1.º: 32 redes (8 arquitecturas x 4 réplicas), 24 aumentos, promediado de máscaras, ranking de candidatos con GBT sobre rasgos morfológicos y watershed. 2.º: pérdida que penaliza el error según el tamaño del objeto. 3.º: Mask R-CNN con 15 TTA | alto | Los ganadores combinaron diversidad (arquitectura, aumentos), no solo semillas, y puntuaron/filtraron candidatos con rasgos de forma. |
| Fort, Hu, Lakshminarayanan, *Deep Ensembles: A Loss Landscape Perspective*, arXiv 1912.02757 [ABIERTO] | Inicializaciones distintas dan modos distintos; las trayectorias de un mismo entrenamiento agrupan sus predicciones en un solo modo | medio | Explica por qué `cont`+`cons_cont` (mismo linaje) aportan poco. |
| Huang et al., *Snapshot Ensembles*, arXiv 1704.00109 [ABIERTO] | M modelos de un entrenamiento con LR cíclico | bajo | Diversidad limitada (Fort). |
| Izmailov et al., *SWA*, arXiv 1803.05407; Tarvainen y Valpola, *Mean Teachers*, arXiv 1703.01780; Wortsman et al., *Model soups*, arXiv 2203.05482 [ABIERTO] | Promediar pesos (SWA, EMA, sopas) da óptimos más planos/estables sin coste de inferencia | bajo | Mejora barata de un modelo; resultados citados son en otros dominios, no prueban ganancia aquí. |
| Abe et al., *Deep Ensembles Work, But Are They Necessary?*, arXiv 2202.06985 [ABIERTO] | La ganancia de un ensamble (incluida OOD) se explica por la precisión dentro de distribución; un modelo mayor la replica | 0 (lectura) | Cautela: si los miembros son casi iguales, el ensamble no añade nada que un modelo mejor no dé. |
| Solovyev et al., *Weighted Boxes Fusion*, arXiv 1910.13302 [ABIERTO] | Fusiona cajas con sus scores en vez de suprimir (NMS) | medio | Para segmentación por píxel no se aplica; la idea (consenso ponderado por score) se traslada a votación por objeto. |
| Warfield et al., STAPLE [SNIPPET]; He, *When Does Consensus Beat Voting?*, arXiv 2607.19402 [ABIERTO, preprint sin revisión] | STAPLE estima sensibilidad/especificidad por segmentador con EM; el preprint afirma que se reduce a voto mayoritario con umbral y falla con desbalance de clases | bajo | Con 2-4 miembros casi iguales y desbalance extremo (filamento <<fondo) STAPLE no promete; voto por objeto sí es barato. |
| Moshkov et al., *TTA for cell segmentation*, Sci. Rep. 2020; Wang et al. arXiv 1807.07356/1810.07884 [Moshkov SNIPPET; Wang ABIERTO] | TTA simple mejora U-Net/Mask R-CNN; la varianza entre vistas estima incertidumbre | bajo | Fusión y efecto por tamaño NO VERIFICADO; la varianza sirve de score (R3). |
| Koehn, EMNLP 2004 [ABIERTO, PDF local]; Bestgen, LREC 2022 [ABIERTO] | Bootstrap emparejado (remuestrear con reemplazo, puntuar ambos sistemas, contar quién gana); informar IC y no solo el número | 0 | Base del criterio de éxito de las recetas. |
| Maier-Hein et al., arXiv 1806.02051 [ABIERTO] | Los rankings no son robustos al conjunto de test, al esquema ni al anotador | 0 | Respalda desconfiar de diferencias de 0,003. |
| Reinke et al. arXiv 2302.01790; *Metrics Reloaded* arXiv 2206.01653 [ABIERTO solo resumen] | Catálogo de trampas y selección de métricas | 0 | Qué dicen sobre bootstrap/agregación: NO VERIFICADO. |
| Sirohi et al., *Uncertainty-aware Panoptic Segmentation*, arXiv 2206.14554 [ABIERTO] | uPQ y pECE: PQ ajustada por incertidumbre/calibración | medio | **"Probabilistic Panoptic Quality" como tal: NO VERIFICADO (no lo encontré).** uPQ es otra cosa y no sirve para el concurso. |
| Kaggle: Sartorius 2.º (WBF), HuBMAP 1.º (un U-Net, 4 pliegues) [SNIPPET; páginas no cargaron] | Ensamble/WBF vs modelo único | - | Detalles NO VERIFICADO. |

## 2. Las 6 recetas (todas sin reentrenar salvo la 6)

Notación: `S,TP,FP,FN` por anotación se guardan una vez; PQ agregado = ΣS/(ΣTP+½ΣFP+½ΣFN), como en `ensemble_fil.py`. **Siempre remuestrear imágenes, no anotaciones** (los anotadores de una imagen están correlacionados). IC = percentil 2,5-97,5 de 2000 remuestreos por imagen; comparación = bootstrap emparejado (mismas imágenes remuestreadas para ambos).

**R1. Puerta estadística (0 coste, sin GPU).** *Hipótesis:* cualquier cambio de posproceso/ensamble con Δ<0,010 no se distingue de ruido en esta validación. *Cambio:* sustituir el "gate 0,4293" por: bootstrap emparejado por imagen (Koehn) con los contadores `S,TP,FP,FN` guardados; guardarlos en `result.json`. *Éxito:* promover solo si el IC95 de Δ excluye 0; para cambios gratuitos y sin riesgo usar no-inferioridad (límite inferior > -0,005). Referencias: Koehn, Bestgen, Maier-Hein 2018.

**R2. Filtro de instancias por puntuación con umbral derivado (sin reentrenar).** *Derivación (interpretación mía):* añadir un objeto con probabilidad q de emparejarse sube el PQ solo si q > PQ/(2·SQ-PQ). Con nuestros números (SQ=PQ/RQ=0,4293/0,625=0,687) q* ≈ 0,45: hay que podar objetos cuya probabilidad de acierto sea <45 %. *Hipótesis:* los 365 FP se concentran en componentes de baja probabilidad media; bajar `thr` a 0,4 para tener candidatos y exigir `media_p ≥ τ` (τ∈{0,6…0,85}) o `p90` elimina ≥10 % de FP perdiendo <10 TP (ΔPQ ≥ +0,008). *Cambio:* en `instances()` calcular por componente media de p, p90, área; mantener si score≥τ(área). Precedentes: Mask Scoring R-CNN (el score de clase no refleja la calidad de máscara), DSB 2018 (ranking de candidatos con rasgos morfológicos). *Éxito:* CV por imagen de 5 pliegues (τ ajustado en 4, evaluado en 1), ΔPQ emparejado IC95 > 0 frente al posproceso actual ajustado igual.

**R3. Discrepancia entre vistas TTA como score (inferencia, sin reentrenar).** *Hipótesis:* la desviación entre las 4 vistas dentro de un componente predice que no empareje (Wang et al.: la varianza de TTA estima incertidumbre aleatoria); AUC de "match" de `media_p - λ·std_vistas` supera en ≥0,02 al de `media_p`. *Cambio:* guardar probs por vista (no solo la media) y por modelo en `val_probs`. *Éxito:* diferencia de AUC con IC95 por imagen > 0; después, usar el score en R2 y exigir ΔPQ como en R2.

**R4. Diagnóstico de diversidad y fusión alternativa (sin reentrenar).** *Hipótesis:* los miembros coinciden en >95 % de sus objetos (mismo linaje; Fort et al.), así que la media no puede ganar más de ~0,005. *Cambio:* (a) medir por pares la coincidencia de objetos con IoU>0,5 y la correlación de errores; (b) comparar media de probabilidades, media de logits, media geométrica y **voto por objeto** (se conserva un componente si ≥k de n miembros tienen otro con IoU>0,5), cada uno con su umbral. *Éxito:* si ninguna fusión supera a M1 con IC95 de Δ>0, declarar el ensamble de linaje igual inútil y dedicar GPU a diversidad real (otra pérdida, resolución o arquitectura, como los ganadores del DSB) o a un modelo mejor (Abe et al.).

**R5. Protocolo de selección de posproceso (0 coste).** *Hipótesis:* elegir el centro de la meseta (thr≈0,5, min_area 120) rinde igual que el argmax en CV anidada y varía menos entre pliegues. *Cambio:* ampliar la rejilla a thr 0,3-0,7, validar por CV anidada sobre imágenes y por los bloques temporales de `split_blocks` (pesimista). *Éxito:* PQ anidado de la regla "meseta" ≥ el del argmax -0,002 y menor sd entre pliegues; informar siempre el IC de R1.

**R6. EMA/SWA de pesos y TTA ampliado (poco reentreno).** *Hipótesis:* un EMA(0,999) de la continuación `cont` (o SWA de los últimos checkpoints) iguala o mejora `best.pt`, que se eligió por PQ de val (sesgo de selección), sin coste en inferencia (Izmailov et al.; Tarvainen y Valpola; Model soups para pesos de linaje común); pasar de 4 a 8 vistas (grupo diédrico) y probar media frente a mediana de vistas. *Éxito:* no-inferioridad (límite inferior de Δ > -0,005) y menor varianza entre semillas; no se puede exigir Δ>0 con 106 imágenes. Moshkov et al. respaldan TTA simple con fusión adecuada, sin detalle verificado.

## 3. Descartes

- **STAPLE:** con 2-4 miembros casi idénticos y fondo >> filamento no hay diversidad de errores que modelar; el preprint abierto dice que se reduce a voto con umbral. Se sustituye por voto por objeto (receta 5).
- **WBF/NMS de cajas:** la salida es una máscara por píxel; no hay cajas ni scores por objeto de la red.
- **Más semillas/instantáneas del mismo linaje:** Fort et al. muestra baja diversidad dentro de una trayectoria y las semillas locales ya dan ±0,005; coste alto, ganancia dentro del ruido.
- **uPQ / PQ probabilística:** no es la métrica del concurso; no aporta al posproceso.
- **Lipton literal (umbral = F1/2):** exige calibración que el lienzo no tiene (se entrena con consenso suave y el umbral óptimo de píxel sale 0,6+); úsese solo como pista de que el umbral de objeto va más bajo que el de píxel.
- **Histéresis:** ya probada localmente (0,424 frente a 0,426).
- **Stacking aprendido sobre 106 imágenes:** sobreajuste casi seguro; solo reglas de 1-3 parámetros con CV anidado.
