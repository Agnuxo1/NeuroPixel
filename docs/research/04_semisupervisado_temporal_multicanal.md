# 04 - Semisupervisado, contexto temporal, fusion multicanal y preentrenamiento solar (2026-10-01)

Metodo: titulos y resumenes abiertos en esta sesion (WebFetch/WebSearch). Lo no abierto se marca NO VERIFICADO. Los numeros del proyecto salen de `docs/FILAMENTOS_2026-09-28.md` (val PQ 0,429 / publico 0,35; techo humano 0,354; canales SDO crudos = empate 0,401 vs 0,406). Hecho = lo dicen las fuentes; Interpretacion = mia.

## 1. Tabla de referencias

| Referencia (abierta) | Idea | Coste | Aplicabilidad |
|---|---|---|---|
| ST++ (Yang et al., arXiv 2106.05095, CVPR 2022) | Autoentrenamiento con aumentos fuertes + reentrenamiento selectivo por estabilidad de prediccion entre checkpoints, a nivel de imagen | Bajo-medio (2-3 rondas) | Alta: ya hay modelo base y miles de GONG sin etiquetar |
| UniMatch (Yang et al., 2208.09910, CVPR 2023) | Consistencia debil-a-fuerte (FixMatch) + perturbacion de rasgos + doble vista fuerte; su resumen cita teledeteccion y medicina | Medio (2 pasadas fuertes) | Alta, pero con aumentos que no rompan la fisica solar |
| FixMatch (2001.07685) | Pseudoetiqueta con umbral de confianza sobre vista debil | Bajo | Media: umbral duro choca con ruido de anotador |
| Mean Teacher (1703.01780) / Temporal Ensembling (1610.02242) | Profesor = media movil de pesos (o de predicciones por epoca) | Bajo | Alta como estabilizador del profesor |
| Noisy Student (1911.04252) | Iterar profesor-alumno con ruido en el alumno; alumno igual o mayor | Alto | Baja: en el proyecto el tamano no aporto (c=96 = 0,407) |
| CPS (2106.01226) | Dos redes se pseudoetiquetan entre si | 2x | Baja: duplica coste, ganancia sobre UniMatch no probada aqui |
| U2PL (2203.03884) | Pixeles poco fiables como negativos por clase (entropia, colas) | Medio | Baja: tarea binaria, sin clases que contrastar |
| Diercke et al. (2402.15407, A&A 2024) | Filamentos Halpha: 955 imagenes ChroTel etiquetadas -> YOLOv5 -> mascaras ruidosas por umbral en 16.759 GONG sin etiquetar -> U-Net | Bajo (precedente) | Alta: precedente directo; precision ~50 %, recall 93 %, falla en filamentos debiles/pequenos |
| Dataset MAGFiLO (Nature Sci. Data; resultado de busqueda, pagina bloqueada) | 10.244 filamentos, 1.593 observaciones GONG 2011-2022, kappa 0,66 | - | Aviso: probablemente sea la fuente del concurso (interpretacion); riesgo de solape con test |
| SegProp (2010.01910), Zhu et al. (1812.01593), Vincent et al. (2503.15676) | Propagacion de etiquetas y consistencia en video aereo/urbano, flujo optico | Medio-alto | Baja directa: la dinamica de filamentos no es flujo optico entre dias; sirve la idea de propagar |
| Automatic detection of filament oscillations I (arXiv 2607.01095) | GONG a 1 imagen/min, coalineado a 12:00 UT; coherencia temporal por persistencia de mascaras | - | Muestra que hay decenas de vistas por dia (cadencia verificada) |
| BF-TrackFormer / DETR+U-Net+seguimiento (solo resumen de busqueda; ScienceDirect dio 403) | Seguimiento de filamentos entre imagenes | - | Concepto util; detalles NO VERIFICADO |
| Karachik y Pevtsov (1307.3317) | Gradiente del campo en la linea neutra: favorece filamentos pero no basta como unico predictor | - | Justifica PIL como prior blando, no filtro duro |
| Surya (Roy et al., 2508.14112) | 366M parametros, 13 canales SDO (8 AIA + 5 HMI), 4096x4096, patch 16 (65.536 tokens), Apache 2.0; tarea ARPIL: IoU 0,768 frente a U-Net 0,688 | Muy alto | Baja: no ve Halpha; util solo para el lado SDO/PIL |
| ARPIL (dentro de Surya) | Receta de PIL: mapas de polaridad con umbral +-50 G, quitar regiones <100 px, dilatar 10 px, intersectar | Trivial | Alta: receta reproducible para el canal PIL |
| SDO-FM (2410.02530) y SDOFMv2 (Springer, bloqueado) | Modelo base multi-instrumento y embeddings en Hugging Face | Alto | Baja; detalles de v2 NO VERIFICADO |
| AIA-JEPA | No se encontro ningun articulo | - | NO VERIFICADO: no usar como referencia |

## 2. Cinco recetas prometedoras

Regla de validacion comun: bloques temporales (GroupKFold por fecha) con huecos de ~1,5 dias entre val y su vecino de train mas cercano, igual que el test; nunca KFold aleatorio por imagen. Comparar siempre contra la validacion fija de 168 anotaciones y la semilla base (ruido +-0,005). Gate de envio: > 0,4293.

**R1. Autoentrenamiento selectivo ST++ con objetivo suave (FIL-010).**
- Hipotesis: anadir GONG publicos sin etiquetar (Diercke usan 16.759, 4 h de cadencia) con pseudoetiquetas suaves del ensamble actual y seleccion por imagen mejora val PQ >= +0,010.
- Cambio: ensamble -> probabilidades en N imagenes GONG; seleccionar el 50 % mas estable (acuerdo entre checkpoints, como ST++); alumno con aumentos fuertes y perdida de consenso suave (ya hay codigo de consenso suave); 2 rondas.
- Riesgos: fuga si las GONG sin etiquetar incluyen fechas de val/test: excluir esas fechas (+-1 dia) y, por si MAGFiLO coincide con el test, no usar etiquetas de MAGFiLO sin revisar la regla 2.6. Confirmacion del profesor sobre si mismo: no seleccionar usando val.
- Exito: val PQ >= 0,439 en dos semillas y publico >= 0,36; abandonar si < +0,005. Cuidado: el modelo ya supera el acuerdo humano, asi que lo que se gana es estabilidad en filamentos pequenos, no mas FN.

**R2. Consistencia debil-fuerte con vistas temporales naturales (UniMatch adaptado).**
- Hipotesis: dos imagenes GONG del mismo dia (minutos de diferencia, cadencia 1/min) son aumentos naturales (seeing, ruido); exigir prediccion consistente mejora la robustez y el PQ en +0,005 a +0,015.
- Cambio: lote no etiquetado = par (t, t+k min) ya coalineado; profesor EMA (Mean Teacher) sobre la vista debil; alumno sobre la fuerte con perturbacion de rasgos (UniMatch). Sin volteos que cambien la quiralidad si se usara quiralidad.
- Riesgos: el limbo y la rotacion solar desplazan el contenido entre vistas: usar k pequeno o derrotar. Sin fuga si los pares excluyen fechas de val.
- Exito: val PQ +0,008 sobre R0 con 3 semillas; si el efecto se confunde con R1, probar aislado.

**R3. Contexto de vecino temporal con derotacion diferencial.**
- Hipotesis: la mediana de 1,5 dias al vecino de train hace que su mascara/prediccion, derotada al instante t, sea un prior util; fusionar predicciones de t-1, t, t+1 sube PQ >= +0,01 (Interpretacion: los filamentos persisten dias; la literatura abierta confirma que existe seguimiento, pero no mide esto).
- Cambio: (a) inferencia: derotar (sunpy, NO VERIFICADO en esta sesion) las probabilidades del modelo en vecinos y promediar con peso por distancia; (b) opcional: canal extra con la mascara derotada del vecino etiquetado, con dropout de canal en entrenamiento.
- Riesgos: la fuga es el peligro mayor. Una imagen de train no puede recibir su propia mascara ni la de duplicados cercanos; en val, usar solo vecinos de train a la distancia real del test. Mascaras de vecino como canal exigen simular el hueco de 1,5 dias en entrenamiento (excluir vecinos a < 12 h).
- Exito: val PQ >= +0,010 con hueco realista; si solo mejora con vecinos a < 12 h, es fuga: descartar.

**R4. Canal de PIL del magnetograma (FIL-008 v2).**
- Hipotesis: un mapa PIL precalculado (receta ARPIL) como canal o cabeza auxiliar mejora el recall de filamentos debiles donde el HMI crudo empato (0,401).
- Cambio: HMI LOS en el instante GONG, derotado y reproyectado a la rejilla GONG; +-50 G, filtro <100 px, dilatacion, interseccion; ademas suavizado y distancia a la PIL como canal continuo; fusion como canal 2 con dropout, o cabeza auxiliar de PIL.
- Riesgos: desalineacion GONG/HMI (escala, rotacion, hora); Karachik y Pevtsov: el gradiente solo no determina filamentos, usar prior blando. Sin fuga: el mapa no usa etiquetas del concurso.
- Exito: val PQ >= +0,008 y mejora en filamentos finos; si empata, cerrar la linea magnetica.

**R5. Preentrenamiento temporal sobre GONG sin etiquetas.**
- Hipotesis: predecir la imagen t+dt derotada (pretexto de avance temporal, el mismo de Surya) sobre >10k GONG da rasgos que superan la inicializacion aleatoria; +0,01 PQ o convergencia con <=50 % de iteraciones.
- Cambio: preentrenar el lienzo/encoder con perdida de siguiente imagen o contraste temporal y afinar con la tarea.
- Riesgos: sin etiquetas no hay fuga, pero excluir fechas de val/test del preentrenamiento si se desea una validacion limpia. Coste GPU por la cola gpuq.
- Exito: igual PQ en 50 % de iteraciones o +0,01 final. Evidencia previa en solar: solo el resultado de Surya en SDO (no en Halpha), asi que es apuesta, no hecho.

## 3. Descartes

- Noisy Student, CPS, U2PL: coste o supuestos (clases, red doble) sin beneficio esperable frente a R1/R2; el tamano de modelo ya no aporto.
- Flujo optico y SegProp entre dias: los filamentos cambian y la rotacion es derotable de forma analitica; el flujo aprendido es ruido.
- Surya/SDO-FM como extractor directo: 4096x4096x13 canales, sin Halpha, 65.536 tokens; coste desproporcionado para ~700 imagenes. Solo valdria destilar un PIL (R4 lo da con umbrales).
- Canales AIA/HMI crudos: ya medidos (empate).
- Pseudoetiquetas con umbral duro tipo FixMatch: los FN restantes son ruido de anotador (313 de 504 son filamentos pequenos marcados por el 31 % de los anotadores); un umbral duro fija ese ruido. Preferir suave.
- AIA-JEPA: sin fuente abierta, no se usa.
- Etiquetas de MAGFiLO como datos extra: riesgo de solape con test y de infringir 2.6/2.8.b; revisar antes.
