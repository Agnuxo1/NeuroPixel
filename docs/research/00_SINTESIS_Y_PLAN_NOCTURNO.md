# RES-001 — Síntesis de la investigación (1-oct-2026) y plan de pruebas

Fuentes: seis informes de `docs/research/01..06_*.md` (solo cita papers abiertos en la sesión; lo demás va marcado «NO VERIFICADO»).
Las cifras marcadas «medido» son nuestras; las demás vienen de la literatura o son derivaciones propias de los investigadores.

## 1. Lo que dicen los seis informes a la vez (convergencias)

1. **El cuello de botella no es el tamaño ni las iteraciones**, sino el desacuerdo entre anotadores. Ningún trabajo científico revisado usa PQ;
   las únicas cifras PQ ajenas vienen de repositorios de participantes, sin auditar (01). Un modelo pequeño no es una desventaja: Flat U-Net
   (arXiv:2502.07259) reporta 0,26 M parámetros con Dice 0,79 frente a 0,69 de una U-Net de 29 M (01).
2. **El PQ de validación es muy ruidoso** (medido, 06): IC95 del PQ 0,426 = [0,400; 0,450] (±0,026) remuestreando por imagen; ensamble de 2 frente a 1:
   Δ = +0,003 [−0,006; +0,013]; de 4 frente a 1: +0,0001 [−0,009; +0,010]. Con la validación actual solo se detectan diferencias ≳ 0,010 y depende
   más de qué imágenes caen en val (sd 0,028 entre particiones) que de la configuración. **Toda decisión debe tomarse con varios folds y bootstrap emparejado por imagen.**
3. **Umbral por objeto (tamaño/puntuación):** añadir un objeto con probabilidad p de emparejarse sube el PQ solo si p supera un umbral. Derivaciones propias de
   dos investigadores: PQ/(2q) ≈ 0,27-0,36 (02) y PQ/(2·SQ−PQ) ≈ 0,45 (06). **Discrepan**: mi comprobación a mano apoya la primera, pero la agregación
   por anotador-imagen puede cambiarla. Se resuelve empíricamente barriendo el umbral de puntuación en val (CPU, minutos).
4. **El umbral global del posproceso está agotado** (medido, 06): meseta plana 0,424-0,429 entre 0,3 y 0,7; la histéresis tampoco mejora.
5. **Los FN pequeños son sobre todo ruido** (medido): los demás anotadores solo los marcan el 31 % de las veces. Cualquier pérdida de recuperación de pequeños
   debe activarse solo con consenso (03). El autoentrenamiento previo en GONG (Diercke et al., arXiv:2402.15407) llegó a recall 93 % con precisión ~50 %: **peligro de amplificar FP** (04).
6. **Riesgo de fuga de datos:** MAGFiLO público parece ser la fuente del concurso (interpretación de 04; un README afirma que contiene etiquetas del test: NO VERIFICADO).
   No entrenar con él. El contexto temporal debe validarse con el hueco real de ~1,5 d y excluyendo vecinos a < 12 h (04).

## 2. Ideas que pasan el filtro (hipótesis falsables)

Criterio común: 3 semillas o 3 folds, efecto = media del Δ PQ frente a la base con IC por bootstrap emparejado; éxito si el IC95 inferior > 0 o Δ ≥ +0,010.

| ID | Prueba | Origen | Coste | GPU | Hipótesis |
|---|---|---|---|---|---|
| P1 | Filtro de objetos por puntuación + umbral por tamaño, calibrado en una mitad y evaluado en la otra | 02 R1/R2, 06 R2, 01 | minutos | no | +0,01 PQ sin tocar la red |
| P2 | Bootstrap emparejado por imagen como puerta de decisión (`06_bootstrap_pq.py`) | 06 R1 | hecho | no | evita falsos positivos de mejora |
| P3 | TTA 8 vistas (D4), teselas con solape | 03 R1, 01, 06 R6 | minutos | inferencia | +0,008 con posproceso congelado (01 avisa: la inferencia nativa por teselas llegó a empeorar) |
| P4 | EMA de pesos (no-inferioridad) | 06 R6, 01 | bajo | sí | ≥ base con menos varianza entre semillas |
| P5 | Control honesto: U-Net de ~91 k parámetros, mismos datos/pérdida/iteraciones | 05 | medio | sí | si ≥ 0,424 el NCA no gana por parámetro (dato clave del informe) |
| P6 | Skeleton Recall en el lienzo fino (esqueleto del GT precalculado) | 03 R2 | medio | sí | FP «trozos» 219 → ≤ 185, PQ +0,01 |
| P7 | Pérdida en pasos intermedios con rollout aleatorio (M1) | 05 | medio | sí | PQ ≥ 0,435 media de 3 semillas (refutada < 0,430) |
| P8 | Pérdida por instancia con peso por tamaño solo con consenso (`--small-frac`/`--consensus-w`, en curso) | 03 R3 | medio | sí | FN pequeños con consenso −15 % |
| P9 | Deformación elástica solo en la etiqueta (1-2 px) | 02 R4 | bajo | sí | +0,005-0,01 por el borde ambiguo |
| P10 | PIL del magnetograma (±50 G, regiones > 100 px, dilatación) como canal suave | 04 R4 | medio | sí | solo ayuda en filamentos débiles; SDO crudo ya empató |
| P11 | Autoentrenamiento suave (ST++) sobre GONG sin etiquetas, con filtro de consistencia | 04 R1 | alto | sí | mejora un bloque temporal independiente; riesgo FP (Diercke) |
| P12 | Contexto de vecino temporal (t−1, t+1) con hueco simulado | 04 R3 | alto | sí | mejora FN débiles; riesgo de fuga alto |

Descartadas por los seis informes: Mask R-CNN (PQ 0,15 en 01), atención pesada y backbones grandes, DenseCRF, clDice/Betti/TopoLoss (caros; nuestros fallos son fragmentos),
StarDist/Cellpose/CenterMask, ELR/Co-teaching/DivideMix (con 91 k parámetros no hay memorización), modelos de anotador (no hay identidad), STAPLE como camino principal,
Noisy Student/CPS/U2PL, Surya/SDO-FM como extractor directo, Lovász/SoftPQ como única novedad, DEQ/Universal Transformers/ViTCA/DiffLogic CA, más semillas del mismo linaje.

## 3. Secuencia propuesta

- **Hoy (CPU, sin GPU):** P1, P2, P3 sobre `val_probs.npz` guardados (+ código de P4-P7).
- **Cadena diurna ya en cola:** cv0_cons, cv0_consw, cv0_small (P8 y consenso) y el ensamble de 3.
- **Noche 01→02:** protocolo estricto de Codex (fold 0, 8k, lr 1e-3, semilla 0): cv0_ema, cv0_unet (control), cv0_skel, cv0_m1.
- **Noche 02→03:** semillas/folds de lo que pase el filtro; después P9, P10 y, solo si algo funciona, P11-P12.
- **Siempre:** una carga larga a la vez por gpuq; registrar cada corrida en `kaggle/HISTORIAL.md`; Δ medido por bootstrap emparejado.

## 4. Resultados medidos del día

- **P1 (filtro de objetos por puntuación + umbral por tamaño), CPU, ensamble 0,4293:** calibrado en una mitad de imágenes y evaluado en la otra, Δ PQ = −0,0004
  (IC95 de particiones [−0,010; +0,007], P(>0) = 0,49). Mejor config global (optimista) 0,4335 (+0,004). **Refutada: el posproceso ya está en su meseta;
  no aporta nada que se distinga del ruido.** El umbral por objeto no es la palanca (consistente con el umbral global agotado de 06).
