# Filamentos solares — conclusiones de la tanda 27-28 de septiembre de 2026

Métrica: Panoptic Quality (PQ) por anotador-imagen, validación fija (168 anotaciones, imágenes únicas);
posproceso ajustado en esa validación (optimista ~+0,03 frente al público). Todos los números están en
`kaggle/HISTORIAL.md` y en `kaggle/filament/runs/*/result.json`.

## Resultado

| Hito | Val PQ | Público |
|---|---|---|
| np_ret_v2 (lienzo a resolución completa) | 0,367 | 0,31 |
| ms_learned (lienzo multiescala + superresolución aprendida, 8k it) | 0,406 | 0,33 |
| ms_learned_cont (+16k it) | 0,426 | 0,34 |
| Ensamble learned_cont + cons_cont | **0,429** | **0,35** |
| Ensamble de 4 (2 largos + 2 semillas) | 0,426 | no enviado (gate 0,4293) |

## Qué funciona y qué no (8k it salvo indicación)

| Idea | Val PQ | Veredicto |
|---|---|---|
| Semillas del modelo base (0 / 1 / 2) | 0,406 / 0,400 / 0,410 | ruido ≈ ±0,005 |
| Multiescala: bilineal / FSR 1 / aprendida | 0,343 / 0,349 / 0,406 | la ganancia es el lienzo fino aprendido |
| FSR largo (24k) / pequeño c=24 / grande c=96 | 0,356 / 0,367 / 0,339 | FSR satura pronto; en reserva (DEC-008) |
| Aprendido grande c=96 | 0,407 | el tamaño no aporta; las iteraciones sí |
| +4k it (lr bajo) bilineal / FSR / aprendido | 0,346 / 0,358 / 0,411 | liebre (FSR) frente a tortuga (aprendido) |
| Objetivo de consenso suave | 0,414 (24k: 0,424) | mejora probable (+1,7 σ); candidato a 40k |
| Filtros clásicos [limbo, Sato, DoG] como entrada | 0,380 | peor (−5 σ): la retina ya aprende esos detectores |
| Canales SDO reales [Hα, AIA 304, HMI] | 0,401 | empate: tal cual no aportan |
| Histéresis en el posproceso | 0,424 frente a 0,426 | no mejora |
| Ensamble con el lienzo grande de reposo | 0,397 | peor |

## Hallazgos

- **Techo humano:** un anotador frente a otro da PQ 0,354 (1196 pares); el consenso de dos frente al tercero,
  0,346. El modelo (0,43) ya supera el acuerdo entre humanos: predice el consenso.
- **Los FN son sobre todo ruido de anotación:** 313 de 504 son filamentos pequeños (área mediana 138 px frente
  a 386) que los demás anotadores solo marcan el 31 % de las veces (frente al 90 %).
- Reglas: datos externos públicos permitidos (2.6); 5 envíos/día; el ganador debe poder explicar el método (2.8.b).
- **Aviso de reglas:** compartir código del concurso fuera de Kaggle durante la competición solo está permitido
  si también se publica en los foros o cuadernos de Kaggle (3.6.b). Revisar antes de cualquier `git push` público.

## Siguiente (2026-09-29)

1. **FIL-011:** consenso suave desde cero hasta 40.000 it, validación cada 2.000 (`kaggle/filament/next_fil011.sh`, ~6,5 h).
2. FIL-008 v2: líneas de inversión de polaridad del magnetograma como canal (en vez de HMI crudo).
3. FIL-010: autoentrenamiento con más imágenes GONG públicas.
4. DOC-001: GIFs (`docs/visual/make_*_gif.py`) e informe de 4 páginas.
