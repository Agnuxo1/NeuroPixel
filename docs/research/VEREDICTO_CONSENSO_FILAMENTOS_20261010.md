# Veredicto preregistrado: consenso frente a baseline (fold 0)

Regla y análisis preregistrados en `docs/research/PREREGISTRO_CONSENSO_FILAMENTOS_20261010.md` (commit 0796d702) y calculados con `docs/research/analyze_consenso_prereg.py`.

| Semilla | Baseline (heldout PQ) | Consenso (heldout PQ) | Δ |
|---|---|---|---|
| 0 | 0,3850 | 0,3953 | +0,0103 |
| 1 | 0,3878 | 0,4011 | +0,0133 |
| 2 | 0,4015 | 0,4014 | −0,0001 |

Δ medio = **+0.0078**; desviación típica de Δ = 0.0070.

**Veredicto (regla 2 del preregistro): SIN APOYO.** Con Δ medio menor de +0,010 y un Δ negativo (semilla 2), la regla declara que no hay evidencia de mejora del consenso suave en este protocolo.

Límites:
- Tres semillas y un solo fold. Con esta dispersión el contraste tiene poca potencia: el veredicto no demuestra que el consenso no mejore con más semillas.
- El Δ de la semilla 0 se observó antes del preregistro; así se declaró en el documento de preregistro.
- No hay bootstrap emparejado por instancia: las salidas por imagen no se guardan en los runs.
- Los dos primeros pares a favor no bastan por sí solos: el preregistro exige tres pares y la regla se aplicó sin cambios.
