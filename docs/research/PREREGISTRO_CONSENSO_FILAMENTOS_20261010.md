# Preregistro: consenso suave frente al baseline en fold 0 (punto 9, 2026-10-10)

Redactado antes de conocer los resultados de `cv0_skel`, `cv0_cons_s1` y `cv0_cons_s2`. Cualquier cambio posterior de esta regla debe registrarse aquí con fecha y motivo.

## Hipótesis

El objetivo de consenso suave (media de los anotadores por imagen, `--consensus`) mejora el PQ de test externo (heldout) frente al baseline con la misma receta, en fold 0, con presupuesto emparejado (8000 iteraciones, lr 1e-3, lowmem, mismas semillas).

## Métrica primaria

`PQ` del bloque `heldout` de `runs/<run>/result.json` (evaluation_set = test, selection_set = calibration; post-proceso fijado por calibración). La línea `RESULT` de los logs es validación y no se usa para comparar.

## Diseño

- Baseline: `cv0_base` (semilla 0), `cv0_base_s1_rerun1` (semilla 1), `cv0_base_s2` (semilla 2).
- Consenso: `cv0_cons` (semilla 0), `cv0_cons_s1` (semilla 1), `cv0_cons_s2` (semilla 2).
- Comparación emparejada por semilla: Δ_s = PQ_cons(s) − PQ_base(s), para s = 0, 1, 2.

## Valores ya observados antes de este preregistro

| Semilla | Base (heldout PQ) | Consenso (heldout PQ) | Δ |
|---|---|---|---|
| 0 | 0,3850 | 0,3953 | +0,0103 |
| 1 | 0,3878 | pendiente | — |
| 2 | 0,4015 | pendiente | — |

Base: media 0,3948, desviación típica ~0,0097 entre semillas. Δ_0 se observó antes de este documento; el resto de Δ aún no.

## Regla de decisión (fijada ahora)

Sea Δ̄ la media de Δ_s sobre las tres semillas.

1. **Apoyo preliminar fuerte:** Δ̄ ≥ +0,010 y Δ_s > 0 para las tres semillas. Acción: encolar semillas adicionales (al menos hasta 5 por brazo) antes de afirmar ninguna mejora. Con tres semillas no se declara mejora.
2. **Sin apoyo:** Δ̄ < +0,005, o cualquier Δ_s < 0 con Δ̄ < +0,010. Acción: se declara que no hay evidencia de mejora del consenso en este protocolo.
3. **Zona intermedia** (+0,005 ≤ Δ̄ < +0,010): no concluyente. Acción: ampliar semillas.

## Límites declarados

- Con n = 3 los intervalos son poco fiables. Una desviación típica de ~0,01 en el baseline significa que una diferencia de 0,010 es del orden del ruido entre semillas. El criterio 1 es un indicio, no una prueba.
- Un solo fold (fold 0) y un solo test externo.
- No hay bootstrap emparejado por instancia: las salidas por imagen no se guardan en los runs.
- Los pesos de la variante (`--consensus-w`, `--skel-w`, `--aux-w`, `--small-frac`) no están preregistrados aquí; cualquier conclusión sobre ellos requiere su propio preregistro.

## Prohibiciones

- Comparar con la línea `RESULT` de los logs.
- Cambiar umbrales, la métrica o la regla después de ver los resultados.
- Añadir variantes a esta comparación.
