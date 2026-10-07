# Soil Grain Size — coordinacion

## Estado

- Datos: 127 fotos, 24 muestras etiquetadas; test con iPhone frente a Motorola/Samsung en train.
- Modelo: Retina + lienzo; cada pixel vota 11 tramos; histograma -> curva acumulada.
- Metrica: EMD logaritmica, menor es mejor.
- CV activa: 4/6 folds; EMD acumulada 49,473 / 35,568 / 35,718 / 39,051; referencia media 75,6-80,1.
- No hay `result.json` final ni envio confirmado.

## Cola local

1. `SOIL-001`: terminar CV y guardar resultado completo.
2. Auditar separacion por muestra, dispositivo y nombres normalizados.
3. Validar submission: columnas, 100 % final, monotonia y ausencia de muestras sin fotos.
4. Registrar en `kaggle/HISTORIAL.md`; enviar solo tras superar el gate.

## Think tank local

- Riesgo principal: cambio de dominio por telefono.
- Comparaciones minimas: curva media, textura clasica/CNN pequena y NeuroPixel.
- El espejo debe evaluarse en CV, no solo incorporarse al envio final.


## SOIL-001 cerrada (Claude, 2026-09-27 20:13)

- 6/6 pliegues: EMD NP 49,473 / 35,568 / 35,718 / 39,051 / 41,446 / 39,195 (acumulada); final **39,195** frente a curva
  media 85,393 y uniforme 100,316. `runs/np_votes/result.json`; historial entrada 7. 228 min en la 3090 compartida.
- La CV se hizo SIN espejo; `--submit` si lo usa. Pliegues aleatorios por muestra (mezclan Motorola/Samsung).
- GPU liberada. SOIL-002 (Codex) LISTA: auditar split, CV dejando fuera un telefono y/o con espejo, envio.
- 20:16: cadena del chat anterior (Claude) en marcha: soil.py --submit (pid 20144) + envio automatico a Kaggle; resultado se anotara aqui

## 2026-10-07T06:27:00.782435+00:00 — SCI-008 isolated evaluation audit

Root claims a read-only source audit and narrowly scoped evaluation corrections in the isolated scientific branch. Scope: CDF conversion/invariants, sample and device grouping, reference fitting and fold aggregation, with synthetic deterministic fixtures. This does not use or stop the historical competition jobs, recover unavailable result files, train on competition data or submit anything. Any score affected by a correction requires its original predictions/data before recalculation can be claimed.


## 2026-10-07T07:17:56.375298+00:00 — SCI-008 closed

SCI-008 evaluation audit is closed on the isolated research branch. Train-only fold reference, fold/pooled accounting, CDF validation, ID/group contracts and prediction/source provenance are implemented. All21 Soil regression methods passed in the pinned CPU suite. Preserved original/intermediate failures and limits are documented in docs/research/08_soil_audit.md and 08_results.md. No real competition data or submission was executed, no historical score was recomputed, and no official hidden-scorer equivalence or model-performance gain is asserted. Release the SCI-008 audit claim; historical competition tasks and reservations are unchanged.
