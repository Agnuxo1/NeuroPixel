# G13: revisión adversarial del borrador BET01 (2026-10-13)

Revisor: subagente cog-critic (Opus, esfuerzo xhigh), de solo lectura e independiente del redactor. Revisó el borrador del commit 4ff27818 contra la guía (§18B–§18E, §18L, §18N, §19) y contra el código del kit. Veredicto: **commit_with_fixes**. Todos los hallazgos se aplicaron en `docs/research/BET01_plan.json` antes de cualquier compromiso criptográfico.

## Comprobado correcto

- Umbral de la posición 1: 0,030396. El reparto 2 × 0,015198 lo respeta.
- Techo de §18D: borrado independiente del ítem, ruido solo en los valores supervivientes y |s| ≤ 1 antes del ruido en todos los modelos. La dinámica posterior es determinista y sin entrada.

## Hallazgos y corrección aplicada

| # | Gravedad | Hallazgo | Corrección |
|---|---|---|---|
| 1 | alta | P2 sesgado por el decoder: un campo que se desvanece lee un ítem «por defecto» con margen. Así P2 pasaba trivialmente y quedaba indefinido sin fallos o con r_hat ≥ 0,6. | El estadístico pasa a ser la fracción de fallos con cambio de cuenca, clasificada por estado (atractor), con reglas para instancias sin atractores, sin fallos o con r_hat fuera de rango. Control de validez: el campo certificado debe dar ≤ 0,5. |
| 2 | alta | P4 casi tautológico: r_hat y el punto de separación medían el mismo suceso con máscaras de borrado. Además, la bisección a 0,005 era imposible con pasos de 1/64. | r_hat sale de perturbaciones isótropas en norma L2 sobre los atractores DEV, convertidas a fracción de borrado. Las máscaras de borrado no entran en r_hat. |
| 3 | alta | Potencia de P4 con n = 8 y ±0,05: 0,28 con SD 0,05. La censura en 0,6 permitía una equivalencia degenerada. | 16 instancias TEST, rejilla e = 1…38 con 64 máscaras por ítem y tolerancia ±0,08. Potencia simulada: 0,96–1,0 con SD ≤ 0,07. Más de 2 instancias censuradas implica no apoyada. |
| 4 | alta | Detector de fugas vacío: η máximo alcanzable ≤ 0,903 en todas las celdas. | Celdas de control (f = 0,9, σ = 4, K = 8: techo de 2,099 bits < 3) y control de entrada sola (f = 1,0). |
| 5 | media | S3 no operativo y esquivable. | Cláusula de restauración del estado definida; code_ML cuenta como no restaurador; cuotas de fallo solo descriptivas. |
| 6 | media | S0 no exigía comparadores competentes (§18N, P2). | Competencia en cada semilla DEV, también para los comparadores. |
| 7 | media | S1 esquivable por la conjunción con la retención, y 256 pasos no separan un transitorio lento de una dirección neutral. | Conjunción eliminada; \|λ\| ≥ 1 − 10⁻⁴ y retención a W2 = 4096. |
| 8 | media | Capacidad infradefinida: K = 64, W2, agregación. | Especificación añadida: W2 = 64, semillas K = 64 aparte, η de la celda = mediana. |
| 9 | media | P1 mezclaba el margen en probabilidad con el margen en logits. | Margen en logits ln 3 y decoder agregado K × 512. |
| 10 | media | SF-free no es el StableField del kit. | Modificaciones declaradas. |
| 11 | baja | P3 casi determinado por K/techo; ±0,05 absoluto es laxo. | ±25 % relativo; la extrapolación queda para G16. |
| 12 | baja | Elementos de la guía ausentes. | Desviaciones declaradas, tiempo de reparación y prevalidación del estimador. |
| 13 | baja | Huecos de especificación (daño en GRU/Hopfield, congelado DEV, prueba t, retención con recuerdo bajo, hora del registro). | Todos especificados. |

## Incertidumbre pendiente

La SD real del punto de separación entre semillas es desconocida. Se estima con las semillas DEV y se fija en el congelado DEV antes de generar la sal de TEST.
