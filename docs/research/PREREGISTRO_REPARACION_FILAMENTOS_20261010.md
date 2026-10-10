# Preregistro: reparación tras daño en el lienzo de filamentos (punto 16, 2026-10-10)

Redactado antes de entrenar ningún modelo nuevo ni de evaluar ninguna lesión. Cualquier cambio posterior de este documento se registra aquí con fecha y motivo.

## Pregunta

¿El entrenamiento con reposo y daño mejora la recuperación del lienzo tras una lesión en test, frente a un control con los mismos pasos variables pero sin daño, y frente al baseline estándar?

## Brazos

| Brazo | Entrenamiento | Semillas | Estado |
|---|---|---|---|
| A (reparación) | `--steps 24 --steps-max 32 --damage-p 0.5` (reposo con daño) | 0, 1, 2 | Por entrenar |
| B (control de pasos) | `--steps 24 --steps-max 32 --damage-p 0` (reposo sin daño) | 0, 1, 2 | Por entrenar |
| C (baseline estándar) | `cv0_base` (24 pasos, sin reposo) | 0, 1, 2 | Existe: `cv0_base`, `cv0_base_s1_rerun1`, `cv0_base_s2` |

El resto de la receta es la de `cv0_base` (fold 0, 8000 iteraciones, lr 1e-3, lowmem, `--ms learned`, mismo manifiesto de fold 0).

## Lesión de evaluación (fija, preregistrada)

- En el test de fold 0 (heldout), en el paso t = 12 de 24, cada celda del estado se anula con probabilidad 0,3 (máscara Bernoulli por píxel, una por muestra).
- Semilla de la máscara: 1000 + id de imagen de test, por imagen; las anotaciones que comparten imagen usan la misma máscara. Enmienda 2026-10-10, antes de cualquier evaluación de lesión: la versión original decía semilla global 0. El cambio asegura que la lesión sea idéntica en los tres brazos aunque cambie el orden o el tamaño de los lotes. No cambia la probabilidad de anulación, el instante ni la métrica.
- El post-proceso (umbral, área mínima, cierre) es el fijado por calibración en cada modelo. No se reajusta sobre el test.
- Pasos de evaluación: 24 para los tres brazos (no los pasos variables del entrenamiento), para que la comparación no dependa del reposo.
- La lesión de evaluación no es la del entrenamiento (que cae en un instante aleatorio y con la misma probabilidad de anulación): se fija aquí para comparar los tres brazos con la misma perturbación.

## Métricas

- PQ_limpio: PQ de test sin lesión (`heldout.PQ` de `result.json`).
- PQ_lesión: PQ de test con la lesión anterior, evaluado con el mismo `best.pt`.
- Índice de recuperación RI = PQ_lesión / PQ_limpio. Métrica primaria.

## Comparaciones y regla de decisión (fijadas ahora)

Sea σ_C la desviación típica de RI entre las tres semillas del brazo C, calculada antes de mirar A ni B.

- ΔRI_AC = media RI(A) − media RI(C); ΔRI_AB = media RI(A) − media RI(B).
- **Apoyo preliminar:** ΔRI_AB > 2σ_C y ΔRI_AC > 2σ_C. Acción: ampliar a 5 semillas por brazo antes de afirmar ninguna mejora.
- **Sin apoyo:** ΔRI_AB ≤ σ_C. Acción: se declara que el daño de entrenamiento no mejora la recuperación en este protocolo.
- **No concluyente:** cualquier caso intermedio.

Las semillas se emparejan por índice (0, 1, 2). El emparejamiento es débil, porque el muestreo de daño cambia la trayectoria de entrenamiento; se declara como límite.

## Límites declarados

- n = 3 por brazo. El criterio de 2σ_C sobre tres semillas es un indicio, no una prueba.
- Una sola lesión (30 % de celdas anuladas, un único instante). No se afirma recuperación frente a otros tipos de daño.
- El brazo A usa en media ~28 pasos frente a 24 de C. El brazo B controla los pasos variables, pero el cómputo no queda emparejado con C.
- Un solo fold y un solo test externo.

## Presupuesto

Seis entrenamientos nuevos (A y B, tres semillas cada uno), de ~90 min cada uno con los pasos variables. Más la evaluación de lesión de los nueve modelos (A, B y C).

## Prohibiciones

- Cambiar la lesión, la métrica o la regla después de ver resultados.
- Seleccionar el checkpoint o el umbral sobre el test.
- Comparar con la línea `RESULT` de los logs (validación).
