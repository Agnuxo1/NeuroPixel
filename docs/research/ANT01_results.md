# ANT01: memory and repair against the closest antecedent (13 October 2026)

**Protocol.** Frozen plan `docs/research/ANT01_plan.json` (SHA-256 `38436579…`, committed in d2700793 before the executor existed). Equal state size: 64 binary cells on an 8×8 grid. Hebbian storage of K patterns in a global Hopfield network and in locally coupled versions (3×3 and 5×5 neighbourhoods), asynchronous sweeps, and an explicit nearest-pattern decoder as the "erasure code with explicit decoder" reference. Grid: K ∈ {1, 2, 4, 8, 12}, erased fraction f ∈ {0.1, 0.25, 0.5}, noise σ ∈ {0, 0.5}, i.i.d. and spatially smooth patterns; 20 memory instances × 10 cues per cell. CPU, NumPy only, 5.9 s. Results: `results/research/ANT01/results.json`.

**Results (exact repair, ranges over f and σ).**

| Patterns | K | Global Hopfield | Local 3×3 | Local 5×5 | Explicit decoder |
|---|---:|---:|---:|---:|---:|
| i.i.d. | 2 | 100% | 44.5–94.5% | 98.5–100% | 100% |
| i.i.d. | 4 | 99.5–100% | 0–2% | 17–46.5% | 100% |
| i.i.d. | 8 | 85–95.5% | 0% | 0–1.5% | 100% |
| i.i.d. | 12 | 38–56% | 0% | 0% | 100% |
| smooth | 2 | 100% | 61–98.5% | 86–99% | 100% |
| smooth | 4 | 55.5–76.5% | 7–21.5% | 7–22% | 100% |
| smooth | 8 | 13.5–26% | 0–4% | 0–4.5% | 100% |
| smooth | 12 | 2–5% | 0–3% | 0–0.5% | 100% |

Both registered predictions held: global Hopfield repaired at least as often as both local versions in all 24 i.i.d. cells with K ≥ 2 (PR1), and the explicit decoder identified the target at least as often as every Hopfield model in all 60 cells (PR2). Spatially smooth, correlated patterns degraded every Hopfield variant, and locality did not help them.

**Interpretation.** At equal state size, restricting Hebbian coupling to a local neighbourhood sharply lowers storage and repair; an explicit decoder dominates every dynamical memory here. By the registered interpretation rule, a memory-repair programme for local fields cannot claim superior storage at equal state size. It must justify locality by other properties, such as cost, local repair or persistence, and compare against these baselines.

**Limits.** These are binary Hebbian Hopfield variants, not the learned NeuroPixel rule with hidden channels, on synthetic patterns. The study is exploratory, with no external custody and no alpha spent. Cues within an instance are not independent: per-cell intervals use the 20 instances as units.
