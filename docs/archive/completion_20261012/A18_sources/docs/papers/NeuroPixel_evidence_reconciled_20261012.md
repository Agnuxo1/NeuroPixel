# NeuroPixel: bounded visual utility and graphics execution

Francisco Angulo de Lafuente. Evidence reconciliation report, 12 October 2026.

## Abstract

Five archived paired fits on UCI optical digits give mean accuracy 94.3573% for NeuroPixel and 90.8403% for the CNN. The mean paired difference is 3.5170 percentage points, with exploratory Student interval [1.9305, 5.1035]. The completed RENDER10 cohort contains 245 registered timing rows and 77 output gates. These results establish bounded task utility and implementation checks; independent external replication remains pending.

## Methods and statistical scope

The saved VIS07 recipe uses five fresh paired initializations, DEV-only selection, a fixed forty-epoch budget and the official 1,797-image test. The models have similar nominal parameter counts but different functional capacity, receptive geometry and operation costs. The interval concerns training variation on one fixed test population, not general population or writer uncertainty. Graphics timing is FP32 on one RTX 3090, with resident inputs, sixteen updates and complete decoding, compared with CUDA eager and CUDA Graph. Recovered timing sessions and technical repeats are not independent scientific replications.

## Visual outcomes

| Seed | NeuroPixel correct | CNN correct | Test examples |
|---|---:|---:|---:|
| 240 | 1715 | 1614 | 1797 |
| 241 | 1688 | 1643 | 1797 |
| 242 | 1690 | 1623 | 1797 |
| 243 | 1688 | 1642 | 1797 |
| 244 | 1697 | 1640 | 1797 |

## Graphics outcomes

| Workload | Best CUDA ms | Compute ms | Best CUDA / compute |
|---|---:|---:|---:|
| C16_grid8 | 2.410533 | 2.270467 | 1.061691 |
| C16_grid32 | 4.131867 | 3.231633 | 1.278569 |
| C16_grid128 | 10.809767 | 5.186767 | 2.084105 |
| C48_grid8 | 3.290033 | 2.971533 | 1.107184 |
| C48_grid32 | 4.006333 | 3.039833 | 1.317945 |
| C48_grid128 | 4.725300 | 9.010633 | 0.524414 |
| RGB8_retina_C16 | 2.916000 | 2.489667 | 1.171241 |

## Energy admission

The original receipt contains 103 rows. The currently supplied evidence contains 105/105 unique registered rows. The endpoint concerns whole-board counters, including background consumption; CPU, RAM, PSU and wall energy are excluded.

| Workload | Method | Median gross J/call | Technical minimum | Technical maximum |
|---|---|---:|---:|---:|
| C16_grid8 | cuda_eager | 0.418605 | 0.302889 | 1.012630 |
| C16_grid8 | cuda_graph | 0.319232 | 0.277135 | 1.228201 |
| C16_grid8 | scalar_render | 0.657943 | 0.419085 | 1.174378 |
| C16_grid8 | vector_render | 0.306644 | 0.287596 | 0.309049 |
| C16_grid8 | render | 0.305650 | 0.281878 | 0.374489 |
| C16_grid32 | cuda_eager | 0.378038 | 0.373301 | 0.491813 |
| C16_grid32 | cuda_graph | 0.448378 | 0.410114 | 0.468060 |
| C16_grid32 | scalar_render | 0.613586 | 0.504776 | 0.709473 |
| C16_grid32 | vector_render | 0.326854 | 0.321464 | 0.335159 |
| C16_grid32 | render | 0.294873 | 0.294285 | 0.352373 |
| C16_grid128 | cuda_eager | 0.641353 | 0.599517 | 0.684021 |
| C16_grid128 | cuda_graph | 0.825419 | 0.737446 | 1.116935 |
| C16_grid128 | scalar_render | 1.555204 | 1.512443 | 1.859479 |
| C16_grid128 | vector_render | 0.485110 | 0.483244 | 0.487097 |
| C16_grid128 | render | 0.519053 | 0.472606 | 0.597111 |
| C48_grid8 | cuda_eager | 0.370683 | 0.360422 | 0.436840 |
| C48_grid8 | cuda_graph | 0.328157 | 0.298602 | 0.329557 |
| C48_grid8 | scalar_render | 2.145381 | 1.758814 | 2.312470 |
| C48_grid8 | vector_render | 0.351540 | 0.342858 | 0.365335 |
| C48_grid8 | render | 0.332533 | 0.326514 | 0.451907 |
| C48_grid32 | cuda_eager | 0.379645 | 0.369789 | 0.399868 |
| C48_grid32 | cuda_graph | 0.383279 | 0.377566 | 0.405080 |
| C48_grid32 | scalar_render | 2.448627 | 2.380153 | 2.545313 |
| C48_grid32 | vector_render | 0.402793 | 0.382909 | 0.436287 |
| C48_grid32 | render | 0.384393 | 0.377301 | 0.407670 |
| C48_grid128 | cuda_eager | 1.817938 | 1.815740 | 2.076691 |
| C48_grid128 | cuda_graph | 1.901175 | 1.846000 | 1.945331 |
| C48_grid128 | scalar_render | 12.019172 | 11.902415 | 12.550950 |
| C48_grid128 | vector_render | 2.554370 | 2.503503 | 2.681610 |
| C48_grid128 | render | 3.545624 | 3.457411 | 3.622194 |
| RGB8_retina_C16 | cuda_eager | 0.357070 | 0.331940 | 0.765507 |
| RGB8_retina_C16 | cuda_graph | 0.316297 | 0.315300 | 0.317273 |
| RGB8_retina_C16 | scalar_render | 0.497052 | 0.489900 | 0.559105 |
| RGB8_retina_C16 | vector_render | 0.317075 | 0.301218 | 0.642073 |
| RGB8_retina_C16 | render | 0.283568 | 0.179576 | 0.335519 |

This is a descriptive complete device cohort across interrupted sessions; it is not a universal efficiency claim.

## Negative findings and open capabilities

The original relational H1 remains unsupported under its registered recipe. A06.1 component attribution used two initializations; general component benefits are unestablished. DEV04 failed its original precision target. The visual model is trained at eight updates; longer continuations reduce accuracy and increase state magnitude. Historical memory, growth and repair audits do not establish competent continual learning or autonomous learned repair. Rendering a learned state does not identify a unique semantic causal mechanism.

## Mathematical extensions

The completion kit proposes nonnormal-Jacobian diagnostics motivated by the Crouzeix result reported in openai/math family 325, and exact finite Boolean-noise controls corresponding to family 119. Their applicability requires the stated hypotheses. The supplied leaky tanh reference field uses a separate classical contraction argument. Its exact-weight certificate concerns ideal real arithmetic and fixed input; it does not prove useful indefinite memory, all-time GPU roundoff control or an empirical advantage of the new mathematics. Formal proof rebuilding remains a distinct gate.

## Reproduction and independent assessment

Use the frozen per-study source commits, original arrays, weights and receipts. Retain failed executions, numerical discrepancies and negative hypotheses. Publish the complete new cohort before drawing an architectural conclusion. Request external reproduction with an independently supplied final task. Repository publication and internal replay are not peer review or outside replication.

## Evidence locations

- `docs/research/VIS07_results.md` and `results/research/SPRINT_review/VIS07_summary.json`.
- `docs/research/RENDER10_results.md` and `results/research/RENDER10_benchmark/recovered_20261009B/independent_summary.json`.
- `docs/research/06_A06_1_results.md`, `DEV04_results.md` and `TASK4_historical_capabilities_audit_20261009.md`.
- `results/research/RENDER11_energy/` and the original frozen energy plan.

This automatically generated report reconciles saved numbers. The final submission also requires the full methods, references, scope-specific mathematical derivations and independently reviewed limitations described in the completion guide.
