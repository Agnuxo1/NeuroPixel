# Sustained device energy: complete registered cohort

The original RTX 3090 execution preserved 103 of 105 registered rows and all 77 output gates before the unchanged 8 GiB RAM floor stopped it. Its raw receipt remains unchanged. The previously completed recovery from Git archive `87c60227f59f2763e474b12e08f22d2c4104aafb` is now admitted locally: 105 unique keys, the 103 original rows identical, and zero repeated closed measurements. No physical measurement occurred during this admission.

Only the RGB8 retina, technical block 2, methods `render` and `vector_render` were absent from the original receipt. Their prior recovery uses the same frozen plan, source and registered inputs. Three technical blocks across interrupted sessions are not independent scientific replications.

Original receipt SHA-256: `883fc6be14663f112d6fdf36f4ef112e35215bddfa7f9699d3c37a85559719a2`. Recovered receipt SHA-256: `bc19c7b3bff67b3fc8fe8544188f63aaf342a7cfaa435f3ef3e89e397ff1039f`. See the [recovered receipt](../../results/research/RENDER11_energy/recovered_20261009B/receipt.json), [admitted table](../../results/research/COMPLETION20261012/energy_complete_summary.json) and [admission proof](../../results/research/COMPLETION20261012/energy_recovery_admission.json).

| Workload | Method | Median gross J/call | Technical minimum | Technical maximum |
|---|---|---:|---:|---:|
| C16_grid8 | cuda_eager | 0.418604790 | 0.302889362 | 1.012629630 |
| C16_grid8 | cuda_graph | 0.319231818 | 0.277135135 | 1.228200980 |
| C16_grid8 | scalar_render | 0.657943396 | 0.419085000 | 1.174377593 |
| C16_grid8 | vector_render | 0.306643836 | 0.287595918 | 0.309048673 |
| C16_grid8 | render | 0.305650442 | 0.281877551 | 0.374489362 |
| C16_grid32 | cuda_eager | 0.378038462 | 0.373300546 | 0.491812950 |
| C16_grid32 | cuda_graph | 0.448378205 | 0.410114286 | 0.468060403 |
| C16_grid32 | scalar_render | 0.613585938 | 0.504776471 | 0.709472727 |
| C16_grid32 | vector_render | 0.326854460 | 0.321463636 | 0.335159420 |
| C16_grid32 | render | 0.294873418 | 0.294285106 | 0.352373057 |
| C16_grid128 | cuda_eager | 0.641352941 | 0.599516854 | 0.684021127 |
| C16_grid128 | cuda_graph | 0.825419192 | 0.737446009 | 1.116935223 |
| C16_grid128 | scalar_render | 1.555203947 | 1.512443038 | 1.859479339 |
| C16_grid128 | vector_render | 0.485109890 | 0.483243781 | 0.487097297 |
| C16_grid128 | render | 0.519052632 | 0.472605882 | 0.597111111 |
| C48_grid8 | cuda_eager | 0.370682796 | 0.360421875 | 0.436840491 |
| C48_grid8 | cuda_graph | 0.328157407 | 0.298602410 | 0.329557143 |
| C48_grid8 | scalar_render | 2.145380597 | 1.758813559 | 2.312470085 |
| C48_grid8 | vector_render | 0.351539906 | 0.342858333 | 0.365335079 |
| C48_grid8 | render | 0.332532710 | 0.326514019 | 0.451907285 |
| C48_grid32 | cuda_eager | 0.379644670 | 0.369788945 | 0.399867725 |
| C48_grid32 | cuda_graph | 0.383278761 | 0.377565611 | 0.405080000 |
| C48_grid32 | scalar_render | 2.448627119 | 2.380153409 | 2.545312925 |
| C48_grid32 | vector_render | 0.402792793 | 0.382909465 | 0.436286517 |
| C48_grid32 | render | 0.384392857 | 0.377301435 | 0.407670330 |
| C48_grid128 | cuda_eager | 1.817937500 | 1.815739645 | 2.076690608 |
| C48_grid128 | cuda_graph | 1.901175000 | 1.846000000 | 1.945331361 |
| C48_grid128 | scalar_render | 12.019171875 | 11.902415385 | 12.550950000 |
| C48_grid128 | vector_render | 2.554369565 | 2.503502762 | 2.681610465 |
| C48_grid128 | render | 3.545624204 | 3.457411043 | 3.622194030 |
| RGB8_retina_C16 | cuda_eager | 0.357070000 | 0.331940367 | 0.765507246 |
| RGB8_retina_C16 | cuda_graph | 0.316296804 | 0.315299559 | 0.317272727 |
| RGB8_retina_C16 | scalar_render | 0.497052326 | 0.489900000 | 0.559104895 |
| RGB8_retina_C16 | vector_render | 0.317074890 | 0.301218107 | 0.642073497 |
| RGB8_retina_C16 | render | 0.283567568 | 0.179575597 | 0.335518868 |

Whole-board NVML cumulative counters include background consumption. Active blocks last at least three seconds, with one-second idle observations before and after. Idle-adjusted estimates, including negative values, remain in the raw receipt and table without clipping. Baseline ranges are descriptive estimates, not confidence intervals or unique process attribution. CPU, RAM, PSU, training, installation and whole-system wall energy remain unmeasured. This complete device cohort does not establish universal efficiency or energy savings.
