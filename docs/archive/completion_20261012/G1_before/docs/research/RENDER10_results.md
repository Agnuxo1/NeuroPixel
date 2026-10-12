# OpenGL 4.6: complete registered timing cohort

All 245 registered rows and 77 output gates are complete. The original 143 rows are preserved exactly; only missing keys were measured. Recovery verifies identical registered input arrays and unchanged source. The interruption separates measurement sessions and qualifies interpretation: these are descriptive technical measurements on one GPU, not independent replications or confidence intervals.

| Workload | CUDA eager ms | CUDA Graph ms | Scalar fragment ms | Vector fragment ms | Compute ms | Best CUDA / compute |
|---|---:|---:|---:|---:|---:|---:|
| C16_grid8 | 4.7211 | 2.4105 | 12.0515 | 2.5491 | 2.2705 | 1.0617× |
| C16_grid32 | 5.3619 | 4.1319 | 16.1152 | 3.5662 | 3.2316 | 1.2786× |
| C16_grid128 | 11.1982 | 10.8098 | 33.1744 | 7.3729 | 5.1868 | 2.0841× |
| C48_grid8 | 5.0039 | 3.2900 | 41.5868 | 3.8431 | 2.9715 | 1.1072× |
| C48_grid32 | 4.7555 | 4.0063 | 45.4874 | 4.1155 | 3.0398 | 1.3179× |
| C48_grid128 | 4.7253 | 9.9665 | 91.2469 | 5.9474 | 9.0106 | 0.5244× |
| RGB8_retina_C16 | 5.8618 | 2.9160 | 14.2554 | 2.8033 | 2.4897 | 1.1712× |

Compute is faster in six point comparisons and slower for C48/grid128. C16/grid128 measures approximately 2.084× relative to the best CUDA comparator. Small differences may be sensitive to technical noise; no universal advantage is claimed.

FP32, TF32 off, resident seeding, sixteen updates and full decoding are held fixed. CUDA eager and CUDA Graph remain strong comparators. Setup, compilation, transfers, training and complete system energy are outside this primary timing. Short counter deltas remain inadequate energy evidence.

The separate sustained RENDER11 energy endpoint preserves 100/105 original blocks after a RAM-floor interruption; do not infer its complete comparative effect without all registered rows. Original failures and queue timeouts remain archived.
