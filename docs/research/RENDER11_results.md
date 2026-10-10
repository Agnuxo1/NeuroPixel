# Sustained GPU device energy: all registered cases

Actual RTX 3090 board counters were measured over three-second active blocks, bracketed by one-second idle observations. All 77 output gates passed; all 105 registered rows and original source hashes were admitted. Three blocks are technical observations, not independent scientific replications.

| Workload | Method | Gross J per call, median [min,max] | Baseline-adjusted range, median J per call | Calls in three blocks |
|---|---|---:|---:|---:|
| C16_grid8 | CUDA eager | 0.418605 [0.302889,1.012630] | [0.048455,0.083926] | 645 |
| C16_grid8 | CUDA Graph | 0.319232 [0.277135,1.228201] | [0.049928,0.054865] | 683 |
| C16_grid8 | scalar fragment | 0.657943 [0.419085,1.174378] | [0.052027,0.133166] | 547 |
| C16_grid8 | vector fragment | 0.306644 [0.287596,0.309049] | [0.018358,0.044907] | 690 |
| C16_grid8 | OpenGL 4.6 compute | 0.305650 [0.281878,0.374489] | [0.044136,0.048050] | 659 |
| C16_grid32 | CUDA eager | 0.378038 [0.373301,0.491813] | [0.051869,0.058792] | 504 |
| C16_grid32 | CUDA Graph | 0.448378 [0.410114,0.468060] | [0.053417,0.086333] | 480 |
| C16_grid32 | scalar fragment | 0.613586 [0.504776,0.709473] | [0.061096,0.160251] | 408 |
| C16_grid32 | vector fragment | 0.326854 [0.321464,0.335159] | [0.040941,0.054158] | 640 |
| C16_grid32 | OpenGL 4.6 compute | 0.294873 [0.294285,0.352373] | [0.036302,0.054030] | 665 |
| C16_grid128 | CUDA eager | 0.641353 [0.599517,0.684021] | [0.166626,0.191699] | 473 |
| C16_grid128 | CUDA Graph | 0.825419 [0.737446,1.116935] | [-0.087928,0.505780] | 658 |
| C16_grid128 | scalar fragment | 1.555204 [1.512443,1.859479] | [0.185380,1.122422] | 431 |
| C16_grid128 | vector fragment | 0.485110 [0.483244,0.487097] | [0.094008,0.158831] | 568 |
| C16_grid128 | OpenGL 4.6 compute | 0.519053 [0.472606,0.597111] | [0.121451,0.130004] | 457 |
| C48_grid8 | CUDA eager | 0.370683 [0.360422,0.436840] | [0.051246,0.052183] | 541 |
| C48_grid8 | CUDA Graph | 0.328157 [0.298602,0.329557] | [0.052714,0.058374] | 675 |
| C48_grid8 | scalar fragment | 2.145381 [1.758814,2.312470] | [-0.081297,1.721803] | 428 |
| C48_grid8 | vector fragment | 0.351540 [0.342858,0.365335] | [-0.083434,0.061705] | 644 |
| C48_grid8 | OpenGL 4.6 compute | 0.332533 [0.326514,0.451907] | [0.053317,0.059762] | 579 |
| C48_grid32 | CUDA eager | 0.379645 [0.369789,0.399868] | [0.034952,0.080334] | 585 |
| C48_grid32 | CUDA Graph | 0.383279 [0.377566,0.405080] | [0.036185,0.110796] | 647 |
| C48_grid32 | scalar fragment | 2.448627 [2.380153,2.545313] | [0.630566,2.120917] | 500 |
| C48_grid32 | vector fragment | 0.402793 [0.382909,0.436287] | [0.054961,0.110471] | 643 |
| C48_grid32 | OpenGL 4.6 compute | 0.384393 [0.377301,0.407670] | [0.073375,0.089408] | 587 |
| C48_grid128 | CUDA eager | 1.817938 [1.815740,2.076691] | [0.502365,0.611944] | 526 |
| C48_grid128 | CUDA Graph | 1.901175 [1.846000,1.945331] | [0.222592,0.805125] | 570 |
| C48_grid128 | scalar fragment | 12.019172 [11.902415,12.550950] | [5.711083,8.745435] | 189 |
| C48_grid128 | vector fragment | 2.554370 [2.503503,2.681610] | [0.767224,0.867727] | 537 |
| C48_grid128 | OpenGL 4.6 compute | 3.545624 [3.457411,3.622194] | [1.004098,1.882396] | 454 |
| RGB8_retina_C16 | CUDA eager | 0.357070 [0.331940,0.765507] | [0.043501,0.064950] | 625 |
| RGB8_retina_C16 | CUDA Graph | 0.316297 [0.315300,0.317273] | [0.049125,0.052289] | 677 |
| RGB8_retina_C16 | scalar fragment | 0.497052 [0.489900,0.559105] | [0.049905,0.143267] | 485 |
| RGB8_retina_C16 | vector fragment | 0.317075 [0.301218,0.642073] | [0.016721,0.055656] | 919 |
| RGB8_retina_C16 | OpenGL 4.6 compute | 0.283568 [0.179576,0.335519] | [0.036174,0.051992] | 848 |

Gross energy includes board background and idle consumption. The baseline range uses the observed before/after idle rates; it is an estimate, not a confidence interval or unique causal attribution. CPU, RAM, PSU and whole-system wall energy are excluded. Resident inference includes seeding, sixteen updates and the complete decoder, but excludes training, compilation, transfers and publication. No universal energy advantage follows.

Minimum recorded available RAM: 8.027 GiB; maximum ending GPU temperature: 47 °C. Original rows and fixed protocol are retained in `results/research/RENDER11_energy/original_20261009A` and `docs/research/RENDER11_energy_plan.json`.
