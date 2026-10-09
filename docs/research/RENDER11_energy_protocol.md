# RENDER11: sustained GPU device energy

This new endpoint addresses the unresolved short NVML deltas. It does not revise earlier timing or zero-counter results. It measures energy at the actual GPU board, not CPU, RAM, PSU or complete wall energy.

Use the same seven RENDER10 workloads and all five methods: CUDA eager, CUDA Graph, scalar fragment, vector fragment and OpenGL 4.6 compute. Preserve FP32, sixteen updates, resident inputs, seeding and complete scanner. Admit the original RENDER09 inputs by hash and exact regenerated-array comparison. All 77 output gates must pass before any energy block; unavailable CUDA Graph invalidates the complete registered comparison.

Three technical blocks per case and method alternate method order. Every active block performs complete synchronized inference calls for at least three seconds, bracketed by one second of idle observation before and after. Record actual NVML cumulative joules, call count, elapsed time, temperatures and available RAM. Report gross board joules per call and a separately labelled baseline-adjusted range using the two idle rates. This range is not a confidence interval and does not uniquely attribute background consumption to the neural process.

Keep all 105 block rows, including unfavourable values. No effect is calculated before the full registered cohort and source checks pass. Counter reset, zero unresolved active energy, missing comparator, RAM below 8 GiB or GPU temperature at least 83°C stops the process and preserves partial evidence. Scheduling uses one shared FIFO GPU actor, CPU two threads and 11 GiB bootstrap admission. No closed training is repeated; blocks are technical observations, not independent scientific replications.

The plan and source/input hashes must be frozen before execution. If resource admission or the deadline prevents execution, the status remains prepared/unexecuted and energy remains unknown. No TDP-based estimate replaces measurement.
