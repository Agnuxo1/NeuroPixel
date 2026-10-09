# OpenGL 4.6 compute: validated execution, incomplete timing

The advanced backend executes the same neural update with OpenGL 4.6 compute shaders, SSBO coefficients, four-channel arithmetic, shared 4×4 pixel tiles and explicit image barriers. It is graphics compute; the scalar and vector fragment renderers remain separate implementations.

## Execution fidelity

The Torch-free physical GPU preflight passed **21 checks across all seven registered workloads**. Complete states and scanner logits meet the original pointwise limit `1e-4 + 1e-5 * abs(reference)`; every categorical scanner decision agrees with the archived native CUDA reference. Inputs and references are admitted by SHA256. The actual NVIDIA context reports 49,152 bytes of available shared memory per workgroup. No training or accuracy selection occurs.

Evidence: [successful receipt](../../results/research/RENDER10_preflight/render_only_20261009B/receipt.json), original inputs/references under `RENDER09_benchmark/original_20261009A`, and `scripts/preflight_RENDER10_render_only_B.py`. Earlier API-query and GLSL image-parameter compilation failures remain preserved. The successful test does not convert those attempts into successful replications or imply that the separate 136-check tensor preflight completed.

## Timing status

The fixed RENDER10 plan compares compute, scalar fragment, vector fragment, native CUDA eager and CUDA Graph on the same seven workloads. All **77 pre-timing parity checks passed**. The timing execution then stopped when available host RAM fell below the unchanged 8 GiB floor. Its partial rows are preserved in [the original receipt](../../results/research/RENDER10_benchmark/original_20261009A/receipt.json).

Recovery verifies the frozen source and original inputs and skips every completed `(case, block, method)` key. The first recovery process exhausted its 20-minute FIFO wait behind another owner's GPU job and terminated before running measurements. Other workloads were not interrupted. The full registered timing cohort is **incomplete**; no RENDER10 speed effect or energy advantage is reported from partial rows. Complete RENDER09 results remain separately available.

The resource interruption and any eventual recovery are operational qualifications, not independent scientific replications. Whole-system wall energy and independent outside replication remain unmeasured.
