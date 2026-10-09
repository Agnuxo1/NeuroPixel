# Neural fields executed through graphics: vision and measured limits

Francisco Angulo de Lafuente — NeuroPixel project. Research manuscript, 9 October 2026. Independent external replication remains pending.

## Abstract

NeuroPixel represents recurrent neural states on a spatial canvas and executes learned local updates through floating-point graphics images. Physical RTX 3090 validation covers states, masks, retina, logits and decisions. Deployment of five learned visual models retains all 320 tested decisions. A fixed five-pair experiment on public UCI optical digits gives 94.36% mean accuracy for NeuroPixel and 90.84% for a CNN with similar nominal parameter count. The paired difference is 3.52 percentage points, with exploratory Student t95 interval [1.93, 5.10]. A seven-workload comparison includes CUDA eager, CUDA Graph and two fragment renderers. Vector rendering measures 1.536× relative to the best CUDA comparator for C16/grid128; advantages are not uniform. An additional OpenGL 4.6 compute backend passes all seven execution checks, but its timing cohort is incomplete. Negative hypotheses, failed precision targets, depth instability and continuous replay discrepancies remain published. These results establish bounded implementation fidelity and task utility, without establishing biological equivalence, universal acceleration or energy savings.

## 1. Question and antecedents

Can a learned spatial neural process execute as evolving graphics images, remain inspectable and offer useful accuracy or runtime advantages under controlled comparisons? A graphics API alone does not establish a speed advantage.

Learned local cellular updates, biological morphogenesis analogies and WebGL/GLSL execution already appear in [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/) (Mordvintsev et al., 2020). [Self-classifying MNIST Digits](https://distill.pub/2020/selforg/mnist/) supplies a classification antecedent. Originality must be assessed through specific mechanisms, implementation choices and demonstrated utility. This work does not claim priority for neural cellular automata or GPU image execution in general.

## 2. Neural field and graphics implementation

For identity field `u` and state `s`, the local update is:

`s[t+1] = s[t] + W2 ReLU(W1 concat(s[t], K*s[t], u) + b1) + b2`.

`K` is a learned depthwise 3×3 perception with two filters per state channel. A learned seed initializes present cells. A retina maps RGB pixels to identity channels. A shared projection and dictionary decode the state. The scanner applies the decoder beyond the supervised output cell; semantic correctness at every displayed cell is not established.

The scalar backend uses RGBA32F texture banks and framebuffer passes. The vector variant packs four channels and coefficients for DOT4 operations. Retina, seeding, perception, residual updates and complete decoding execute through image operations. The separate OpenGL 4.6 compute backend uses SSBO weights, shared pixel tiles and image barriers, explicitly distinguished from fragment rasterization.

The extended fragment preflight passed 136 physical GPU checks. Learned deployment compares all five selected visual models on the fixed first 64 official test rows each: 320 decisions agree with native CUDA and original CPU predictions, with the registered pointwise logit and mean NLL limits retained. The Torch-free compute preflight passes 21 state/logit/decision checks across seven workloads. These are engineering validations, not new training realizations or outside replications.

## 3. Controlled reading and binding

The original H1 binding comparison remains unsupported. Later task-specific successes do not reverse that decision.

DEV04 uses twelve fresh bodies across three partitions and four initializations, with two readout heads per body. All twelve attention candidates pass their competence gates. The attention–local difference is 16.244 percentage points, exploratory t95 interval [10.586, 21.902]. Its half-width, 5.658 points, exceeds the original 5-point target: the precision gate fails.

GLOB05 adds twelve active global uniform heads on the frozen bodies. Parameter dimensions, labels, batches and registered update budget are held fixed; original heads are not refitted. Query-dependent channel gating activates Q/K/V without query-specific spatial weighting. Attention minus this control is 6.278 points, interval [4.108, 8.449]. This different contrast does not repair DEV04. Functional capacity and FLOPs remain unequal; global controls do not all pass competence gates. All original cases and 3,538,944 exact replayed decisions are retained.

## 4. External pixel experiment

The public [UCI optical digits dataset](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits), Alpaydin and Kaynak (1998), DOI 10.24432/C50P49, contains 8×8 pixel counts in the known range 0–16. Its metadata documents different writers for official TRAIN and TEST; individual writer identifiers were not independently verified. Exact images are unique and disjoint within and across the official splits. A balanced 300-image DEV subset comes only from the 3,823 TRAIN images, leaving 3,523 training images. TEST contains 1,797 images. Scaling by 16 is fixed in advance.

Five paired seeds, 240–244, start fresh. NeuroPixel uses retina width 8, identity width 8, 16 state channels, hidden width 32, eight updates and eleven decoder classes including PAD. It has 5,056 nominal parameters; the CNN has 5,039. Functional geometry and operation counts differ. Both receive identical training examples, orders, labels and forty epochs, using batch size 64, Adam learning rate 0.001 and gradient clipping at 1. There is no extra seed, extended budget or hyperparameter search. DEV selection uses correct count, then NLL, then earliest epoch. All ten selected models are sealed before test scoring.

| Seed | NeuroPixel correct / 1,797 | CNN correct / 1,797 |
|---|---:|---:|
| 240 | 1,715 | 1,614 |
| 241 | 1,688 | 1,643 |
| 242 | 1,690 | 1,623 |
| 243 | 1,688 | 1,642 |
| 244 | 1,697 | 1,640 |

Mean accuracy is 94.3573% versus 90.8403%. The mean paired difference is 3.51697 points; exploratory Student t95, df=4, gives [1.93046, 5.10349]. This describes variation across training realizations on a shared fixed test population, not uncertainty across arbitrary writers, datasets or devices. All five NeuroPixel models exceed 90%. Nominal parameter proximity does not establish architectural causality or state-of-the-art rank. The public corpus is not a blind custody evaluation.

## 5. Dynamics and scanner limits

Registered depth controls evaluate T=0, 4, 8, 16 and 32 with input held fixed. Accuracy degrades beyond the trained depth of eight, while state RMS grows markedly. The validated visual runtime resets per input and uses eight updates. These finite observations do not establish indefinite stable memory, an attractor or an all-time divergence theorem.

A functional CPU code path reproduces 98,835 categorical decisions. All thirty base and transformed-input evaluations pass logit and mean NLL checks. Ten extended T16/T32 comparisons fail the original rule `1e-4 + 1e-5*abs(reference)` and remain archived. A fixed 128-row same-host diagnostic reproduces both code paths and archived endpoints exactly on its recorded host. That subset does not erase the broader failed execution or uniquely identify its cause.

SCN06 shows that identical current scanner outputs can hide a state difference that changes a future answer. Displaying only the top token and confidence also compresses distinct logit vectors. Five authenticated learned read matrices have ambient rank 16 and nullity 32; this does not identify reached or semantically causal latent directions. Colour inspection is useful observation, while complete causal interpretation and biological correspondence remain unproven.

## 6. Runtime and energy

Benchmarks use the same RTX 3090, driver 581.29, FP32 and disabled TF32. Comparators include deterministic native CUDA eager and CUDA Graph. Seven workloads cover C16/C48 at grids 8, 32 and 128, plus RGB retina at grid 8. Each performs sixteen updates and complete decoding from resident inputs. Seven blocks with three inner calls are technical repeats, not independent scientific replications.

The scalar renderer is 3.14–12.25× slower than the best CUDA comparator. The vector renderer improves that implementation. In C16/grid128 its median is 6.6585 ms against the best CUDA median of 10.2256 ms, a 1.536× ratio. Four of seven vector ratios are at or below one; two positive ratios are small and may be sensitive to technical noise. All workloads are reported.

RENDER10 passes all 77 parity checks before timing, then stops at the unchanged 8 GiB RAM floor with 143 of 245 rows saved. A recovery process times out in the shared GPU queue before execution. No complete RENDER10 speed or energy effect is inferred. Recovery may only measure missing registered keys while preserving inputs, source and completed rows.

Five NeuroPixel training and DEV-selection runs cost 413.318 seconds wall time and 826.410 seconds process CPU time. Five CNN runs cost 34.237 and 68.427 seconds. The visual accuracy advantage has greater measured training cost. Resident timing excludes installation, scheduling, compilation, transfers, storage and verification.

Short NVML deltas do not resolve per-inference energy adequately. A zero reading does not mean zero consumption. GPU device counters exclude CPU, RAM, PSU and whole-system wall energy. Cloud training energy and complete wall energy were not measured; TDP multiplied by time is not substituted for measurement.

## 7. Reproduction and open requirements

Frozen plans, source commits, weights, datasets, predictions, raw states, receipts, Git object identities and SHA256 manifests are supplied. Failed attempts and negative findings remain available. GIFs derive from actual GPU trajectories and numerical tables; playback speed and display mappings are declared. Publication to GitHub main is authorized by the project owner and does not constitute peer review.

Runtime guards require at least 8 GiB available host RAM before heavy neural loading, two CPU threads and at most two scientific executors. Local GPU work follows the shared FIFO without interrupting other owners. Exact environments are recorded per study. Closed fits are not rerun to improve intervals; operational retries are not counted as scientific realizations.

Code is MIT licensed; UCI optical digits is attributed under CC BY 4.0. See [the evidence index](../research/RESULTS_INDEX_20261009.md), [reproduction guide](../research/REPRODUCE_20261009.md) and [cost and limits](../research/COST_AND_LIMITS_20261009.md).

The implemented neural field demonstrates bounded fidelity, competence on the specified visual task and workload-dependent acceleration after vectorization. Independent outside replication, complete energy accounting, robust continual memory and an exceptional original contribution confirmed by others remain unestablished.
