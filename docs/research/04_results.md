# Item 4 — matched neural pilot results

## Outcome

All eight declared runs completed under the frozen protocol. **No NeuroPixel advantage was observed**, and none of the eight configurations reached the predeclared 95% binding threshold on the training-partition probe. The selected Transformer achieved category-level retrieval, with nominal-role binding close to the elementary 50% expectation. These are descriptive results from initialization 7; the H1 decision remains reserved for item 5.

The experiment used 1,024 updates of 64 examples, answer cross-entropy only, two learning rates per family and a final checkpoint. This is a limited optimization budget. An unfavorable result here is not a proof of architectural incapacity; it also does not establish that simply extending training would solve the deficit.

## Selected configurations

Binding is the equally weighted mean of AGENT and PATIENT accuracy. Validation selected both the rate for each family and the strongest non-NeuroPixel reference before final examples were opened.

| Family | Parameters | Selected LR | Validation binding | Training-probe binding | Final binding | Final global accuracy |
|---|---:|---:|---:|---:|---:|---:|
| NeuroPixel | 29,824 | 0.003 | 9.277% | 8.594% | 9.326% | 18.872% |
| Adapted NCA | 30,384 | 0.001 | 9.863% | 7.812% | 11.914% | 14.990% |
| Spatial ConvGRU | 27,835 | 0.003 | 12.305% | 14.062% | 13.574% | 13.623% |
| Relative Transformer | 29,547 | 0.001 | 47.949% | 51.953% | 48.389% | 74.194% |

The selected reference is the relative-position Transformer at LR 0.001. The descriptive NeuroPixel-minus-reference binding difference is **−39.0625 percentage points**. No family or unsuccessful rate trial is omitted.

## Final role-specific accuracy

Each role has 1,024 examples, for 4,096 examples per selected checkpoint.

| Family | AGENT | ACTION | PATIENT | PLACE |
|---|---:|---:|---:|---:|
| NeuroPixel | 8.789% | 30.859% | 9.863% | 25.977% |
| Adapted NCA | 12.109% | 18.555% | 11.719% | 17.578% |
| Spatial ConvGRU | 17.773% | 10.840% | 9.375% | 16.504% |
| Relative Transformer | 53.906% | 100.000% | 42.871% | 100.000% |

The Transformer has 100% ACTION and PLACE accuracy, but only 48.389% mean nominal-role accuracy. Its saved conditional example-bootstrap interval for binding is 46.240–50.586%, which includes the 50% category-control expectation. This interval concerns sampled examples under one checkpoint; it is not an interval over training runs. The exact schema-aware controls in item 3 already solve the task perfectly.

Inspection of saved validation predictions found that the selected Transformer chose a visible filler of the correct category in all 2,048 examples, while answering only 491/1,024 nominal queries correctly. This describes category-level behavior without proving that its internal computation ignores positions. The independent validation diagnostic is provided separately. Other families also show partial category learning, so the observed failure is not evidence that the entire data/training pipeline is disconnected.

## All learning-rate trials

| Family | LR | Validation binding | Training-probe binding | Validation cross-entropy | Training seconds |
|---|---:|---:|---:|---:|---:|
| NeuroPixel | 0.001 | 8.203% | 8.594% | 2.444692 | 26.664 |
| NeuroPixel | 0.003 | 9.277% | 8.594% | 2.459427 | 25.460 |
| Adapted NCA | 0.001 | 9.863% | 7.812% | 2.676160 | 25.080 |
| Adapted NCA | 0.003 | 9.082% | 8.984% | 2.234744 | 25.059 |
| Spatial ConvGRU | 0.001 | 10.254% | 12.109% | 2.227519 | 24.801 |
| Spatial ConvGRU | 0.003 | 12.305% | 14.062% | 2.134307 | 23.612 |
| Relative Transformer | 0.001 | 47.949% | 51.953% | 0.433867 | 11.518 |
| Relative Transformer | 0.003 | 45.312% | 53.125% | 0.427074 | 11.267 |

The NCA rate selected by binding is not necessarily the rate with the lowest cross-entropy. The frozen endpoint order was preserved. Every trial is flagged as optimization-budget limited. The probe uses freshly sampled examples from the training partition in evaluation mode, not the exact minibatches used for fitting.

## Verification, provenance and resources

- Training source: `5ab415ec825e931b69fc22571c47cdb74a4fc592`; protocol `NP-SCI-20261006-v1` is unchanged.
- Target environment: Python 3.13.7, PyTorch 2.6.0+cu124, CUDA 12.4, RTX 3090, four CPU threads. The target compatibility suite passed **88 tests**, with one unavailable-CIFAR skip.
- An unrelated globally installed pytest plugin caused the first attempt to fail before test collection and scientific training. Its logs are retained. The isolated retry disabled automatic plugin loading; no installed package or scientific configuration was changed.
- The selection event was recorded at `2026-10-06T22:28:58.539148+00:00`, before final data access, recorded at `22:28:58.653985+00:00`. Selection SHA-256: `a62300cfccc8d01f9d2777042c20a2620ba0ebfbd24d1b2b0fe4a7ea3463673b`.
- Final dataset content SHA-256: `23bca94c224b195da94ab1794801fdae131007a77bd594c2167a085227e7dba2`.
- All **60 transferred files** matched their byte hashes. A separate verifier, which does not execute models or generate examples, checked eight training records, source/environment consistency, checkpoints, saved predictions, metric counts, selection and event ordering. It reported **zero discrepancies**; its 20 corruption/valid-fixture tests passed.
- Total recorded training time over eight trials: **173.462 seconds**. This excludes some setup, evaluation, queue waiting and test overhead; it is not a complete end-to-end cost comparison.
- Across recorded resource samples, minimum available RAM was **8.282 GiB**, minimum free GPU memory **22.343 GiB**, and maximum GPU temperature **49 C**. The queued slot was released at completion.

Hash and timestamp agreement establishes internal artifact consistency; it is not independent laboratory replication or third-party preregistration. Inference latency and physical energy have not been measured in this item.

## Remaining interpretation limits

The two NCA models use random firing during training and full synchronous updates during evaluation. The resulting state dynamics can differ. Its causal contribution to this poor result has not been isolated, and ConvGRU also performs poorly without that mask difference. No post-test tuning or additional final evaluation was performed to rescue the result.

The adapted NCA has 2,048 nominal weights receiving zero identity inputs; the ConvGRU can propagate state dependencies up to radius two per update; the Transformer uses two global-attention blocks. Close parameter counts and equal sample exposure do not equalize capacity, receptive fields, arithmetic or optimization difficulty.

Item 4 closes implementation, limited tuning, execution and verification of the declared matched pilot. Broad claims about a family of architectures, a competitive fully optimized benchmark, or an intrinsic NeuroPixel advantage remain unestablished. The next required work is the frozen initialization and split replication panels, followed by mechanism ablations.

## Evidence files

- `results/research/04_experiment/pilot/pilot_summary.json`: selected results and timing.
- `results/research/04_experiment/pilot/`: all eight weights, validation predictions, selected final predictions, curves, datasets and configurations.
- `results/research/04_experiment/metadata/`: first-attempt and successful-retry logs/status.
- `results/research/04_verification.json`: independent artifact audit.
- `docs/research/04_runtime_incidents.json`, `04_deployment.json`, `04_reference_checks.md`: incident, deployment and primary reference records.
