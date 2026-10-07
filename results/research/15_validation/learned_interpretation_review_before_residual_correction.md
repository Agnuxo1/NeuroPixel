# Item 15B: interpretation review of the completed learned-state panel

Review date: 2026-10-07. Scope: read-only interpretation of the archived JSON reports, not a new numerical audit or model run.

Scientific source: `771975b4833bfcf20e06f848253912c699b2152e`.
Final archive: `205507b71e2f7abb00ce97aa1dbec8321fc0af26`.
Archive root: `results/research/15_cloud_runs/37654405152-1-preflight`.

Read: the final archive manifest; `audit/audit.json`; `native/gates_summary.json`; `native/report.json`; and the relevant rest-state wording in README.md at the scientific source. The manifest binds the 235,154-byte audit report to SHA-256 `519a1e329ffbb5638c387b3d8644680430144223c49ecd83fb88159907a9e3c5`. This review read the saved summaries; it did not independently deserialize or recalculate the raw state arrays.

## Finding and admissible wording

No interpretive blocker was identified in the bounded conclusion communicated by the report author.

The saved audit reports verified status, 5,761 checks and zero issues. All five T16 compatibility gates passed, with no prediction mismatch on the 64 replayed rows per checkpoint. These are output-compatibility observations across the recorded Python/thread change, not evidence that the original unarchived hidden states were reproduced exactly.

All 40 trajectories completed their specified horizon without a state guard crossing. The complete panel contains 20 pairs and 10,280 observed endpoint states. Each pair concerns one fixed validation query and one literal checkerboard perturbation; the four queries share one exposed scene within each checkpoint.

The audited final RMS-distance gains range from 1.556726149116686 to 6.058957512860499. Every pair's observed maximum gain occurs at continuation time 256. The corresponding absolute RMS distances remain approximately 0.000156 to 0.000606. Reporting both scales avoids equating a relative gain above one with a large absolute separation or a chaotic regime.

All 40 observed state-RMS maxima also occur at the last retained time. Across the base trajectories, initial RMS is approximately 0.427-0.573 and final RMS approximately 2.459-9.589. Nonzero final adjacent-state increments show that the recorded endpoints are not exact fixed points of the observed float32 update. They do not prove that later convergence is impossible.

A suitable report sentence is:

> Under the fixed future inputs, all selected perturbation pairs showed net finite-horizon amplification, and the largest observed state magnitudes and pair gains occurred at the final sampled time. These observations do not establish asymptotic divergence, Lyapunov instability, chaos or a global sensitivity bound, and they do not demonstrate convergence to a resting attractor.

## Interpretive restrictions

- A maximum at the final time is not, by itself, evidence of monotonic growth at every intervening step. Do not infer monotonicity from the endpoint/peak summaries.
- Staying finite and below the operational cap throughout this run is not an all-time boundedness certificate. No cap crossing also does not establish contraction.
- Amplification in the measured RMS metric and direction does not rule out eventual decay, a different contraction metric, or other attracting behaviour. Conversely, finite endpoint accuracy would not certify state convergence.
- The held future input is identical within each pair. Different queries and checkpoints define different conditional maps; the four role queries are not four independent initializations of one common input-conditioned map.
- NLL does not worsen in every case. For example, the base patient query for seed42 changes from 2.2954328949836085 to 1.5490684258990723. Other cases worsen substantially. Report the complete readout results rather than replacing them with a universal deterioration claim.
- Known-query readouts are descriptive diagnostics. Four correlated queries do not provide a new generalization estimate or establish useful stable memory. The result does not identify the cause of earlier weak competence or reopen H1.
- The README's historical rest-state training/visualization language should not be presented as an existing convergence guarantee. Normalized darkness, small activity summaries, finite accuracies and all-time state stability are different observations.

## Primary context already read

Miller and Hardt, *Stable Recurrent Models*, sections 2-3, https://arxiv.org/pdf/1805.10369 : their uniform same-input contraction condition is stronger than general Lyapunov stability. It is not inherited by an unconstrained residual update.

Revay and Manchester, *Contracting Implicit Recurrent Neural Networks*, section 1.1, equation (3), example 1 and theorem 1, https://arxiv.org/pdf/1912.10402 : contraction can be assessed in a specified metric; Euclidean transient amplification alone is not a general impossibility result for contraction in other metrics. No such certificate was constructed for these learned checkpoints.

No new search, inference, numerical recount, training, binary deserialization or reference mutation was performed for this review. It is an internal interpretive review, not an external replication.
