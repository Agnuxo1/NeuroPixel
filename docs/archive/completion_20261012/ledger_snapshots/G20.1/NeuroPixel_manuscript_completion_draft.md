# NeuroPixel completion manuscript — working draft

Francisco Angulo de Lafuente. Working draft started 13 October 2026 for the completion programme; it becomes the final manuscript in task G12. Evidence levels follow `docs/research/claims_ledger.json`; no statement here exceeds them.

## Antecedents that could void the novelty claim

Every claim about dynamics, memory or repair must answer the following antecedents. "Not established" means that no registered NeuroPixel result answers the question yet; it is not a negative result.

| Antecedent | What it already establishes | Question NeuroPixel must answer | Current answer and evidence |
|---|---|---|---|
| Hopfield (1982) | Associative memory as energy minima; pattern completion, a form of repair | What does a local spatial field add over a Hopfield network with the same state? | Direct comparison in G20.2 (global versus local coupling at equal state size). No NeuroPixel memory-repair advantage is established (H05-01, H06-04, HT-14). |
| Amit, Gutfreund and Sompolinsky (1985) | Hopfield capacity near 0.138 N for random patterns | Does a local rule beat or approach that capacity per cell? | Not established; G20.2 measures the local-coupling cost. |
| Turing (1952) | Pattern formation by local reaction-diffusion | Does the learned rule produce something a fitted reaction-diffusion system does not? | Not established; no reaction-diffusion comparator has been run. |
| Jaeger (2001); Maass, Natschläger and Markram (2002) | Reservoirs: echo-state property and fading memory | Is the contractive reference field (StableField) an end-to-end trained reservoir, and what does training add? | Not established; StableField is implemented with a contraction certificate (P-MATH-01) but not compared with a reservoir. |
| Boyd and Chua (1985) | Fading-memory operators are approximable (Volterra series) | Basis of the bound in section 18C | Used as the deductive basis of P1 in the planned memory study; no result yet. |
| Miller and Hardt (2019) | Stable recurrent models are approximated by truncated feedforward networks | A contractive field cannot claim a long-memory advantage: test against an equivalent truncated feedforward model | Not established; required before any long-memory claim for contractive fields. |
| Bai, Kolter and Koltun (2019); Winston and Kolter (2020) | Equilibrium models; guaranteed unique fixed point (monDEQ) | How does the field differ from a DEQ or monDEQ with the same budget? | Not established. |
| Lohmiller and Slotine (1998) | Contraction analysis in general metrics | Tool to certify partial or basin-wise contraction | Used in the completion kit's certificate (ideal arithmetic, fixed input; P-MATH-01). |
| Gilpin (2019) | Cellular automata are convolutional neural networks | Priority of the "cellular automaton = CNN" formulation | Acknowledged: NeuroPixel claims no priority for this formulation (P-ARCH-01). |
| Mordvintsev et al. (2020); Randazzo et al. (2020) | Learned neural cellular automata that grow, regenerate and classify digits, executed on GPUs/WebGL | What is new beyond NCA growth, regeneration and self-classification? | NeuroPixel's VIS07 digit result (D-VIS07-01) is an NCA classification on a different corpus; it is not new in kind. Registered robustness differs by perturbation (D-VIS07-05). |
| Seung (1996); Khona and Fiete (2022) | Continuous attractors and neural integrators; memory along neutral directions | Does field memory use basins or neutral directions? | Not established; this is prediction P2 of the candidate central hypothesis (section 18B). |
| Kirkpatrick et al. (2017); van de Ven, Tuytelaars and Tolias (2022) | Catastrophic forgetting and the three incremental-learning scenarios | Classify any continual-learning result (G7) by scenario | Historical copies preserve old tensors (HT-12) but no competent continual learning is established. |

The novelty claim is limited to what none of these contain, with the evidence that shows it. If the central result turns out to be a known special case, the abstract will say so.
