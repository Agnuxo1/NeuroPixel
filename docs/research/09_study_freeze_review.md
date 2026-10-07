# Item 9 Stage-B candidate freeze review

**Conclusion: no blocking discrepancy in the reviewed candidate.** This is an independent read-only review within the same assistant team. It does not execute `load_plan`, the scientific trainer, models, tests or generators, and it is not a record of a Stage-B result or launch.

The reviewed candidate is `docs/research/09_study_plan.json`, SHA-256 `7e9089ae013abbc4fa042e30606d5d1329dc8b7e89281fcab080addc832c7792`, with freeze timestamp `2026-10-07T08:22:31.852462+00:00`. Read-only checks completed at `2026-10-07T08:23:38.241783+00:00`. The authoring checkout's HEAD was the Stage-A source `e63764ccb924ab23d65c2deb94b477387d102c82`; root will create and verify the new Stage-B commit. This review does not claim that the candidate had already been committed or that an Actions checkout had passed its live source guard.

## Candidate identity and compatibility

All **83 implementation bindings** were compared with the actual candidate bytes and matched. All **49 Stage-A bindings** are present and byte-identical to their prior values. The recipe remains SHA-256 `7450b1ede40542c5ff940842e42eb2725ac4420946baad6bac920e779a214b97`. The required package Python files, worker, controller, workflows, runtime inputs, recipe and both contract-test files are covered.

The candidate satisfies the conditions read in `scripts/research_item9_worker.py:146` (`load_plan`): schema/item/phase/status, exact recipe path, stages `contract_tests`, `train`, `final`, exact two-file test inventory, positive integer test count, past timezone-aware freeze instant, valid relative source/hash bindings and recipe identity. Its resource-policy object equals the existing `policy("study")`: two numerical threads, one inter-operation thread, aggregate ceiling four, 8 GiB available RAM, 13,800-second worker limit with a 180-second final-archive reserve, and a 240-minute workflow timeout. The 34 declared test methods were recounted statically from the two bound test files. This is a collection expectation; live collection, execution and skips are checked by the worker.

The unchanged study workflow triggers only on changes to the study-plan path on the declared isolated source branch. It invokes `--phase study`, installs the same pinned CPU runtime as Stage A and uses the existing shared concurrency group. No analysis-runtime package substitution is introduced into training: cloud numerical execution remains Python 3.12.8/Torch 2.6.0+cpu/NumPy 2.2.6, whereas the separately frozen offline analysis runtime is Python 3.12.14/NumPy 2.3.5/SciPy 1.17.0. That latter object matches the independent analyzer's explicit contract.

## Stage-A receipt and primary population

The bound evidence receipt is `results/research/09_validation/stage_a_evidence_receipt.json`, SHA-256 `a1d53e0b064e812317d93e6e8749bae4728730fecfbf359edf0249f2a72f6ee4`. Its copied selection and complete six-run objects match the downloaded raw Stage-A objects exactly, including canonical serialized hashes and byte counts. The raw archive identity is `8311c052796aeef5a52484b597cf5d9675ca315e`, run `37589908205`, job `112688648396`. Its manifest identity agrees with the prior audited archive.

The review independently reapplied the declared ordering to all four completed pilot candidates: descending validation binding, ascending validation cross entropy, then ascending learning rate. The winners are NeuroPixel `0.001` and Relative Transformer `0.003`. The candidate, copied selection and development identities agree. Both memorization records remain present; they do not alter the selection rule. The receipt binds the already retained operational and scientific audit receipts. This review does not substitute a new remote-provenance claim for those prior archive checks.

`neuropixel/research/complex_binding_protocol.py:78` (`expected_primary`) derives exactly the following ordered population:

| Initialization | First run | Second run | Updates per run |
|---:|---|---|---:|
| 40 | `neuropixel_seed40`, LR 0.001 | `relative_transformer_seed40`, LR 0.003 | 4,096 |
| 41 | `neuropixel_seed41`, LR 0.001 | `relative_transformer_seed41`, LR 0.003 | 4,096 |
| 42 | `neuropixel_seed42`, LR 0.001 | `relative_transformer_seed42`, LR 0.003 | 4,096 |
| 43 | `neuropixel_seed43`, LR 0.001 | `relative_transformer_seed43`, LR 0.003 | 4,096 |
| 44 | `neuropixel_seed44`, LR 0.001 | `relative_transformer_seed44`, LR 0.003 | 4,096 |

The plan's `expected_primary_runs` matches that derivation exactly. That plan field is descriptive: the existing trainer and final gate derive their actual inventory from the hash-bound recipe and authenticated selected rates. The agreement was therefore checked directly, rather than assuming that the worker consumes this field. The independent analyzer derives the same seed-major, family-minor inventory and checks completion, configuration, checkpoint references and the paired input-stream records.

## Final access and limits

The source contract retains the required sequence: all ten runs complete; `create_final_inventory` verifies their summaries and checkpoint/log/probe/validation artifacts; the worker publishes the inventory and its ten checkpoint/summary references and confirms their archived identities; it then writes the exclusive archive receipt. `claim_final_access` verifies those bindings and writes the exclusive consume-once access record before the controller generates the final population. The offline analyzer checks the corresponding local hash links and event ordering before opening final prediction artifacts. Remote push confirmation remains the worker's operational responsibility, checked again by the later archive audit.

The candidate is compatible with this sequence. The final source must still be committed with every bound file tracked and the plan bytes unchanged; `source_record` checks actual HEAD, committed plan blob, source hashes, tracked differences and unexpected untracked files before execution and again afterward. Adding this review does not excuse any of those checks. Source/root integration and launch remain separate actions owned by root.

There is no new rate search, budget extension, stopping rule, checkpoint selection, final-population change or hypothesis revision in this candidate. The later ten-run results and final-access chronology are still unobserved. The closed earlier H1 remains not supported; the Stage-B experiment is the previously specified exploratory two-event diagnostic. Cost estimates are operational projections, and neither local access receipts nor same-team review establish external blinding, custody or external replication.
