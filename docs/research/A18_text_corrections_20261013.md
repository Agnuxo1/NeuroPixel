# Public text corrections after the claims review (A18.3, 13 October 2026)

The A18.2 review compared every public sentence with retained receipts. The corrections below change wording only; no result, receipt or protocol was edited. The exact pre-correction bytes of every document are archived in `docs/archive/completion_20261012/A18_sources/`, and `results/research/COMPLETION20261012/A18_text_corrections.json` records the SHA-256 of each file before and after.

| File | Line | Claim | Reason |
|---|---:|---|---|
| `docs/research/06_A06_1_results.md` | 38 | H06-04 | Receipt audit_saved.json gives -0.27145 pp, which rounds to -0.271; no retained file supports -0.272. |
| `docs/research/04_results.md` | 78 | H04-04 | The cited verifier output is missing; the statement now names the evidence that exists. |
| `docs/research/05_results.md` | 5 | H05-01..H05-09 | Verification statements refer to bytes that are not retained; level stated explicitly. |
| `docs/research/VIS07_results.md` | 25 | D-VIS07-02 | RMS source not retained; the retained diagnostic values are stated. |
| `docs/research/OPT03_ROLE_results.md` | 32 | D-ROLE-02 | Admission value not found in retained files. |
| `docs/research/RENDER08_results.md` | 3 | G-R08-01 | RENDER08 has one (scalar) renderer method; the column label said 'vector'. |
| `docs/research/RENDER09_results.md` | 15 | G-R09-01 | Two other workloads have ratios above 1; the sentence said none did. |
| `README.md` | 57 | G-R08-01 | Receipt medians give 3.138-12.244x. |
| `docs/papers/NeuroPixel_manuscript_20261009.md` | 63 | G-R08-01 | Receipt medians give 3.138-12.244x. |
| `docs/research/REPRODUCE_20261009.md` | 17 | G-R11-02 | Stale count: the original receipt stores 103 rows. |
| `README.md` | 96 | P-AVAIL-01 | Availability was overstated. |
| `docs/papers/NeuroPixel_manuscript_20261009.md` | 73 | P-AVAIL-01 | Availability was overstated. |
| `docs/research/RESULTS_INDEX_20261009.md` | 17 | P-AVAIL-01 | Availability was overstated. |
| `docs/research/REPRODUCE_20261009.md` | 11 | P-AVAIL-01 | Availability was overstated. |
| `README.md` | 21 | P-ARCH-01 | The table reports internal L1-L2 measurements, not demonstrated general capabilities. |
| `docs/papers/NeuroPixel_manuscript_20261009.md` | 79 | P-GPU-01 | Wording aligned with the registered evidence level. |
| `README.md` | 37 | P-ARCH-01 | Pre-scientific figure labelled with its evidence level. |
| `README.md` | 43 | H05-05 | Historical figure labelled with its evidence level and the prospective negative result. |

Evidence levels for every public claim are in `docs/research/claims_ledger.json` (73 claims; maximum declared level L2). Claims at L0 `blocked_external` depend on the owner recovering the item-5 executor archive.
