# Item 10 — TRAIN acquisition and independent recount

## Status and scope

The frozen acquisition phase completed successfully on 7 October 2026. This report admits preparation of the neural preflight; it does not report a trained model, TEST accuracy or successful generalization. Item 10 remains active. Items 11–30 remain unopened, and the closed historical H1 is unchanged.

The external benchmark is original English 1k bAbI task 4, two-argument relations, from the externally authored benchmark described by Weston et al. The scientific question will concern models freshly trained on this benchmark with a literal token-grid adapter. It will not be a zero-shot evaluation of an earlier NeuroPixel checkpoint, a test of real-world language, or evidence of independent laboratory replication.

## Frozen source and execution

| Identity | Value |
|---|---|
| Source commit | `99ecaeefdd00dffd3dae444a781ad009c19dff5d` |
| Acquisition plan SHA-256 | `87260de70352c99906e51745cd284c4ad2a8ff069013b78e1bd062e0b6a602c8` |
| Freeze | 2026-10-07T10:44:50+00:00 |
| Actions run / job | 37609533224 / 112753126084 |
| Final evidence commit | `19c8c40ab50141ba128623e7bbbc04cfb2460f31` |
| Raw directory | `results/research/10_cloud_runs/37609533224-1-acquisition` |
| Final manifest | 34 files; 23,778,454 bytes excluding the manifest |

The plan bound twelve source, protocol, workflow and test files. The worker verified their hashes and clean tracked source before and after execution. Twenty-four collected test methods passed, with no failures or skips. Pytest recorded 82 passed call reports: 24 parent methods plus 58 subtests. The JUnit header also reports 82, and must not be described as 82 independent methods.

The worker took 12.401776782 seconds before its final archive operation. The contract stage took 1.002593099 seconds including shutdown, and acquisition took 1.502504172 seconds. Three one-second child-supervision samples had a minimum available RAM of 14.427669525146484 GiB; worker admission was 14.304248809814453 GiB. These are sampled available-memory values, not continuous minima, process peak memory or energy measurements. Final archive confirmation was recorded at 10:46:13.003458+00:00. The standard public CPU runner was released. No model, GPU or paid computation was used.

## Data source and provenance limitation

The original-author HTTP endpoint returned 404 Not Found. The prospectively listed HTTPS S3 mirror succeeded:

- Archive: `babi_tasks_1-20_v1-2.tar.gz`, 11,745,123 bytes.
- Archive SHA-256: `84f5296ab9a1ad0dc9464e08c491d65cd08830fca3acae9ab86f75e0fb81573c`.
- Selected member: `tasks_1-20_v1-2/en/qa4_two-arg-relations_train.txt`.
- TRAIN bytes: 117,470.
- TRAIN SHA-256: `c9370502d85fa382f32d1fea2111e817f81ae9e3e79eb2924a1a5392a90417fa`.

Both endpoint outcomes and the exact downloaded bytes were retained. An original-versus-mirror comparison cannot be performed when the original endpoint returns no archive. The mirror has documented primary-code provenance; its content is not independently authenticated by a successful original-host download. The declared dataset license and attribution are recorded separately in the acquisition plan and provenance note.

Only the exact TRAIN member was selected, extracted, decoded, tokenized and scored. No TEST member was selected or parsed. Traversing a compressed TAR can internally decompress other bytes, so this procedure is not physical custody of unread compressed TEST bytes. Public availability also precludes any claim of external blind custody.

## TRAIN structure and leakage controls

There are 1,000 episodes and 1,000 question-answer records, with two visible facts per record. The normalized ordered inputs contain 919 unique values: 81 rows repeat an input. Input-defined connected grouping, joining source episodes, full normalized stories and shared inputs, produces 752 groups.

The prospectively fixed salt `neuropixel-item10-dev-v1` and nominal validation fraction 0.1 yield 895 optimization-TRAIN records and 105 validation records. Their normalized input intersection is empty. There are no conflicting raw-input labels or conflicting encoded-input labels. The split is grouped; its realized record fraction is not required to equal exactly 10%.

Vocabulary and geometry were fitted only on optimization TRAIN. The resulting grid is 4 by 8, with the question in row 2 and readout at coordinate (3, 7). Vocabulary size is 18, including PAD and UNK. All official TRAIN inputs fit without overflow or input OOV; every gold answer is in the vocabulary and has occurred as a TRAIN answer. Maximum grid distance to readout is seven local steps. A declared sixteen-step model can therefore receive information from all cells geometrically; this does not show that it learns the required computation.

Gold answer strings and supporting-fact identifiers do not select or alter the visible facts. Punctuation, fact order and question wording remain in the literal encoding. No symbolic solver constructs neural inputs.

## Exact control results

All learned shortcut tables were fitted on the 895 optimization-TRAIN records. The raw symbolic control is an unlearned grammar implementation.

| Control | TRAIN correct / 895 | DEV correct / 105 | All official TRAIN correct / 1,000 |
|---|---:|---:|---:|
| Majority answer | 161 | 19 | 180 |
| Exact ordered-input memory | 895 | 19 | 914 |
| Question-only memory | 287 | 20 | 307 |
| Bag-of-words memory | 730 | 44 | 774 |
| Bag-of-words naive Bayes | 159 | 15 | 174 |
| Visible-fact answer frequency | 313 | 32 | 345 |
| Fact answer frequency excluding query tokens | 578 | 64 | 642 |
| Raw symbolic relation solver | 895 | 105 | 1,000 |

The frequency control excluding question-mentioned candidates reaches 64/105, or 60.95238095%, on DEV. Bag-of-words memory reaches 44/105, or 41.90476190%. These observed development baselines are relevant because substantial accuracy can arise without full argument-order competence. They are not final benchmark results. The symbolic solver's 100% validates this selected task grammar and saved gold answers; it is not neural performance.

## Independent verification and limits

Root independently reconstructed the raw parser records and hashes, 752 connected groups, ordered split memberships, vocabulary, all 1,000 encoded inputs, all eight controls' predictions, and 24 population-by-control score cells in JavaScript. All 3,176 recorded checks passed. The calculation used raw TRAIN text and an independent implementation; it did not import the Python model or dataset module, draw study RNG, or read TEST.

Root separately verified byte lengths and SHA-256 values for seventeen fetched text/JSON artifacts against the final manifest. The full provider-decoded Actions log is preserved as UTF-8: 30,070 bytes, SHA-256 `94a8da1625e1e2b9d63a47f2305cae911807d1b80e35f0b1cd8ac8ccf1c5ff4c`. This identifies the decoded text, not the original HTTP transfer bytes.

This root check did not unpack the binary source ZIP or compressed benchmark archive. The planned separate CPU auditor must check those archive identities and saved scientific outputs. The independent JavaScript naive-Bayes calculation agrees on every selected label; it does not claim byte-identical intermediate floating-point scores.

Before neural execution, review found and corrected a worker/controller receipt-name collision and strengthened authentication of the six expected preflight configurations and completed-update counts. Those are prospective integration fixes, with source drafts retained. They are not failed model-training runs and do not change the experimental algorithm.

## Next sequential action

Freeze the exact neural recipe, runtime, source bindings, synthetic-fixture exposure ledger and finite preflight resource allowance. Execute two fixed small-set memorization fits and four fixed learning-rate pilots only after all controller/data contracts pass. Retain all runs and audit them before any primary-study admission. Do not open the official TEST member until the later ten-checkpoint training inventory has been completed and durably archived.

## Primary sources

- Weston et al., *Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks*: https://arxiv.org/pdf/1502.05698.
- Facebook bAbI QA loader, original URL and exact original TRAIN/TEST member mapping: https://huggingface.co/datasets/facebook/babi_qa/blob/main/babi_qa.py.
- Facebook dataset card, declared dataset license: https://huggingface.co/datasets/facebook/babi_qa/blob/main/README.md.
- Original generator README, limits of regenerating the published data: https://github.com/facebookarchive/bAbI-tasks/blob/master/README.rst.
- Ray example documenting the S3 dataset mirror: https://github.com/ray-project/ray/blob/master/python/ray/tune/examples/pbt_memnn_example.py. Only its source URL provenance is used; its TRAIN-plus-TEST vocabulary fitting is not adopted.
