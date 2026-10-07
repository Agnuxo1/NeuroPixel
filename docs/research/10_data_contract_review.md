# Item 10: raw-QA contract test review

This is a prospective source review and test inventory, not a result from the official bAbI dataset. No original TRAIN, DEV or TEST payload was opened for this work. No model was constructed, no numerical training was run, and the tests have not been executed or compiled locally.

## Source and command

- Data module reviewed: Git blob 73170979b2b50282ab04648e231f1bff641f1235, intended path neuropixel/research/babi_qa.py.
- Independent tests: Git blob b4f47a171758638bd91cfb75233f76f81ae3e0a4, 24,863 UTF-8 bytes, intended path tests/test_babi_qa.py.
- Exactly 24 unittest methods. The suite uses only the Python standard library and loads the data module directly, avoiding unrelated package imports.
- Run from the repository root: python -m unittest discover -s tests -p test_babi_qa.py -v.
- The root coordinator must preserve the first execution log and any failures. Static inspection is not a passing execution receipt.

## Independent expectations and coverage

| Area | Methods | Contract witnesses |
| --- | ---: | --- |
| Parser | 7 | Literal expected prior facts and original line IDs; earlier QA answers and future statements excluded; strict positive contiguous IDs; malformed fields/support IDs rejected; raw CRLF hashes; punctuation-preserving normalization; label-blind input identity; conflicting labels retained but fitting blocked. |
| Grouping | 3 | Same episode, identical full story and identical visible prefix connect transitively; disjoint complete TRAIN/DEV membership; group/partition identities unaffected by record order or gold changes; diagnostic conflict grouping retains both records; empty partitions are errors. |
| Encoder | 5 | Literal token rows and punctuation; row order preserved; output remains PAD; metadata-independent pure encoding; frozen input UNK versus output support; TRAIN-label vocabulary admission; overflow rejected without truncation; malformed canvases rejected. |
| Scoring | 4 | Manually specified accuracy and per-answer counts; unsupported gold remains wrong in the full accuracy denominator; supported-only CE with explicit missing entries elsewhere; empty subset gives null accuracy/CE; invalid lengths, types, masks and nonfinite losses rejected. |
| Controls | 5 | Literal direct/inverse/object-question answers, explicit ambiguity/unsupported grammar and no transitive inference; majority/exact/question-only differences; fact-only frequency and separately declared query-exclusion control; unordered BOW lookup and Naive Bayes invariance contrasted with ordered exact memory. |

Expected values are literal semantic examples or simple manually specified counts. The tests do not use the symbolic solver to manufacture their own expected answers. The grouping tests check the required equivalence classes and membership invariants; they do not reproduce the implementation's hashing algorithm as a second asserted calculation.

The split uses its declared default salt neuropixel-item10-dev-v1 and DEV fraction 0.1. A fixed background of 128 separate invented episodes gives the population test many groups without retrying salts or sampling until a preferred split appears. The exact fraction of groups or records assigned to DEV is not asserted to equal 10 percent.

## Fixture exposure ledger

DEVELOPMENT_FIXTURES is an exported deterministic tuple of input-only mappings:

    {"facts": ["ordered raw statement", ...], "question": "current raw question"}

It includes the fixed 128 grouping episodes and every valid context passed by the tests to parsing, encoding or a control, including alternative statement order, a shared-prefix episode, OOV inputs, encoder overflow, unsupported symbolic grammar and unordered-token control inputs. No entry has an answer, support IDs, model prediction or prospective performance result. The ledger conservatively includes a few contexts that are constructed for admission failures or source review rather than successful encoding.

DEVELOPMENT_FIXTURE_SCOPE records that the examples are freshly invented, without RNG or official source payloads. Invalid numbered/tabbed parser mutations remain in the source tests; they are not presented as valid performance examples. Exact raw-string duplicate contexts are deduplicated in the exported tuple. Distinct strings can share the declared normalized-input identity, which downstream novelty accounting must deduplicate using its frozen tokenizer.

The later study must bind the exported ledger, the entire official TRAIN input population including assigned DEV, and any additional memorization fixtures before TEST access. Its exact-unseen subsets must state whether the key is normalized raw input or frozen encoded input; an OOV collision is not evidence of raw semantic identity. Official all-TEST counts must remain available, and an empty unseen subset has a null metric, not zero accuracy.

## Interpretation and remaining controller responsibility

The parser retains all prior statements, without selecting supporting statements or injecting relation triples, gold labels or support IDs into the neural input. TRAIN answers legitimately contribute output vocabulary during encoder fitting. A word that appeared only in TRAIN input can still be an output-supported label; support is membership in the full nonreserved vocabulary, not membership in the narrower list of observed TRAIN answer labels.

The frequency controls are fitted using TRAIN-derived answer categories, then read only the visible facts. One separately named variant excludes candidates mentioned in the question. BOW controls discard order deliberately. Their performance on original QA4 remains unknown until the authorized data analysis; behavior of a rewritten generator is not evidence of the original archive's shortcut rate.

This module has no checkpoint-selection or final-access gate. These data tests therefore do not claim that final evaluation is gated, that supplied records truly came from TRAIN, or that checkpoints were selected without TEST. Those properties require the separately reviewed controller, source bindings, acquisition receipts and archive-before-final gate. The public fitting functions accept the explicit population supplied by callers.

Passing this suite would establish the tested input and metric contracts on invented fixtures. It would not establish competence, generalization outside the bAbI generator, representational sufficiency for unknown official records, or absence of actual TRAIN/TEST duplicates.
