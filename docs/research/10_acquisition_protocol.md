# Item 10 — Prospective acquisition and TRAIN-only audit

## Question and scope

Can the current single-token NeuroPixel interface be adapted faithfully to an externally authored QA benchmark, and does its official training data admit a defensible prospective comparison? This first execution answers the provenance and representation questions. It performs no neural training, no final-test parsing, and no model-performance evaluation.

The chosen target is bAbI task 4, original English 1k, whose raw story and current question predict one location-word answer. It is a different author's synthetic benchmark. A later successful experiment would demonstrate performance under this external task and recipe; it would not demonstrate zero-shot transfer from item 9, out-of-generator bAbI generalization, open-world language understanding, or Nobel eligibility. H1 and the completed negative item-9 competence result remain closed.

The source review is [10_babi_provenance.md](10_babi_provenance.md). Its original-author URL is documented by the published facebook/babi_qa loader. A separately identified HTTPS mirror is documented in Ray's example. Only URL provenance is used from that example; its TRAIN+TEST vocabulary/length adaptation is not copied.

## Acquisition before learning

The JSON execution plan fixes the following source order:

1. The original author's HTTP endpoint: http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz
2. The documented HTTPS mirror: https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz

Both are attempted, without credentials. Each successful compressed archive is retained with byte count, SHA-256, final URL, response metadata and an observed redirect chain. Failed and partial attempts are retained. Redirects are limited prospectively to the exact archive paths on the author host spellings and the two specified S3 host forms; arbitrary redirect destinations are rejected. A metadata URL or the unrelated ParlAI checksum is not an upstream checksum for these bytes.

Downloads have a 32-MiB compressed cap per source and a deadline checked between reads, with a 30-second socket timeout; a blocking read can extend that per-download check. The independently supervised child has an outer hard stage limit. TAR traversal admits at most 5,000 member headers and 1 GiB of declared uncompressed contents. No archive path is extracted to the filesystem through extractall.

The sole payload selected for parsing is:

`tasks_1-20_v1-2/en/qa4_two-arg-relations_train.txt`

It must be a unique regular member no larger than 2 MiB. The exact bytes are retained. TAR/gzip traversal can decompress intervening bytes internally; no TEST member is selected, decoded as example text, tokenized, fed to a model or scored. The shared compressed package is public and contains TEST, so this is an internal access workflow, not independent blind custody.

The first source yielding the exact selected member is provisionally selected. When both sources yield it, their TRAIN payloads must agree byte for byte; the comparison is saved before any rejection. Compressed archive hash equality is also reported, without requiring compressed containers to be identical. If only the mirror is available, that narrower provenance is reported. Availability failures are not scientific outcomes.

## Raw text and gold separation

The reader preserves all numbered statements preceding the current question in the same episode and their order. It removes administrative numeric line identifiers from model inputs. Earlier questions, earlier answers, future statements, answer fields, supporting-fact IDs, source IDs and split names do not enter the encoder. Supporting IDs are checked as metadata but never select a fact.

The normalized representation lowercases words and retains every non-whitespace punctuation token. This normalization is not a byte-perfect restoration of case or whitespace. Raw file and line hashes remain available for provenance. The normalized input hash uses ordered fact-token lists and the current question only; the gold label cannot affect it.

A contradictory answer for the same normalized input is retained in parsed records and a conflict receipt. Diagnostic grouping may still run, but fitting is stopped before the encoder or shortcut models are fitted. Conflicts are never silently deleted.

## Development grouping fixed before TRAIN inspection

Development is derived only from the selected official TRAIN member. The fixed salt is `neuropixel-item10-dev-v1`; the nominal validation fraction is 0.1. The assignment is by deterministic SHA-256 of input-defined connected components, not an adaptive search over salts.

Components connect all queries of an episode, duplicate full normalized stories, and duplicate normalized visible inputs. Transitive connections are preserved. Every component goes wholly to TRAIN or DEV. Its actual sizes are reported; exact 90/10 row counts are not imposed by cutting components. Empty partitions cause a recorded stop. Shared individual facts, vocabulary, templates or renamed/isomorphic scenes are not excluded by this grouping and are not claimed to be out of distribution.

Vocabulary, answer-category shortcut statistics and geometry are fitted solely on assigned optimization-TRAIN. DEV may later select among a prospectively fixed learning-rate inventory, but does not fit these preprocessing objects. The original TEST contributes neither vocabulary nor dimensions.

## Encoding admission

Each raw statement occupies one row, in order. The question occupies the penultimate row and a blank output row is last. Padding is token 0 and unknown input words map to token 1. TRAIN words, punctuation and supervised TRAIN answer tokens define the rest of the sorted vocabulary.

Height is the maximum number of optimization-TRAIN facts plus two. Width is the largest optimization-TRAIN token count for a statement or question. Predeclared caps are height 8 and width 16; exceeding either stops admission. The readout is the bottom-right cell. With these caps every input lies within Chebyshev distance 15 of the readout, which is within the planned T=16 geometric influence radius. That is reachability, not evidence of learned information transport.

No row or token is silently truncated. The acquisition stage requires all official TRAIN/DEV inputs to fit the frozen TRAIN-derived geometry and reports every unknown input token. A gold token present anywhere in the full TRAIN vocabulary is representable by a full-vocabulary output head even if never observed as a TRAIN answer; those two coverage notions are reported separately. An unsupported gold can never be credited as an UNK answer. Accuracy keeps it in the denominator; any later CE uses an explicit supported-label denominator.

## Controls fixed before data acquisition

All learned controls use assigned optimization-TRAIN and fixed lexical tie breaking. Each is reported separately:

- TRAIN-majority answer.
- Exact normalized-input memory, with TRAIN-majority fallback.
- Question-only memory, with the same fallback.
- Exact unordered-token multiset memory over all facts and the question.
- Multinomial naive Bayes over that same unordered token collection, Laplace alpha=1.
- Most frequent TRAIN-answer-category token in facts, ignoring question frequency.
- A separate frequency variant excluding category tokens mentioned in the question.
- An unlearned raw-text grammar solver for direct and inverse QA4 relations.

The symbolic solver consumes only raw facts and question; it reports unsupported or ambiguous inputs and does not repair them, use support IDs, overwrite gold, or supply representations to neural models. These controls can reveal a shortcut; they do not establish neural learning.

The audit retains exact raw and encoded duplicate counts, gold conflicts, label coverage, geometry, all control predictions and scores on TRAIN, DEV and the official TRAIN union. Such DEV inspection is development exposure and will be disclosed before a later final evaluation.

## Contract tests and admission sequence

Twenty-four independently authored stdlib unit methods use invented contexts only. They cover original numeric IDs, prior-only fact visibility, raw byte hashes, gold/support isolation, connected-component grouping, conflict retention, TRAIN-only fitting, punctuation/order fidelity, output blankness, padding holes, OOV and overflow behavior, empty denominators, CE masks and distinct shortcut semantics. Valid fixture contexts are exported as `DEVELOPMENT_FIXTURES` for a prospective exposure ledger. Invalid parser mutations remain in the test source.

The frozen workflow executes contract tests first. All 24 collected methods must pass with no skipped tests before acquisition starts. JUnit subtest counters, if present, are not interpreted as additional independent test methods. A failed test stops the phase and retains its source, log and receipt. The current static review is explicitly not a compilation or execution result.

No checkpoint-gate test is claimed by this data-only suite. That belongs to the later scientific controller before any final access.

## Resource and evidence policy

This execution uses one standard public ubuntu-24.04 CPU runner, Python 3.12.14, psutil 7.2.2 and pytest 9.1.1. Numerical thread environment variables are fixed to 1, Git packing/index work to 1, aggregate active CPU reservation at most 4, and the available-memory admission floor is 8 GiB. No GPU, paid runner, account/billing change, cache or artifact-storage upload is requested.

The worker limit is 1,200 seconds total, reserving the final 180 seconds for stopped evidence archival; the workflow limit is 25 minutes. Child stages are supervised every second with heartbeats at most 60 seconds apart and interim archival at most 300 seconds apart while a child runs. Available RAM is sampled, not continuously measured; archive-only intervals are not described as continuous memory monitoring. Only owned child process groups are terminated on failure or deadline.

The committed plan binds every used source/config/test input by SHA-256. A source ZIP, plan, raw downloads, failed attempts, parsed TRAIN, split/encoder objects, control predictions, test results, child logs and resource observations are copied to immutable final evidence on the dedicated results branch. The source branch remains isolated. A final stopped archive says whether the attempt completed or failed; an archive's existence alone is not success.

After reviewing this phase, a separate complete scientific recipe and run inventory may be frozen. No neural run or TEST result is authorized by this acquisition plan alone.
