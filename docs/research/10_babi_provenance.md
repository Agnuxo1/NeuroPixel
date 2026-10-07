# Item 10 — External benchmark provenance: bAbI task 4

**Scope and access date:** 2026-10-07. This is a source/documentation review for item 10, not an experiment or a frozen training recipe. No benchmark train or final-test example file was downloaded or inspected by this reviewer. Published paper illustrations and repository documentation were read; withholding dataset files does not imply that every public illustration is historically unexposed.

## Selected source and exact member names

The selected benchmark is the **original English 1k bAbI task 4**, Two Argument Relations. The archive URL explicitly recorded by the published `facebook/babi_qa` loader is:

`http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz`

The exact original member names are:

| Purpose | Archive member |
|---|---|
| Original 1k training | `tasks_1-20_v1-2/en/qa4_two-arg-relations_train.txt` |
| Original final test | `tasks_1-20_v1-2/en/qa4_two-arg-relations_test.txt` |

The same loader separately names `tasks_1-20_v1-2/en-valid/qa4_train.txt`, `qa4_valid.txt` and `qa4_test.txt`. That provided development variant must not silently replace the selected full-1k source. If development is carved from the selected training member, its grouping, membership and exposure are part of the prospective protocol.

The publisher card reports 1,000 training and 1,000 test story records for `en-qa4`. That is metadata, not an independently recounted sample size here; parsing must record both stories and question examples rather than treating all dataset configurations as having one question per story.

The original archive has **not been downloaded or hash-verified in this review**. Its actual bytes, download outcome, redirects if any, and selected-member SHA256 values must be recorded by the authorized acquisition stage. A named URL and version do not substitute for artifact identity. The publisher's research landing page redirected to a sign-in page during this review; the author's directory did not load through the search tool. Neither observation establishes that the archive URL itself is unavailable. No unverified HTTPS spelling or guessed alternative host is asserted as an official archive URL.

## Why task 4 is relevant

Weston et al., *Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks*, arXiv:1502.05698, §3, printed p.3, identifies task 4 with subject/object discrimination: questions with the same words in different orders can require different answers. Table 1, printed p.4, illustrates single location-word answers. The output should therefore be learned from the actual training answer vocabulary, not assumed to be a four-direction classifier.

The paper evaluates correct versus incorrect answers and distinguishes QA-only training from training that also supplies supporting-fact supervision (§5, printed pp.6–9). A raw-text, answer-only experiment belongs to the former setting. Results from stronger supervision or external NLP features are not matched baselines for it. The paper's discussion presents synthetic prerequisite tasks for learning methods, not a certificate of general language understanding.

This makes task 4 a bounded external **synthetic QA** benchmark for training the architecture from scratch. It is not zero-shot transfer of item-9 checkpoints, a real-world domain evaluation, or a continuation of the closed item-5 H1. Item 9's low NeuroPixel binding score and failed competence screen remain unchanged.

## Licensing and distributions that must remain distinct

The dataset card under the publisher's `facebook/babi_qa` namespace and its loader declare **Creative Commons Attribution 3.0** for the dataset. This review records that published declaration; it has not inspected an embedded license inside the original archive. Preserve dataset attribution, the original paper and the source URL in any acquired-data manifest.

The `facebookarchive/bAbI-tasks` repository's `LICENSE.md` is a BSD notice specifically for the **software**. Its README explicitly warns that the Lua generator is a rewrite and cannot reproduce the published dataset exactly. The reviewed repository commit is `ccd8fd6af76d35346109e7bb5f7de9138d055e01`. Running that generator would create a different dataset and would not satisfy this original-benchmark selection.

The first-party ParlAI builder, at commit `a29567f7ce76992fd1f03c51ba9e3b155a37ea51`, points to `http://parl.ai/downloads/babi/babi.tar.gz` with SHA256 `f7f0bee187efca0d81c3daac1b162cda4eb7f9505dee5ad6846eabbed3dbf92e`. However, its teacher reads `tasks_1-20_v1-2/en-valid{exsz}-nosf/` members. That is a separate distribution with different paths; its checksum is **not** an expected checksum for the selected original archive. The current selection excludes the ParlAI repack.

## Admissible adaptation and evaluation boundaries

The model input should consist of the original visible story context and current question, processed by a deterministic text tokenizer and declared canvas layout. The loader shows that answers and supporting sentence IDs are separate tab-delimited fields. Neither may enter the neural input or choose which sentences are presented. Earlier question-answer records must not accidentally expose their gold answers when constructing later contexts. Converting the text to gold subject/object triples, marking the supporting fact, or placing the correct answer in the canvas would replace the requested learning problem.

Vocabulary, output candidates, handling of unknown words and canvas limits must be determined from the authorized training/development process and frozen before final access. Any overlength or unknown-answer case needs the declared behavior and denominator; silent filtering or outcome-driven changes are not valid.

The full official test remains the benchmark endpoint. Before opening it, declare an input-only duplicate key and a secondary test subset absent from the complete training/development exposure ledger. Report all-test, seen-input and unseen-input counts and scores, including a null result if the unseen denominator is zero. These are diagnostic strata, not a newly generated external domain. Exact textual novelty does not establish novel semantic composition. No claim about the amount of task-4 overlap is made from blogs or uninspected files.

Use a matched neural comparator receiving the same raw inputs, with training-only selection, and distinguish simple question-only or order-insensitive controls from a symbolic parser. A parser can validate task semantics; it cannot serve as evidence that a neural system learned the task. Uncertainty should preserve shared-story and exact-duplicate dependence and separate training-realization variation from fixed-checkpoint example resampling.

## Alternatives considered

| Benchmark | Verified public interface and split distinction | Reason not selected for this bounded adaptation |
|---|---|---|
| SCAN | The author repository supplies command-to-action sequences and standard simple, length and held-out-primitive splits. Reviewed commit: `c4b756cbc010d75c912f16c42c8f15dc6b7e6c8f`. Its LICENSE is a BSD notice headed for CommAI-env software; no separate dataset license text was verified. | Native evaluation needs complete output sequences and a declared decoder/stopping rule. Reducing targets to one action or presenting gold prefixes would not preserve the original task. |
| COGS | Natural-language input maps to structured semantic output. Author documentation distinguishes in-distribution dev/test from the generalization set and recommends case-level reporting. Reviewed commit: `165a7b669eade971fa47bf568a2e51925360fed8`, with the 2022 correction to 50 generalization labels; repository MIT license. | It needs sequence/structured prediction and exact semantic evaluation beyond the current single-answer interface. The corrected revision must be named in any later study. No generalization data file was opened here. |

These remain serious benchmarks, but adopting either now would require a different output architecture and a separate prospective scope. bAbI4 allows the original answer task to remain intact with a smaller input-adapter change. No performance claim follows from that engineering feasibility judgment.

## Primary sources read

1. Weston et al. Original paper, §3, Table 1, §5 and §6: [PDF](https://arxiv.org/pdf/1502.05698).
2. Publisher dataset card, license, schema and split metadata: [facebook/babi_qa README](https://huggingface.co/datasets/facebook/babi_qa/blob/main/README.md). Published loader, original URL/member paths and field parsing: [babi_qa.py](https://huggingface.co/datasets/facebook/babi_qa/raw/main/babi_qa.py).
3. Generator author's warning and software license, pinned: [README](https://github.com/facebookarchive/bAbI-tasks/blob/ccd8fd6af76d35346109e7bb5f7de9138d055e01/README.rst), [LICENSE](https://github.com/facebookarchive/bAbI-tasks/blob/ccd8fd6af76d35346109e7bb5f7de9138d055e01/LICENSE.md).
4. First-party repack documentation in executable source, read without running it: [ParlAI build.py](https://github.com/facebookresearch/ParlAI/blob/a29567f7ce76992fd1f03c51ba9e3b155a37ea51/parlai/tasks/babi/build.py), [agents.py](https://github.com/facebookresearch/ParlAI/blob/a29567f7ce76992fd1f03c51ba9e3b155a37ea51/parlai/tasks/babi/agents.py).
5. SCAN author repository: [README](https://github.com/brendenlake/SCAN/blob/c4b756cbc010d75c912f16c42c8f15dc6b7e6c8f/README.md), [LICENSE](https://github.com/brendenlake/SCAN/blob/c4b756cbc010d75c912f16c42c8f15dc6b7e6c8f/LICENSE).
6. COGS author repository: [README](https://github.com/najoungkim/COGS/blob/165a7b669eade971fa47bf568a2e51925360fed8/README.md), [LICENSE](https://github.com/najoungkim/COGS/blob/165a7b669eade971fa47bf568a2e51925360fed8/LICENSE).

The review used primary papers and publisher/author metadata. It did not execute code, retrieve held-out example files, create a new generator, or establish acquisition success for the original archive.

## Acquisition addendum: documented HTTPS mirror

Added on 2026-10-07, before acquisition or inspection of the selected archive. The Ray project’s [`pbt_memnn_example.py`](https://github.com/ray-project/ray/blob/master/python/ray/tune/examples/pbt_memnn_example.py), read as primary implementation documentation, explicitly requests `https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz` and gives the original author-hosted HTTP archive as its manual fallback. This supports admitting the S3 URL as a documented benchmark mirror. It does not establish ownership by the dataset authors, an independently authenticated upstream checksum, or byte identity with the original host.

The prospective acquisition attempts both observed URLs, with a 32 MiB compressed-byte cap per attempt. Preserve each raw archive or retained partial download, its SHA-256, source/final URL, recorded redirect history and any failure. If both yield the named English 1k QA4 TRAIN member, compare both full-archive hashes and exact TRAIN payload bytes; disagreement requires review before fitting. If only the mirror succeeds, report that provenance explicitly. Other TAR entries may be traversed as header metadata, but their payloads must not be extracted, decoded, tokenized or scored in this phase.

The Ray example is evidence for the URL only. Its vocabulary and maximum-length calculations use both training and test stories, so its adapter must not be reused here. The present adaptation fits its vocabulary and geometry from the assigned optimization-TRAIN records and handles development or final incompatibilities through declared admission rules.
