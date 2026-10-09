# Item 1 — Primary prior art and contribution boundaries

**Status: scoped literature review complete; empirical novelty is not established.**

**Review and access date:** 2026-10-06.

**Literature cutoff:** 2026-10-06.

**Task:** SCI-001, programme item 1 only.

**Audited source:** `Agnuxo1/NeuroPixel`, commit `7da18d1cbd36c40c48228118abb20a7b5ab032f4`.

**Working branch:** `research/scientific-validation-2026-10-06`.

**Companion documents:** [mechanism specification](01_mechanism_specification.md) and [structured references and complete query ledger](references.json).

## 1. Finding and scope

NeuroPixel has a concrete, testable implementation in the family of spatially local recurrent neural networks. Its shared cellular rule, stochastic residual updates, persistent input access, tied input/output dictionary and auxiliary supervision have identifiable antecedents or close functional comparators. Learned regeneration, symbolic grid processing, NCA adaptation, cellular memory and NCA-related language modeling also have primary research precedents. The evidence therefore supports describing the particular combination as a **candidate contribution whose distinctive effect remains to be isolated**. It does not establish a new scientific principle, an exclusive architectural category or worldwide priority. [R01] [R02] [R03] [R06] [R07] [R09] [R11] [R12] [R14]

This review compares mechanisms and the evidence actually reported for them. A source showing a related capability is not an independent replication of NeuroPixel. Similar biological vocabulary does not show equivalent computation, and a successful artificial task does not validate a biological mechanism.

Fourteen primary works are included. Twelve were inspected as full texts; two have claims restricted to accessible primary abstracts. The recent coverage includes papers first released or published during 2024–2026. This is a **targeted scoping review and citation audit**, not an exhaustive systematic review or a patent search. It does not determine the earliest public disclosure date of every NeuroPixel claim. Consequently, papers newer than a particular implementation may be current comparators without being chronological prior art for that implementation.

## 2. Reproducible review protocol

The following criteria were applied during the review and consolidated here; the review was not preregistered.

### 2.1 Inclusion and exclusion

Include a work if it was publicly identifiable by the cutoff date, has a mechanism relevant to a listed NeuroPixel claim, and has an accessible primary manuscript or author abstract sufficient to support the recorded statement. Relevant mechanisms are local shared recurrence, task-conditioned cellular computation, input preservation, tied decoding, distributed binding, regeneration, memory, novelty-controlled allocation and NCA-related language or symbolic reasoning.

Use official publication records, author manuscripts and institutional author archives for scientific claims. Secondary search results may identify a lead, but a lead must be resolved to its primary source. Exclude unrelated uses of “NCA” or “Neuropixels,” unverified identifiers, and social or repository performance claims as evidence of independent scientific validation. When only an abstract is accessible, exclude claims that require unavailable methodological details.

Count a preprint and its proceedings version as one work. Record first posting, inspected revision and publication dates separately where verified. Do not infer a posting date from the date-like component of an identifier. An access error is recorded as an access limitation, not proof that a paper is false.

### 2.2 Search and verification procedure

Two public search routes were used: **S1** denotes `system1_search_query`; **S2** denotes `system2_search_query`. Routine discovery used S2, while uncertain identities, current literature and gaps were checked with S1. Searches were followed by direct primary-source opening and citation chaining. Neural GPU was checked through the references of the recent language work; EngramNCA was followed to its own manuscript and author page.

All searches occurred on 2026-10-06. The machine-readable ledger records **44 query strings in 18 batches**, their engine, the applied recency or domain filter, and a separate direct-retrieval record. A batch is a documentation grouping, not a claim about the number of API calls. Representative entries are:

| Search purpose | Exact query examples | Route and filters |
|---|---|---|
| Foundational NCA | `site.distill.pub 2020 growing neural cellular automata Mordvintsev` | S1 |
| Tied decoding and novelty | `Press Wolf Using Output Embedding to Improve Language Models 2017 ACL`; `Carpenter Grossberg 1987 massively parallel architecture self organizing neural pattern recognition machine vigilance PDF site:bu.edu` | S2 |
| Recent adaptation and memory | `"neural cellular automata" "continual learning" 2024 2025 2026`; `"Neural Cellular Automata" "associative memory"` | S2 |
| Recent NCA language | `"Neural cellular automata" "language" "2024" "2025"` | S1; 730-day recency filter |
| Independent language cross-check | `"neural cellular automata" language 2024 2025` | S2; arxiv.org, openreview.net, aclanthology.org |
| Disputed identity | `"TextNCA"`; `"2608.02050" arxiv` | S1 |
| Latest memory mechanism | `"Online Task Adaptation via Self-Organisation"`; `"2609.29281" site:arxiv.org` | S1 |
| Publication verification | `"Attention Pooling Enhances NCA-Based Classification of Microscopy Images" site:link.springer.com`; `"TextNCA" site:aclanthology.org` | S2 and S1 respectively |

The full ledger and all retrieval URLs are in [references.json](references.json). Search rankings and access availability can change, so rerunning the strings reproduces the procedure rather than an immutable result set. Exhaustive hit counts were not retained; no screening-flow totals or recall estimate are asserted.

### 2.3 Evidence labels

**Full text** means the primary manuscript was inspected for the mechanism and experimental scope used here. **Abstract** means only the primary author abstract and bibliographic record support the claims. These labels concern retrieval depth, not a quality score or a guarantee of replication. Publication status is recorded separately; a recent arXiv manuscript is not labeled peer reviewed merely because authors report future acceptance.

Every reference record contains metadata, primary URL, supported claim, verification date, access limitations and an original summary shorter than 100 words.

## 3. Mechanism inventory used for the comparison

The detailed equations and code boundaries are in the [companion specification](01_mechanism_specification.md). The comparison uses the following directly inspected implementation facts.

| NeuroPixel mechanism | Audited behavior | Consequence for the literature comparison |
|---|---|---|
| Local rule | Learned depthwise 3×3 perception, followed by shared 1×1 layers and residual recurrence. | Compare against NCA and convolutional recurrent models at the operation level. |
| Dimensions | Default token embedding width is 16; recurrent state width is 48. | “16-channel token colour” must not be confused with 16 total state channels. |
| Update randomness | A per-cell Bernoulli mask is applied during training; ordinary evaluation updates every cell. | Record train/evaluation differences when comparing stability and cost. |
| Dictionary and decoder | State is projected into the dictionary space, then multiplied by the same dictionary transpose used for input. | Weight tying and decoder rank are distinct experimental factors. |
| Input availability | The identity embedding is supplied on every recurrent step. Each ordinary forward call initializes a new state. | Recovery can use the retained input; persistence across calls needs a separate protocol. |
| “School” supervision | Selected intermediate states are trained to decode the original input tokens at occupied cells. | This is additional supervision; readability and answer accuracy are separate outcomes. |
| Canvas allocation | A script uses token-readability novelty to choose or create independent canvases. | This is external model allocation, not growth of one cellular tissue. |
| HRR helper | Circular-convolution binding utilities exist separately; they are not called by the audited core forward method. | An available helper does not establish the mechanism used by a trained model. |

These facts follow from [the model](../../neuropixel/model.py), [phase-3 training and allocation](../../scripts/phase3.py), [the scanner](../../neuropixel/scanner.py) and [the HRR helper](../../neuropixel/hrr.py). They describe implementation, not measured advantage.

## 4. Primary mechanism and evidence matrix

Dates below separate older foundations from current comparators. They are not a reconstructed priority timeline for NeuroPixel.

| Reference and inspected status | Established mechanism | Evidence and scope | Boundary relevant to NeuroPixel |
|---|---|---|---|
| [R01] Growing NCA, 2020; full text | Local shared perception/MLP updates, stochastic residual dynamics and damage training. | Learned visual growth, persistence and regeneration. | Repair and the basic cellular update are established ingredients. |
| [R02] Self-classifying MNIST, 2020; full text | Cellular classification with a persistent image channel and learned local perception. | Digit classification and response to changed images. | Input preservation is a close comparator; its implementation differs from token reconstruction loss. |
| [R03] Press and Wolf, 2017 proceedings; full text | Shared input embedding and output vocabulary weights. | Language-model experiments and parameter reduction. | Tied decoding alone is not a new component. |
| [R04] Carpenter and Grossberg, 1987; full text | Vigilance-controlled mismatch, category search and recruitment. | ART architecture and analysis under its assumptions. | Provides a stability–plasticity precedent, not proof of equivalence to the canvas controller. |
| [R05] Plate, 1995; abstract | Circular-convolution binding with approximate retrieval and cleanup. | Distributed associations, variable binding and structured representations. | HRR is existing machinery; verify whether an evaluated model actually uses it. |
| [R06] Neural GPU, 2015 first posting; 2016 v3 full text | Repeated convolutional gated updates on a 2D state. | Algorithm learning and length extrapolation, with substantial model search. | Shared local recurrence is an established route to algorithmic tasks. |
| [R07] NCAdapt, 2024 preprint / WACV 2025; full text | Frozen NCA backbone plus domain-specific adaptation capacity. | Continual hippocampus segmentation across MRI domains. | Expansion and retention have an NCA comparator; allocating whole canvases is a different implementation. |
| [R08] Active NCA, 2026 journal; abstract | Movable sensors coupled to a cellular system. | Attention and robustness in a three-class digit setting. | Its active sensing differs from fixed-coordinate input and from a global attention head. |
| [R09] NCA for ARC-AGI, 2025; full text | Task-specific learned cellular grid transformations. | Reports 23 solved tasks in a filtered set of 172 feasible public training tasks. | This denominator must not be presented as an unrestricted ARC score. |
| [R10] Attention-pooling NCA, 2025 preprint / 2026 chapter; full text | Local NCA features with a learned spatial pooling head. | Classification across eight microscopy datasets. | The cellular rule is local; final pooling aggregates globally. |
| [R11] EngramNCA, 2025; full text | Private cellular codes and learned propagation of represented morphology. | Morphology transfer, composition and regeneration. | Morphology memory does not directly establish symbolic fact recall. |
| [R12] TextNCA, 2026; full text | Causal 1D local attention, gated recurrence and stage-wise weight sharing. | SlimPajama training; WikiText-103 evaluation; weaker reported perplexity than its Transformer references. | Real NCA-style language-model precedent, with an architecture distinct from NeuroPixel's 2D rule. |
| [R13] Lee et al., 2026; full text | NCA trajectories generate synthetic token pretraining data. | Subsequent language and reasoning evaluation of Transformers. | The trained language architecture remains a Transformer. |
| [R14] Online Task Adaptation, 2026; full text | 2D NCA, persistent input and per-cell associative fast memory. | Supervised adaptation on held-out image-classification tasks after meta-training. | Separate transient state, fast memory and slow weights when interpreting “memory.” |

The matrix supports **component-level antecedents and specific comparison requirements**. It does not show that any listed system is identical to NeuroPixel or that the full combination has already been evaluated.

## 5. What remains a defensible contribution hypothesis

A useful candidate claim is: **under a specified task distribution and resource budget, combining a shared local rule, a tied low-dimensional token interface, repeated input conditioning and intermediate token supervision improves a particular measured property.** The property could be accuracy, retention, recovery, sample efficiency or readable intermediate representations. Each needs a separate operational definition.

The mechanism specification already identifies the decisive controlled comparisons. The literature review motivates them without executing later programme items:

| Candidate attribution | What must be separated in later evidence |
|---|---|
| Benefit from dictionary tying | Tying versus an untied decoder with the same factorization; report the parameter difference. |
| Benefit from the school objective | Extra token supervision versus architecture, including its interaction with tying. |
| Internal preservation or repair | Persistent input access versus surviving internal information; lesion and recovery protocols must match. |
| Benefit from recurrent computation | Number of local propagation steps, weight sharing, receptive field and training cost. |
| Novelty-driven learning | Vocabulary unfamiliarity, task novelty, routing quality and the total capacity of allocated models. |
| Explanatory scanner | What can be decoded versus what causally controls the answer; test intervention predictions. |
| General reasoning or language | Synthetic generator shortcuts, systematic held-out combinations, distance and length extrapolation, and actual language modeling. |

A result against one small baseline does not isolate the cause of an advantage. Equal parameter count does not imply equal training compute, inference work, access to targets or hyperparameter search. Conversely, a null result is scientifically usable: it may show that the combination is unnecessary for the tested effect.

The term “colour” describes the token representation in this implementation. The learned 16-dimensional interface and optional RGB anchors should be specified mathematically and tested as such. Renaming an embedding, displaying labels as colours, or encoding identifiers as RGB does not by itself establish a new computation. This is an implementation-level inference; no exhaustive survey of physical colour computation is claimed.

No source in this review independently replicates NeuroPixel. No experiment was run for this document, and no claim of a biological memory mechanism, autonomous task discovery or broad cognitive competence is certified by the review.

## 6. Identity, chronology and access checks

**TextNCA is verified.** The arXiv identifier `2608.02050` resolves to the stated title, authors and version dated 2026-08-03. Its training corpus is SlimPajama, with WikiText-103 evaluation. Parameters are shared within stages; the successive stages have different parameters. Author pages report a forthcoming EMNLP acceptance, but official proceedings were not verified, so this review records an arXiv preprint and does not adopt its priority claim. [R12]

**NCA-generated pretraining is a separate claim.** The Lee et al. paper uses cellular trajectories to train a Transformer. Its title cannot support saying that its final language architecture is an NCA. [R13]

**Dates are version-aware.** NCAdapt first appeared in 2024 and has a WACV 2025 entry. The microscopy paper first appeared in 2025; its MLMI 2025 chapter was published online on 2026-01-02. The ARC manuscript first appeared on 2025-06-18; 2025-12-02 is its third revision. Active NCA has an earlier bioRxiv record and a verified 2025 revision, but the exact first posting day was not independently recovered. Its journal record gives online publication on 2026-03-03. [R07] [R08] [R09] [R10]

**Access limitations are explicit.** HRR's primary abstract was readable, while the retrieved PDF extraction was unusable. Active NCA's journal abstract supports the bounded claims used here. NCAdapt's full text was verified through arXiv when direct CVF retrieval failed; official indexed CVF metadata supports the proceedings record. EngramNCA's arXiv full text was read, while its associated ALIFE publication information was verified only on the authors' page. [R05] [R07] [R08] [R11]

**Online Task Adaptation has a successful full-text retrieval.** Some review attempts returned an error or only an abstract. A later direct search-service `open` of [the versioned arXiv HTML](https://arxiv.org/html/2609.29281v1) succeeded, and its cellular update, associative memory, supervised adaptation and classification evaluation were inspected. The reference record preserves the successful URL, method and retrieval identifiers, alongside the inconsistent access observed by other reviewers. It is therefore included as full-text evidence, with recent-preprint status. [R14]

A code-only “neural-hard-drive” repository was inspected as a lead but excluded from the core academic evidence. Related work on tied word vectors was verified as corroboration without adding a duplicate mechanism row. These decisions are recorded in the structured references.

## 7. Completion decision

The item-1 literature deliverable is complete for its declared scope:

- Fourteen primary works have resolvable identities, evidence levels, dates, URLs and bounded claims.
- The core NeuroPixel mechanisms are mapped to specific antecedents and meaningful differences.
- Recent NCA adaptation, memory, language and symbolic reasoning literature is included.
- Search strings, filters, inclusion/exclusion criteria and retrieval limitations are recorded.
- Worldwide priority and a distinct empirical advantage remain **unestablished**.
- No training, implementation change or execution of programme items 2 onward was performed for this review.

The operational conclusion is **components known; combination testable; distinctive benefit pending matched comparisons, replication and ablations**. Completion of the review does not depend on obtaining a positive novelty result.

## References

Complete author lists, dates, publication status, access notes and original summaries are in [references.json](references.json).

- **R01:** Mordvintsev et al. (2020). [Growing Neural Cellular Automata][R01].
- **R02:** Randazzo et al. (2020). [Self-classifying MNIST Digits][R02].
- **R03:** Press and Wolf (2017). [Using the Output Embedding to Improve Language Models][R03].
- **R04:** Carpenter and Grossberg (1987). [A Massively Parallel Architecture for a Self-Organizing Neural Pattern Recognition Machine][R04].
- **R05:** Plate (1995). [Holographic Reduced Representations][R05].
- **R06:** Kaiser and Sutskever (2015; inspected revision 2016). [Neural GPUs Learn Algorithms][R06].
- **R07:** Ranem et al. (2024; WACV 2025). [NCAdapt][R07].
- **R08:** Kvalsund et al. (2026). [Sensor movement drives emergent attention and scalability in active neural cellular automata][R08].
- **R09:** Xu and Miikkulainen (2025). [Neural Cellular Automata for ARC-AGI][R09].
- **R10:** Yang et al. (2025; chapter online 2026). [Attention Pooling Enhances NCA-Based Classification of Microscopy Images][R10].
- **R11:** Guichard et al. (2025). [EngramNCA: a Neural Cellular Automaton Model of Memory Transfer][R11].
- **R12:** Mittal et al. (2026). [TextNCA][R12].
- **R13:** Lee et al. (2026). [Training Language Models via Neural Cellular Automata][R13].
- **R14:** Proroković (2026). [Online Task Adaptation via Self-Organisation][R14].

[R01]: https://distill.pub/2020/growing-ca/
[R02]: https://distill.pub/2020/selforg/mnist/
[R03]: https://aclanthology.org/E17-2025/
[R04]: https://sites.bu.edu/steveg/files/2016/06/CarGro1987CVGIP.pdf
[R05]: https://pubmed.ncbi.nlm.nih.gov/18263348/
[R06]: https://arxiv.org/html/1511.08228v3
[R07]: https://arxiv.org/html/2410.23368v1
[R08]: https://pubmed.ncbi.nlm.nih.gov/41807901/
[R09]: https://nn.cs.utexas.edu/downloads/papers/xu.alife25.pdf
[R10]: https://link.springer.com/chapter/10.1007/978-3-032-09513-8_56
[R11]: https://arxiv.org/html/2504.11855v1
[R12]: https://arxiv.org/html/2608.02050v1
[R13]: https://arxiv.org/html/2603.10055v1
[R14]: https://arxiv.org/html/2609.29281v1
