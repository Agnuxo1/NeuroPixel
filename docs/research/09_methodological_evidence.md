# Item 9 — Methodological evidence for explicit two-event role binding

Consulted: **2026-10-07 UTC**. Scope: primary literature and prospective design reasoning for item 9 only. This note contains no model outcomes, generated performance datasets or executed scientific simulations. The executable recipe and decision rules belong to [09_protocol.md](09_protocol.md) and its frozen execution plans. Item-5 H1 remains closed. This bounded review does not establish scientific priority.

## 1. What the diagnostic can establish

The proposed task asks a model to retrieve a filler using the conjunction of an explicit event identifier and an explicit role. Each canvas contains eight labelled facts: two events, each with agent, action, patient and place. Knowing the role alone leaves two candidate fillers; knowing the event and noun category alone leaves its agent and patient unresolved. A correct response to all queries and their controlled transformations supplies behavioral evidence about these associations under this particular generator.

The distinction between a filler and its binding is formalized in tensor-product representations [S1]. Controlled generalization splits in SCAN, COGS and CFQ illustrate why the exact withheld combination matters [S2–S4]. CLEVR and HANS motivate generator-based gold answers and controls against solutions that exploit superficial regularities [S5–S6]. The design below applies these lessons; it does not reproduce those benchmarks or require their representations.

Explicit role and event markers substantially simplify the problem. This investigation does not test syntactic role inference, implicit event identification, anaphora, repeated entities, recursive structure, more than two events, perception from photographs or general systematicity. Success cannot establish that the learned state implements tensor products, discrete variables, a biological binding mechanism or human-like reasoning. Failure under the frozen budget leaves this operational capability unestablished; it does not prove that the architecture could never learn it.

## 2. Primary sources and reading limits

All six entries below were consulted through **targeted sections of primary full texts**, not only abstracts. Metadata pages were used to verify titles, venues and preprint versions. Summaries are original paraphrases; section/page references identify the material actually used. No claim is made to have examined every appendix or replicated any published experiment.

| ID | Primary contribution relevant here | Supported design implication and limit |
|---|---|---|
| S1 | Smolensky (1990), §3.1, printed pp.186–188, formalizes retrieval from role–filler bindings. Linearly independent role vectors permit exact unbinding through their duals; retrieval using the role vectors themselves depends on their geometry and may incur interference. | Distinguish filler identity from its assignment, and distinguish the two instances of the same role by event. This formal result concerns the specified representation; behavioral accuracy cannot show that an NCA learned it. |
| S2 | Lake and Baroni (2018), §2 and Experiment 3, define an unambiguous command/action mapping and test composed uses of primitives whose training contexts are controlled. Performance depends on the generalization split. | State exactly what is new at test time. New bags under one fixed two-event template are not evidence for arbitrary novel structures, unseen grammar rules or length extrapolation. The paper's failures concern the studied settings and models. |
| S3 | Kim and Linzen (2020), §§3.2–3.5 and 4, distinguish lexical/grammatical-role generalizations from structural and recursive extensions, using generated semantic interpretations and separate generalization types. | Report role/query cases separately and avoid collapsing lexical novelty, role transfer and structural depth into one claim. Our explicit labels remove much of the syntactic inference and semantic parsing required by COGS. |
| S4 | Keysers et al. (2020), §2, formulate distribution-based compositionality assessment with similar distributions of atomic elements and different distributions of their compounds. | Audit primitive coverage alongside withheld groups. A hash split of bags is not a measured maximum-compound-divergence split; this study implements neither CFQ derivation graphs nor its divergence optimization. |
| S5 | Johnson et al. (2017), §3 and §4.1, use functional programs for precise gold answers, filter ill-posed or degenerate questions, and examine question-only biases. Balancing answer types does not remove every conditional shortcut. | Validate the rendered scene/query with an independent parser; evaluate category and query-only controls. A symbolic grid does not reproduce CLEVR's visual-perception or multi-step reasoning demands. |
| S6 | McCoy et al. (2019), §2/Table 2 and §3, construct examples that support or contradict specific syntactic heuristics, including lexical-overlap strategies that ignore relevant word order. | Test matched content with changed bindings and unchanged content with changed queries. These are targeted falsification controls, not proof that every shortcut is eliminated. Our retrieval task is not natural-language inference. |

### Source records

- **S1 — Paul Smolensky (1990).** *Tensor product variable binding and the representation of symbolic structures in connectionist systems.* Artificial Intelligence 46(1–2), 159–216. DOI: [10.1016/0004-3702(90)90007-M](https://doi.org/10.1016/0004-3702(90)90007-M). [Institutional author record](https://www.microsoft.com/en-us/research/publication/tensor-product-variable-binding-representation-symbolic-structures-connectionist-systems/); [primary full-text PDF read](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/02/Smolensky-90-AI-Tensor-Product-Variable-Binding-and-the-Representation-of-Symbolic-Structures-in-Connectionist-Systems.pdf). Reading centered on §3.1, Definition 3.2 and Theorem 3.3; the relevant formal retrieval distinction is in printed pp.186–188.
- **S2 — Brenden Lake and Marco Baroni (2018).** *Generalization without Systematicity: On the Compositional Skills of Sequence-to-Sequence Recurrent Networks.* ICML, PMLR 80, 2873–2882. [Publisher record](https://proceedings.mlr.press/v80/lake18a.html); [published PDF read](https://proceedings.mlr.press/v80/lake18a/lake18a.pdf). Read task construction in §2, Experiment 3 and the discussion of the scope of the tested generalizations.
- **S3 — Najoung Kim and Tal Linzen (2020).** *COGS: A Compositional Generalization Challenge Based on Semantic Interpretation.* EMNLP, 9087–9105. [Publisher record](https://aclanthology.org/2020.emnlp-main.731/); [published PDF read](https://aclanthology.org/2020.emnlp-main.731.pdf). Read §§3.2–3.5, 4 and the lexical-generalization discussion in §5.2.1.
- **S4 — Daniel Keysers et al. (2020).** *Measuring Compositional Generalization: A Comprehensive Method on Realistic Data.* ICLR 2020. [Venue PDF endpoint](https://openreview.net/pdf?id=SygcCnNKwr) returned an access challenge; it was not the text read. Used the authors' [arXiv:1912.09713v2 PDF](https://arxiv.org/pdf/1912.09713v2), revised 2020-06-25, focusing on §2 and the definitions of atom and compound divergence. [Version metadata](https://arxiv.org/abs/1912.09713v2).
- **S5 — Justin Johnson et al. (2017).** *CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning.* CVPR 2017. [Venue record](https://openaccess.thecvf.com/content_cvpr_2017/html/Johnson_CLEVR_A_Diagnostic_CVPR_2017_paper.html). The CVF PDF endpoint failed during this consultation. Used the authors' [arXiv:1612.06890v1 PDF](https://arxiv.org/pdf/1612.06890v1), submitted 2016-12-20, principally §3 on question generation, §4.1 on baselines, and the subsequent bias discussion. [Version metadata](https://arxiv.org/abs/1612.06890v1).
- **S6 — R. Thomas McCoy, Ellie Pavlick and Tal Linzen (2019).** *Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural Language Inference.* ACL, 3428–3448. DOI: [10.18653/v1/P19-1334](https://doi.org/10.18653/v1/P19-1334). [Publisher record](https://aclanthology.org/P19-1334/); [published PDF read](https://aclanthology.org/P19-1334.pdf). Read §2, Table 2, and §3/§3.1 on diagnostic construction and controls.

## 3. Task semantics, grouping and coverage

The protocol uses a 10×8 token canvas with eight distinct fact rows selected from rows 0–8. Each fact is a horizontal `[EVENT][ROLE][FILLER]` triple. The query event and role occupy row 9, columns 5 and 6; the blank output is column 7. Two new event identifiers extend the existing vocabulary to 37 tokens. Four distinct nouns, two distinct verbs and two distinct places make every role's two event-specific answers different.

A grouping key must depend only on the **canonical unordered filler bag**: sort noun IDs within their category, likewise verb and place IDs, then serialize the three category lists with an explicit, versioned format. It must not depend on assignment, row order, query, transformation or model prediction. There are

\[
\binom{12}{4}\binom{10}{2}\binom{8}{2}=623{,}700
\]

possible bags. SHA-256 modulo 100 with the declared residue ranges assigns the group to a partition. The 70/15/15 ranges specify bucket proportions, not exact realized population or sampled proportions. Save actual unique group lists, counts, content hashes and token/role/event support. All assignments and queries of a bag, including future redraws of its layout, retain that partition.

This boundary prevents the same complete filler bag from entering different partitions. Shared individual tokens and smaller subcombinations remain intentional. Thus the held-out unit is a new bag of familiar primitives within a familiar two-event grammar, not an unseen primitive, an unseen single-event triple, or an unseen structural rule. No CFQ-style compound-divergence claim follows from this hash construction. Report that scope even if every neural score is high.

The protocol keeps 2,048 training bags, 128 validation bags and 256 final bags, each unique within its partition. Training redraws assignments/layouts/queries from its bags. Validation selection uses **base only**, 128×8=1,024 rows. A single assignment per final bag supplies the final six-condition panel, 256×8×6=12,288 nominal rows per checkpoint. The final count describes saved measurements, not independent observations. In particular, query switching can reproduce another base query on the same scene.

The exact parser must infer the answer from visible triples and query tokens alone. It may know the public rendering grammar; it must not consume labels, hidden assignments, scenario IDs or generator metadata. Missing or duplicate event–role keys, malformed triples or a non-unique queried answer are contract failures. Agreement between a visible-input parser and separately generated gold labels is an integrity prerequisite, not a learned-model result. Generation and validation code must be checked independently enough to avoid merely repeating one mistaken lookup.

## 4. Counterfactual criteria

These transformations are declared before performance inspection and are determined by scene semantics. They are not selected according to model success or failure. Every condition retains all eight queries; eligible denominators for changed-gold and invariant cases remain separate.

| Condition | Expected gold relation to its paired base query | Diagnostic scope |
|---|---|---|
| `base` | The filler bound to the queried event and role | Absolute retrieval performance |
| `swap_queried_agent_patient` | Different answer for agent/patient queries; unchanged for action/place | Sensitivity to the relevant noun binding; four changed-gold and four invariant cases per bag |
| `swap_other_agent_patient` | Unchanged for every query | Insensitivity to a specified change in the other event |
| `relabel_events` | Unchanged after both fact event labels and the query label are exchanged | Invariance to a consistent renaming of the two event identifiers |
| `query_switch` | Different for every query because same-role fillers are distinct across events | Sensitivity to which event is requested |
| `layout_permutation` | Unchanged for every query | Invariance to the declared reassignment of fact positions |

Swapping the agent and patient changes the visible associations while preserving the filler bag. Query switching preserves the facts but changes the selector. These interventions address different shortcuts. A globally renamed event and query must not be scored as requiring an answer change. The controls address their declared transformations only; random positions do not prove independence from every possible spatial cue.

For each fixed checkpoint, retain absolute accuracy, role-wise accuracy, cross-entropy, all-eight correctness and raw prediction tables. The primary endpoint is base agent/patient macro accuracy with equal event and noun-role weighting. Since every bag contains all four event-by-noun-role queries, first averaging those four indicators per bag and then averaging bags is equivalent to that balanced macro endpoint.

Counterfactual scoring must include **both base and variant correct**. Prediction equality on invariant cases can be perfect for a constant wrong predictor; inequality on changed-gold cases can be satisfied by two wrong answers. Report equality/inequality with the corresponding joint correctness and exact eligible counts. Do not multiply marginal accuracies to estimate joint success or treat the variants as independent trials. All-eight correctness is a stricter scenario-level secondary endpoint, not eight new independent replications.

## 5. Controls and analytical expectations

The declared controls distinguish retrieving any plausible filler from resolving the queried association. Uniform controls can be scored through the exact probability assigned to the correct token; random draws are unnecessary. The following expectations are mathematical consequences of this generator's distinct fillers and balanced queries, not measured neural baselines or literature performance claims.

| Control | Information deliberately omitted | Expected global accuracy | Expected agent/patient accuracy |
|---|---|---:|---:|
| Visible-input symbolic lookup | None of the explicit associations | 100% on valid generated inputs | 100% |
| Role-only matching | Event identity; uniform over its two matching fillers | 50% | 50% |
| Event-aware category matching | Agent versus patient distinction; uniform over the two nouns in that event | 75% | 50% |
| Global bag/category matching | Both event and noun-role association | 37.5% | 25% |

The last row averages 25%, 50%, 25% and 50% across agent, action, patient and place. Event-aware category matching resolves action and place exactly but guesses between two nouns. These rates require the stated candidate sets; they are not the expectation of an arbitrary uniform draw over all 37 tokens. The parser is a grammar-informed algorithmic control, not a parameter- or compute-matched neural model.

A per-query-role frequency/majority control is fitted on training answers only, using the declared lowest-token-ID tie break. Report its actual performance instead of assuming balanced global marginals remove finite-sample or conditional biases. Neither validation nor final labels may fit these frequencies. A strong score against the weakest bag control alone does not establish event–role conjunction: role-only or event-aware category strategies are stronger specified alternatives.

## 6. Independence, selection and uncertainty

For sampling uncertainty, the unit is the **bag/scenario**. Resample its complete query-by-condition block and use the same resampled indices for paired comparisons. The 12,288 rows contain only 256 sampled final bags; they do not provide 12,288 independent units. Any additional assignments of a reused bag would remain in that cluster. Bootstrap intervals describe the fixed checkpoints' performance over this sampled scenario distribution, subject to its sampling assumptions and finite support. They do not include training or development-selection variation.

Training initializations answer a separate uncertainty question. The protocol has five primary initializations per family after a development-only pilot; report every run and all paired NeuroPixel-minus-Transformer differences. Its t interval uses five paired differences (df=4), conditional on the selected rates, shared final dataset and fixed recipe. Limited seed count and possible non-normal outcome distributions constrain its interpretation. Combining all conditions or bags with seeds as if they were independent training replications would misstate precision.

The protocol's two-stage freeze, retained unsuccessful pilot configurations, complete checkpoint inventory and consume-once final gate are necessary safeguards for this investigation. Public recipes and seeds make inputs reproducible; they do not constitute external custody or prove analyst blindness. Newly generated item-9 instances also do not retroactively restore the historical H1 holdout. The training memorization fixture is an optimization diagnostic, not evidence of compositional generalization. Its result and failures must remain visible without post-final budget changes.

The predeclared competence screen and all secondary conditions should be reported in full, including failures. Even meeting the screen establishes only the specified explicit two-event behavior. A difference from the Transformer is a separate paired descriptive result with its uncertainty; neither the screen nor favorable secondary subsets supports a universal architecture ranking or an inferred internal binding mechanism.

## 7. Reproducible search and coverage limits

The bounded search used the following literal web queries on 2026-10-07, followed by primary publisher/institution/author links:

1. `Smolensky 1990 tensor product variable binding representation symbolic structures pdf`
2. `Lake Baroni 2018 generalization systematicity compositionality SCAN PMLR`
3. `COGS compositional generalization challenge semantic interpretation Kim Linzen 2020 ACL`
4. `Keysers Measuring Compositional Generalization Comprehensive Method Realistic Data ICLR 2020 pdf`
5. `CLEVR diagnostic dataset compositional language elementary visual reasoning Johnson 2017 CVPR pdf biases`
6. `Geirhos shortcut learning deep neural networks 2020 paper arxiv`

The HANS publisher record/PDF was then opened directly as a more task-specific primary empirical source for heuristic counterexamples. The Geirhos et al. review was a background candidate, not one of the six evidentiary sources; its opening is not represented as a targeted full-text reading. The CFQ and CLEVR access failures and the exact author-preprint versions used are recorded above.

Inclusion required an original formal treatment or original diagnostic dataset/method with directly applicable design lessons and accessible relevant full-text sections. Search snippets, abstract-only support, derivative summaries and merely keyword-related sources were not used for central propositions. The six-source selection is sufficient for this bounded diagnostic rationale; it is not a systematic review of all binding architectures, compositional benchmarks or shortcut mechanisms. No worldwide novelty conclusion follows from it. The task-specific grouping, numerical expectations and inference limits in this note are our analytical design reasoning, clearly distinct from the cited papers' own experiments.
