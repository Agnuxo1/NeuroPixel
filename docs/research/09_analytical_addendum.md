# Item 9 — Additional source reading and analytical checks

This addendum was prepared after the Stage-A source freeze and before Stage-B admission. It changes no frozen training, selection, dataset or evaluation rule. The conclusions below are analytical expectations and methodological interpretations, not measured neural outcomes.

## Primary-source access completed by root

The version-suffixed arXiv PDF endpoints for CFQ and CLEVR repeatedly returned access errors in this session. Following the authors' official project/publication links succeeded with the `.pdf` endpoints:

- Keysers et al., *Measuring Compositional Generalization: A Comprehensive Method on Realistic Data*, ICLR 2020, arXiv v2 dated25 June2020: https://arxiv.org/pdf/1912.09713.pdf . Root read the abstract, introduction and sections2–3, including the atom/compound split principles and rule-based construction. The official Google publication page is https://research.google/pubs/measuring-compositional-generalization-a-comprehensive-method-on-realistic-data/ .
- Johnson et al., *CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning*, arXiv v1 dated20 December2016, presented at CVPR2017: https://arxiv.org/pdf/1612.06890.pdf . Root read section4.7 and the discussion, and checked the version marker. The official project page is https://cs.stanford.edu/people/jcjohns/clevr/ .

These readings supplement the frozen methodological review. CFQ motivates keeping primitives available while testing genuinely different compositions; merely reserving eight-filler bags in the current two-event grammar does not establish a maximum-compound-divergence benchmark. CLEVR illustrates why evaluating controlled attribute combinations and explicit task factors helps diagnose behavior. The current study is neither a reproduction of these benchmarks nor an evaluation of language or visual reasoning. Those sources do not predict the outcome of NeuroPixel's new task.

## Duplicate-input expectation

Each final bag provides8 event/role queries under6 conditions, hence48 nominal rows. The8 `query_switch` canvases are precisely the8 base canvases in a different query order. They therefore add paired behavioral checks but no new unique input. Under the implemented distinct-filler grammar, the two agent/patient exchange conditions each contribute8 new canvases, joint event relabeling contributes8, and the explicitly nonidentity layout permutation contributes8. Thus the source-derived prediction is40 unique canvases per bag, with32 appearing once and8 appearing twice. Across256 disjoint bags, the prediction is12,288 nominal rows and10,240 unique canvases per checkpoint. The final audit must count actual bytes and check duplicate targets/predictions; it must not substitute these expectations for measurement.

The uniqueness argument uses visible fact triples: an agent/patient exchange changes which filler belongs to a role, event relabeling changes which event label accompanies each filler, and layout permutation retains the original triple set while moving it. Distinct fillers prevent accidental equivalence of these changed triple sets. Distinct bags prevent cross-bag equality. The only prescribed exact repetition is the selector switch versus the corresponding base query.

## Size of the rendering space

For one specified bag there are4!×2!×2!=96 semantic assignments to the four noun slots, two action slots and two place slots. Eight labeled facts occupy eight distinct rows from nine, with six allowed starts each, yielding9!×6^8=609,499,054,080 layouts. Their product is58,511,909,191,680 rendered scenes; eight possible queries give468,095,273,533,440 rendered query inputs per bag. These are counts derived from the declared grammar, not enumerated cases or completed tests. A large rendering space does not itself establish difficult reasoning, generalization, novelty or practical value. The memorization diagnostic fixes32 rows; the ordinary training sampler resamples arrangements of its2,048 fixed bags.

## Uncertainty and control interpretation

The five paired primary seeds change model initialization and the privately seeded minibatch/firing streams. Their t interval therefore describes variation across five paired training realizations under this complete recipe, not isolated initialization noise. The shared seed pairs the input stream across families; it cannot make the two families' internal stochastic operations identical.

Uniform partial-information controls have exact marginal expected answer probabilities. A joint all-eight success probability for a stochastic control additionally needs a specified coupling of its answers; multiplying eight probabilities silently assumes independent draws. No such new assumption is introduced. The deterministic symbolic and train-role-majority controls can be evaluated directly for joint correctness. Whole-scenario bootstrap uncertainty and variability over five trained runs remain separate conditional quantities.

## Offline analysis implementation references

Before the Stage-B freeze, root also consulted the versioned primary documentation for [NumPy2.3 PCG64](https://numpy.org/doc/2.3/reference/random/bit_generators/pcg64.html), [NumPy2.3 quantile](https://numpy.org/doc/2.3/reference/generated/numpy.quantile.html) and [SciPy1.17 Student t](https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.stats.t.html). The independently implemented offline analysis will explicitly bind Python3.12.14, NumPy2.3.5 and SciPy1.17.0, one numerical thread, PCG64 with seed94001 and the declared2,000 resamples. It will retain the exact sampled bag-index array and use linear interpolation for percentile endpoints. This independently versioned arithmetic environment is separate from the unchanged neural training/inference environment. Bit-generator stream documentation is not treated as an unrestricted guarantee for every Generator distribution call across environments; the saved index artifact identifies the actual resampling used.

The separate numerical tolerances follow the quantities being compared: predictions, targets, counts and source/data identities require exact agreement; saved float32 logits may have a declared float32 roundoff allowance for repeated-input comparisons; NLL is computed by the trainer in float64 from those saved float32 logits and can be independently recomputed with a tighter double-precision arithmetic tolerance. No tolerance is selected using final performance results.
