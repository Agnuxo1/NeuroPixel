# Item 9 — Additional recent variable-binding precedent

This literature addendum was prepared after Stage-B source publication and before any final outcome was inspected. It changes no recipe, model, endpoint, budget, selection rule or final-access condition. The running source remains `83fe135e301f76bc0c74e30c66bb18e067ca5959`.

## Primary source and reading scope

Yiwei Wu, Atticus Geiger and Raphaël Millière, **How Do Transformers Learn Variable Binding in Symbolic Programs?**, ICML 2025, PMLR 267:67284–67299. [Proceedings record](https://proceedings.mlr.press/v267/wu25j.html); [author manuscript, arXiv v2, 30 May 2025](https://arxiv.org/html/2505.20896v2). Root read the task, training, causal-intervention and discussion sections, plus appendices on sampling, training seeds, generalization and linear probes. The direct arXiv PDF retrieval failed because its approximately 14.5-MB response exceeded the web reader limit; the full-text HTML was accessible. This is not an execution or replication of their code.

The authors study dereferencing synthetic assignment chains with distractors. They report a progression from shallow heuristics to accurate binding and use activation interchange to investigate internal information flow. Their 12-layer, 37.8-million-parameter Transformer differs substantially from this study's reference. Their appendices also describe failures on some changed-length cases and limits of linear state probes. These qualifications matter alongside their successful results.

## Consequences for this investigation

The paper is a relevant precedent for learned variable binding and for distinguishing behavioral success from an internally tested mechanism. It supplies no NeuroPixel result. The present task instead asks for a filler matching one of two explicitly tagged events and four roles, with no reference-chain traversal. Its Transformer has two layers and 30,157 parameters; NeuroPixel has 29,856 parameters. See the unchanged [recipe](09_experiment_recipe.json) and [Stage-A report](09_preflight_results.md).

Consequently, comparing the studies' headline accuracies would conflate tasks, scale, data and optimization. Our small neural reference is not a representative test of every Transformer. A negative fixed-budget result cannot establish that the architecture family is incapable of binding; a positive result cannot by itself establish priority over this literature. Input counterfactuals in item 9 test behavior. They do not replace interventions on the trained internal computation, which would require their own specified experiment.

## Search boundary

Two additional broad searches, restricted to a 730-day freshness window, used the terms `"neural cellular automata" "compositional" reasoning` and `"neural cellular automata" "variable binding"`. A targeted PMLR search verified the proceedings entry. Irrelevant forums and secondary descriptions were not used as evidence. These searches add a directly pertinent primary source; absence of another retrieved NCA result is not proof of novelty or an exhaustive priority search.
