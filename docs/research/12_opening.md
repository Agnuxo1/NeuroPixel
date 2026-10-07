# Item 12 — Continual learning: opening record

Opened 2026-10-07T13:33:44+00:00, only after item11 closure commit 3ad305390bf9f9d07d18120f5d641ceb38edd28e.

## Scientific questions

1. Does an old task retain competent behavior after subsequent learning, and what was its baseline competence?
2. Which object is preserved: one shared model, independent expert weights, state, dictionary or router?
3. Does end-to-end retrieval preserve the old capability when task identity is not supplied?
4. How do task order, representation overlap, optimizer reuse, replay, cumulative storage and compute affect the interpretation?
5. Which existing results can be independently audited, and which additional bounded simulations can resolve a concrete question without replacing an unsuccessful earlier protocol?

## Method

Read current implementation and historical/earlier audited results first; consult primary continual-learning work for metric and scenario definitions. Preserve exact prior findings, including low baseline competence. Define any new numerical hypothesis, controls, seeds, metrics, admission criteria and finite budget before execution. Source/API invariants may be tested with labeled fixtures; an invariant of frozen parameters alone is not end-to-end continual-learning success. A retention assay requires adequate initial acquisition. No later-item work belongs to this opening.

## Boundaries and resources

The local executor remains unavailable. No numerical job is admitted at this record. Source/analysis reservation90 minutes; aggregate active CPU<=4 and available RAM>=8GiB for any later numerical admission, with thread limits, supervision and an archival reserve specified prospectively. Only previously authorized standard public CPU execution may be used. No GPU, paid compute, main merge, external communications, billing or access changes.
