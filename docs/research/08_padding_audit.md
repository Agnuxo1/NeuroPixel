# Item 8 — Effective PAD dictionary correction

**Source inspected before correction:** `ef25ed6497cd3554d8eb78a0ef278f7b589e1e1f`. The original `neuropixel/model.py` is preserved byte-for-byte in [padding_model_before.py](../../results/research/08_validation/padding_model_before.py), SHA-256 `564045fff799185e15b921873225d344014a4d4bcae6df6deacaf42f6e4bd701` (6,610 bytes), with [a provenance receipt](../../results/research/08_validation/padding_source_before.json).

**Validation status at authoring:** source inspection and Python AST parsing completed. Local Torch is unavailable. The deterministic reproductions and regression tests below are authored for the declared CPU validation job and have **not been executed locally**. No training study, task sampling, checkpoint collection, inference benchmark or energy measurement was performed. This note must be read together with the eventual cloud execution receipt; authored assertions are not passing test results.

## Defect and the intended boundary

The original class declares `nn.Embedding(..., padding_idx=0)`, but its `dictionary()` reads `self.embed.weight` directly. `forward()` then calls `F.embedding(canvas, self.dictionary())` without a functional padding argument. The embedding module's special lookup route is not invoked. The resulting PAD identity enters every local update through reinjection, even though initial seeding is multiplied by `canvas != 0`. `lens_logits()` also multiplies by the complete dictionary transpose, exposing its PAD row to the lens objective. The answer head masks the PAD answer logit, but that does not block the input/reinjection route or the separate lens route.

Consequently, an initially zero PAD parameter can receive a gradient and become nonzero after an optimizer step. An explicitly grounded RGB row 0 or a nonzero row from an older checkpoint also passes directly through the original dictionary. Merely checking the initial seed or the newly initialized table does not exercise these paths. In particular, the historical `test_grounded_dictionary_fixed_colors` retains a dictionary tensor made before its backward call and does not perform an optimizer step; it cannot establish an invariant over updates.

The corrected boundary is the **effective dictionary**, not the serialized parameter storage:

| Property | Corrected behavior |
|---|---|
| Effective PAD identity | All `c_id` channels of `dictionary()[0]` are exactly zero. The constant row is applied after grounded RGB composition. |
| PAD gradient through the dictionary | Zero for `embed.weight[0]`; grounded row-0 values also do not influence the effective output. |
| Other dictionary rows | The existing grounded first-three-channel mapping and learned remaining channels are unchanged. Their local derivatives are unchanged by this projection. |
| Old stored PAD weights | Loading remains strict and schema-compatible. Nonzero row 0 is projected out without silently rewriting checkpoint parameters. |
| Resumed optimizer state | Old momentum or weight decay can still move the raw stored PAD parameter. It cannot reintroduce an effective PAD identity through `dictionary()`. |
| Consumers | `forward`, initial seeding, reinjection, tied answer readout, `lens_logits`, and `phase3.np_stream` use the shared effective dictionary. No separate lookup-only repair is required. |

The implementation assembles the existing table and returns `cat([zeros_like(table[:1]), table[1:]], dim=0)`. A constant row and a slice make the boundary independent of the stored PAD value; there is no in-place mutation, optimizer hook, new parameter, buffer, constructor option or state-dict key. Fixing only `F.embedding(..., padding_idx=0)` would leave direct tied readout and grounded/nonzero PAD output untouched, so it would not cover the whole contract.

## Deterministic reproductions and regression inventory

The owned [test file](../../tests/test_padding_invariant.py) loads the preserved original module only after verifying its SHA-256. Its fixture uses four tokens, four identity channels, two state channels, two hidden units, a `2×2` grid, two updates and firing rate 1. All parameters are set explicitly; construction runs inside an isolated CPU RNG context. There is no sampled task or accuracy result.

The fixture sets the first update channel to `ReLU(1 + identity[0])`, disables dependence on state/neighbors, and reads that channel back through a dictionary whose non-PAD first coordinates are `[1, -1, 0.5]`. The output cell is empty. Before the first step, its state coordinate becomes 2 and answer logits for the three non-PAD tokens are `[2, -2, 1]`. With target token 2, answer CE has a positive derivative with respect to the input PAD coordinate through reinjection. A separate fixed all-ones state supplies the lens fixture: its PAD logit participates in CE normalization for a non-PAD target, also yielding a nonzero PAD-row derivative. Each legacy route makes exactly one SGD step and records the initial row, gradient and changed row. These are analytical expectations until the fixture runs.

| Test | Failure or compatibility property exercised |
|---|---|
| `test_preserved_source_reproduces_both_legacy_gradient_paths` | Positively demonstrates nonzero PAD gradients and post-SGD effective PAD in the hash-pinned old implementation, separately for answer reinjection and lens readout. |
| `test_actual_answer_and_lens_optimizer_steps_preserve_effective_pad` | Runs those real objective/backward/SGD paths on the correction; requires zero PAD gradient/effective row while non-PAD rows still learn. |
| `test_checkpoint_nonzero_pad_is_projected_without_rewriting_stored_weights` | Strictly loads an old-style state dictionary with deliberately nonzero PAD; compares state, logits, lens, frames and activity against a zero-PAD copy and checks raw weights remain unchanged. |
| `test_resumed_adamw_momentum_cannot_reintroduce_effective_pad` | Transfers a legacy one-step AdamW state and checkpoint into the correction; even continued raw PAD movement cannot defeat the effective boundary. |
| `test_grounded_pad_is_zero_and_other_values_and_gradients_are_unchanged` | Gives row 0 a nonzero grounded RGB value; checks its suppression and the exact non-PAD grounded/learned gradient paths. |
| `test_nonpad_function_and_gradients_match_legacy_when_pad_has_no_objective` | On an entirely occupied canvas with answer CE only, checks exact legacy/core outputs and parameter gradients in a case where PAD has no active role. |
| `test_forward_lens_and_stream_use_the_same_zero_pad_boundary` | Inspects actual update inputs, compares one-frame stream and forward outputs, and checks an empty subsequent stream frame plus the PAD lens logit. |
| `test_zero_identity_and_initial_seed_do_not_imply_zero_recurrent_activity` | Uses explicit recurrent biases to produce activity on an initially empty grid despite zero PAD identity and initial seed. |
| `test_camera_observation_is_not_discarded_as_padding` | Confirms a camera-marked cell with token ID 0 remains observed and receives its RGB-derived seed; unobserved cells remain initially zero. |
| `test_research_nca_explicit_legacy_policy_survives_core_correction` | With nonzero PAD weights, checks `ResearchNCA(freeze_pad=False)` against the old source and its existing opt-in `freeze_pad=True` against the corrected core. |

Declared cloud entry points, to run under the reviewed one-thread CPU/RAM policy rather than locally here:

```bash
python tests/test_padding_invariant.py --legacy-demonstration results/research/08_validation/padding_legacy_demonstration.json
python -m pytest -q tests/test_padding_invariant.py
```

The demonstration refuses to overwrite its JSON. Preserve any failed run or assertion and its environment; do not describe a fixture as having passed merely because it was authored or parsed. The core test suite's existing initial-zero and grounding tests remain useful but do not replace these optimizer and loaded-weight cases. No official PyTorch documentation page was retrieved for this note; the future pinned Torch runtime execution supplies direct evidence of the hypothesized gradient behavior.

## Compatibility and interpretation

This correction is a **new source version with changed inference and training semantics**. A legacy checkpoint whose effective PAD row was nonzero can produce different recurrence, outputs or lens probabilities under the corrected core. Even from an initially zero row, the revised training trajectory can differ because PAD no longer learns through the dictionary. Although the mapping for non-PAD rows is unchanged, downstream losses and therefore their training gradients can change when legacy PAD previously contributed. Do not relabel old results as corrected-model results or claim general numerical equivalence.

`neuropixel/research/models.py` is deliberately unchanged. `ResearchNCA` owns its dictionary, forward and lens methods, and its historical default `freeze_pad=False` preserves the audited legacy policy. Its existing `freeze_pad=True` provides a separate explicit projection. The item-6 tests named “legacy factory” call the research factory, not the corrected core, so that historical reference remains appropriate. Frozen historical controller source hashes should continue to reject the modified core when old exact-source reproduction is requested; do not update their anchors merely to make the new source appear historical. Use the historical commit/snapshot and matching environment for historical reproduction.

Zero PAD identity means neither a zero state forever nor zero work. Initial non-camera seeding is zero on empty cells because of the presence mask, but biases and propagation from occupied neighbors can make those cells active later. Camera cells with ID 0 are explicitly present visual observations. The implementation still runs dense convolutions across the grid, and its mean update magnitude is not an energy or hardware sparsity measurement. This repair establishes a token-identity/gradient boundary only; it does not introduce a new dynamics programme or validate a performance claim.
