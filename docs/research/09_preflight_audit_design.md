# Item 9: offline preflight audit design

This auditor is an editorial and verification artifact written after the preflight source was frozen. It does not change the experiment recipe, training, candidate population, selection rule or performance population. It reads the completed Stage-A archive, never executes a model, imports Torch, replays a random generator, opens a final-performance population or writes into the raw archive.

The source identity is `e63764ccb924ab23d65c2deb94b477387d102c82`, tree `2c4de6ea03dbc916ab2d252dd42fc96c76178588`. The preflight plan SHA-256 is `7802ee68ae8ce19bf17e6135815e0c796216696e05c8d322347a314a2c88df04`; the recipe SHA-256 is `7450b1ede40542c5ff940842e42eb2725ac4420946baad6bac920e779a214b97`. These are explicit constants in the new helper, not identities inferred from the results being checked.

## Invocation and failure preservation

```sh
python scripts/research_item9_preflight_audit.py \
  --input /absolute/path/to/completed-preflight-raw \
  --source-root /absolute/path/to/frozen-source \
  --output /absolute/path/outside-raw/preflight_audit_01.json
```

The output must not exist and must be outside the input directory. Its exclusive JSON receipt records the auditor's SHA-256, all opened artifact identities, local runtime, check count and first exception if any. An unsuccessful audit exits nonzero and retains the failure receipt. Correcting an auditor assumption requires a new receipt and identification of the correction; the original receipt is never overwritten. The first actual execution has not occurred at the time this design is written. Only syntax compilation is planned before that first execution; no fabricated result fixture is used to present a successful audit.

All numerical thread environment variables are set to one before importing NumPy. The auditor admits only when Linux `MemAvailable` is at least 8 GiB, checks it again between populations and runs, and uses no model execution or stochastic computation. Its working arrays are the saved development datasets and predictions.

## Checks before reading numerical outcomes

The archive must be marked final and completed, contain exactly its manifest inventory, and match every declared byte count and SHA-256. A read-only Git inventory binds the source ZIP's complete regular-file inventory to the specified commit and tree. Each ZIP member is compared both with its Git blob identity and the frozen checkout bytes. The ZIP commit comment, copied plan and recipe, all 49 bound source hashes, and source-before/source-after receipts must agree. No final access, final inventory, final dataset or final prediction artifact may exist.

The two declared stages must be complete with return code zero, pinned runtime versions, CPU Torch, two intra-operation threads and one inter-operation thread. The helper checks every saved test boundary, the 34 collected unique test IDs against the frozen test AST, and the 34 corresponding successful JUnit parent elements without errors, failures or skips. Both actual-model integration methods must be among those IDs. JUnit suite counters are retained separately: nested unittest subtest reports may increase a suite aggregate without creating additional unique parent testcase elements. The auditor does not reinterpret that aggregate as a count of independently collected tests or reconstruct an unsaved per-subtest event stream.

The resource-monitor samples are recounted, require the 8 GiB floor, and retain their sampling limitations. Freeze, source checks, stages and final archive timestamps must be ordered. Stage and worker timers are compared with the frozen budgets. These are checks of saved observations and supervision receipts, not continuous memory, process or energy measurements.

## Development population and input truth

The auditor implements its own visible-canvas decoder. It does not import the generator or consult scenario metadata to obtain the gold answer. It checks the 10 by 8 grammar: eight nonoverlapping adjacent event-role-filler triples with all event-role keys, category-correct distinct fillers, and the explicit event/role query at row 9, columns 5 and 6. The output cell is PAD. Gold is recovered from the visible queried key.

Separately, saved scenario metadata is validated and rendered directly to verify correspondence with the saved canvas. Canonical semantic bag hashes, scenario, pair and record IDs, array types, query order, group indices, content digests and support counts are reconstructed. The training population has 2,048 groups; validation has 128; the probe and memorization groups are the first 64 and four training groups, with their own recorded layouts. The base-only panels therefore contain 1,024, 512 and 32 query rows. Validation groups must be disjoint from training, use their declared hash bucket and exclude every group in the frozen fixture-exposure ledger.

The exposure check compares the saved ledger with its frozen source artifact and recounts its declared candidate and constructed unions, including the manual fixture group. It does not replay Python's RNG or establish that no other unrecorded human or historical exposure occurred. The training-only majority is recounted from training facts. Exact, role-only, event-category and bag-category control expected probabilities are checked against their analytical values for every saved validation query; the majority control is checked against its recounted training rule.

## Six runs, saved predictions and selection

Exactly two memorization runs and four pilot runs must be present, indexed, successful and retained. Their IDs, family, initialization, learning rate, model configuration, parameter count, optimizer recipe, scheduled budget and source/data context are compared with the frozen recipe. All artifacts are hashed. Training logs must contain the complete finite sequential update inventory, valid training-group indices, finite gradients and monotonically increasing elapsed times. The aggregate input hash is independently reconstructed from the logged batch digests. The four pilots must have identical logged input-digest/group-index sequences. Memorization batches must match the full saved 32-example panel at every update.

This proves agreement of the recorded training streams, not a reconstruction of every sampled canvas, initialization, firing mask or gradient. Checkpoints are inspected as ZIP serialization containers and hashed without unpickling or loading tensors. Tensor values and embedded configurations are therefore not independently deserialized by this helper. The cloud model-integration test's checkpoint roundtrip is a separate executed fixture, not a reload of every pilot checkpoint.

Saved prediction logits must be finite float32 values, aligned exactly with the corresponding dataset targets, roles, groups and record IDs. The auditor reconstructs argmax predictions and computes stable binary64 log-sum-exp negative log likelihood. Numerical comparisons use absolute tolerance `1e-10` and relative tolerance `1e-12`; categorical identities and counts are exact. It recounts global accuracy, per-role accuracy, macro accuracy, agent/patient binding, cross entropy and all-eight-query accuracy, and compares every endpoint metric with its run descriptor. No uncertainty interval is attached to the eight dependent queries within a group.

Memorization checks must appear at the prescribed 128-update intervals and satisfy the frozen consecutive-check rule or reach the maximum budget. Intermediate checks are verified from their saved scalar metrics; only final saved predictions permit an independent numerical recount. Their flags remain separate from pilot selection. The four pilot candidates are ranked within family by descending validation binding, then ascending validation cross entropy, then ascending learning rate. The helper verifies the saved winner, complete population, data identities, full six-run index hash, and chronology before the selection receipt. Training probes remain diagnostics.

## Stage-A to Stage-B identity and cost scope

The report preserves the exact recipe, selection and six-run-index hashes and development identities required by the separate two-stage authenticator. Before the Stage-B plan freezes, root must bind the downloaded raw Git archive identity and copy the complete authenticated selection and preflight-run objects into the prospective handoff receipt. This helper does not replace that receipt, grant final access or select any new hyperparameter.

Timing estimates use only elapsed times already recorded in the successful pilot logs. They report the mean and median per-update differences after the first update, the measured post-training remainder, and simple projections to five 4,096-update runs per family. Those selected-pilot projections describe their stated reference; they are not a worst-case admission bound. A separate conservative operational estimate may use the slowest observed pilot and additional safety factors. Neither estimate is a measurement of Stage B. Final evaluation's 122,880 prediction rows, tests, source/data setup, active archival and changes in hardware or load need separate allowance. No budget or scientific setting is altered from these descriptive computations.

This is an independent implementation within the same assistant team, with shared artifacts and local custody. It is not external replication, a blinded analysis or an independently controlled archive.
