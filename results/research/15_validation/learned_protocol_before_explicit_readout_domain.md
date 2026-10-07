# Item 15B — retrospective learned-state finite dynamics

## Question and status

How do the states and readouts of the five retained item9 NeuroPixel checkpoints behave under longer recurrence and one precisely defined state perturbation? This is a descriptive diagnostic on exposed development inputs. It does not retrain models, reopen H1, estimate generalization accuracy, establish a cause of weak task acquisition, or certify all possible inputs and times.

The constructed census in item15A is complete: scientific source ff66dc4bacd19219bf4f754e40944396e7ce8c5b and final archive ff34cc4f44c535e5f4d00276e1e3be194ec70f21. Its report is15_census_results.md. Item15 remains the only open item. This distinct phase is frozen before loading the retained binary checkpoints or scoring their extended dynamics. Four known queries are fixed by index; no query or checkpoint may be selected or discarded after seeing trajectories.

## Original evidence and input recovery

Original scientific source83fe135e301f76bc0c74e30c66bb18e067ca5959; final original archive b64d0e2ba222df76caf72bc9870c7602873e7236, rooted at results/research/09_cloud_runs/37593731891-1-study. The original final manifest has SHA-25649cda016746d6f020e22da2e10e09f8706b1f17f15491baf438d6952484f78e6.

The plan binds22 files totaling1,777,007bytes: that manifest, one validation NPZ, and a checkpoint, validation-prediction NPZ, configuration JSON and run JSON for each seed40–44. All bytes and SHA-256 hashes are fixed in the input map. Recovery copies only regular100644 Git blobs from the original commit into a new owned input directory; it checks object size, copied-file size, SHA and original-manifest membership before any numerical deserialization. It makes no reference changes. The original results history is already available to the worker's archive machinery.

Validation data contains1,024 rows. The source orders base-condition rows by bag, event and role. Fixed indices0,1,2,3 are therefore the four roles of event0 of the first scene; stored metadata must confirm that before continuation. They are four correlated queries of one scene, not four independent content groups. Validation and its prior predictions have already been exposed. No new test set is created or consulted.

Each native model has vocabulary37, output(9,7), identity dimension16, state channels48, hidden width128, grid10x8, defaultT16 and29,856 parameters. The model source is exactly the same blob b66e2f575c6ae937d8cad807b4dd0e48531850f7 as in item9. Load the original checkpoint with torch.load(weights_only=True,map_location='cpu'), compare its configuration with the archived configuration and run metadata, and load the state dictionary strictly. Grounding was absent; nonpersistent g_rgb and g_mask buffers must reconstruct as zeros and be recorded.

## Compatibility gate before all continuations

This worker uses Python3.12.14, Torch2.6.0+cpu and one numerical thread. The original run used Python3.12.8 and two numerical threads. Report this runtime change explicitly. A compatibility check can establish output agreement within the declared tolerance; it cannot establish identity of the historical hidden state because no corresponding original hidden-state array was archived.

For every checkpoint, run the first64 validation examples in one T16 evaluation batch, matching the original evaluation batch size. Retain all64 new and reference logits, predictions, metadata and the four selected new final states. Every logit must be finite and satisfy abs(new−reference)<=1e-4+1e-5*abs(reference). All64 predictions must agree exactly. Preserve maximum absolute/scaled discrepancies and every changed prediction, including discrepancies inside the numerical logit tolerance.

Run and save all five compatibility checks before any extended trajectory. Continuation is admitted only if all five pass. If any comparison fails, preserve all five gates, set compatibility_gate_passed=false and extended_panel_executed=false, and finish the measured gating outcome without selecting a compatible subset. Such a recorded negative gate may be successfully audited; it is not a passing compatibility claim. Unexpected source, deserialization, schema, resource or execution failures remain operational failures. No tolerance, seed or query substitution is allowed after outcomes.

## Initial intervention and held-input recurrence

Let s16 be the selected final float32 state from the same64-example gate forward. Define a fixed direction at every channel and location, d[c,r,k]=(-1)^(c+r+k), with zero-based indices. This direction uses no labels, randomness, fitted parameters or outcome selection.

Compute nominal amplitude a=1e-4*max(1,RMS(s16)), using float64 arithmetic. The base branch begins at s16. The perturbed branch begins at float32(float64(s16)+a*d). Preserve the nominal amplitude, both represented initial states, actual float64 difference delta_real=double(perturbed)−double(base), and its positive finite RMS. Normalize later paired distance by the RMS of this represented difference. Report the perturbation's actual magnitude; quantization is not silently replaced by the nominal value.

There are five trained initializations, four fixed queries per initialization and two branches per query:40 trajectories forming20 pairs. Each branch is run individually with the same fixed future canvas, parameters, dictionary and eval mode; stochastic training firing is inactive. A native post-update1 hook installs the saved initial state, after which up to256 ordinary native updates are retained. Diagnostic time tau0 is the installed T16 state; tau256 corresponds to effective original depth272. The discarded initialization update incurred by each native hook call is an implementation cost, not another retained state transition.

No input removal, reseeding contribution after the intervention, gradient update, noise resampling or parameter projection is part of the continuation. The perturbation is one structured direction. Its pair gain is not an operator norm, a global Lipschitz bound, a Lyapunov exponent or a worst-case sensitivity.

## Records, guards and interpretation

Retain full float32 states for every observed endpoint, plus observation and finiteness masks. Explicit NaN padding denotes unobserved endpoints and must never be treated as measurements or passed numerical states. Keep all observations preceding a stop and the first state that trips a state/finiteness guard. There is at most40x257=10,280 retained endpoint slots; the count actually observed may be smaller.

Stop an individual trajectory at its first nonfinite state or maximum absolute state above1e12. Report the first stopping time and reason separately for each branch. A state threshold is an operational boundary, not a definition of mathematical instability or a proof of asymptotic divergence. A paired gain is defined only on the shared observed finite horizon; later missing observations are censored, not evidence of stability or zero gain. CPU deadline and RAM violations remain operational failures with partial evidence.

For every observed finite endpoint retain logits, token prediction, float64 NLL for the archived label, state L2 norm, RMS, maximum absolute component and adjacent-step L2 increments (RMS can be derived by dividing by the square root of 3,840 components). Time0 has no observed predecessor in this aligned continuation. Record paired RMS distance and gain using the actual initial perturbation. Summaries cover maxima, first peak times, final available values, full-horizon availability and guard status. These measures describe known-input readouts, not a new accuracy benchmark. No confidence interval is computed by treating four related queries as independent samples.

Preserve all named parameters and buffers before and after inference and the RNG state after checkpoint loading. Full state slots alone occupy at most31,580,160bytes per checkpoint and157,900,800bytes across five, before compression, masks and other arrays. Those are planned array sizes, not measured compressed file sizes. Process one model at a time and write separate seed artifacts. Do not fill failed or unexecuted measurements with successful defaults.

## Independent checks

A fresh checkpoint-binding process loads the original state dictionaries with weights_only=True and independently compares configurations and exact parameter arrays with the native saved gate and continuation arrays. It performs no model inference. It verifies the restored nonpersistent zero buffers and hashes every native JSON/NPZ it consumes. This supplies a separate check that the numerical traces are bound to the specified learned weights.

A separately authored NumPy process reads the saved arrays and receipts without Torch or native-producer imports. It recomputes the compatibility decision, initial perturbation, observation masks, guard times, metrics, NLL, pair gains, summary counts and unchanged-array checks. It verifies the checkpoint-binding receipt and its original-input and native-file hash maps. It does not itself deserialize a Torch checkpoint or rerun the complete recurrent dynamics.

Float64 scalar metrics use atol1e-9 plus rtol1e-12. Reconstructing the float32 readout with NumPy float64 uses a prospectively fixed allowance based on128*eps_float32 times the absolute-product scale of the two linear readout stages, including bias and propagated first-stage rounding, plus1e-5. PAD logit remains exactly−10000. This scale accounts for cancellation and finite float32 products; the code fixes the precise formula before execution. It is not a theorem certifying all forward dynamics. Source and input hashes, raw arrays and actual rounding differences are retained.

The existing post-update/snapshot test is the unchanged worker contract gate. It does not count as another learned result. The root review checks the saved summaries and archive provenance; all these checks are internal to this work, not an independent external laboratory replication.

## Finite resources and completion

One standard public CPU job: numerical1, interop1, Git1, aggregate active CPU ceiling4 and available RAM floor8GiB. Worker1020s includes180s reserved for final archive. The contract gate allows120s; the learned stage600s contains recovery120s, native240s, checkpoint binding60s and NumPy audit120s. The outer deadline remains binding even if individual component maxima cannot all be consumed. Workflow20min with finite bootstrap deadline1170s; resource supervision every1s, heartbeat60s and partial archive cadence300s during long stages.

Retain the source ZIP, frozen plan, recovered inputs, compatibility gates, conditional trajectories, binding/audit receipts, test events, logs, resource observations, failures and final manifest. No paid compute, GPU, main merge, external contact, nomination or access/billing change is requested. Verify completed provider state and archive before release.

A completed compatible panel can support only finite observations for these five checkpoints, this scene and this perturbation. A failed compatibility gate leaves the extended panel unexecuted. Neither outcome establishes useful stable memory, generalization, universal convergence or a Nobel-level discovery. Item15 closes only after all feasible evidence and limitations are reported; items16–30 remain unopened until then.
