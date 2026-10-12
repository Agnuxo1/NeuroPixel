"""A18.2.H: semantic review of the historical claim candidates (studies 04, 05, 06 A06.1,
continuation reconciliation and the task-4 audit).

Every frozen candidate of batch A18.2.H receives exactly one classification. Claims are written
in English, bound to the protocol and receipts that exist in this repository, and every quoted
number is checked against those receipts. Missing raw evidence is recorded, never replaced by a
summary. No model is loaded, trained or evaluated; saved predictions are only recounted.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/research/COMPLETION20261012"
PART = ROOT / "docs/research/claims_ledger_parts/A18.2.H.json"
AUTHORS = ["Francisco Angulo de Lafuente"]
R04 = "docs/research/04_results.md"
R05 = "docs/research/05_results.md"
R06 = "docs/research/06_A06_1_results.md"
RC = "docs/research/CONTINUATIONS_RECONCILED_20261008.md"
RT = "docs/research/TASK4_historical_capabilities_audit_20261009.md"
A06 = "results/research/06_A06_1_recovery/"
P04 = "results/research/04_experiment/pilot/"
T4 = "results/research/TASK4_review/"
SUMMARY05 = "results/research/05_experiment/item5_recovery_receipt_20261007.json"

# ---------------------------------------------------------------- claims
COMMON_SYNTH = "Synthetic RoleTask generator only; no independently designed or external task"
NO_EXT = "No independent laboratory replication and no externally dated public preregistration"
MISSING05 = ("Raw evidence unavailable: the 28-run CPU archive (215 files, executor commit 0a56621), "
             "05_analysis.json (sha256 11fd91ee...) and 05_independent_check.json (sha256 adef6b11...) "
             "are absent from this repository, local disks and remote branches")
RECOVERY05 = ("docs/research/05_results.md is itself a 2026-10-07 recovery copy whose byte identity with the "
              "executor report could not be verified")

def c(cid, statement, population, status, level, limits, protocol=None, receipts=(), checks=(),
      stop_reason=None, context=(), study=""):
    return dict(id=cid, study=study, statement=statement, population=population, status=status,
                declared_level=level, limits=list(limits), protocol=protocol, receipt_paths=list(receipts),
                value_checks=list(checks), stop_reason=stop_reason, context_paths=list(context))

POP04 = ("Item-4 matched pilot on the synthetic RoleTask (8x8 grid, 35 tokens, split 0, initialization 7); "
         "NeuroPixel, adapted NCA, spatial ConvGRU and relative-position Transformer at 27,835-30,384 parameters; "
         "2,048 validation and 4,096 final examples per selected checkpoint")
POP05 = ("Item-5 replication on the synthetic RoleTask (split 0, initializations 10-14; exploratory splits 0, 101, 202 "
         "at initialization 10); four fixed recipes from the item-4 pilot, 1,024 updates, answer-only cross-entropy, CPU")
POP06 = ("Exact frozen local A06.1 ablation recipe (commit a009f2a9) on the synthetic RoleTask; two initializations "
         "(seeds 20/21), one split; NeuroPixel variants only")
POPT4 = "Saved evidence of the historical cloud items 7-15 recovered from fifteen pinned commits"
EP04 = "docs/research/04_execution_plan.json"
EP05 = "docs/research/05_execution_plan.json"
EP06 = "docs/research/06_execution_plan.json"
L04 = [COMMON_SYNTH, NO_EXT,
       "Descriptive pilot with one initialization; the H1 decision was reserved for item 5",
       "Limited optimization budget: 1,024 updates of 64 examples and two learning rates per family",
       "Similar parameter counts and equal sample exposure do not equalize capacity, receptive field, FLOPs or optimization difficulty",
       "The independent verifier output results/research/04_verification.json cited by the report is not available; "
       "the transferred artifacts were re-verified here only by their transfer-manifest hashes"]
L06 = [COMMON_SYNTH, NO_EXT,
       "Replication unit is the initialization: n=2, one split; paired t95 intervals have df=1",
       "Exploratory: test and scientific context had prior exposure; no confirmatory family was preregistered",
       "Equal updates, not equal FLOPs or total cost; low competence of every variant (all 26 probes budget_limited)"]

CLAIMS = [
    c("H04-01", "In the item-4 matched pilot (protocol NP-SCI-20261006-v1, initialization 7, two learning rates per "
      "family, 1,024 updates of 64 examples, answer-only cross-entropy), no NeuroPixel binding advantage was observed: "
      "the validation-selected NeuroPixel configuration reached 9.326% final binding versus 48.389% for the selected "
      "relative-position Transformer (descriptive difference -39.0625 percentage points), and none of the eight "
      "configurations reached the predeclared 95% training-probe binding threshold.",
      POP04, "experiment_complete", 2, L04, EP04,
      [P04 + "pilot_summary.json", P04 + "selection.json", "results/research/COMPLETION20261012/item4_transfer_verification.json"],
      [("9.326", P04 + "pilot_summary.json"), ("48.389", P04 + "pilot_summary.json"), ("11.914", P04 + "pilot_summary.json"),
       ("13.574", P04 + "pilot_summary.json"), ("18.872", P04 + "pilot_summary.json"), ("74.194", P04 + "pilot_summary.json"),
       ("29824", P04 + "pilot_summary.json"), ("30384", P04 + "pilot_summary.json"), ("27835", P04 + "pilot_summary.json"),
       ("29547", P04 + "pilot_summary.json"), ("9.277", P04 + "selection.json"), ("47.949", P04 + "selection.json"),
       ("45.312", "results/research/04_experiment/item4_evidence.zip"), ("12.305", P04 + "selection.json"), ("2.444692", "results/research/04_experiment/item4_evidence.zip"),
       ("0.433867", P04 + "selection.json"), ("173.462", P04 + "pilot_summary.json")], study="04"),
    c("H04-02", "In the item-4 pilot, the selected relative Transformer answered every ACTION and PLACE final query "
      "correctly (100%) but reached only 48.389% mean nominal-role (AGENT/PATIENT) accuracy; its conditional "
      "example-bootstrap binding interval, 46.240-50.586%, includes the 50% category-only expectation.",
      POP04, "experiment_complete", 2,
      L04 + ["The bootstrap interval is conditional on one checkpoint and its examples; it is not an interval over training runs"],
      EP04, [P04 + "pilot_summary.json"],
      [("53.906", P04 + "pilot_summary.json"), ("42.871", P04 + "pilot_summary.json"), ("100.000", P04 + "pilot_summary.json"),
       ("46.240", P04 + "pilot_summary.json"), ("50.586", P04 + "pilot_summary.json"), ("30.859", P04 + "pilot_summary.json"),
       ("25.977", P04 + "pilot_summary.json"), ("8.789", P04 + "pilot_summary.json"), ("9.863", P04 + "pilot_summary.json")],
      study="04"),
    c("H04-03", "In a post hoc inspection of saved item-4 validation predictions, the selected relative Transformer chose a "
      "visible filler of the queried category in all 2,048 validation examples while answering 491 of 1,024 nominal "
      "(AGENT/PATIENT) queries correctly.",
      POP04, "experiment_complete", 2,
      L04 + ["Post hoc descriptive diagnostic, not a frozen endpoint; it does not establish the internal computation or positional invariance",
             "Recounted here from the retained validation predictions and dataset; the report's separate diagnostic file is not in the repository"],
      EP04, ["results/research/04_experiment/item4_evidence.zip", "results/research/COMPLETION20261012/historical_value_checks_H.json"],
      [("recount:item4_visible_category", "2048"), ("recount:item4_nominal_correct", "491")], study="04"),
    c("H04-04", "Item-4 execution provenance: training source 5ab415ec with protocol NP-SCI-20261006-v1 unchanged; the "
      "selection was frozen at 2026-10-06T22:28:58.539148Z before final data access at 22:28:58.653985Z (final dataset "
      "content sha256 23bca94c...); total recorded training time over the eight trials was 173.462 s, excluding setup, "
      "evaluation, queue waiting and tests; all 60 transferred files match their byte hashes.",
      POP04, "engineering_complete", 1,
      [NO_EXT, "Hash and timestamp agreement establishes internal artifact consistency only",
       "Training time is not an end-to-end cost; inference latency and physical energy were not measured",
       "The 88-test compatibility count, resource minima and the independent verifier's 20 fixture tests are reported by the "
       "report and worker records but were not re-executed here"],
      None, [P04 + "pilot_summary.json", "results/research/04_experiment/item4_evidence.zip",
             "results/research/COMPLETION20261012/item4_transfer_verification.json"],
      [("173.462", P04 + "pilot_summary.json"), ("text:2026-10-06T22:28:58.539148", P04 + "pilot_summary.json"),
       ("text:22:28:58.653985", P04 + "pilot_summary.json"),
       ("text:23bca94c224b195da94ab1794801fdae131007a77bd594c2167a085227e7dba2", P04 + "pilot_summary.json"),
       ("text:5ab415ec825e931b69fc22571c47cdb74a4fc592", P04 + "pilot_summary.json"),
       ("60", "results/research/COMPLETION20261012/item4_transfer_verification.json")], study="04"),
    c("H05-01", "Under the frozen protocol NP-SCI-20261006-v1 (RoleTask, 1,024 updates, answer-only cross-entropy, fixed "
      "item-4 recipes, CPU recovery), H1 - that NeuroPixel improves nominal role binding over the fixed relative "
      "Transformer - was not supported: across five primary initialization pairs, mean binding was 12.393% for "
      "NeuroPixel and 49.395% for the reference, paired difference -37.002 percentage points, 95% t interval "
      "-42.030 to -31.974 pp; NeuroPixel was below the reference in all five pairs.",
      POP05, "blocked_external", 0,
      [MISSING05, RECOVERY05, COMMON_SYNTH, NO_EXT,
       "Does not prove equivalence, universal inferiority or inability to learn; optimization adequacy unresolved",
       "CPU recovery of a GPU-selected recipe; device invariance not established"],
      EP05, [], [("12.393", SUMMARY05), ("49.395", SUMMARY05), ("-37.002", SUMMARY05), ("-42.030", SUMMARY05),
                 ("-31.974", SUMMARY05)], context=[SUMMARY05], study="05"),
    c("H05-02", "In the item-5 exploratory split panel (splits 0, 101, 202 at initialization 10), the paired "
      "NeuroPixel-minus-Transformer binding difference was -34.375 pp (SD 0.517 pp; descriptive 95% t interval "
      "-35.659 to -33.091 pp, df=2), reported separately from H1.",
      POP05, "blocked_external", 0, [MISSING05, RECOVERY05, COMMON_SYNTH, NO_EXT,
       "Three conditional observations sharing the split-0 checkpoint; not pooled with the initialization panel"],
      EP05, [], [], context=[SUMMARY05], study="05"),
    c("H05-03", "In item 5, NeuroPixel showed no stable advantage over the adapted NCA or the ConvGRU: all four descriptive "
      "t intervals of those contrasts cross zero (for example NP minus ConvGRU +2.676 pp, interval -2.354 to 7.705 pp, "
      "five initializations), while each of the seven per-checkpoint comparisons against the fixed Transformer has a "
      "negative conditional bootstrap interval.",
      POP05, "blocked_external", 0, [MISSING05, RECOVERY05, COMMON_SYNTH, NO_EXT,
       "No multiplicity adjustment; conditional bootstrap intervals do not estimate training-run variance"],
      EP05, [], [], context=[SUMMARY05], study="05"),
    c("H05-04", "All 28 item-5 training probes fell below the predeclared 95% training-binding threshold "
      "(budget-limited flag); NeuroPixel probe binding ranged 10.547-18.750% and the relative Transformer 47.656-58.984%.",
      POP05, "blocked_external", 0, [MISSING05, RECOVERY05,
       "The flag is an operational optimization diagnostic; it does not identify the cause or show that more updates would solve the task"],
      EP05, [], [("text:\"all_28_training_probes_below_0_95\": true", SUMMARY05)], context=[SUMMARY05], study="05"),
    c("H05-05", "Retrospective reassessment of six retained historical scaling checkpoints: each reconstructed clean global "
      "score matched its historical value at four decimals; historical NeuroPixel reached 96.131% mean binding versus "
      "7.320% for the original TinyTransformer (three coupled seeds), a result excluded from H1.",
      "Six retained historical checkpoints (NeuroPixel 29,824 and TinyTransformer 44,483 parameters, coupled seeds 0-2), "
      "2,000 regenerated RoleTask examples per split",
      "blocked_external", 0,
      ["Raw evidence unavailable: results/research/05_historical/ outputs, archived checkpoints and "
       "05_historical_verification.json are absent from this repository and local disks", RECOVERY05,
       "Retrospective; historical recipe differs (80/20 split, school loss 0.3, different optimizer, batch and Transformer); "
       "no causal attribution to one component", "Historical example hashes and package versions unavailable",
       COMMON_SYNTH, NO_EXT],
      None, [], [], study="05"),
    c("H05-06", "The item-5 GPU attempt stopped at recorded update 768 of its first run (NeuroPixel, initialization 10, "
      "split 0) through the unchanged 8 GiB RAM guard (available RAM 7.99 GiB), with zero completed training runs and "
      "no validation or final evaluation; it is preserved separately and excluded from all statistics.",
      "Item-5 GPU attempt, 2026-10-06 23:09:38-23:10:18 UTC", "programme_stopped", 0,
      ["Incident record only; its cost is separate from the CPU panel", NO_EXT],
      EP05, ["results/research/05_experiment/item5_gpu_interrupted.zip", "results/research/05_experiment/item5_queue_exit.json"],
      [("text:7.99", "results/research/05_experiment/item5_gpu_interrupted.zip"),
       ("text:768", "results/research/05_experiment/item5_gpu_interrupted.zip")],
      stop_reason="ResourceStop: available RAM 7.99 GiB below the unchanged 8 GiB guard at update 768", study="05"),
    c("H05-07", "The item-5 analyzer verified 28 training and 28 final records with zero issues (analysis sha256 "
      "11fd91ee...), and a second implementation independently recounted 114,688 final predictions and reproduced the "
      "five paired differences, t interval and H1 decision within 1e-12.",
      POP05, "blocked_external", 0, [MISSING05, RECOVERY05,
       "Internal artifact consistency and a second recount implementation; not independent laboratory reproduction"],
      None, [], [("114688", SUMMARY05), ("text:11fd91ee4ddb8155fc9127e5f6b826535654842190f048fecd23c3dd6f31e436", SUMMARY05),
                 ("text:adef6b11f29a0c00c58f1cb10a95453ba056eea5e711d8462179bcfafcff1453", SUMMARY05)],
      context=[SUMMARY05], study="05"),
    c("H05-08", "A post hoc item-5 validation diagnostic found that each of seven relative-Transformer checkpoints chose "
      "a visible filler of the queried category in all 2,048 validation examples while its nominal binding ranged "
      "43.750-47.559%.",
      POP05, "blocked_external", 0, [MISSING05, RECOVERY05,
       "Post hoc descriptive diagnostic; does not alter selection or H1 and does not establish an internal mechanism"],
      None, [], [], study="05"),
    c("H05-09", "The item-5 CPU recovery recorded 2,876.993 s of training timers over 28 runs, 2,979.207 s worker elapsed "
      "time and 0.518 GiB peak child RSS; inference latency and physical energy were not measured.",
      POP05, "blocked_external", 0, [MISSING05, RECOVERY05,
       "Training timers exclude worker overhead, the GPU attempt, historical reassessment and auditing; not total project cost"],
      None, [], [("2876.993", SUMMARY05)], context=[SUMMARY05], study="05"),
    c("H06-01", "The exact frozen A06.1 ablation and attribution recipe (commit a009f2a9, manifest sha256 4bc25ed4...) "
      "completed its full inventory - 26 trainings, 34 evaluations, six growth trajectories with 18 stages and eight "
      "gate fits - and closed with general component benefit, optimization sufficiency and architectural competence "
      "not established; it does not rescue H1.",
      POP06, "experiment_complete", 2, L06, EP06,
      [A06 + "closure_receipt.json", A06 + "audit_saved.json", A06 + "original.zip"],
      [("text:not_established", A06 + "closure_receipt.json"), ("4362", A06 + "closure_receipt.json"),
       ("text:4bc25ed4acce0c95f41d0a595289f854e0330fe237aee1210a8787378cdf6b01", A06 + "closure_receipt.json")],
      study="06_A06_1"),
    c("H06-02", "In A06.1, none of the three factorial components established a benefit: shared dictionary/readout "
      "+2.014 pp (paired 95% CI -10.394 to +14.423, df=1), school 0.3 versus 0 +0.439 pp (-13.675 to +14.554), "
      "reinjection -1.135 pp (-11.062 to +8.791).",
      POP06, "experiment_complete", 2,
      L06 + ["The tying ablation also changes the parameter count", "Intervals including zero do not demonstrate equivalence"],
      EP06, [A06 + "audit_saved.json", A06 + "ci_receipt.json"],
      [("2.014", A06 + "audit_saved.json"), ("-10.394", A06 + "audit_saved.json"), ("14.423", A06 + "audit_saved.json"),
       ("0.439", A06 + "audit_saved.json"), ("-1.135", A06 + "audit_saved.json"), ("8.791", A06 + "audit_saved.json")],
      study="06_A06_1"),
    c("H06-03", "In A06.1, recurrent depth effects were not established: training at T4 versus T16 +2.832 pp (paired 95% "
      "CI -2.752 to +8.416, df=1), T1 versus T16 +0.146 pp; truncating the same T16 weights to T1 or T4 changed binding "
      "by +0.732 and +0.293 pp with intervals including zero.",
      POP06, "experiment_complete", 2,
      L06 + ["Training at another depth and truncating the same weights are different interventions"],
      EP06, [A06 + "audit_saved.json"],
      [("2.832", A06 + "audit_saved.json"), ("-2.752", A06 + "audit_saved.json"), ("8.416", A06 + "audit_saved.json"),
       ("0.146", A06 + "audit_saved.json"), ("0.732", A06 + "audit_saved.json"), ("0.293", A06 + "audit_saved.json")],
      study="06_A06_1"),
    c("H06-04", "In A06.1, training with damage produced small uncertain effects: +0.732 pp in clean evaluation (paired "
      "95% CI -38.974 to +40.439), +1.392 pp under lesion, interaction +0.659 pp (-0.271 to +1.590); autonomous repair "
      "or recovery of useful memory was not demonstrated.",
      POP06, "experiment_complete", 2, L06, EP06, [A06 + "audit_saved.json"],
      [("-38.974", A06 + "audit_saved.json"), ("40.439", A06 + "audit_saved.json"), ("1.392", A06 + "audit_saved.json"),
       ("0.659", A06 + "audit_saved.json"), ("-0.271", A06 + "audit_saved.json"), ("1.590", A06 + "audit_saved.json")],
      study="06_A06_1"),
    c("H06-05", "In A06.1, extending training from 1,024 to 8,192 updates raised binding by +17.310 pp with school 0 "
      "(paired 95% CI +0.868 to +33.751, df=1, unadjusted) and +15.283 pp with school 0.3 (-48.620 to +79.186); the four "
      "8,192-update endpoints reached 27.832%, 34.082%, 24.951% and 18.115%, and all 26 probes remained budget_limited.",
      POP06, "experiment_complete", 2,
      L06 + ["No contrast remains below 0.05 after the retrospective Holm sensitivity",
             "Greater exposure does not resolve competence or separate optimization, representation and dynamics"],
      EP06, [A06 + "audit_saved.json"],
      [("17.310", A06 + "audit_saved.json"), ("0.868", A06 + "audit_saved.json"), ("33.751", A06 + "audit_saved.json"),
       ("15.283", A06 + "audit_saved.json"), ("-48.620", A06 + "audit_saved.json"), ("79.186", A06 + "audit_saved.json"),
       ("27.832", A06 + "original.zip"), ("34.082", A06 + "original.zip"), ("24.951", A06 + "original.zip"),
       ("18.115", A06 + "original.zip")], study="06_A06_1"),
    c("H06-06", "In A06.1 growth and routing, the fixed bank with the historical router reached 25.651% mean binding "
      "(23.242% and 28.060%) and the learned gate 26.335%; the historical router exceeded a uniform mixture by +17.546 pp "
      "(paired 95% CI +3.069 to +32.022, unadjusted) and a random router by +16.178 pp, but not the learned gate "
      "(-0.684 pp); novelty (-0.163 pp) and resonance (-12.207 pp) did not improve on the fixed bank; no competent "
      "adaptive growth or general routing superiority was established.",
      POP06, "experiment_complete", 2,
      L06 + ["Gates learn with additional labels over frozen experts; topics carry lexical signals",
             "Resonance and novelty banks differ in retained capacity and update policy"],
      EP06, [A06 + "audit_saved.json"],
      [("25.651", A06 + "original.zip"), ("23.242", A06 + "audit_saved.json"), ("28.060", A06 + "audit_saved.json"),
       ("26.335", A06 + "original.zip"), ("17.546", A06 + "audit_saved.json"), ("3.069", A06 + "audit_saved.json"),
       ("32.022", A06 + "audit_saved.json"), ("16.178", A06 + "audit_saved.json"), ("-0.684", A06 + "audit_saved.json"),
       ("-0.163", A06 + "audit_saved.json"), ("-12.207", A06 + "audit_saved.json")], study="06_A06_1"),
    c("H06-07", "A retrospective Holm sensitivity over the 40 rederived A06.1 main and routing contrasts left none below 0.05.",
      POP06, "engineering_complete", 1,
      ["The contrast family was defined during the audit; it is not a preregistered confirmatory family", NO_EXT],
      None, [A06 + "multiplicity_sensitivity.json"], [("40", A06 + "multiplicity_sensitivity.json")], study="06_A06_1"),
    c("H06-08", "The saved-evidence audit of A06.1 passed 4,362 checks with zero issues - recounting 139,264 final core, "
      "53,248 validation and 98,304 router predictions - and its 204 Wilson and conditional-bootstrap interval controls "
      "also passed; an independent recomputation of the factorial effects matched the frozen analysis means.",
      POP06, "engineering_complete", 1,
      ["Internal audit of saved evidence; no training, selection or new inference", NO_EXT,
       "The original 142-pass/2-skip regression count is reconstructed from a log excerpt without a JUnit file"],
      None, [A06 + "audit_saved.json", A06 + "audit_intervals.json", A06 + "closure_receipt.json", A06 + "regression_excerpt.txt"],
      [("4362", A06 + "closure_receipt.json"), ("204", A06 + "closure_receipt.json"), ("139264", A06 + "audit_saved.json"),
       ("53248", A06 + "audit_saved.json"), ("98304", A06 + "audit_saved.json")], study="06_A06_1"),
    c("H06-09", "CPU replay of A06.1 checkpoints reproduced decisions exactly for 34/34 core evaluations and 8/8 routing "
      "banks, with 44 checkpoints and 448 tensors finite; the original 1e-4 tolerance failed in five continuous "
      "comparisons of four long checkpoints (maximum absolute NLL error 0.000413895, confidence 0.000118315), the cause "
      "was not isolated, and the tolerance was not relaxed retrospectively.",
      POP06, "engineering_complete", 1,
      ["Linux-Windows numerical reproduction is not bitwise", "Replay uses CPU with two threads and no weight updates", NO_EXT],
      None, [A06 + "audit_replay.json", A06 + "numeric_diagnostic.json"],
      [("text:issues_found", A06 + "audit_replay.json"), ("0.000413895", A06 + "audit_replay.json"),
       ("0.000118315", A06 + "audit_replay.json"), ("448", A06 + "audit_replay.json")], study="06_A06_1"),
    c("HC-01", "Reconciliation of the local A06.1 continuation with the cloud item-6 study recovered 400 files from the "
      "cloud archive pinned at commit 15e76456 without size or hash discrepancies; 37 cloud execution sources match "
      "their manifest and 17 files are identical to local equivalents; the 26 nominal configs, 34 cases and final core "
      "dataset coincide, while checkpoint bytes and recorded results differ between executions (mean tying effect "
      "+2.014 pp in A06.1 versus -0.244 pp in the cloud report) and neither value is selected or averaged.",
      "Local A06.1 run 37588312570 and cloud item-6 run 37566497890", "engineering_complete", 1,
      ["The cause of the result differences is not identified", "Seeds 20/21 are not counted as four independent initializations",
       "No result merging, retraining or new prospective contrast", "The cloud value -0.244 pp is quoted from the cloud report and not re-verified here", NO_EXT],
      None, ["coord/recovery/reconciliation-20261008/reconciliation_receipt.json",
             "coord/recovery/reconciliation-20261008/source_comparison.json"],
      [("400", "coord/recovery/reconciliation-20261008/reconciliation_receipt.json"),
       ("37", "coord/recovery/reconciliation-20261008/reconciliation_receipt.json"),
       ("17", "coord/recovery/reconciliation-20261008/reconciliation_receipt.json")], study="CONTINUATIONS"),
    c("HT-01", "The task-4 audit recovered 763 original files (401,580,342 bytes) from fifteen pinned commits, verified "
      "commits, Merkle trees, every blob and all original manifests, and two internal programs passed 984 and 2,984 "
      "checks with zero discrepancies without loading models, training or generating predictions.",
      POPT4, "engineering_complete", 1,
      ["Internal audit of saved evidence, not a new independent scientific replication", NO_EXT],
      None, [T4 + "recovery_inventory.json", T4 + "saved_evidence_audit.json", T4 + "control_recount.json"],
      [("984", T4 + "saved_evidence_audit.json"), ("2984", T4 + "control_recount.json"),
       ("count:763", T4 + "recovery_inventory.json"), ("sumbytes:401580342", T4 + "recovery_inventory.json")], study="TASK4"),
    c("HT-07", "Historical item 7: nine internally disjoint partitions were confirmed and the eligible TRAIN union covers "
      "1,319 of 1,320 triples.", POPT4, "engineering_complete", 1,
      ["Does not identify the minibatches actually seen and does not create a historically clean holdout; contracts are optional", NO_EXT],
      None, [T4 + "control_recount.json"], [("1319", T4 + "control_recount.json")], study="TASK4"),
    c("HT-08", "Historical item 8: metric and PAD corrections and regressions are documented, with the original suite "
      "recording 186 passes and one declared skip.", POPT4, "engineering_complete", 1,
      ["No empirical improvement or competitive result; corrected historical sources and the current frozen core are different versions", NO_EXT],
      None, [T4 + "saved_evidence_audit.json"], [("186", "results/research/08_cloud_runs")], study="TASK4"),
    c("HT-09", "Historical item 9 (complex binding, five pairs): NeuroPixel 9.628906% versus relative Transformer "
      "49.355469% binding, difference -39.726563 pp (nominal interval -41.254103 to -38.199022).",
      POPT4 + "; item 9 synthetic complex-binding population", "experiment_complete", 2,
      ["Both recipes insufficiently competent", "Nominal interval over training realizations of the same population; "
       "no multiplicity adjustment", "Synthetic population with prior exposure", NO_EXT],
      "results/research/09_cloud_runs/37593731891-1-study/execution_plan.json",
      ["results/research/09_cloud_runs/37593731891-1-study/study/final_summary.json", "results/research/09_offline_runs/37603398040-1-offline/scientific_audit_summary.json", T4 + "saved_evidence_audit.json"],
      [("9.628906", "results/research/09_offline_runs/37603398040-1-offline/scientific_audit_summary.json"), ("49.355469", "results/research/09_offline_runs/37603398040-1-offline/scientific_audit_summary.json"),
       ("-39.726563", T4 + "saved_evidence_audit.json"), ("-41.254103", T4 + "saved_evidence_audit.json"),
       ("-38.199022", T4 + "saved_evidence_audit.json")], study="TASK4"),
    c("HT-10", "Historical item 10 (bAbI QA4, five pairs, trained from scratch): NeuroPixel 50.40% versus Transformer "
      "88.66% accuracy, difference -38.26 pp (nominal interval -52.385695 to -24.134305).",
      POPT4 + "; item 10 bAbI QA4 synthetic external task", "experiment_complete", 2,
      ["Synthetic external task learned from scratch; not zero-shot transfer or general language",
       "Nominal interval without multiplicity adjustment", NO_EXT],
      "results/research/10_cloud_runs/37615277149-1-study/execution_plan.json",
      ["results/research/10_cloud_runs/37615277149-1-study/pre_final_archive_receipt.json", T4 + "saved_evidence_audit.json"],
      [("50.40", "results/research/10_cloud_runs"), ("88.66", "results/research/10_cloud_runs"),
       ("-38.26", "results/research/10_cloud_runs"), ("-52.385695", "results/research/10_cloud_runs"),
       ("-24.134305", "results/research/10_cloud_runs")], study="TASK4"),
    c("HT-11", "Historical item 11 (memory): the selected NeuroPixel scored 38/48 versus 48/48 for the GRU against a "
      "46/48 requirement, so admission was false; the main five-pair panel, the delay-8 contrast and learned "
      "interference were not executed.", POPT4 + "; item 11 memory admission", "programme_stopped", 0,
      ["No figure is reported for the unexecuted panels", NO_EXT],
      "results/research/11_cloud_runs/37626660573-1-preflight/execution_plan.json",
      [T4 + "control_recount.json", "results/research/11_cloud_runs/37626660573-1-preflight/preflight_summary.json"],
      [("38", T4 + "control_recount.json"), ("48", T4 + "control_recount.json")],
      stop_reason="Admission failed: selected NeuroPixel 38/48 below the 46/48 requirement; main panel not executed",
      study="TASK4"),
    c("HT-12", "Historical item 12 (continual learning): eight native copies preserve the old tensors exactly while the new "
      "expert changes, and a negative alias was detected.", POPT4, "engineering_complete", 1,
      ["Copying weights increases storage and does not demonstrate useful acquisition of eight tasks or competent routing", NO_EXT],
      None, [T4 + "control_recount.json", T4 + "saved_evidence_audit.json"], [("count:8", T4 + "control_recount.json#isolation_updated_tensor_counts")],
      study="TASK4"),
    c("HT-13", "Historical item 13 (new symbols and distractors): original contracts and census are saved, and a presence "
      "shortcut solves the positive new-word panel.", POPT4, "engineering_complete", 1,
      ["A new token is not a newly learned meaning; the transformation can break the original disjunction", NO_EXT],
      None, [T4 + "saved_evidence_audit.json"], [], study="TASK4"),
    c("HT-14", "Historical item 14 (repair): an independent categorical recount of all 324 repair rows, with immediate "
      "damage and held, removed or changed sources, matches the saved evidence.", POPT4, "engineering_complete", 1,
      ["Assigned-weight controls separate assisted reconstruction, retention and redundancy; not learned autonomous repair", NO_EXT],
      None, [T4 + "control_recount.json"], [("324", T4 + "control_recount.json")], study="TASK4"),
    c("HT-15", "Historical item 15 (stability): over five checkpoints, one exposed scene and 256 additional steps, RMS "
      "growth ranged 4.293-18.431 and perturbation gain 1.557-6.059.", POPT4, "engineering_complete", 1,
      ["Finite over the observed horizon only; no convergence, attraction, useful memory, asymptotic divergence or universal stability", NO_EXT],
      None, [T4 + "saved_evidence_audit.json"],
      [("4.293", T4 + "saved_evidence_audit.json"), ("18.431", T4 + "saved_evidence_audit.json"),
       ("1.557", T4 + "saved_evidence_audit.json"), ("6.059", T4 + "saved_evidence_audit.json")], study="TASK4"),
]

DISCREPANCIES = [
    dict(source=R06, line=38, claim="H06-04", report_text="[-0,272; +1,590]",
         receipt=A06 + "audit_saved.json", receipt_field="/core_contrasts/damage interaction/t95_df1[0]",
         receipt_value=-0.0027145054220029576,
         finding="Lower interval endpoint rounds to -0.271 pp in the receipt; no retained file supports -0.272. "
                 "The ledger uses the receipt value; the report text needs an archived correction in A18.3."),
    dict(source=R05, line=5, claim="H05-07", report_text="both artifact checks verified",
         receipt=None, receipt_field=None, receipt_value=None,
         finding="The verified analysis and independent recount files are not in the repository or on local disks; "
                 "the statement cannot be checked against retained bytes."),
    dict(source=R04, line=59, claim="H04-04", report_text="separate verifier ... zero discrepancies",
         receipt=None, receipt_field=None, receipt_value=None,
         finding="results/research/04_verification.json is missing; only the 60-file transfer manifest was re-verified."),
]

# ---------------------------------------------------------------- line decisions
# class: claim, data_row, limit, method, provenance -> linked claim ids; editorial, governance -> reason
ED_HEADER = "table header row: column labels, not an assertion"
ED_LINKS = "link list or reference definition: navigation only; linked artifacts are bound in the claims"
ED_FORMULA = "displayed formula: method notation bound to the H1 claim protocol"
ED_FIGURE = "figure embed or caption: visual restatement of bound data rows"
ED_POINTER = "evidence-file pointer: navigation to artifacts bound in the claims"
GOV = "project-process statement (task ordering or closure governance), not a scientific or technical claim"

def rng(a, b, step=1):
    return list(range(a, b + 1, step))

D = {R04: {}, R05: {}, R06: {}, RC: {}, RT: {}}
def put(src, lines, cls, target):
    for line in (lines if isinstance(lines, list) else [lines]):
        assert line not in D[src], (src, line)
        D[src][line] = (cls, target)

put(R04, 5, "claim", ["H04-01", "H04-02"]); put(R04, 7, "limit", ["H04-01"]); put(R04, 11, "method", ["H04-01"])
put(R04, [13, 26, 39], "editorial", ED_HEADER); put(R04, rng(15, 18), "data_row", ["H04-01"]); put(R04, 20, "claim", ["H04-01"])
put(R04, 24, "method", ["H04-02"]); put(R04, rng(28, 31), "data_row", ["H04-02"]); put(R04, 33, "claim", ["H04-02"])
put(R04, 35, "claim", ["H04-03"]); put(R04, rng(41, 48), "data_row", ["H04-01"]); put(R04, 50, "method", ["H04-01"])
put(R04, [54, 55, 56, 57, 58, 59, 61], "provenance", ["H04-04"]); put(R04, 60, "claim", ["H04-04"])
put(R04, 63, "limit", ["H04-04", "H04-01"]); put(R04, [67, 69], "limit", ["H04-01"]); put(R04, 71, "claim", ["H04-01"])
put(R04, rng(75, 79), "editorial", ED_POINTER)

H5 = ["H05-01", "H05-02", "H05-03", "H05-04", "H05-05", "H05-07", "H05-08", "H05-09"]
put(R05, 1, "limit", H5); put(R05, 5, "claim", ["H05-01", "H05-07"]); put(R05, 7, "claim", ["H05-01", "H05-07"])
put(R05, 9, "claim", ["H05-04", "H05-05"]); put(R05, [11, 28, 41, 76, 91, 98, 127, 135, 145, 158, 164, 178, 191, 210, 232, 242],
                                                 "editorial", ED_HEADER)
put(R05, 13, "data_row", ["H05-05"]); put(R05, 14, "data_row", ["H05-06"]); put(R05, 15, "data_row", ["H05-01"])
put(R05, 16, "limit", ["H05-01"]); put(R05, [22, 24, 26], "method", ["H05-01"]); put(R05, rng(30, 33), "method", ["H05-01"])
put(R05, 35, "limit", ["H05-01"]); put(R05, 37, "method", ["H05-01"]); put(R05, [43, 44], "method", ["H05-01", "H05-02"])
put(R05, 46, "method", ["H05-01", "H05-02"]); put(R05, 50, "method", ["H05-01", "H05-07"]); put(R05, 52, "method", ["H05-04"])
put(R05, 56, "claim", ["H05-06"]); put(R05, 58, "provenance", ["H05-01"]); put(R05, [60, 62], "limit", ["H05-01"])
put(R05, [68, 70], "provenance", ["H05-05"]); put(R05, 74, "claim", ["H05-05"]); put(R05, rng(78, 83), "data_row", ["H05-05"])
put(R05, 85, "claim", ["H05-05"]); put(R05, 89, "method", ["H05-05"]); put(R05, rng(93, 96), "data_row", ["H05-05"])
put(R05, [100, 101], "data_row", ["H05-05"]); put(R05, [103, 109, 111], "limit", ["H05-05"]); put(R05, 107, "claim", ["H05-05"])
put(R05, 115, "claim", ["H05-01"]); put(R05, [119, 121, 123], "claim", ["H05-07"]); put(R05, rng(129, 133), "data_row", ["H05-01"])
put(R05, rng(137, 140), "data_row", ["H05-01"]); put(R05, 143, "method", ["H05-01"]); put(R05, rng(147, 150), "data_row", ["H05-01"])
put(R05, 152, "claim", ["H05-01"]); put(R05, 156, "method", ["H05-02"]); put(R05, rng(160, 162), "data_row", ["H05-02"])
put(R05, rng(166, 169), "data_row", ["H05-02"]); put(R05, 172, "claim", ["H05-02"]); put(R05, 176, "claim", ["H05-03"])
put(R05, rng(180, 183), "data_row", ["H05-03"]); put(R05, 185, "claim", ["H05-03"]); put(R05, 189, "claim", ["H05-04"])
put(R05, rng(193, 196), "data_row", ["H05-04"]); put(R05, 198, "limit", ["H05-04"]); put(R05, 200, "claim", ["H05-08"])
put(R05, [204, 206], "editorial", ED_FIGURE); put(R05, rng(212, 218), "data_row", ["H05-09"]); put(R05, 220, "limit", ["H05-09"])
put(R05, [224, 230], "method", ["H05-01"]); put(R05, [226, 227, 228], "editorial", ED_FORMULA)
put(R05, rng(234, 238), "data_row", ["H05-01"]); put(R05, 240, "method", ["H05-01"]); put(R05, rng(244, 249), "data_row", ["H05-01"])
put(R05, 251, "claim", ["H05-01"]); put(R05, 253, "method", ["H05-03"]); put(R05, 255, "limit", ["H05-01", "H05-02"])
put(R05, 259, "claim", ["H05-01"]); put(R05, 261, "limit", ["H05-01", "H05-04"]); put(R05, 263, "limit", ["H05-01", "H05-05"])
put(R05, [267] + rng(269, 279) + rng(281, 302) + rng(304, 316), "editorial", ED_LINKS)

put(R06, 3, "claim", ["H06-01"]); put(R06, [7, 8, 9, 10, 11, 15], "provenance", ["H06-01"]); put(R06, 12, "claim", ["H06-01"])
put(R06, [13, 14], "claim", ["H06-08"]); put(R06, 19, "method", ["H06-01"])
put(R06, 21, "limit", ["H06-02", "H06-03", "H06-04", "H06-05", "H06-06"]); put(R06, 25, "method", ["H06-02"])
put(R06, 27, "editorial", ED_HEADER); put(R06, rng(29, 31), "data_row", ["H06-02"]); put(R06, rng(32, 35), "data_row", ["H06-03"])
put(R06, rng(36, 38), "data_row", ["H06-04"]); put(R06, [39, 40], "data_row", ["H06-05"]); put(R06, rng(41, 45), "data_row", ["H06-06"])
put(R06, 47, "claim", ["H06-08"]); put(R06, [51, 52, 53], "claim", ["H06-02"]); put(R06, 54, "claim", ["H06-03"])
put(R06, 55, "claim", ["H06-04"]); put(R06, 56, "claim", ["H06-06"]); put(R06, 60, "claim", ["H06-05"])
put(R06, 62, "claim", ["H06-07"]); put(R06, 64, "limit", ["H06-02", "H06-03", "H06-04", "H06-05", "H06-06"])
put(R06, 66, "claim", ["H05-01", "H06-01"]); put(R06, 70, "editorial", ED_POINTER); put(R06, 72, "limit", ["H06-09"])
put(R06, 74, "claim", ["H06-09"]); put(R06, 76, "claim", ["H06-08"]); put(R06, 78, "editorial", ED_LINKS)

put(RC, 3, "limit", ["HC-01"]); put(RC, 5, "editorial", ED_HEADER); put(RC, rng(7, 14), "provenance", ["HC-01"])
put(RC, [16, 18, 20], "claim", ["HC-01"]); put(RC, 22, "governance", GOV); put(RC, 24, "editorial", ED_POINTER)

put(RT, [3, 5], "claim", ["HT-01"]); put(RT, 7, "editorial", ED_HEADER)
for line, cid in zip(rng(9, 17), ["HT-07", "HT-08", "HT-09", "HT-10", "HT-11", "HT-12", "HT-13", "HT-14", "HT-15"]):
    put(RT, line, "data_row", [cid])
put(RT, 19, "limit", ["HT-09", "HT-10", "H05-01"]); put(RT, 21, "editorial", ED_POINTER); put(RT, 23, "governance", GOV)

# ---------------------------------------------------------------- verification helpers
def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

_cache: dict[str, tuple[list[float], str]] = {}
NUM = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")

def _walk(value, out):
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        out.append(float(value))
    elif isinstance(value, dict):
        for item in value.values():
            _walk(item, out)
    elif isinstance(value, list):
        for item in value:
            _walk(item, out)
    elif isinstance(value, str):
        out.extend(float(x) for x in NUM.findall(value))

def _load_bytes(name: str, data: bytes, numbers: list[float], texts: list[str]):
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return
    texts.append(text)
    if name.endswith(".json"):
        try:
            _walk(json.loads(text), numbers)
            return
        except json.JSONDecodeError:
            pass
    numbers.extend(float(x) for x in NUM.findall(text))

def evidence(rel: str) -> tuple[list[float], str]:
    if rel in _cache:
        return _cache[rel]
    path = ROOT / rel
    numbers: list[float] = []
    texts: list[str] = []
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    for file in files:
        if file.suffix == ".zip":
            with zipfile.ZipFile(file) as bundle:
                for name in bundle.namelist():
                    if name.endswith((".json", ".jsonl", ".log", ".txt", ".csv")):
                        _load_bytes(name, bundle.read(name), numbers, texts)
        elif file.suffix in (".json", ".jsonl", ".log", ".txt", ".csv", ".md") and file.stat().st_size < 8_000_000:
            _load_bytes(file.name, file.read_bytes(), numbers, texts)
    _cache[rel] = (numbers, "\n".join(texts))
    return _cache[rel]

def number_ok(claimed: str, numbers: list[float]) -> bool:
    value = float(claimed)
    decimals = len(claimed.split(".")[1]) if "." in claimed else 0
    tolerance = 0.5 * 10 ** (-decimals) + 1e-12
    for x in numbers:
        for scaled in (x, x * 100.0, x / 100.0):
            if abs(scaled - value) <= tolerance:
                return True
    return False

def recount_item4() -> dict:
    """Recount the selected Transformer's saved validation predictions; no model is executed."""
    with zipfile.ZipFile(ROOT / "results/research/04_experiment/item4_evidence.zip") as bundle:
        data = np.load(io.BytesIO(bundle.read("pilot/datasets/split0_validation_n2048_seed61001.npz")))
        pred = np.load(io.BytesIO(bundle.read("pilot/relative_transformer_s0_i7_lr0.001/validation_predictions.npz")))
        selection = json.loads(bundle.read("pilot/selection.json"))
    assert selection["primary_reference_family"] == "relative_transformer"
    assert selection["selected_configs"]["relative_transformer"]["learning_rate"] == 0.001
    canvas, target = data["canvas"], data["target"]
    prediction, role = pred["prediction"], pred["role"]
    assert np.array_equal(pred["target"], target)
    def category(t):  # vocabulary: 0 empty, 1-4 roles, 5-16 nouns, 17-26 verbs, 27-34 places
        return np.select([(t >= 5) & (t <= 16), (t >= 17) & (t <= 26), (t >= 27) & (t <= 34)], [1, 2, 3], 0)
    visible = np.array([prediction[i] in canvas[i] for i in range(len(prediction))])
    same = category(prediction) == category(target)
    nominal = np.isin(role, [0, 2])
    return dict(selected_reference="relative_transformer_s0_i7_lr0.001", examples=int(len(prediction)), visible_same_category=int((visible & same).sum()),
                nominal_queries=int(nominal.sum()), nominal_correct=int((prediction == target)[nominal].sum()))

# ---------------------------------------------------------------- build
def main() -> int:
    candidates = json.loads((OUT / "claim_candidates.json").read_text(encoding="utf-8"))["candidates"]
    batches = json.loads((OUT / "claim_review_batches.json").read_text(encoding="utf-8"))["batches"]
    batch = next(b["candidate_ids"] for b in batches if b["task"] == "A18.2.H")
    by_id = {cand["id"]: cand for cand in candidates}
    claim_ids = {claim["id"] for claim in CLAIMS}
    problems: list[str] = []
    rows, used = [], set()
    units: dict[str, list[str]] = {cid: [] for cid in claim_ids}
    for cid in batch:
        cand = by_id[cid]
        decision = D.get(cand["source_path"], {}).get(cand["source_line"])
        if decision is None:
            problems.append(f"unclassified candidate {cid} {cand['source_path']}:{cand['source_line']}")
            continue
        used.add((cand["source_path"], cand["source_line"]))
        cls, target = decision
        row = dict(candidate_id=cid, source_path=cand["source_path"], source_line=cand["source_line"],
                   source_sha256=cand["source_sha256"], classification=cls)
        if cls in ("editorial", "governance"):
            row["exclusion_reason"] = target
        else:
            unknown = [t for t in target if t not in claim_ids]
            if unknown:
                problems.append(f"{cid}: unknown claim ids {unknown}")
            row["claims"] = target
            for t in target:
                units.setdefault(t, []).append(cid)
        rows.append(row)
    for src, lines in D.items():
        for line in lines:
            if (src, line) not in used:
                problems.append(f"decision without candidate {src}:{line}")
    for cid in claim_ids:
        if not any(cand_cls in ("claim", "data_row") and cid in row.get("claims", [])
                   for row in rows for cand_cls in [row["classification"]]):
            problems.append(f"{cid}: no asserting source unit")

    recount = recount_item4()
    checks_out, ledger_claims = [], []
    for claim in CLAIMS:
        results = []
        for spec, rel in claim["value_checks"]:
            if spec.startswith("recount:"):
                key = {"recount:item4_visible_category": "visible_same_category",
                       "recount:item4_nominal_correct": "nominal_correct"}[spec]
                ok = str(recount[key]) == rel
                results.append(dict(check=spec, expected=rel, observed=recount[key], ok=ok))
            elif spec.startswith("count:") or spec.startswith("sumbytes:"):
                path, _, key = rel.partition("#")
                data = json.loads((ROOT / path).read_text(encoding="utf-8"))
                if spec.startswith("count:"):
                    observed = len(data[key]) if key else len(data["files"])
                else:
                    observed = sum(int(item.get("bytes", item.get("size", 0))) for item in data["files"])
                ok = observed == int(spec.split(":")[1])
                results.append(dict(check=spec, source=rel, observed=observed, ok=ok))
            elif spec.startswith("text:"):
                ok = spec[5:] in evidence(rel)[1]
                results.append(dict(check=spec, source=rel, ok=ok))
            else:
                ok = number_ok(spec, evidence(rel)[0])
                results.append(dict(check=spec, source=rel, ok=ok))
            if not ok:
                problems.append(f"{claim['id']}: value check failed {spec} in {rel}")
        checks_out.append(dict(claim=claim["id"], checks=results))
    (OUT / "historical_value_checks_H.json").write_text(
        json.dumps(dict(schema=1, item4_recount=recount, claims=checks_out), indent=2) + "\n", encoding="utf-8", newline="\n")
    for claim in CLAIMS:
        entry = dict(id=claim["id"], statement=claim["statement"], population=claim["population"],
                     status=claim["status"], declared_level=claim["declared_level"], limits=claim["limits"],
                     receipts=[dict(path=p, sha256=sha256(ROOT / p)) for p in claim["receipt_paths"]],
                     stop_reason=claim["stop_reason"], review_batch="A18.2.H", study=claim["study"],
                     source_units=sorted(units.get(claim["id"], [])))
        if claim["protocol"]:
            entry["protocol"] = dict(path=claim["protocol"], sha256=sha256(ROOT / claim["protocol"]))
        if claim["context_paths"]:
            entry["context_records"] = [dict(path=p, sha256=sha256(ROOT / p),
                                             role="summary_record_not_raw_evidence") for p in claim["context_paths"]]
        ledger_claims.append(entry)

    classes: dict[str, int] = {}
    for row in rows:
        classes[row["classification"]] = classes.get(row["classification"], 0) + 1
    review = dict(schema=1, task="A18.2.H", candidates=len(batch), classified=len(rows), classes=classes,
                  claims=len(CLAIMS), discrepancies=DISCREPANCIES, problems=problems,
                  scope=("Semantic review of batch A18.2.H. Claims are artifact-bound; study 05 and the historical "
                         "reassessment remain L0 blocked_external because their raw evidence is unavailable."),
                  decisions=rows)
    (OUT / "claim_review_H.json").write_text(json.dumps(review, indent=2, ensure_ascii=False) + "\n",
                                             encoding="utf-8", newline="\n")
    PART.parent.mkdir(parents=True, exist_ok=True)
    PART.write_text(json.dumps(dict(schema=1, authors=AUTHORS, claims=ledger_claims), indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8", newline="\n")
    print(json.dumps(dict(candidates=len(batch), classified=len(rows), classes=classes, claims=len(CLAIMS),
                          item4_recount=recount, problems=problems), indent=1))
    return 0 if not problems else 1

if __name__ == "__main__":
    sys.exit(main())
