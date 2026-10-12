"""A18.2.P: semantic review of README, manuscripts and new-phase documents.

New claims are registered here; lines restating claims already registered in batches H, D and G
are linked to them and their quoted numbers are re-checked against receipts or recounts.
"""
from __future__ import annotations

import hashlib
import json
import sys
import zipfile

import completion_review_claims_D as DMOD
import completion_review_claims_G as GMOD
from completion_claim_review_lib import (ED_FIGURE, ED_HEADER, ED_LINKS, ED_POINTER, GOV, NO_EXT, ROOT, Decisions,
                                         claim, ledger_ids, run_review)

R = "docs/research/"
RR = "results/research/"
C = RR + "COMPLETION20261012/"
README = "README.md"
MS = "docs/papers/NeuroPixel_manuscript_20261009.md"
EV = "docs/papers/NeuroPixel_evidence_reconciled_20261012.md"
G0D = R + "COMPLETION_G0_inventory_20261012.md"
REP = R + "REPRODUCE_20261009.md"
IDX = R + "RESULTS_INDEX_20261009.md"
EXTERNAL = ledger_ids("A18.2.H.json", "A18.2.D.json", "A18.2.G.json")
ED_LICENSE = "license and data attribution: legal notice, not a scientific or technical claim"
ED_BYLINE = "author byline and document date"
POPIMPL = "The NeuroPixel software in this repository (PyTorch model and three OpenGL backends)"
VIEWER = RR + "VIS07_live_viewer/"


def recounts() -> dict:
    out = dict(DMOD.vis07_recounts())
    out.update(GMOD.RECOUNTS)
    for name in ("actual_neural_trajectory", "advanced_GPU_backends", "all_GPU_workloads",
                 "depth_limits_and_state_growth", "external_vision_all_seeds"):
        out[f"sha12|{name}"] = hashlib.sha256((ROOT / f"docs/animations/{name}.gif").read_bytes()).hexdigest()[:12]
    out["sha12|thinking"] = hashlib.sha256((ROOT / "docs/img/thinking.gif").read_bytes()).hexdigest()[:12]
    with zipfile.ZipFile(ROOT / RR / "VIS07MAIN_cloud/37908785698/raw.zip") as bundle:
        records = json.loads(bundle.read("result.json"))["training_records"]
    for family in ("nca", "cnn"):
        out[f"vis07_{family}_wall"] = sum(r["wall_seconds"] for r in records if r["family"] == family)
        out[f"vis07_{family}_cpu"] = sum(r["cpu_seconds"] for r in records if r["family"] == family)
    ratios = [GMOD.RECOUNTS[k] for k in GMOD.RECOUNTS if k.startswith("R08|") and k.endswith("|ratio")]
    out["R08|slowdown_min"], out["R08|slowdown_max"] = 1 / max(ratios), 1 / min(ratios)
    for case in ("C16_grid8", "C16_grid32", "C16_grid128", "C48_grid8", "C48_grid32", "C48_grid128", "RGB8_retina_C16"):
        out[f"R10|{case}|best_cuda"] = min(GMOD.RECOUNTS[f"R10|{case}|cuda_eager"], GMOD.RECOUNTS[f"R10|{case}|cuda_graph"])
    return out


RECOUNTS = recounts()

CLAIMS = [
    claim("P-ARCH-01", "NeuroPixel represents a recurrent neural state on a spatial canvas updated by a learned local residual "
          "rule, s[t+1] = s[t] + W2 ReLU(W1 concat(s[t], K*s[t], u) + b1) + b2, with a learned depthwise 3x3 perception K, a "
          "learned seed, an RGB retina for identity channels and a shared projection/dictionary decoder; it is implemented in "
          "PyTorch and in three graphics backends: a scalar RGBA32F fragment renderer, a four-channel vector fragment renderer "
          "and an OpenGL 4.6 compute backend with SSBO weights and shared pixel tiles.",
          POPIMPL, "engineering_complete", 1,
          ["Implementation description only; not a biological brain or a complete causal explanation",
           "The scanner applies the decoder beyond the supervised output cell; semantic correctness at every displayed cell is not established",
           "Learned cellular updates and GPU/WebGL image execution have primary antecedents (Mordvintsev et al., 2020); no priority is claimed",
           "Memory and rest-state behaviour shown in pre-scientific figures are not established by current evidence", NO_EXT],
          None, ["neuropixel/rendering/nca_gl.py", "neuropixel/rendering/nca_gl_vector.py",
                 "neuropixel/rendering/nca_gl_compute.py", "neuropixel/model.py"],
          [("text:RGBA32F", "neuropixel/rendering/nca_gl.py"), ("text:SSBO", "neuropixel/rendering/nca_gl_compute.py")],
          study="IMPLEMENTATION"),
    claim("P-GPU-01", "Graphics execution reproduces the neural computation within the registered checks: the extended fragment "
          "preflight passed 136 physical RTX 3090 checks (states, masks, retina, logits and decisions); learned deployment "
          "reproduced all 320 tested decisions of the five selected visual models (first 64 official test rows each) against "
          "native CUDA and original CPU predictions; the Torch-free compute preflight passed 21 checks across seven workloads.",
          POPIMPL + "; RTX 3090, FP32", "engineering_complete", 1,
          ["Engineering validations on fixed subsets, not new training realizations or outside replications",
           "Pointwise logit and mean NLL limits as registered; no claim beyond the tested rows", NO_EXT],
          None, [RR + "RENDER09_preflight/original_20261009A/receipt.json", RR + "RENDER08_preflight/local_gpu_20261009B/receipt.json",
                 RR + "VIS07_learned_gpu/receipt.json", RR + "RENDER10_preflight/render_only_20261009B/receipt.json"],
          [("len:136", RR + "RENDER09_preflight/original_20261009A/receipt.json#checks"),
           ("len:136", RR + "RENDER08_preflight/local_gpu_20261009B/receipt.json#checks"),
           ("len:21", RR + "RENDER10_preflight/render_only_20261009B/receipt.json#checks"),
           ("len:5", RR + "VIS07_learned_gpu/receipt.json#checks"), ("64", RR + "VIS07_learned_gpu/receipt.json")],
          study="GPU_FIDELITY"),
    claim("P-VIEW-01", "The interactive VIS07 viewer exposes all sixteen raw channels, an RGB projection, per-cell decoder "
          "probabilities and reset/depth controls; archive mode passed 50 HTTP checks and browser controls, live physical GPU "
          "integration passed 27 trajectory gates plus 52 HTTP checks including the ten TRAIN resets, and a separate CPU/Mesa "
          "execution passed the same internal endpoint.",
          "VIS07 viewer software on the RTX 3090 host and a separate CPU/Mesa runner", "engineering_complete", 1,
          ["Software integration checks; not evidence of semantic interpretability", NO_EXT], None,
          [VIEWER + "offline_20261009B/http_verification.json", VIEWER + "live_20261009A/http_verification.json",
           VIEWER + "live_20261009A/server/receipt.json", RR + "VIS07_viewer_software_review/37936500542/http_verification.json"],
          [("len:50", VIEWER + "offline_20261009B/http_verification.json#checks"),
           ("len:52", VIEWER + "live_20261009A/http_verification.json#checks"),
           ("len:27", VIEWER + "live_20261009A/server/receipt.json#checks"),
           ("len:52", RR + "VIS07_viewer_software_review/37936500542/http_verification.json#checks")], study="VIEWER"),
    claim("P-ANIM-01", "The README animations are captures of actual model states linked by SHA-256 in docs/animations/manifest.json, "
          "not synthetic artwork; the trained-trajectory GIF is an actual GPU-rendered capture of the selected VIS07 model "
          "(registered seed 240, fixed TRAIN digit 0), slowed for inspection, with colours projecting raw channels.",
          "Five README GIFs under docs/animations/", "engineering_complete", 1,
          ["Display colours do not change model states or prove semantic causality",
           "The separate scanner figure docs/img/thinking.gif comes from the pre-scientific figure script and is not in the scientific manifest",
           NO_EXT], None, ["docs/animations/manifest.json", RR + "VIS07_learned_gpu/receipt.json"],
          [("recount:sha12|actual_neural_trajectory", "3cc78dacc2d1"), ("recount:sha12|advanced_GPU_backends", "efe22459cffc"),
           ("recount:sha12|all_GPU_workloads", "87a3572584c1"), ("recount:sha12|depth_limits_and_state_growth", "09e0cca97300"),
           ("recount:sha12|external_vision_all_seeds", "50243fe28a7c"),
           ("text:actual_rendered_train0_seed240_frames.npz", "docs/animations/manifest.json")], study="ANIMATIONS"),
    claim("P-COST-01", "VIS07 training plus DEV selection cost 413.318 s wall and 826.410 s process CPU time for the five "
          "NeuroPixel runs versus 34.237 s and 68.427 s for the five CNN runs: the visual accuracy advantage carried a greater "
          "measured training cost.",
          "VIS07 main cohort, five paired seeds", "experiment_complete", 2,
          ["Excludes installation, scheduling, compilation, transfers, storage and verification; no energy measurement", NO_EXT],
          R + "VIS07_execution_plan.json", [RR + "VIS07MAIN_cloud/37908785698/raw.zip"],
          [("recount:vis07_nca_wall", "413.318"), ("recount:vis07_nca_cpu", "826.410"),
           ("recount:vis07_cnn_wall", "34.237"), ("recount:vis07_cnn_cpu", "68.427")], study="VIS07"),
    claim("P-AVAIL-01", "Frozen plans, source commits, checkpoints, datasets, saved predictions or states, failed attempts and "
          "SHA-256 manifests are retained for the indexed studies - in this repository or, for large recovery archives, on the "
          "project's local disks - except the item-5 CPU archive, its analysis files and the historical reassessment outputs, "
          "which are unavailable.",
          "All indexed NeuroPixel study reports", "engineering_complete", 1,
          ["Large recovery archives (for example OPT03_QTRAIN_U16_recovery and READ03_recovery) remain outside the Git tree",
           "Publication and internal replay are not peer review or outside replication", NO_EXT],
          None, [C + "G0_inventory_summary.json", C + "historical_receipt_import.json", C + "dependency_import_A18.2.D.json",
                 C + "study05_raw_evidence_recovery_package.json"],
          [("1797", C + "G0_inventory_summary.json")], study="AVAILABILITY"),
    claim("P-MATH-01", "The completion kit implements nonnormal-Jacobian diagnostics motivated by the Crouzeix result (openai/math "
          "family 325), exact finite Boolean-noise controls (family 119) and a leaky-tanh reference field with an exact-weight "
          "contraction certificate; its 38 contract and 25 extension tests passed in the pinned environment.",
          "completion_kit modules in this repository", "engineering_complete", 1,
          ["Applicability requires the stated hypotheses", "The certificate concerns ideal real arithmetic and fixed input",
           "No useful indefinite memory, all-time GPU roundoff control or empirical advantage of the new mathematics is shown",
           "Formal proof rebuilding is a separate gate (task M3)", NO_EXT],
          None, ["completion_kit/math_diagnostics.py", "completion_kit/stable_field.py", "docs/execplan/evidence/P0.3_contracts.txt",
                 "docs/execplan/evidence/P0.3_extensions.txt"],
          [("text:Ran 38 tests", "docs/execplan/evidence/P0.3_contracts.txt"), ("text:Ran 25 tests", "docs/execplan/evidence/P0.3_extensions.txt")],
          study="COMPLETION_KIT"),
    claim("P-G0-01", "At base commit d8721a59 all 1,797 raw blobs of the publication manifest match their Git blob identities "
          "and SHA-256 values, all 44 declared origin commits are available with the registered blobs, 417 further artifacts are "
          "hashed and 19 historical reports are linked to plans, governors, execution identifiers and saved recounts, with "
          "missing dependencies named.",
          "Frozen publication tree at commit d8721a59", "engineering_complete", 1,
          ["Archive commits are not ancestors of main; preservation copies blobs and does not establish replication",
           "A report or source commit is not a substitute for missing original data", NO_EXT],
          None, [C + "G0_inventory_summary.json", C + "original_blob_audit.json", C + "origin_provenance.json",
                 C + "study_traceability_map_checked.json"],
          [("1797", C + "G0_inventory_summary.json"), ("44", C + "G0_inventory_summary.json"),
           ("417", C + "G0_inventory_summary.json"), ("19", C + "G0_inventory_summary.json")], study="G0"),
    claim("P-G0-02", "The release audit at the G0 baseline correctly exited with status 1 and preserved four issues (incomplete "
          "published energy evidence, 105 announced energy rows without an admitted receipt, a narrative stating 100 original "
          "rows against 103 stored, and incompatible RENDER10 statements in the manuscript).",
          "Frozen publication tree at commit d8721a59", "engineering_complete", 1,
          ["Issues later reconciled in G1 are preserved in the before-audit; some stale wording remains (see A18.2.P discrepancies)", NO_EXT],
          None, [C + "release_audit_before.json"], [("len:4", C + "release_audit_before.json#issues")], study="G0"),
    claim("P-T0-01", "A scoped zero-depth (T=0) forward contract is installed in a new model version with its AST scope verified; "
          "its runtime behaviour in the historical Torch environment and the related CPU regressions have not been executed "
          "(task G2.2, blocked by the unchanged 8 GiB RAM guard), so no behavioural claim is made.",
          "NeuroPixel model forward pass at T=0 in the completion branch", "engineering_complete", 1,
          ["Runtime pending: implementation receipts only", "The frozen original checkout is preserved separately", NO_EXT],
          None, ["completion_kit/model_forward_fix.py", "docs/execplan/evidence/G2.1_patch_scope.json"], [], study="T0"),
]
UNSOURCED = {"P-T0-01"}  # new-phase claim created after the A18.1 source freeze

V = "recount:vis07_"
RESTATE = [
    ("README:26", V + "nca_mean", "94.36"), ("README:26", V + "cnn_mean", "90.84"), ("README:26", V + "diff_mean", "3.52"),
    ("README:26", V + "diff_lo", "1.93"), ("README:26", V + "diff_hi", "5.10"), ("README:27", "recount:R09|C16_grid128|ratio", "1.536"),
    ("README:28", "6.278", RR + "GLOB05_review/37887762153/"), ("README:47", "recount:R10|rows", "245"),
    ("README:57", "recount:R08|slowdown_min", "3.14"), ("README:57 receipt value; text says 12.25", "recount:R08|slowdown_max", "12.24"),
    ("MS:7", "recount:R09|C16_grid128|ratio", "1.536"), ("MS:31", "16.244", RR + "DEV04_review/37845945011/"),
    ("MS:31", "5.658", RR + "DEV04_review/37845945011/"), ("MS:33", "3538944", RR + "GLOB05_review/37887762153/closure_receipt.json"),
    ("MS:49", V + "nca_mean", "94.3573"), ("MS:49", V + "cnn_mean", "90.8403"), ("MS:49", V + "diff_mean", "3.51697"),
    ("MS:49", V + "diff_lo", "1.93046"), ("MS:49", V + "diff_hi", "5.10349"), ("MS:55", "98835", RR + "VIS07REPLAY_cloud"),
    ("MS:63", "recount:R09|C16_grid128|render", "6.6585"), ("MS:63", "recount:R09|C16_grid128|cuda_eager", "10.2256"),
    ("MS:65", "recount:R10|C16_grid128|ratio", "2.084"), ("EV:7", V + "diff_mean", "3.5170"), ("EV:7", V + "diff_lo", "1.9305"),
    ("EV:7", V + "diff_hi", "5.1035"), ("G0:9", V + "diff_mean", "3.5169727323"), ("G0:9", V + "diff_lo", "1.9304594488"),
    ("G0:9", V + "diff_hi", "5.1034860159"), ("IDX:5", "calc:3968=984+2984", RR + "TASK4_review"),
    ("IDX:6", "10616832", RR + "READ03_review/37831051538/closure_receipt.json"),
    ("IDX:12", "len:35", RR + "RENDER08_benchmark/original_20261009A/receipt.json#checks"),
    ("IDX:12", "recount:R08|rows", "147"), ("IDX:13", "recount:R09|rows", "196"),
    ("IDX:13", "len:56", RR + "RENDER09_benchmark/original_20261009A/receipt.json#checks"),
    ("REP:21", "103", C + "energy_recovery_admission.json"),
]
EV_LINES = (ROOT / EV).read_text(encoding="utf-8").splitlines()
for number in range(27, 34):
    case, best, compute, ratio = [x.strip() for x in EV_LINES[number - 1].strip().strip("|").split("|")]
    RESTATE += [(f"EV:{number}", f"recount:R10|{case}|best_cuda", best), (f"EV:{number}", f"recount:R10|{case}|render", compute),
                (f"EV:{number}", f"recount:R10|{case}|ratio", ratio)]
for number in range(41, 76):
    case, method, med, low, high = [x.strip() for x in EV_LINES[number - 1].strip().strip("|").split("|")]
    RESTATE += [(f"EV:{number}", f"recount:R11|{case}|{method}|median", med),
                (f"EV:{number}", f"recount:R11|{case}|{method}|min", low),
                (f"EV:{number}", f"recount:R11|{case}|{method}|max", high)]

D = Decisions()
D.put(README, [1, 2, 3, 11, 12, 13, 15, 33, 34, 35, 39, 40, 41, 49, 53, 65, 73, 79, 80, 81], "editorial", ED_FIGURE)
D.put(README, [9, 47], "claim", ["P-ARCH-01", "G-R10-01"]); D.put(README, [17, 19, 51, 55, 67, 75, 97], "claim", ["P-ANIM-01"])
D.put(README, 23, "editorial", ED_HEADER); D.put(README, 25, "data_row", ["P-GPU-01"])
D.put(README, 26, "restatement", ["D-VIS07-01"]); D.put(README, 27, "restatement", ["G-R09-01"])
D.put(README, 28, "restatement", ["D-GLOB05-01"]); D.put(README, 29, "restatement", ["H05-01", "D-DEV04-01", "D-VIS07-02"])
D.put(README, 31, "limit", ["P-ARCH-01", "D-VIS07-01", "G-R09-01", "D-GLOB05-01"]); D.put(README, 37, "limit", ["P-ARCH-01"])
D.put(README, 43, "limit", ["H05-01"]); D.put(README, 57, "restatement", ["G-R08-01", "G-R09-01"])
D.put(README, 59, "limit", ["G-R08-01", "G-R11-01"]); D.put(README, 63, "method", ["D-VIS07-01"])
D.put(README, 69, "limit", ["D-VIS07-01"]); D.put(README, 77, "restatement", ["D-VIS07-02", "D-VIS07-03"])
D.put(README, 83, "claim", ["P-ANIM-01"]); D.put(README, 85, "restatement", ["D-SCN06-01"]); D.put(README, 89, "claim", ["P-VIEW-01"])
D.put(README, 91, "restatement", ["G-R11-02", "G-R11-01"]); D.put(README, [93, 94, 95], "editorial", ED_LINKS)
D.put(README, 96, "claim", ["P-AVAIL-01"]); D.put(README, 98, "provenance", ["P-AVAIL-01"]); D.put(README, 100, "provenance", ["P-GPU-01"])
D.put(README, 104, "editorial", ED_LICENSE)

D.put(MS, 3, "limit", ["D-VIS07-01"]); D.put(MS, 7, "restatement", ["P-GPU-01", "D-VIS07-01", "G-R09-01", "G-R10-01"])
D.put(MS, 11, "limit", ["G-R09-01"]); D.put(MS, 13, "limit", ["P-ARCH-01"]); D.put(MS, [17, 19, 21], "method", ["P-ARCH-01"])
D.put(MS, 23, "claim", ["P-ARCH-01"]); D.put(MS, 25, "claim", ["P-GPU-01"]); D.put(MS, 29, "restatement", ["H05-01"])
D.put(MS, 31, "restatement", ["D-DEV04-01"]); D.put(MS, 33, "restatement", ["D-GLOB05-01", "D-GLOB05-02"])
D.put(MS, [37, 39], "method", ["D-VIS07-01"]); D.put(MS, 41, "editorial", ED_HEADER)
D.put(MS, [43, 44, 45, 46, 47, 49], "restatement", ["D-VIS07-01"]); D.put(MS, 53, "restatement", ["D-VIS07-02"])
D.put(MS, 55, "restatement", ["D-VIS07-03"]); D.put(MS, 57, "restatement", ["D-SCN06-01", "D-SCN06-02"])
D.put(MS, 61, "method", ["G-R08-01", "G-R09-01", "G-R10-01"]); D.put(MS, 63, "restatement", ["G-R08-01", "G-R09-01"])
D.put(MS, 65, "restatement", ["G-R10-01", "G-R10-02"]); D.put(MS, 67, "claim", ["P-COST-01"]); D.put(MS, 69, "limit", ["G-R11-01"])
D.put(MS, 73, "claim", ["P-AVAIL-01", "P-ANIM-01"]); D.put(MS, 75, "governance", GOV); D.put(MS, 77, "editorial", ED_LICENSE)
D.put(MS, 79, "restatement", ["P-GPU-01", "D-VIS07-01", "G-R09-01"])
D.put(MS, 83, "restatement", ["G-R10-01", "G-R10-02", "P-VIEW-01", "G-R11-02"])

D.put(EV, 3, "editorial", ED_BYLINE); D.put(EV, 7, "restatement", ["D-VIS07-01", "G-R10-01"])
D.put(EV, 11, "method", ["D-VIS07-01", "G-R10-01"]); D.put(EV, [15, 25, 39], "editorial", ED_HEADER)
D.put(EV, range(17, 22), "restatement", ["D-VIS07-01"]); D.put(EV, range(27, 34), "restatement", ["G-R10-01"])
D.put(EV, 37, "restatement", ["G-R11-02", "G-R11-01"]); D.put(EV, range(41, 76), "restatement", ["G-R11-01"])
D.put(EV, 77, "limit", ["G-R11-01"])
D.put(EV, 81, "restatement", ["H05-01", "H06-01", "D-DEV04-01", "D-VIS07-02", "HT-12", "HT-14", "D-SCN06-01"])
D.put(EV, 85, "claim", ["P-MATH-01"]); D.put(EV, [89, 98], "governance", GOV); D.put(EV, range(93, 97), "editorial", ED_POINTER)

D.put(G0D, [3, 5], "claim", ["P-G0-01"]); D.put(G0D, 7, "claim", ["P-G0-02"])
D.put(G0D, 9, "restatement", ["D-VIS07-01", "D-VIS07-03", "HC-01"]); D.put(G0D, 11, "editorial", ED_POINTER)

D.put(REP, 3, "method", ["P-AVAIL-01"]); D.put(REP, 4, "governance", GOV); D.put(REP, 5, "method", ["D-VIS07-03", "P-AVAIL-01"])
D.put(REP, 6, "claim", ["P-ARCH-01"]); D.put(REP, 7, "claim", ["P-ANIM-01"])
D.put(REP, 9, "restatement", ["G-R10-01", "G-R10-02", "P-GPU-01"]); D.put(REP, 11, "claim", ["P-AVAIL-01"])
D.put(REP, 13, "limit", ["P-AVAIL-01", "D-VIS07-01"]); D.put(REP, 17, "restatement", ["G-R10-01", "P-VIEW-01", "G-R11-02"])
D.put(REP, [21, 27], "restatement", ["G-R11-02"]); D.put(REP, 24, "editorial", ED_POINTER)

D.put(IDX, 3, "editorial", ED_HEADER); D.put(IDX, 5, "restatement", ["HT-01"]); D.put(IDX, 6, "restatement", ["D-READ03-01", "D-READ03-02"])
D.put(IDX, 7, "restatement", ["D-DEV04-02"]); D.put(IDX, 8, "restatement", ["D-GLOB05-02"])
D.put(IDX, 9, "restatement", ["D-SCN06-01", "D-SCN06-02"]); D.put(IDX, 10, "restatement", ["D-VIS07-01", "D-VIS07-03"])
D.put(IDX, 11, "claim", ["P-ANIM-01"]); D.put(IDX, 12, "restatement", ["G-R08-01"]); D.put(IDX, 13, "restatement", ["G-R09-01"])
D.put(IDX, 14, "restatement", ["G-R10-01", "G-R10-02", "P-GPU-01"]); D.put(IDX, 15, "restatement", ["G-R11-02"])
D.put(IDX, 17, "claim", ["P-AVAIL-01"]); D.put(IDX, 21, "restatement", ["G-R10-01", "P-VIEW-01", "G-R11-02"])

OVERSTATED = ("Availability is overstated: the item-5 CPU archive, 05_analysis.json, 05_independent_check.json and the historical "
              "reassessment outputs are unavailable, and several dependencies existed only outside the tree until A18.2.H/D imports.")
DISCREPANCIES = [
    dict(source=README, line=96, claim="P-AVAIL-01", report_text="Frozen plans, original checkpoints, datasets, raw predictions/states ... under results/research/", finding=OVERSTATED),
    dict(source=MS, line=73, claim="P-AVAIL-01", report_text="Frozen plans, source commits, weights, datasets ... are supplied", finding=OVERSTATED),
    dict(source=IDX, line=17, claim="P-AVAIL-01", report_text="All raw archives and their original Git blobs are included/indexed", finding=OVERSTATED),
    dict(source=REP, line=11, claim="P-AVAIL-01", report_text="The publication manifest maps all raw archives", finding=OVERSTATED),
    dict(source=README, line=57, claim="G-R08-01", report_text="3.14-12.25x slower",
         finding="Receipt medians give 3.138-12.244x (C48/grid8: 35.0129/2.8595 ms); 12.25 should read 12.24. Same in the manuscript line 63."),
    dict(source=REP, line=17, claim="G-R11-02", report_text="RENDER11 retained100/105 original energy blocks",
         finding="Stale: the original receipt stores 103 of 105 rows (G-R11-02; same document line 21 says 103)."),
    dict(source=README, line=37, claim="P-ARCH-01", report_text="pre-scientific capability figure",
         finding="README still embeds pre-scientific capability (memory, rest-state, growing canvases) and scaling figures; captions "
                 "qualify them, but A18.3 must decide whether the figures exceed the evidence level."),
]

if __name__ == "__main__":
    sys.exit(run_review("A18.2.P", CLAIMS, D, DISCREPANCIES, "A18.2.P.json",
                        "Semantic review of batch A18.2.P (README, manuscripts, new-phase documents); restatements are "
                        "linked to claims registered in batches H, D and G and re-checked.",
                        recounts=RECOUNTS, external_ids=EXTERNAL, restatements=RESTATE, unsourced=UNSOURCED))
