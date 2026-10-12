"""A18.2.D: semantic review of the development, scanner and vision claim candidates
(DEV04, GLOB05, OPT03-D/MODE/ROLE/QTRAIN/QTRAIN-U16, READ03, SCN06, VIS07)."""
from __future__ import annotations

import sys

from completion_claim_review_lib import (ED_FIGURE, ED_HEADER, ED_LINKS, ED_POINTER, GOV, NO_EXT, Decisions,
                                         claim, run_review)

R = "docs/research/"
DEVPOP = ("Internal development RoleTask universe (partitions/policies 101-103, body initializations 400-411), "
          "four queries per context, school 0.3, recipe selected after READ03; the 264 original test compositions excluded")
OPTPOP = ("Historical NeuroPixel backbone (29,824 parameters, T16) on exposed RoleTask development pools "
          "(probe 2,048 / validation 4,096 examples); three initializations per condition, one split")
DEV = "Internal development result on an exposed universe; not a fair new comparison against the Transformer and no final test"
N3 = "Replication unit is the initialization (n=3 per condition); exploratory t95 df2 intervals, unadjusted, untruncated"
NOH1 = "Original H1 remains not supported under its recipe; this result does not reinterpret it"
NOCOST = "Update timers exclude other operations; complete cost and physical energy not measured"

DEVR = "results/research/DEV04_review/37845945011/"
GLR = "results/research/GLOB05_review/37887762153/"
ODR = "results/research/OPT03_root_review/37696760073/"
QTR = "results/research/OPT03_QTRAIN_review/37728384380/"
U16 = "results/research/OPT03_QTRAIN_U16_review/37771588722/"
R3R = "results/research/READ03_review/37831051538/"
QC = "results/research/OPT03_QTRAIN_recovery/37728384380/cohort/QTRAIN_paired_school0.3_seed200"

CLAIMS = [
    claim("D-DEV04-01", "DEV04 reproduced the READ03-selected recipe on 12 new bodies with 24 readouts: query attention "
          "minus the local control gave +16.244 pp mean validation accuracy (exploratory t95 interval 10.586 to 21.902 pp, "
          "half-width 5.658 pp); the competence gate passed (12/12 candidates with probe >=95% and validation >=90%) but "
          "the registered precision target (half-width <=5 pp with a positive lower bound) was not met.",
          DEVPOP, "experiment_complete", 2,
          [DEV, NOH1, NO_EXT, "Three policy means with df=2; policies reuse one development semantic universe and overlap",
           "Both readouts have 3,168 nominal active parameters but different functional spaces and FLOPs; shortcuts not excluded",
           "Attention over visible state has prior art; a gain does not demonstrate exceptional originality", NOCOST],
          R + "DEV04_execution_plan.json", [DEVR + "closure_receipt.json", DEVR + "final_receipts_verification.json"],
          [("16.244", DEVR), ("10.586", DEVR), ("21.902", DEVR), ("5.658", DEVR), ("15.988", DEVR), ("18.639", DEVR),
           ("14.105", DEVR), ("24.524", "results/research/DEV04_review"), ("32.257", "results/research/DEV04_review"),
           ("99.780", "results/research/DEV04_review"), ("67.261", "results/research/DEV04_review")], study="DEV04"),
    claim("D-DEV04-02", "DEV04 provenance and audit: original run 37845945011 (nine successes and three transport failures "
          "before training), operational recovery 37871777801 retraining only the three failures and internal endpoint "
          "verification 37876687599; an independent replay matched all 9,732,096 individual decisions exactly with mean NLL "
          "within 1e-4, and an orthogonal integer-denominator recount passed 41 checks without discrepancy.",
          DEVPOP, "engineering_complete", 1,
          ["Internal verification within the project, not external replication", "9,732,096 decisions are not experimental subjects", NO_EXT],
          None, [DEVR + "closure_receipt.json", DEVR + "final_receipts_verification.json"],
          [("9732096", DEVR + "closure_receipt.json"), ("41", DEVR), ("text:2484f3c5917e474ca3755bcdbbaca7d322077615ba47d51653116042d980f20a", DEVR + "closure_receipt.json")],
          study="DEV04"),
    claim("D-GLOB05-01", "GLOB05 trained 12 new global heads on the 12 frozen DEV04 bodies: query attention minus "
          "query-conditioned uniform global aggregation gave +6.278 pp (exploratory t95 4.108 to 8.449 pp, half-width "
          "2.171 pp), and the active global control minus the original local control gave +9.966 pp (5.868 to 14.063 pp, "
          "descriptive); the global control failed the joint gate on 8 of the 12 bodies.",
          DEVPOP + "; 12 frozen DEV04 bodies", "experiment_complete", 2,
          [DEV, NOH1, NO_EXT, "Not an identified causal mediation: aggregation, gating and readout functions differ",
           "The DEV04 attention-local precision failure (half-width 5.658 > 5 pp) is preserved, not repaired; intervals not combined",
           "Masks, examples, heads and stages are not independent replicates"],
          R + "GLOB05_execution_plan.json", [GLR + "closure_receipt.json", GLR + "final_receipts_verification.json"],
          [("6.278", GLR), ("4.108", GLR), ("8.449", GLR), ("2.171", GLR), ("9.966", GLR), ("5.868", GLR), ("14.063", GLR),
           ("86.072", "results/research/GLOB05_review"), ("98.767", "results/research/GLOB05_review"),
           ("text:\"global_gate_all_twelve\": false", GLR + "closure_receipt.json")], study="GLOB05"),
    claim("D-GLOB05-02", "GLOB05 independent replay verified 3,538,944 exact decisions in 3,120 checks with zero incidents "
          "and maximum mean-NLL error 0, and an orthogonal statistical recount passed 52 checks.",
          DEVPOP, "engineering_complete", 1, ["Internal verification, not external replication", NO_EXT], None,
          [GLR + "closure_receipt.json", GLR + "final_receipts_verification.json"],
          [("3538944", GLR + "closure_receipt.json"), ("3120", GLR + "closure_receipt.json"), ("52", GLR + "closure_receipt.json")],
          study="GLOB05"),
    claim("D-SCN06-01", "In the SCN06 assigned-control experiment (canonical run 37896280187), two assigned states with "
          "identical current scanner outputs changed the next answer only under the assigned hidden-to-visible coupling, "
          "while the inactive-coupling control stayed unchanged; top-token/confidence views also compressed distinct logit "
          "vectors; 13 native and 14 independent arithmetic checks passed. The scanner therefore shows a useful but "
          "incomplete view of the artificial state.",
          "Assigned (hand-set) NeuroPixel states and couplings; scanner outputs versus next-step answers", "experiment_complete", 2,
          ["Assigned counterexamples prove instrument limits; they do not discover a learned semantic circuit",
           "A redundant dispatch (run 37896335193) trained nothing and produced identical NPZ bytes: one experiment, not two replications",
           "PAD -10000 is finite suppression, not absolute exclusion", "Not a biological brain scan or complete causal explanation", NO_EXT],
          R + "SCN06_execution_plan.json",
          ["results/research/SCN06_cloud/37896280187/receipt.json", "results/research/SCN06_review/recovery_receipt.json"],
          [("13", "results/research/SCN06_review"), ("14", "results/research/SCN06_review")], study="SCN06"),
    claim("D-SCN06-02", "A separately frozen matrix census of five authenticated old learned models passed 75 checks: all "
          "read/non-PAD decoder matrices had rank 16 and ambient nullity 32 at the declared float64 threshold.",
          "Five authenticated historical learned NeuroPixel models; decoder matrices", "experiment_complete", 2,
          ["Measures ambient instantaneous projections, not reached-state dimensionality or learned semantic directions",
           "Multi-time observability is a different question", NO_EXT],
          R + "SCN06_rank_execution_plan.json", ["results/research/SCN06_rank_census/receipt.json"],
          [("75", "results/research/SCN06_rank_census/receipt.json"), ("16", "results/research/SCN06_rank_census/receipt.json"),
           ("32", "results/research/SCN06_rank_census/receipt.json")], study="SCN06"),
    claim("D-VIS07-01", "On the public UCI optical handwritten-digits test set (1,797 images, writer-separated; selection on "
          "DEV300 only before TEST), five NeuroPixel NCA realizations averaged 94.357% accuracy versus 90.840% for a CNN of "
          "close nominal size (5,056 versus 5,039 parameters): paired difference +3.517 pp, exploratory t95 interval 1.930 "
          "to 5.103 pp; all five NCA realizations exceeded 90%.",
          "UCI optdigits public data, writer-separated test set of 1,797 images; five paired training seeds 240-244",
          "experiment_complete", 2,
          ["Interval over five trainings on the same population; no blind custody and no independent replication",
           "TEST was opened by the project itself, not by an external custodian",
           "Depth, geometry and FLOPs differ despite close nominal parameter counts", "Not universal superiority over CNNs",
           "Does not rescue the binding H1", NO_EXT],
          R + "VIS07_execution_plan.json", ["results/research/VIS07MAIN_cloud/37908785698/receipt.json"],
          [("recount:vis07_nca_mean", "94.357"), ("recount:vis07_cnn_mean", "90.840"),
           ("recount:vis07_diff_mean", "3.517"), ("recount:vis07_diff_lo", "1.930"),
           ("recount:vis07_diff_hi", "5.103"), ("1715", "results/research/VIS07MAIN_cloud"),
           ("1614", "results/research/VIS07MAIN_cloud"), ("1697", "results/research/VIS07MAIN_cloud"),
           ("5056", "results/research/VIS07MAIN_cloud"), ("5039", "results/research/VIS07MAIN_cloud")], study="VIS07"),
    claim("D-VIS07-02", "With trained VIS07 weights, retaining 0, 4, 8, 16 or 32 updates gave 22.582%, 78.219%, 94.357%, "
          "53.100% and 16.661% mean test accuracy over the five seeds; in the retained 128-row depth diagnostic the mean "
          "state RMS grew from 0.543 at T0 to 149.3 at T16 and 134,843 at T32: extending beyond the trained T8 strongly "
          "degrades accuracy and magnitude.",
          "The ten VIS07 models evaluated at alternative update depths on the same test images", "experiment_complete", 2,
          ["Finite diagnostic endpoints, not proof of infinite divergence",
           "The report RMS column (0.540034 ... 138914) is not reproducible from retained receipts; the retained diagnostic covers the first 128 official test rows", "No arbitrary stability or durable memory shown", NO_EXT],
          R + "VIS07_execution_plan.json", ["results/research/VIS07_depth_review/37913653666/receipt.json"],
          [("recount:vis07_depth0", "22.582"), ("recount:vis07_depth4", "78.219"), ("recount:vis07_depth8", "94.357"),
           ("recount:vis07_depth16", "53.100"), ("recount:vis07_depth32", "16.661"), ("recount:vis07_rms0", "0.543"),
           ("recount:vis07_rms16", "149.3"), ("recount:vis07_rms32", "134843")],
          study="VIS07"),
    claim("D-VIS07-03", "VIS07 functional replay preserved 98,835 exact decisions and passed logits/NLL for the 30 "
          "base/transformation conditions, but ten continuous T16/T32 comparisons failed the frozen tolerance although "
          "their decisions matched; the failed receipt is preserved and the tolerance was not raised.",
          "VIS07 checkpoints replayed under base and transformation conditions", "engineering_complete", 1,
          ["Replay is not declared fully successful", "Same-host path diagnostic is separate", NO_EXT], None,
          ["results/research/VIS07REPLAY_cloud/37911162172/receipt.json"],
          [("98835", "results/research/VIS07REPLAY_cloud")], study="VIS07"),
    claim("D-VIS07-04", "VIS07 saved-file recount passed 110 checks with zero discrepancies over ten new models trained for "
          "40 epochs (2,240 updates each); recorded training+DEV wall time was 447.555 s and CPU process time 894.837 s.",
          "VIS07 main cohort", "engineering_complete", 1, [NOCOST, NO_EXT], None,
          ["results/research/VIS07MAIN_cloud/37908785698/receipt.json"],
          [("recount:vis07_wall", "447.555"), ("recount:vis07_cpu", "894.837"),
           ("2240", "results/research/VIS07MAIN_cloud")], study="VIS07"),
    claim("D-OPT03D-01", "OPT03-D (12 new trainings: seeds 100-102 x effective PAD zero yes/no x firing 0.5/1.0, 8,192 updates, "
          "school 0.3): no condition passed the frozen gate (95% probe and 90% validation in three seeds); mean validation "
          "binding was 23.568%, 16.732%, 28.662% and 26.221%, and the factorial contrasts PAD zero minus legacy +7.292 pp, "
          "firing 1.0 minus 0.5 -4.639 pp and their interaction +4.395 pp all had paired 95% intervals (df=2) including zero.",
          OPTPOP + "; seeds 100-102", "experiment_complete", 2,
          [N3, NOH1, NO_EXT, "Negative development result: neither PAD projection nor higher firing resolved low competence",
           "PAD also removes 16 effective degrees of freedom; firing changes randomness and expected dose"],
          R + "OPT03_execution_plan.json",
          [ODR + "analysis_executed_plan.json", ODR + "development_phase_receipt.json",
           "results/research/OPT03_cloud_recovery/37696760073/original.zip"],
          [("23.568", ODR), ("16.732", ODR), ("28.662", ODR), ("26.221", ODR), ("7.292", ODR), ("-4.639", ODR), ("4.395", ODR),
           ("-11.985", ODR), ("26.568", ODR), ("23.763", ODR), ("30.013", ODR)], study="OPT03_D"),
    claim("D-OPT03D-02", "OPT03-D provenance: the local plan (sha256 bf54be3c..., 141 CRLF line endings) equals the executed "
          "plan (sha256 f1caada1..., LF) byte for byte after the single CRLF-to-LF conversion; a complementary audit passed "
          "860 checks and recounted final metrics from the twelve final checkpoints (73,728 replay predictions); minimum "
          "observed RAM was 14.2327 GiB and recorded training times sum to 14,763.44 s.",
          OPTPOP, "engineering_complete", 1,
          ["Per-example original decisions and intermediate endpoint weights were not archived: no example-level equivalence is claimed",
           "Training times are not complete cost or physical energy", NO_EXT], None,
          [ODR + "plan_byte_provenance_check.json", ODR + "checkpoint_audit_executed_plan.json"],
          [("860", ODR), ("calc:73728=12*(2048+4096)", ODR), ("text:f1caada1af01d0078877b55800e3e7928e7bc8832993b8c189e0ffae2cdadb67", ODR + "plan_byte_provenance_check.json"),
           ("text:bf54be3c4c8e30885b330e67e92195622e28c85a49e317358d106699d40008df", ODR + "plan_byte_provenance_check.json")],
          study="OPT03_D"),
    claim("D-MODE-01", "OPT03-MODE (12 existing OPT03-D checkpoints, no new training): evaluating with firing 0.5 instead of "
          "full updates changed validation-subset binding by +11.100 pp for models trained at 0.5 and -5.200 pp for models "
          "trained at 1.0; the train-firing x evaluation-mode interaction was +16.300 pp (t95 df2 -4.107 to +36.708 pp); the "
          "preregistered descriptive screen (>=5 pp, three positive signs) passed on both panels while the intervals include "
          "zero, and the best subset value, 53.711%, stays far below the 90% gate.",
          OPTPOP + "; first 128 examples per role of each development pool", "experiment_complete", 2,
          [N3, "Eight masks are not new initializations; rollouts are averaged, not ensembled or selected",
           "Does not identify a single cause of insufficient learning; data subset historically exposed", NO_EXT],
          R + "OPT03_mode_plan.json", ["results/research/OPT03_modes_review/audit.json"],
          [("calc:11.100~0.001=mean([20.361-17.969, 53.711-27.344, 31.982-23.047, 35.107-26.953, 39.844-27.734, 41.846-33.203])", "results/research/OPT03_modes"),
           ("calc:-5.200~0.001=mean([8.398-14.844, 6.689-10.156, 13.672-17.188, 9.570-10.156, 14.844-23.047, 25.391-34.375])", "results/research/OPT03_modes"),
           ("16.300", "results/research/OPT03_modes_review/audit.json"), ("-4.107", "results/research/OPT03_modes_review/audit.json"),
           ("36.708", "results/research/OPT03_modes_review/audit.json"), ("8.789", "results/research/OPT03_modes_review/audit.json"),
           ("15.592", "results/research/OPT03_modes_review/audit.json"), ("53.711", "results/research/OPT03_modes")],
          study="OPT03_MODE"),
    claim("D-MODE-02", "The OPT03-MODE audit independently recounted 110,592 decisions and NLL values from the saved arrays in "
          "1,052 checks with no incidents; checkpoints were hash-verified before and after and no optimizer or gradient was created.",
          OPTPOP, "engineering_complete", 1, ["Internal audit, not external replication", NO_EXT], None,
          ["results/research/OPT03_modes_review/audit.json"],
          [("110592", "results/research/OPT03_modes_review/audit.json"), ("1052", "results/research/OPT03_modes_review/audit.json")],
          study="OPT03_MODE"),
    claim("D-ROLE-01", "OPT03-ROLE (12 existing weights, no training): joint accuracy on the original and the AGENT<->PATIENT "
          "flipped query over the same context stayed low - 7.503%, 0.521%, 2.572% and 2.344% for the four PAD/firing "
          "conditions - while 13.2-63.7% of predictions changed under the flip and 29.6-80.5% named one of the two candidate "
          "nouns; relocating whole pairs kept 49.9-82.3% of nominal predictions unchanged.",
          OPTPOP + "; validation subset of 512 examples (128 per role)", "experiment_complete", 2,
          [N3, "Query sensitivity alone is not correct role assignment; no single cause attributed to all models",
           "Measures geometry dependence of these weights and subsets; does not separate key reading, propagation or noise", NO_EXT],
          R + "OPT03_role_plan.json", ["results/research/OPT03_roles_review/audit.json"],
          [("7.503", "results/research/OPT03_roles/summary.json"), ("0.521", "results/research/OPT03_roles_review/audit.json"),
           ("2.572", "results/research/OPT03_roles/summary.json"), ("2.344", "results/research/OPT03_roles/summary.json"),
           ("13.216", "results/research/OPT03_roles/summary.json"), ("63.672", "results/research/OPT03_roles/summary.json"),
           ("29.557", "results/research/OPT03_roles/summary.json"), ("80.501", "results/research/OPT03_roles/summary.json"),
           ("49.886", "results/research/OPT03_roles/summary.json"), ("82.292", "results/research/OPT03_roles/summary.json")],
          study="OPT03_ROLE"),
    claim("D-ROLE-02", "OPT03-ROLE stopped by RAM after 11 checkpoints; a partial audit recounted 1,050 checks/162,816 "
          "predictions, the resume evaluated only the pending checkpoint with the operational start margin lowered from 8.75 "
          "to 8.5 GiB while keeping the 8 GiB scientific minimum, and the full-cohort audit "
          "passed 1,384 checks with zero incidents over 165,888 newly recounted predictions.",
          OPTPOP, "engineering_complete", 1, ["Internal audit; incident, first refusal and resume transcript preserved", NO_EXT],
          None, ["results/research/OPT03_roles_review/audit.json"],
          [("1384", "results/research/OPT03_roles_review/audit.json"), ("165888", "results/research/OPT03_roles_review/audit.json"),
           ("162816", "results/research/OPT03_roles_review")], study="OPT03_ROLE"),
    claim("D-QT-01", "OPT03-QTRAIN (12 new trainings: seeds 200-202 x four queries per context versus one repeated query x "
          "school 0/0.3, 8,192 updates): no condition met the frozen joint-query competence criterion; mean joint validation "
          "accuracy was 1.428%, 28.225%, 0.138% and 0.220%, with four queries/school 0.3 reaching 74.677% on seed 200 but "
          "2.466% and 7.532% on seeds 201/202; the three prospective contrasts (queries +14.647 pp, school +13.439 pp, "
          "interaction +26.715 pp) were positive on all seeds but their intervals include zero.",
          OPTPOP + "; seeds 200-202", "experiment_complete", 2,
          [N3, "Contrasts dominated by seed 200; no reproducible population benefit or equivalence", NOH1,
           "Equal updates do not mean equal FLOPs; school adds supervision and compute", NO_EXT],
          R + "OPT03_QTRAIN_plan.json", [QTR + "analysis/summary.json", QTR + "audit_full.json", QTR + "closure_receipt.json"],
          [("1.428", QTR), ("28.225", QTR), ("0.138", QTR), ("0.220", QTR), ("74.677", QTR), ("2.466", QTR), ("7.532", QTR),
           ("14.647", QTR), ("13.439", QTR), ("26.715", QTR), ("-35.416", QTR), ("64.711", QTR), ("76.599", QTR),
           ("86.865", QTR)], study="OPT03_QTRAIN"),
    claim("D-QT-02", "OPT03-QTRAIN verification: the full audit passed 4,338 checks with zero incidents over 1,622,016 "
          "recounted decisions; checkpoint replay at 1,024/4,096/8,192 updates reproduced all decisions exactly (1,932 checks, "
          "maximum mean-NLL error 0); a supplementary recount passed 28 checks at 1e-12, and the first analytic recount failure "
          "(critical-value difference up to 1.254e-11) is preserved.",
          OPTPOP, "engineering_complete", 1, ["Internal verification; a second executor and recount algorithm are not external replication", NO_EXT],
          None, [QTR + "audit_full.json", QTR + "final_receipts_verification.json", QTR + "statistics_initial_failure.json"],
          [("4338", QTR + "audit_full.json"), ("1622016", QTR + "audit_full.json"), ("1932", QTR), ("text:1.2535750215647568e-11", QTR + "statistics_initial_failure.json")],
          study="OPT03_QTRAIN"),
    claim("D-QT-03", "In OPT03-QTRAIN the mean answer loss fell between the last two 1,024-update windows in all twelve cases "
          "(four queries/school 0.3: seed 200 0.376 to 0.290), and seed 200's dense joint endpoint rose from 1.904% at 4,096 "
          "updates to 60.010% at 8,192.",
          OPTPOP, "experiment_complete", 2,
          ["Curves do not demonstrate convergence, a single cause or architectural incapacity; dense and matched modes kept separate", NO_EXT],
          R + "OPT03_QTRAIN_plan.json", [QTR + "analysis/summary.json", QC + "/progress.json"],
          [("recount:qtrain_s200_loss_prev", "0.376"), ("recount:qtrain_s200_loss_last", "0.290"), ("1.904", QC), ("60.010", QC)], study="OPT03_QTRAIN"),
    claim("D-U16-01", "OPT03-QTRAIN-U16 continued the 12 QTRAIN states from 8,192 to 16,384 updates (exact optimizer and RNG "
          "restoration): the additional budget improved the joint primary endpoint by +19.554 pp (exploratory paired t95 df2 "
          "14.862 to 24.247 pp), positive in all three seeds, yet no condition met the competence gate; four queries/school 0.3 "
          "rose from 28.225% to 84.674% joint validation (88.593/80.621/84.808%) with probes 88.086/80.469/84.607%, all below 95%.",
          OPTPOP + "; continued QTRAIN states, seeds 200-202", "experiment_complete", 2,
          [N3, "Budget jointly changes updates, example exposure and compute; not a pure optimizer improvement",
           "The favorable cell's 92.069% single-query accuracy is not compared as a win over the historical Transformer", NOH1, NO_EXT],
          R + "OPT03_QTRAIN_U16_plan.json", [U16 + "analysis/summary.json", U16 + "audit_full.json", U16 + "closure_receipt.json"],
          [("19.554", U16), ("14.862", U16), ("24.247", U16), ("84.674", U16), ("88.593", U16), ("80.621", U16), ("84.808", U16),
           ("18.361", U16), ("0.903", U16), ("4.289", U16), ("88.086", U16), ("80.469", U16), ("84.607", U16), ("calc:92.069~0.001=mean([94.287, 89.960, 91.962])", U16),
           ("80.176", U16), ("63.184", U16), ("0.151", U16), ("0.142", U16)], study="OPT03_QTRAIN_U16"),
    claim("D-U16-02", "Among the seven prospective OPT03-QTRAIN-U16 contrasts, school 0.3 minus 0 (+24.144 pp, t95 10.353 to "
          "37.934), queries x school (+44.821 pp, 23.666 to 65.976) and budget x queries (+34.274 pp, 13.967 to 54.581) "
          "excluded zero, while four minus one query (+31.785 pp, -8.128 to 71.697), budget x school (+21.410 pp) and the "
          "triple interaction (+36.212 pp) included zero.",
          OPTPOP + "; continued QTRAIN states", "experiment_complete", 2,
          [N3, "Seven unadjusted intervals are not seven independent confirmations; including zero is not absence of effect",
           "Tying, reinjection, recurrence, damage and growth were not intervened in this phase", NO_EXT],
          R + "OPT03_QTRAIN_U16_plan.json", [U16 + "analysis/summary.json"],
          [("24.144", U16), ("10.353", U16), ("37.934", U16), ("44.821", U16), ("23.666", U16), ("65.976", U16),
           ("34.274", U16), ("13.967", U16), ("54.581", U16), ("31.785", U16), ("-8.128", U16), ("71.697", U16),
           ("21.410", U16), ("36.212", U16), ("-125.007", U16), ("197.432", U16)], study="OPT03_QTRAIN_U16"),
    claim("D-U16-03", "OPT03-QTRAIN-U16 verification: audit 4,105 checks with zero incidents over 2,654,208 decisions; replay "
          "2,196 checks with all decisions exact and maximum mean-NLL error 0; independent statistical recount 60 checks at "
          "1e-12; minimum available RAM 14.172 GiB and timed windows summing to 16,464.282 s.",
          OPTPOP, "engineering_complete", 1, [NOCOST, "Internal verification, not external replication", NO_EXT], None,
          [U16 + "audit_full.json", U16 + "final_receipts_verification.json", U16 + "closure_receipt.json"],
          [("4105", U16 + "audit_full.json"), ("2654208", U16 + "audit_full.json"), ("2196", U16), ("14.172", U16),
           ("16464.282", U16)], study="OPT03_QTRAIN_U16"),
    claim("D-READ03-01", "READ03 fitted 36 new heads on 12 frozen U16 states: query attention minus the query-conditioned "
          "local control gave +54.225 pp (exploratory t95 37.575 to 70.875 pp), minus uniform global aggregation +55.070 pp "
          "and minus the parent readout +58.464 pp, while local minus parent was +4.239 pp and uniform minus parent "
          "+3.394 pp; only four-queries/school-0.3 attention passed the joint gate in all three seeds.",
          OPTPOP + "; 12 frozen U16 states, bodies/dictionaries/decoder frozen", "experiment_complete", 2,
          [N3, NOH1, "Head-versus-parent comparisons combine reading capacity and extra labels; not a pure attention effect",
           "No equivalence margin was registered; no final test opened; not a confirmatory winner", NOCOST, NO_EXT],
          R + "READ03_execution_plan.json", [R3R + "analysis/summary.json", R3R + "closure_receipt.json"],
          [("54.225", R3R), ("37.575", R3R), ("70.875", R3R), ("55.070", R3R), ("58.464", R3R), ("4.239", R3R),
           ("3.394", R3R), ("48.218", R3R), ("61.455", R3R), ("99.762", R3R), ("99.835", R3R)], study="READ03"),
    claim("D-READ03-02", "READ03 replay reproduced the 1,024/4,096/8,192 endpoints and recounted predictions and metrics with "
          "exact decisions and unchanged 1e-4 NLL tolerance; an independent statistical recount passed 27 checks with zero "
          "discrepancies, and the initial verifier packaging and Windows-path failures are preserved.",
          OPTPOP, "engineering_complete", 1, ["Internal verification, not external scientific replication", NO_EXT], None,
          [R3R + "closure_receipt.json", R3R + "final_receipts_verification.json"],
          [("27", R3R), ("text:d273b8284d9a387b49266dc4ec62cfad76aefaa11c5f82c7ab6a97ba0e431ea0", R3R + "closure_receipt.json")],
          study="READ03"),
]

D = Decisions()
F = R + "DEV04_results.md"
D.put(F, [3, 5], "claim", ["D-DEV04-01"]); D.put(F, [9, 26], "editorial", ED_HEADER)
D.put(F, range(11, 23), "data_row", ["D-DEV04-01"]); D.put(F, [28, 29, 30], "data_row", ["D-DEV04-01"])
D.put(F, 32, "method", ["D-DEV04-01"]); D.put(F, 36, "provenance", ["D-DEV04-02"]); D.put(F, [38, 40], "claim", ["D-DEV04-02"])
D.put(F, [44, 46], "limit", ["D-DEV04-01"]); D.put(F, 48, "limit", ["D-DEV04-01", "D-DEV04-02"]); D.put(F, 50, "governance", GOV)
D.put(F, 54, "editorial", ED_POINTER); D.put(F, 56, "editorial", ED_LINKS)
F = R + "GLOB05_results.md"
D.put(F, [3, 5, 7, 30], "claim", ["D-GLOB05-01"]); D.put(F, 11, "editorial", ED_HEADER)
D.put(F, range(13, 25), "data_row", ["D-GLOB05-01"]); D.put(F, [28, 32], "method", ["D-GLOB05-01"])
D.put(F, 36, "provenance", ["D-GLOB05-02"]); D.put(F, [38, 40], "claim", ["D-GLOB05-02"]); D.put(F, 42, "limit", ["D-GLOB05-01"])
F = R + "SCN06_results.md"
D.put(F, 3, "claim", ["D-SCN06-01"]); D.put(F, 5, "limit", ["D-SCN06-01"]); D.put(F, 7, "claim", ["D-SCN06-02"])
D.put(F, 9, "claim", ["D-SCN06-01", "D-SCN06-02"])
F = R + "VIS07_results.md"
D.put(F, 3, "claim", ["D-VIS07-01"]); D.put(F, [5, 17], "editorial", ED_HEADER); D.put(F, range(7, 12), "data_row", ["D-VIS07-01"])
D.put(F, 13, "limit", ["D-VIS07-01"]); D.put(F, range(19, 24), "data_row", ["D-VIS07-02"]); D.put(F, 25, "claim", ["D-VIS07-02"])
D.put(F, 27, "claim", ["D-VIS07-03"]); D.put(F, 29, "claim", ["D-VIS07-04"]); D.put(F, 31, "provenance", ["D-VIS07-01"])
F = R + "OPT03_D_results.md"
D.put(F, [3, 28, 46], "claim", ["D-OPT03D-01"]); D.put(F, [7, 19], "method", ["D-OPT03D-01"])
D.put(F, [9, 13, 40], "claim", ["D-OPT03D-02"]); D.put(F, 15, "provenance", ["D-OPT03D-02"]); D.put(F, [21, 30], "editorial", ED_HEADER)
D.put(F, [23, 24, 25, 26, 32, 33, 34], "data_row", ["D-OPT03D-01"]); D.put(F, [36, 48], "limit", ["D-OPT03D-01"])
D.put(F, 42, "limit", ["D-OPT03D-02"]); D.put(F, 50, "governance", GOV); D.put(F, 52, "editorial", ED_POINTER)
F = R + "OPT03_MODE_results.md"
D.put(F, [3, 5], "method", ["D-MODE-01"]); D.put(F, [9, 17], "editorial", ED_HEADER)
D.put(F, [11, 12, 13, 19, 20, 21, 22], "data_row", ["D-MODE-01"]); D.put(F, [15, 30], "claim", ["D-MODE-01"])
D.put(F, 24, "limit", ["D-MODE-01"]); D.put(F, 28, "claim", ["D-MODE-02"]); D.put(F, 32, "governance", GOV)
D.put(F, 34, "provenance", ["D-MODE-01"])
F = R + "OPT03_ROLE_results.md"
D.put(F, [3, 5, 9, 11], "method", ["D-ROLE-01"]); D.put(F, 13, "editorial", ED_HEADER)
D.put(F, [15, 16, 17, 18], "data_row", ["D-ROLE-01"]); D.put(F, [20, 24, 38], "claim", ["D-ROLE-01"])
D.put(F, 26, "limit", ["D-ROLE-01"]); D.put(F, [30, 34], "claim", ["D-ROLE-02"]); D.put(F, 32, "provenance", ["D-ROLE-02"])
D.put(F, 40, "governance", GOV); D.put(F, 42, "provenance", ["D-ROLE-01"])
F = R + "OPT03_QTRAIN_results.md"
D.put(F, [3, 29, 31, 43, 71], "claim", ["D-QT-01"]); D.put(F, [7, 9, 11, 20, 35], "method", ["D-QT-01"])
D.put(F, [13, 14, 15, 16], "provenance", ["D-QT-02"]); D.put(F, [22, 37], "editorial", ED_HEADER)
D.put(F, [24, 25, 26, 27, 39, 40, 41], "data_row", ["D-QT-01"]); D.put(F, 45, "limit", ["D-QT-01"])
D.put(F, 49, "claim", ["D-QT-03"]); D.put(F, 51, "limit", ["D-QT-03"]); D.put(F, [55, 56, 57, 58], "claim", ["D-QT-02"])
D.put(F, 59, "limit", ["D-QT-02"]); D.put(F, [61, 64], "editorial", ED_POINTER); D.put(F, [67, 69], "editorial", ED_FIGURE)
F = R + "OPT03_QTRAIN_U16_results.md"
D.put(F, [3, 22, 24, 50, 52, 69], "claim", ["D-U16-01"]); D.put(F, [7, 9, 11], "method", ["D-U16-01"])
D.put(F, [15, 32], "editorial", ED_HEADER); D.put(F, [17, 18, 19, 20, 34], "data_row", ["D-U16-01"])
D.put(F, [26, 46, 56], "editorial", ED_FIGURE); D.put(F, 30, "method", ["D-U16-02"])
D.put(F, range(35, 41), "data_row", ["D-U16-02"]); D.put(F, 42, "claim", ["D-U16-02"]); D.put(F, 44, "limit", ["D-U16-02"])
D.put(F, [54, 63, 64], "claim", ["D-U16-03"]); D.put(F, [60, 61, 62, 65], "provenance", ["D-U16-03"])
D.put(F, 67, "editorial", ED_POINTER)
F = R + "READ03_results.md"
D.put(F, 3, "claim", ["D-READ03-01"]); D.put(F, [5, 40, 46], "limit", ["D-READ03-01"]); D.put(F, [9, 21], "editorial", ED_HEADER)
D.put(F, list(range(11, 16)) + list(range(23, 35)), "data_row", ["D-READ03-01"]); D.put(F, [17, 36], "method", ["D-READ03-01"])
D.put(F, [42, 54], "claim", ["D-READ03-02"]); D.put(F, [44, 50], "provenance", ["D-READ03-02"])
D.put(F, 52, "editorial", ED_POINTER); D.put(F, 56, "governance", GOV)


def vis07_recounts() -> dict:
    """Recount VIS07 summary values from the retained per-seed receipts; no model is executed."""
    import json
    import zipfile
    from completion_claim_review_lib import CALC_NAMES, ROOT
    with zipfile.ZipFile(ROOT / "results/research/VIS07MAIN_cloud/37908785698/raw.zip") as bundle:
        result = json.loads(bundle.read("result.json"))
    scores = result["scores"]

    def acc(family, condition):
        rows = sorted((r for r in scores if r["family"] == family and r["condition"] == condition), key=lambda r: r["seed"])
        assert len(rows) == 5 and all(r["n"] == 1797 for r in rows)
        return [100.0 * r["correct"] / r["n"] for r in rows]

    nca, cnn = acc("nca", "base"), acc("cnn", "base")
    diff = [a - b for a, b in zip(nca, cnn)]
    mean = CALC_NAMES["mean"]
    out = dict(vis07_nca_mean=mean(nca), vis07_cnn_mean=mean(cnn), vis07_diff_mean=mean(diff),
               vis07_diff_lo=CALC_NAMES["t95lo"](diff), vis07_diff_hi=CALC_NAMES["t95hi"](diff),
               vis07_wall=sum(r["wall_seconds"] for r in result["training_records"]),
               vis07_cpu=sum(r["cpu_seconds"] for r in result["training_records"]))
    for depth in (0, 4, 8, 16, 32):
        out[f"vis07_depth{depth}"] = mean(acc("nca", f"held_depth{depth}"))
    rows = json.loads((ROOT / "results/research/VIS07_depth_review/37913653666/receipt.json").read_text(encoding="utf-8"))["rows"]
    for depth in (0, 16, 32):
        values = [r["state_rms"] for r in rows if r.get("depth", 0) == depth]
        assert len(values) == 5
        out[f"vis07_rms{depth}"] = mean(values)
    assert len(result["training_records"]) == 10
    curve = json.loads((ROOT / "results/research/OPT03_QTRAIN_recovery/37728384380/cohort/"
                        "QTRAIN_paired_school0.3_seed200/progress.json").read_text(encoding="utf-8"))["curve"]
    prev = [w["answer_loss_mean"] for w in curve if 6144 < w["update"] <= 7168]
    last = [w["answer_loss_mean"] for w in curve if 7168 < w["update"] <= 8192]
    assert len(prev) == len(last) == 8
    out["qtrain_s200_loss_prev"], out["qtrain_s200_loss_last"] = mean(prev), mean(last)
    return out

DISCREPANCIES: list[dict] = [
    dict(source=R + "VIS07_results.md", line=19, claim="D-VIS07-02", report_text="RMS column 0.540034 / 1.83129 / 7.30792 / 150.423 / 138914",
         receipt="results/research/VIS07_depth_review/37913653666/receipt.json", receipt_field="rows[*].state_rms",
         finding="Not reproducible from retained receipts: the retained depth diagnostic (first 128 official test rows, five seeds) "
                 "gives means 0.543 / ... / 149.3 / 134,843. The report's RMS source is not retained; the ledger states the retained values."),
    dict(source=R + "OPT03_ROLE_results.md", line=32, claim="D-ROLE-02", report_text="admision real del reanudador registro 8,6749GiB",
         receipt=None, receipt_field=None, receipt_value=None,
         finding="The 8.6749 GiB admission value is not present in the retained ROLE review files; removed from the ledger statement."),
]

if __name__ == "__main__":
    sys.exit(run_review("A18.2.D", CLAIMS, D, DISCREPANCIES, "A18.2.D.json",
                        "Semantic review of batch A18.2.D; development results are internal and exploratory, "
                        "no level above L2 is justified without external custody or dated public commitment.",
                        recounts=vis07_recounts()))
