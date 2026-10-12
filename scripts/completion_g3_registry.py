"""G3 registry: separation unit, selector, test exposure and leakage review for every evaluation set in use.

Fields marked 'checked' are read from retained receipts by this script; the rest are declared
facts with the receipt that states them. No model is loaded and no TEST is opened.
"""
from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RR = ROOT / "results/research"
OUT = RR / "COMPLETION20261012/G3_split_registry.json"


def load(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def from_zip(rel: str, member: str) -> dict:
    with zipfile.ZipFile(ROOT / rel) as bundle:
        return json.loads(bundle.read(member))


vis_zip = "results/research/VIS07MAIN_cloud/37908785698/raw.zip"
gate = from_zip(vis_zip, "gate/selection_gate.json")
access = from_zip(vis_zip, "final_access.json")
source = from_zip(vis_zip, "source_record.json")
depth = load("results/research/VIS07_depth_review/37913653666/receipt.json")
learned = load("results/research/VIS07_learned_gpu/receipt.json")
newrel = load("results/research/NEWREL20261013/manifest.json")
newrel_audit = load("results/research/COMPLETION20261012/G3_newrel_family_audit.json")
kit_audit = load("results/research/COMPLETION20261012/G3_kit_generator_family_audit.json")
checks = dict(
    vis07_all_ten_sealed_before_test=gate["status"] == "all_ten_selected_models_sealed_before_official_test"
    and len(gate["selected_models"]) == 10,
    vis07_final_access_consumed_once=access["status"] == "consumed",
    vis07_original264_not_scored=source["original264_test_scored"] is False,
    dev04_test_not_scored=load("results/research/DEV04_review/37845945011/closure_receipt.json")["test_scored"] is False,
    glob05_original_test_not_scored=load("results/research/GLOB05_review/37887762153/closure_receipt.json")["original_test_scored"] is False,
    read03_test_not_accessed=load("results/research/READ03_review/37831051538/closure_receipt.json")["test_accessed"] is False,
    newrel_train_dev_family_overlap_zero=newrel_audit["overlaps"]["dev|train"] == 0 and newrel_audit["overlaps"]["dev_long|train"] == 0,
    kit_default_family_overlap_zero=all(v == 0 for k, v in kit_audit["panels"]["overlaps"].items() if k != "test_base|test_long"),
)
assert all(checks.values()), checks

TAXONOMY = ("Kapoor and Narayanan (2023), Patterns 4(9):100804: L1.1 no test set; L1.2 pre-processing on train and test; "
            "L1.3 feature selection on train and test; L1.4 duplicates; L2 illegitimate features; L3.1 temporal leakage; "
            "L3.2 non-independence between train and test; L3.3 sampling bias in the test distribution")
registry = dict(
    schema=1, created_utc=datetime.now(timezone.utc).isoformat(), leakage_taxonomy=TAXONOMY, checked=checks,
    selector=dict(rule="most DEV correct, then lowest DEV NLL, then earliest epoch; never TEST, never best seed",
                  implementation="scripts/completion_g3_splits.py:select_checkpoint", tests="tests/test_completion_g3_splits.py"),
    sealing=("All models of a cohort are digested (completion_kit.commitments digest) and the digest commitment is published "
             "before any new TEST is revealed; procedure and external timing are task G17.1"),
    exposure_rule="Every TEST access is logged here with its purpose; a revealed TEST is never used for selection or tuning (Dwork et al. 2015)",
    datasets=[
        dict(name="UCI optical digits (VIS07)", unit="writer (per UCI metadata; individual writer ids not independently verified)",
             splits="official TRAIN 3,823 -> 3,523 train + 300 DEV (balanced, TRAIN only); official TEST 1,797",
             custody="public corpus; no blind custody",
             test_accesses=[
                 dict(purpose="single registered scoring after all ten DEV-selected models were sealed",
                      at_utc=datetime.fromtimestamp(access["at_unix"], timezone.utc).isoformat(), receipt=vis_zip + ":final_access.json"),
                 dict(purpose="post-reveal depth diagnostic, no selection", subset=depth["subset"],
                      receipt="results/research/VIS07_depth_review/37913653666/receipt.json"),
                 dict(purpose="post-reveal GPU fidelity capture, no selection", subset=learned["subset"],
                      receipt="results/research/VIS07_learned_gpu/receipt.json"),
                 dict(purpose="post-reveal functional replay of base and transformation conditions, no selection",
                      receipt="results/research/VIS07REPLAY_cloud/37911162172/receipt.json")],
             leakage=dict(L1_1="no: official held-out TEST", L1_2="no: fixed scaling by 16, no fitted statistics",
                          L1_3="no feature selection", L1_4=("exact-image uniqueness and disjointness are asserted by the manuscript; "
                          "no retained receipt; re-verification needs the public UCI files (owner-approved download)"),
                          L2="no: pixel counts only; labels and class tokens are not inserted as input",
                          L3_1="not applicable", L3_2="writer separation documented by UCI metadata, not independently verified",
                          L3_3="TEST from the same collection: in-distribution only; no external-population claim"),
             status="historical exposed TEST; usable only as already-reported evidence; no new confirmatory claim"),
        dict(name="RoleTask original split (items 4-5)", unit="agent/verb/patient triple composition",
             splits="1,320 triples: 924 train / 132 validation / 264 final (protocol NP-SCI-20261006-v1)",
             custody="locally generated; no blind custody",
             test_accesses=[dict(purpose="item-4 pilot and item-5 final evaluations (historical)",
                                 receipt="docs/research/05_results.md (raw evidence partly unavailable; see A18.2.H.ext)")],
             leakage=dict(L1_4="triples disjoint within each split by construction; fillers and roles shared by design",
                          L3_2="compositional hold-out only; same generator", L3_3="single synthetic generator"),
             later_studies="the 264 final compositions stay excluded: VIS07, DEV04, GLOB05 and READ03 receipts record no scoring",
             note="Split disjointness was not re-executed here: it needs the Torch generator, deferred under the 8 GiB RAM guard"),
        dict(name="Development pools of OPT03/READ03/DEV04/GLOB05", unit="RoleTask partitions and seeds 77001/77002",
             custody="exposed development data reused across studies", test_accesses=[],
             leakage=dict(L3_2="reused, exposed pools: results are development-only and are registered as such"),
             status="never a TEST; no confirmatory use"),
        dict(name="NEWREL20261013 (ordered relation composition)", unit=newrel["separation_unit"],
             splits={k: v["families"] for k, v in newrel["panels"].items()}, manifest="results/research/NEWREL20261013/manifest.json",
             custody="train/DEV local; TEST to be generated by the designated evaluator with a committed seed (G17.1)",
             test_accesses=[],
             evaluator_spec=dict(generator="scripts/completion_g3_splits.py (family_id, rename) with completion_kit.datasets.encode/oracle",
                                 entities=12, relations=3, grid=8, depths="2-3 base and 4 long, paired",
                                 must_exclude=newrel["excluded_families"], seed="chosen and committed by the evaluator"),
             leakage=dict(L1_1="TEST pending evaluator", L1_4="family-level disjointness verified (0 overlaps)",
                          L2="no: answers computed by the symbolic oracle and decodable from the input alone; never encoded",
                          L3_2="structural-family unit removes renamed variants across splits",
                          L3_3="single synthetic generator; external validity requires other tasks (G5)")),
        dict(name="Kit default relational dataset", unit=kit_audit["kit_grouping"],
             family_audit="results/research/COMPLETION20261012/G3_kit_generator_family_audit.json",
             note="0 structural-family overlaps at the default seed; test_base and test_long share graphs by design"),
    ],
)
OUT.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
print(json.dumps(dict(checked=checks, datasets=len(registry["datasets"])), indent=1))
