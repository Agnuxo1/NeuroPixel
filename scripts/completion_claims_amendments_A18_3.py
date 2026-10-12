"""A18.3 amendments to the claims registry, from the independent verifier review
(docs/execplan/evidence/A18.3_independent_review.json, verdict pass_with_fixes).

The closed batch fragments are not edited: completion_build_claims_ledger.py applies these
amendments on top of them. Every new or changed number is checked against receipts here.
"""
from __future__ import annotations

import json
import statistics
import zipfile

import completion_review_claims_D as DMOD
import completion_review_claims_G as GMOD
import completion_review_claims_H as HMOD
import completion_review_claims_P as PMOD
from completion_claim_review_lib import CALC_NAMES, NO_EXT, ROOT, evidence, number_ok, sha256, single_check

RR = "results/research/"
VIS = RR + "VIS07MAIN_cloud/37908785698/raw.zip"
REVIEW = "docs/execplan/evidence/A18.3_independent_review.json"


def vis07_transform_recounts() -> dict:
    with zipfile.ZipFile(ROOT / VIS) as bundle:
        scores = json.loads(bundle.read("result.json"))["scores"]
    out = {}
    for condition in ("half_intensity", "shift_right_crop"):
        acc = {}
        for family in ("nca", "cnn"):
            rows = sorted((r for r in scores if r["family"] == family and r["condition"] == condition), key=lambda r: r["seed"])
            assert len(rows) == 5 and all(r["n"] == 1797 for r in rows)
            acc[family] = [100.0 * r["correct"] / r["n"] for r in rows]
            out[f"{condition}|{family}|mean"] = statistics.mean(acc[family])
            out[f"{condition}|{family}|min"], out[f"{condition}|{family}|max"] = min(acc[family]), max(acc[family])
        diff = [a - b for a, b in zip(acc["nca"], acc["cnn"])]
        out[f"{condition}|diff|mean"] = statistics.mean(diff)
        out[f"{condition}|diff|lo"], out[f"{condition}|diff|hi"] = CALC_NAMES["t95lo"](diff), CALC_NAMES["t95hi"](diff)
    return out


def r11_overlap() -> int:
    """Workloads where the technical ranges of the best render-family and best CUDA methods overlap."""
    overlap = 0
    for case in ("C16_grid8", "C16_grid32", "C16_grid128", "C48_grid8", "C48_grid32", "C48_grid128", "RGB8_retina_C16"):
        def best(methods):
            m = min(methods, key=lambda x: GMOD.RECOUNTS[f"R11|{case}|{x}|median"])
            return GMOD.RECOUNTS[f"R11|{case}|{m}|min"], GMOD.RECOUNTS[f"R11|{case}|{m}|max"]
        (a_lo, a_hi), (b_lo, b_hi) = best(("vector_render", "render")), best(("cuda_eager", "cuda_graph"))
        overlap += a_lo <= b_hi and b_lo <= a_hi
    return overlap


RECOUNTS = vis07_transform_recounts()
RECOUNTS["R11|overlapping_workloads"] = r11_overlap()

NEW_CLAIMS = [
    dict(id="D-VIS07-05", study="VIS07", review_batch="A18.3", status="experiment_complete", declared_level=2,
         statement=("In the same frozen VIS07 protocol, two registered test-time transformation conditions were scored on the "
                    "official 1,797-image test: at half input intensity NeuroPixel averaged 88.737% versus 67.056% for the CNN "
                    "(paired difference +21.681 pp, exploratory t95 8.440 to 34.921), whereas with a one-column right shift and crop NeuroPixel averaged 55.826% "
                    "versus 85.120% (paired difference -29.293 pp, t95 -45.696 to -12.890; NeuroPixel seeds ranging 42.7-70.2%): the NCA was far less "
                    "robust than the CNN to the shift and more robust to the intensity change."),
         population="UCI optdigits public data, writer-separated test set of 1,797 images; five paired training seeds 240-244",
         limits=["Two registered transformation conditions only; descriptive five-pair results without blind custody",
                 "Not reported in the public texts until the A18.3 correction (omitted, not withheld by selection)",
                 "Depth, geometry and FLOPs differ between the models", NO_EXT],
         protocol=dict(path="docs/research/VIS07_execution_plan.json"), receipts=[VIS], source_units=[],
         public_text_after_A18_3=[dict(path="docs/research/VIS07_results.md", line=15), dict(path="README.md", line=26),
                                  dict(path="README.md", line=29), dict(path="docs/papers/NeuroPixel_manuscript_20261009.md", line=49)],
         checks=[("recount:half_intensity|nca|mean", "88.737"), ("recount:half_intensity|cnn|mean", "67.056"),
                 ("recount:half_intensity|diff|mean", "21.681"), ("recount:shift_right_crop|nca|mean", "55.826"),
                 ("recount:shift_right_crop|cnn|mean", "85.120"), ("recount:shift_right_crop|diff|mean", "-29.293"), ("recount:shift_right_crop|diff|lo", "-45.696"),
                 ("recount:shift_right_crop|diff|hi", "-12.890"), ("recount:half_intensity|diff|lo", "8.440"),
                 ("recount:half_intensity|diff|hi", "34.921"),
                 ("recount:shift_right_crop|nca|min", "42.7"), ("recount:shift_right_crop|nca|max", "70.2")]),
]

# claim id -> list of edits: ("replace", old, new) on the statement, ("limit", text), ("level", status, level),
# ("receipt", path), ("units", [candidate ids]), ("check", spec, rel)
EDITS = {
    "D-VIS07-01": [("replace", "(1,797 images, writer-separated;", "(1,797 images, writer-separated according to the UCI "
                    "metadata, individual writer identifiers not independently verified;"),
                   ("limit", "Robustness to a one-column shift was far worse than the CNN (55.826% versus 85.120%; D-VIS07-05)"),
                   ("receipt", VIS), ("receipt", RR + "SPRINT_review/VIS07_summary.json"),
                   ("units", ["S001-L0063", "S003-L0037", "S003-L0039"])],
    "D-VIS07-03": [("receipt", RR + "VIS07REPLAY_cloud/37911162172/raw.zip")],
    "D-VIS07-04": [("receipt", VIS), ("receipt", RR + "SPRINT_review/VIS07_summary.json")],
    "D-SCN06-01": [("receipt", RR + "SCN06_review/recovery_receipt.json"), ("receipt", RR + "SCN06_cloud/37896280187/raw.zip"),
                   ("limit", "A single constructed (assigned-state) experiment; there are no statistical units")],
    "D-SCN06-02": [("replace", "A separately frozen matrix census", "A separately frozen retrospective matrix census"),
                   ("level", "engineering_complete", 1),
                   ("limit", "Deterministic retrospective census; level lowered to L1 by the A18.3 review")],
    "H06-06": [("replace", "and a random router by +16.178 pp,", "and a random router by +16.178 pp (interval -2.434 to "
                "+34.791, including zero),"),
               ("limit", "Retrospective Holm sensitivity over 40 contrasts: none survives; the uniform-mixture contrast has "
                "unadjusted p = 0.0413 and Holm p = 1.0"),
               ("check", "-2.434", RR + "06_A06_1_recovery/audit_saved.json"),
               ("check", "34.791", RR + "06_A06_1_recovery/audit_saved.json"),
               ("check", "0.0413", RR + "06_A06_1_recovery/multiplicity_sensitivity.json")],
    "H06-03": [("replace", "T1 versus T16 +0.146 pp;", "T1 versus T16 +0.146 pp (-14.123 to +14.416);"),
               ("check", "-14.123", RR + "06_A06_1_recovery/audit_saved.json"),
               ("check", "14.416", RR + "06_A06_1_recovery/audit_saved.json")],
    "H04-03": [("level", "engineering_complete", 1),
               ("limit", "Recount of retained predictions performed in the completion phase; level lowered to L1 by the A18.3 review")],
    "HT-09": [("remove_receipt", RR + "09_cloud_runs/37593731891-1-study/study/final_summary.json")],
    "D-QT-02": [("receipt", RR + "OPT03_QTRAIN_review/37728384380/closure_receipt.json")],
    "D-QT-03": [("limit", "Descriptive curve values from one seed; no statistical inference")],
    "P-COST-01": [("limit", "Sums of recorded timers; no statistical units")],
    "D-OPT03D-02": [("receipt", RR + "OPT03_cloud_recovery/37696760073/original.zip")],
    "D-ROLE-02": [("receipt", RR + "OPT03_roles_review/partial_audit.json")],
    "D-READ03-01": [("replace", "gave +54.225 pp (exploratory", "gave +54.225 pp averaged over the four original conditions "
                     "(exploratory"),
                    ("replace", "only four-queries/school-0.3 attention passed the joint gate in all three seeds.",
                     "only four-queries/school-0.3 attention passed the joint gate in all three seeds; attention failed it in "
                     "the other three conditions.")],
    "D-U16-01": [("limit", "Values are in the training-matched firing-0.5 mode; dense-mode joint accuracy for the same cell is "
                  "lower (80.176/79.785/63.184%)"),
                 ("check", "79.785", RR + "OPT03_QTRAIN_U16_review/37771588722/")],
    "G-R11-01": [("limit", "Technical min-max ranges of the best render and best CUDA methods overlap in 4 of 7 workloads; the "
                  "C48 grid8/grid32 differences are about 1%"),
                 ("check", "recount:R11|overlapping_workloads", "4")],
    "H05-05": [("limit", "Three coupled seeds cannot isolate the variances; no historical significance decision was made"),
               ("limit", "Describes these trained models, not Transformers as an architectural family"),
               ("limit", "Seed-2 scores are normalized JSON snapshots from the owner's existing project")],
    "H04-01": [("limit", "An unfavorable result is not proof of architectural incapacity; adapted NCA (11.914%) and ConvGRU "
                "(13.574%) also scored low"),
               ("limit", "The NCA models use random firing in training and synchronous updates in evaluation; the causal "
                "contribution of that difference is not isolated")],
    "D-OPT03D-01": [("limit", "The bound protocol is the local CRLF copy (sha256 bf54be3c...); the executed LF plan (f1caada1...) "
                     "is byte-identical after CRLF-to-LF conversion (D-OPT03D-02)")],
    "P-MATH-01": [("replace", "its 38 contract and 25 extension tests passed in the pinned environment.",
                   "the kit test suites (38 contract and 25 extension tests), which include tests of these modules, passed in "
                   "the pinned environment.")],
    "P-T0-01": [("limit", "New-phase claim created after the A18.1 source freeze; not yet stated in the public texts")],
}
COVERAGE_NOTES = [dict(candidate="S003-L0075", decision="kept as governance",
                       reason="Restates the operational rule that closed fits are not rerun; process statement, not a result")]


def batch_value_checks() -> dict:
    out = {}
    for module in (DMOD, GMOD, PMOD):
        for item in module.CLAIMS:
            out[item["id"]] = item["value_checks"]
    for item in HMOD.CLAIMS:
        out[item["id"]] = item["value_checks"]
    return out


def located_receipts(checks) -> list[str]:
    """For checks made against a directory, bind the first file that holds each non-trivial number."""
    found = []
    for spec, rel in checks:
        if spec.split(":")[0] in ("recount", "text", "len", "calc") or (ROOT / rel).is_file() or not (ROOT / rel).is_dir():
            continue
        if "." not in spec and abs(float(spec)) < 100:
            continue
        for file in sorted(p for p in (ROOT / rel).rglob("*") if p.is_file()):
            relative = file.relative_to(ROOT).as_posix()
            if number_ok(spec, evidence(relative)[0]):
                found.append(relative)
                break
    return found


def apply(claims: list[dict]) -> tuple[list[dict], dict]:
    by_id = {c["id"]: c for c in claims}
    log, problems, checks = [], [], []
    all_checks = batch_value_checks()
    for cid, item in by_id.items():
        have = {r["path"] for r in item["receipts"]}
        for path in located_receipts(all_checks.get(cid, [])):
            if path not in have:
                item["receipts"].append(dict(path=path, sha256=sha256(ROOT / path)))
                have.add(path)
                log.append(dict(claim=cid, edit="receipt_located", path=path))
    for cid, edits in EDITS.items():
        item = by_id[cid]
        for edit in edits:
            kind = edit[0]
            if kind == "replace":
                if item["statement"].count(edit[1]) != 1:
                    problems.append(f"{cid}: replace target not found once")
                    continue
                item["statement"] = item["statement"].replace(edit[1], edit[2])
            elif kind == "limit":
                item["limits"].append(edit[1])
            elif kind == "level":
                item["status"], item["declared_level"] = edit[1], edit[2]
            elif kind == "receipt":
                if edit[1] not in {r["path"] for r in item["receipts"]}:
                    item["receipts"].append(dict(path=edit[1], sha256=sha256(ROOT / edit[1])))
            elif kind == "remove_receipt":
                kept = [r for r in item["receipts"] if r["path"] != edit[1]]
                if len(kept) != len(item["receipts"]) - 1 or not kept:
                    problems.append(f"{cid}: receipt to remove not found or last receipt")
                item["receipts"] = kept
            elif kind == "units":
                item["source_units"] = sorted(set(item.get("source_units", [])) | set(edit[1]))
            elif kind == "check":
                checks.append((cid, edit[1], edit[2]))
            log.append(dict(claim=cid, edit=kind, detail=[str(x) for x in edit[1:]]))
        item.setdefault("amendments", []).append("A18.3")
    for new in NEW_CLAIMS:
        assert new["id"] not in by_id
        entry = {k: v for k, v in new.items() if k != "checks"}
        entry["protocol"] = dict(path=new["protocol"]["path"], sha256=sha256(ROOT / new["protocol"]["path"]))
        entry["receipts"] = [dict(path=p, sha256=sha256(ROOT / p)) for p in new["receipts"]]
        entry["stop_reason"] = None
        claims.append(entry)
        checks += [(new["id"], spec, rel) for spec, rel in new["checks"]]
        log.append(dict(claim=new["id"], edit="new_claim"))
    results = []
    for cid, spec, rel in checks:
        if spec.startswith("recount:"):
            observed = RECOUNTS[spec[8:]]
            ok = number_ok(rel, [float(observed)])
            results.append(dict(claim=cid, check=spec, expected=rel, observed=observed, ok=ok))
        else:
            ok, result = single_check(spec, rel)
            results.append(dict(claim=cid, **result))
        if not ok:
            problems.append(f"{cid}: amendment check failed {spec} {rel}")
    record = dict(schema=1, task="A18.3", review=REVIEW, review_sha256=sha256(ROOT / REVIEW), recounts=RECOUNTS,
                  edits=log, checks=results, coverage_notes=COVERAGE_NOTES, problems=problems)
    return claims, record
