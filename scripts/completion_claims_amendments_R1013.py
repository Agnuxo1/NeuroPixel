"""R1013 amendments: bring the registry up to the work closed after A18 (G2.2 runtime, G17.1 custody rehearsal,
G18 programme statistics) before those results are stated in the README; every number is checked against its receipt."""
from __future__ import annotations

import json

from completion_claim_review_lib import ROOT, sha256

G22 = "results/research/COMPLETION20261012/G2.2_runtime_receipt_attempt3.json"
G17 = "results/research/G17_REHEARSAL_20261013/rehearsal_receipt.json"
G17_DOC = "docs/research/G17_custody_procedure_20261013.md"
G18 = "docs/research/program_registry.json"
G18_SENS = "results/research/COMPLETION20261012/G18_sensitivity.json"
G18_DOC = "docs/research/G18_program_statistics_20261013.md"
LIMIT = "No independent laboratory replication and no externally dated public preregistration"


def _load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def apply(claims: list[dict]) -> tuple[list[dict], dict]:
    by_id = {c["id"]: c for c in claims}
    runtime, rehearsal, registry = _load(G22), _load(G17), _load(G18)
    steps = runtime["steps"]
    checks = [
        ("G2.2 admitted at >= 8 GiB", runtime["admission"]["status"] == "admitted"
         and runtime["admission"]["measured_available_ram_gib"] >= 8.0),
        ("G2.2 all steps passed", runtime["all_passed"] and all(s["exit_code"] == 0 for s in steps)),
        ("G2.2 3 original T0 tests", "Ran 3 tests" in steps[1]["stderr_tail"] and "OK" in steps[1]["stderr_tail"]),
        ("G2.2 7 runtime contracts", "7 passed" in steps[2]["stdout_tail"]),
        ("G2.2 Torch 2.6.0 CPU", "2.6.0" in steps[0]["stdout_tail"] and runtime["device"].startswith("CPU")),
        ("G17.1 rehearsal steps ok", rehearsal["rehearsal"] and not rehearsal["real_data"] and rehearsal["all_steps_ok"]
         and len(rehearsal["steps"]) == 9),
        ("G18 online Bonferroni 0.05", registry["error_control"]["total_alpha"] == 0.05
         and abs(registry["error_control"]["first_thresholds"][0] - 0.030396355092701333) < 1e-15),
        ("G18 no equality claims", registry["equivalence_policy"]["current_equality_claims"] == 0),
    ]
    t0 = by_id["P-T0-01"]
    t0["statement"] = (
        "A scoped zero-depth (T=0) forward contract is installed in a new model version with its AST scope verified, and "
        "its runtime was checked on CPU with Torch 2.6.0 under the unchanged 8 GiB guard: the 3 original T0 tests and 7 "
        "new contracts passed, including bit-identical outputs of the patched and frozen forward passes at positive depth "
        "(evaluation and training masks with the same random state), PAD handling, class indices, finiteness and "
        "persistence; the original core suite also passed.")
    t0["limits"] = [
        "CPU runtime only; the affected graphics backend controls (task G2.3) are pending",
        "The frozen original checkout is preserved separately",
        LIMIT]
    t0["receipts"] = t0["receipts"] + [dict(path=G22, sha256=sha256(ROOT / G22))]
    t0.setdefault("amendments", []).append("R1013")
    new = [
        dict(id="P-CUST-01", study="G17", review_batch="R1013", status="engineering_complete", declared_level=1,
             statement=("A commit-reveal custody procedure for future TEST sets was documented and rehearsed with dummy "
                        "material: commitment, cohort digest and verification passed in all 9 rehearsal steps and "
                        "tampered openings were rejected; no real TEST set has yet been placed in third-party custody."),
             population="Procedure rehearsal with dummy material only",
             limits=["Rehearsal only; independent evaluator and external timestamp pending owner approval (G17.1.ext)",
                     "A claimed commitment time is not proof of time", LIMIT],
             protocol=dict(path=G17_DOC, sha256=sha256(ROOT / G17_DOC)),
             receipts=[dict(path=G17, sha256=sha256(ROOT / G17))], stop_reason=None, source_units=[]),
        dict(id="P-PROG-01", study="G18", review_batch="R1013", status="engineering_complete", declared_level=1,
             statement=("A programme registry fixes familywise error control for every future confirmatory hypothesis "
                        "(online Bonferroni, total alpha 0.05, first threshold 0.0304) and forbids equivalence claims "
                        "without a pre-registered margin; descriptive robustness analyses (hierarchical bootstrap for "
                        "DEV04/GLOB05, four-specification curves for VIS07) agree in sign with the registered intervals."),
             population="Programme-level statistics over the existing exploratory studies",
             limits=["All earlier studies remain exploratory and spend no alpha",
                     "Fewer than ten clusters: the hierarchical bootstrap is descriptive", LIMIT],
             protocol=dict(path=G18_DOC, sha256=sha256(ROOT / G18_DOC)),
             receipts=[dict(path=G18, sha256=sha256(ROOT / G18)), dict(path=G18_SENS, sha256=sha256(ROOT / G18_SENS))],
             stop_reason=None, source_units=[]),
    ]
    claims.extend(new)
    problems = [label for label, ok in checks if not ok]
    return claims, dict(schema=1, task="R1013", checks=[dict(check=label, ok=ok) for label, ok in checks],
                        edited_claims=["P-T0-01"], new_claims=[c["id"] for c in new], problems=problems)
