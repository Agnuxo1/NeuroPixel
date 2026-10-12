"""M2 amendments: register the MATH119 finite-check result (new claim, checked against its results)."""
from __future__ import annotations

import json

from completion_claim_review_lib import ROOT, number_ok, sha256

RESULTS = "results/research/MATH119/results.json"
PLAN = "docs/research/MATH119_plan.json"


def apply(claims: list[dict]) -> tuple[list[dict], dict]:
    r = json.loads((ROOT / RESULTS).read_text(encoding="utf-8"))
    effs = [t["efficiency"] for t in r["E3_selected"]["test"]]
    checks = [("27846 cases, no violation", r["cases"] == 27846 and not r["E1_violations"]),
              ("closed forms within 1e-12", r["passed"]["E2"]), ("distribution checks", r["passed"]["E4"]),
              ("kit agreement", r["passed"]["E5"]),
              ("selected rule 77 T1", r["E3_selected"]["rule"] == 77 and r["E3_selected"]["steps"] == 1),
              ("efficiency range 0.751-0.954", number_ok("0.751", [min(effs)]) and number_ok("0.954", [max(effs)]))]
    claim = dict(
        id="MATH119-01", study="MATH119", review_batch="M2", status="experiment_complete", declared_level=2,
        statement=("In exact finite enumeration over 3-8 uniform bits and BSC noise p in {0.02, 0.05, 0.1, 0.2, 0.3, 0.45}, "
                   "none of 27,846 evaluated one-bit functions (analytic controls and all elementary cellular-automaton "
                   "readouts after one to three steps) exceeded I(f(X);Y) <= 1 - h2(p); dictator, parity and constant "
                   "controls matched their closed forms within 1e-12; the DEV-selected non-trivial local readout (rule 77, "
                   "one step) retained 0.751-0.954 of the ceiling at the TEST noise levels."),
        population="Uniform independent bits (n = 3-8) under a binary symmetric channel; one-bit deterministic readouts",
        limits=["Finite numerical checks of examples; not a proof of the general theorem",
                "Not applicable to ten-class labels, UCI images, spatial dropout or dependent neural states",
                "The formal proof is checked separately (task M3)",
                "No independent laboratory replication and no externally dated public preregistration"],
        protocol=dict(path=PLAN, sha256=sha256(ROOT / PLAN)), receipts=[dict(path=RESULTS, sha256=sha256(ROOT / RESULTS))],
        stop_reason=None, source_units=[])
    claims.append(claim)
    problems = [label for label, ok in checks if not ok]
    return claims, dict(schema=1, task="M2", results=RESULTS, results_sha256=sha256(ROOT / RESULTS),
                        checks=[dict(check=label, ok=ok) for label, ok in checks], new_claims=["MATH119-01"], problems=problems)
