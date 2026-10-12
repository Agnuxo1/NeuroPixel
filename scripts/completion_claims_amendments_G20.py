"""G20 amendments: register the ANT01 antecedent comparison (new claim, checked against its results)."""
from __future__ import annotations

import json

from completion_claim_review_lib import ROOT, number_ok, sha256

RESULTS = "results/research/ANT01/results.json"
PLAN = "docs/research/ANT01_plan.json"


def _range(results, family, k, model):
    values = [c["models"][model]["E1_exact_repair"]["mean"] for c in results["cells"]
              if c["pattern_family"] == family and c["K"] == k]
    return min(values), max(values)


def apply(claims: list[dict]) -> tuple[list[dict], dict]:
    results = json.loads((ROOT / RESULTS).read_text(encoding="utf-8"))
    checks = [("PR1 held in 24/24", results["predictions"]["PR1"]["held"] == 24 == results["predictions"]["PR1"]["cells"]),
              ("PR2 held in 60/60", results["predictions"]["PR2"]["held"] == 60 == results["predictions"]["PR2"]["cells"])]
    for label, (family, k, model, low, high) in {
            "rand K8 global": ("rand", 8, "global_hopfield", "85", "95.5"),
            "rand K4 local r1": ("rand", 4, "local_r1", "0", "2"),
            "rand K8 local r2": ("rand", 8, "local_r2", "0", "1.5"),
            "smooth K8 global": ("smooth", 8, "global_hopfield", "13.5", "26")}.items():
        lo, hi = _range(results, family, k, model)
        checks.append((label, number_ok(low, [100 * lo]) and number_ok(high, [100 * hi])))
    claim = dict(
        id="ANT01-01", study="ANT01", review_batch="G20", status="experiment_complete", declared_level=2,
        statement=("At equal state size (64 binary cells), a global Hebbian Hopfield network exactly repaired 85-95.5% of "
                   "erased and noisy cues with eight stored i.i.d. patterns, while restricting the same Hebbian coupling to "
                   "3x3 neighbourhoods failed from four patterns (0-2%) and to 5x5 neighbourhoods from eight (0-1.5%); "
                   "spatially smooth patterns degraded all variants (global 13.5-26% at eight patterns), and an explicit "
                   "nearest-pattern decoder identified the target in every condition. Both registered predictions held."),
        population="Synthetic binary patterns (i.i.d. and smoothed) on an 8x8 grid; K 1-12, erased fraction 0.1-0.5, noise 0-0.5; 20 memory instances x 10 cues per condition",
        limits=["Binary Hebbian Hopfield variants, not the learned NeuroPixel rule with hidden channels",
                "Exploratory antecedent comparison; spends no alpha; no external custody",
                "Cues within a memory instance are not independent; instances are the units",
                "No independent laboratory replication and no externally dated public preregistration"],
        protocol=dict(path=PLAN, sha256=sha256(ROOT / PLAN)), receipts=[dict(path=RESULTS, sha256=sha256(ROOT / RESULTS))],
        stop_reason=None, source_units=[], public_text_after_G20=[dict(path="docs/research/ANT01_results.md", line=7)])
    claims.append(claim)
    problems = [label for label, ok in checks if not ok]
    return claims, dict(schema=1, task="G20.2", results=RESULTS, results_sha256=sha256(ROOT / RESULTS),
                        checks=[dict(check=label, ok=ok) for label, ok in checks], new_claims=["ANT01-01"], problems=problems)
