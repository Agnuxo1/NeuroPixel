"""G13 checks: BET01 decision matrix, sensitivity, eligible capacity grid and plan completeness (CPU, no neural import)."""
import itertools
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from completion_kit.laws import capacity_ceiling_bits  # noqa: E402

MATRIX = ROOT / "docs/research/BET01_decision_matrix.json"
PLAN = ROOT / "docs/research/BET01_plan.json"
LEDGER = ROOT / "docs/research/claims_ledger.json"
REGISTRY = ROOT / "docs/research/program_registry.json"
OUT = ROOT / "results/research/G13_BET01/bet01_checks.json"
CLAIM_ID = re.compile(r"\b(?:H0[456]|HT|HC|D-[A-Z0-9]+|G-R\d+|P-[A-Z0-9]+|ANT01|MATH119)-\d{2}\b")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def check_matrix(matrix: dict, claim_ids: set[str]) -> tuple[dict, list[str]]:
    problems, totals, cited = [], {}, set()
    criteria = list(matrix["criteria"])
    for name, candidate in matrix["candidates"].items():
        scores = candidate["scores"]
        if sorted(scores) != sorted(criteria):
            problems.append(f"{name}: criteria mismatch")
        for criterion, cell in scores.items():
            if not (isinstance(cell["score"], int) and 0 <= cell["score"] <= 3):
                problems.append(f"{name}.{criterion}: score outside 0-3")
            if len(cell["evidence"].strip()) < 20:
                problems.append(f"{name}.{criterion}: evidence missing")
            for claim in CLAIM_ID.findall(cell["evidence"]):
                cited.add(claim)
                if claim not in claim_ids:
                    problems.append(f"{name}.{criterion}: cites unknown claim {claim}")
        totals[name] = sum(cell["score"] for cell in scores.values())
    eligible = [n for n, c in matrix["candidates"].items() if c["eligible"]]
    ranking = sorted(eligible, key=lambda n: -totals[n])
    winner, runner = ranking[0], ranking[1]
    # Smallest number of one-point changes (any cells, scores kept in 0-3) that lets another eligible candidate tie or pass.
    def room(name: str, up: bool) -> list[int]:
        return [(3 - c["score"]) if up else c["score"] for c in matrix["candidates"][name]["scores"].values()]
    overturn = {}
    for other in ranking[1:]:
        gap = totals[winner] - totals[other]
        capacity = sum(room(winner, False)) + sum(room(other, True))
        overturn[other] = gap if gap <= capacity else None
    single = []
    for name in eligible:
        for criterion in criteria:
            for delta in (-1, 1):
                new = matrix["candidates"][name]["scores"][criterion]["score"] + delta
                if 0 <= new <= 3:
                    changed = dict(totals, **{name: totals[name] + delta})
                    best = max(eligible, key=lambda n: (changed[n], n == winner))
                    if best != winner:
                        single.append(f"{name}.{criterion}{delta:+d}")
    report = dict(totals=totals, eligible=eligible, ranking=ranking, winner=winner, runner_up=runner,
                  lead=totals[winner] - totals[runner],
                  one_point_changes_to_tie=overturn,
                  single_changes_that_flip_winner=single,
                  fragile=bool(single), cited_claims=sorted(cited))
    if winner != matrix.get("chosen", winner):
        problems.append("matrix winner differs from declared choice")
    return report, problems


def capacity_grid() -> list[dict]:
    cells = []
    for f, sigma, k in itertools.product((0.1, 0.3, 0.5, 0.7), (1.0, 2.0, 4.0), (8, 64)):
        erased = round(f * 64)
        ceiling = capacity_ceiling_bits(64, 8, erased, sigma, 1.0)["ceiling_bits"]
        attainable = math.log2(k) / ceiling
        cells.append(dict(f=f, sigma=sigma, K=k, erased_cells=erased, ceiling_bits=round(ceiling, 6),
                          max_attainable_eta=round(min(1.0, attainable), 6), eligible=attainable >= 0.05))
    return cells


def check_plan(plan: dict, registry: dict) -> list[str]:
    problems = []
    for key in ("hypothesis", "population", "endpoints", "predictions", "error_control", "stop_rules",
                "fixed_constants", "capacity_grid", "choice"):
        if key not in plan:
            problems.append(f"plan lacks {key}")
    if sorted(plan.get("predictions", {})) != ["P1", "P2", "P3", "P4"]:
        problems.append("plan must register P1-P4")
    spent = sum(p.get("alpha", 0) for p in plan["predictions"].values())
    threshold = registry["error_control"]["first_thresholds"][plan["error_control"]["programme_registry_position"] - 1]
    if spent > threshold + 1e-12:
        problems.append(f"alpha {spent} exceeds stream threshold {threshold}")
    if abs(plan["error_control"]["stream_threshold"] - threshold) > 1e-12:
        problems.append("stream threshold in plan differs from registry")
    for rule in ("S0_gate_P2", "S1_no_multistability", "S2_eta_floor", "S3_comparator_dominance"):
        if rule not in plan["stop_rules"]:
            problems.append(f"stop rule {rule} missing")
    if not plan["hypothesis"]["text"].startswith("In locally coupled cellular fields"):
        problems.append("hypothesis text differs from guide 18B")
    return problems


def main() -> int:
    matrix, plan = load(MATRIX), load(PLAN)
    claim_ids = {c["id"] for c in load(LEDGER)["claims"]}
    matrix["chosen"] = plan["choice"]["chosen"]
    matrix_report, problems = check_matrix(matrix, claim_ids)
    grid = capacity_grid()
    problems += check_plan(plan, load(REGISTRY))
    eligible = [c for c in grid if c["eligible"]]
    if len(eligible) < 4:
        problems.append("fewer than four eligible capacity cells")
    report = dict(schema=1, task="G13", matrix=matrix_report, capacity_grid=grid,
                  eligible_capacity_cells=len(eligible), problems=problems, valid=not problems)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(dict(totals=matrix_report["totals"], winner=matrix_report["winner"], lead=matrix_report["lead"],
                          flips=matrix_report["single_changes_that_flip_winner"],
                          to_tie=matrix_report["one_point_changes_to_tie"],
                          eligible_cells=len(eligible), problems=problems), ensure_ascii=False))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
