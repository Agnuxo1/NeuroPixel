"""Merge the A18.2 review fragments into docs/research/claims_ledger.json and record full coverage.

Fails unless every frozen candidate (795) is classified exactly once across the four batches and
every claim id is unique. The merged ledger is then audited with completion_kit.claims_ledger.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/research/COMPLETION20261012"
BATCHES = ["H", "D", "G", "P"]

candidates = json.loads((OUT / "claim_candidates.json").read_text(encoding="utf-8"))["candidates"]
claims, seen, covered, problems = [], set(), Counter(), []
for tag in BATCHES:
    part = json.loads((ROOT / f"docs/research/claims_ledger_parts/A18.2.{tag}.json").read_text(encoding="utf-8"))
    review = json.loads((OUT / f"claim_review_{tag}.json").read_text(encoding="utf-8"))
    if review["problems"]:
        problems.append(f"batch {tag} has open problems")
    for row in review["decisions"]:
        covered[row["candidate_id"]] += 1
    for item in part["claims"]:
        if item["id"] in seen:
            problems.append("duplicate claim " + item["id"])
        seen.add(item["id"])
        claims.append(item)
missing = [c["id"] for c in candidates if covered[c["id"]] != 1]
if missing:
    problems.append(f"{len(missing)} candidates not classified exactly once")
assert not problems, problems

ledger = dict(schema=1, authors=["Francisco Angulo de Lafuente"],
              scope=("Every public claim of the README, manuscripts, indexed study reports and completion-phase documents "
                     "frozen in A18.1, bound to retained artifacts. Declared levels never exceed what the artifacts support; "
                     "no external timestamp, custody or independence is asserted."),
              claims=claims)
(ROOT / "docs/research/claims_ledger.json").write_text(json.dumps(ledger, indent=2, ensure_ascii=False) + "\n",
                                                       encoding="utf-8", newline="\n")
summary = dict(candidates=len(candidates), classified_exactly_once=len(candidates) - len(missing), claims=len(claims),
               by_level=dict(sorted(Counter(c["declared_level"] for c in claims).items())),
               by_status=dict(sorted(Counter(c["status"] for c in claims).items())),
               max_declared_level=max(c["declared_level"] for c in claims))
(OUT / "claims_ledger_build.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
print(json.dumps(summary, indent=1))
