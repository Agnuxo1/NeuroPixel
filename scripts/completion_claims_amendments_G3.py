"""G3 amendments to the claims registry (applied after the A18.3 amendments)."""
from __future__ import annotations

from completion_claim_review_lib import ROOT, sha256

EDITS = {
    "D-VIS07-01": [("limit", "Exact-image uniqueness and disjointness across the official splits are asserted by the manuscript "
                    "without a retained receipt; re-verification needs the public UCI files (G3 leakage review, L1.4)")],
}
SOURCE = "results/research/COMPLETION20261012/G3_split_registry.json"


def apply(claims: list[dict]) -> tuple[list[dict], dict]:
    by_id = {c["id"]: c for c in claims}
    log = []
    for cid, edits in EDITS.items():
        for kind, text in edits:
            assert kind == "limit"
            by_id[cid]["limits"].append(text)
            log.append(dict(claim=cid, edit=kind, detail=[text]))
        by_id[cid].setdefault("amendments", []).append("G3")
    return claims, dict(schema=1, task="G3", source=SOURCE, source_sha256=sha256(ROOT / SOURCE), edits=log, problems=[])
