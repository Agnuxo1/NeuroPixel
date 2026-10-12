"""Claims ledger: each declared evidence level must be backed by verifiable artefacts."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .common import read_json, sha256_file, write_json

HEX64 = re.compile(r"^[0-9a-f]{64}$")
LEVELS = {
    0: "idea or registered hypothesis",
    1: "implemented software with verifiable receipts",
    2: "complete registered experiment, negatives included",
    3: "prospective claim supported within scope",
    4: "independent blind external replication",
    5: "same principle supported in two or more registered domains",
    6: "new quantitative prediction confirmed on third-party data",
    7: "independent use by others (recorded, never self-declared)",
}
MINIMUM_LEVEL = {
    "hypothesis_registered": 0, "blocked_external": 0, "programme_stopped": 0,
    "engineering_complete": 1, "experiment_complete": 2, "claim_not_supported": 2,
    "ready_for_external_review": 2, "claim_supported_within_scope": 3,
    "claim_replicated_external": 4, "principle_generalized": 5, "prediction_confirmed_external": 6,
}
SUPPORTED_STATUSES = {"claim_supported_within_scope", "claim_replicated_external",
                      "principle_generalized", "prediction_confirmed_external"}
COMPLETED_STATUSES = SUPPORTED_STATUSES | {"experiment_complete", "claim_not_supported",
                                           "ready_for_external_review", "blocked_external", "programme_stopped"}
SELF_NAMES = {"me", "self", "author", "authors", "owner", "team", "internal"}


def _receipt_ok(root: Path, receipt: object, problems: list[str], label: str) -> bool:
    """Verify one registered artefact inside the root by its SHA-256."""
    if not isinstance(receipt, dict) or not isinstance(receipt.get("path"), str):
        problems.append(label + ": receipt needs path and sha256")
        return False
    path = (root / receipt["path"]).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        problems.append(label + ": missing or escaping artefact " + receipt["path"])
        return False
    if sha256_file(path) != receipt.get("sha256"):
        problems.append(label + ": bytes differ from registered sha256 " + receipt["path"])
        return False
    return True


def _commitment_ok(root: Path, record: object, problems: list[str], label: str, *, dated: bool) -> bool:
    """Link a commitment to the verified public record that publishes it."""
    if not isinstance(record, dict) or not HEX64.match(str(record.get("commitment", ""))):
        problems.append(label + ": commitment record with 64 hex digits required")
        return False
    if not _receipt_ok(root, record.get("public_record"), problems, label + " public record"):
        return False
    published = read_json(root / record["public_record"]["path"])
    if not isinstance(published, dict) or published.get("commitment") != record["commitment"]:
        problems.append(label + ": public record does not publish this commitment")
        return False
    if dated and not str(record.get("timestamp_evidence") or "").strip():
        problems.append(label + ": external timestamp evidence required")
        return False
    return True


def _independent(name: object, authors: set[str]) -> bool:
    """Reject empty names, self-references and listed authors."""
    text = str(name or "").strip().casefold()
    return bool(text) and text not in authors and text not in SELF_NAMES


def supported_level(claim: dict, root: Path, authors: set[str]) -> tuple[int, list[str]]:
    """Return the highest cumulative level the artefacts justify and the gaps found."""
    gaps: list[str] = []
    receipts = claim.get("receipts") or []
    if (not isinstance(receipts, list) or not receipts
            or not all([_receipt_ok(root, r, gaps, "receipt") for r in receipts])):
        return 0, gaps + ["L1 needs at least one verified receipt"]
    if not _receipt_ok(root, claim.get("protocol"), gaps, "protocol"):
        return 1, gaps + ["L2 needs a verified frozen protocol"]
    if claim.get("status") not in COMPLETED_STATUSES:
        return 1, gaps + ["L2 needs a completed experiment status"]
    if claim.get("status") not in SUPPORTED_STATUSES or not _commitment_ok(
            root, claim.get("preregistration"), gaps, "preregistration", dated=True):
        return 2, gaps + ["L3 needs a supported status and a published, externally dated commitment"]
    replications = [item for item in claim.get("external_replications") or []
                    if isinstance(item, dict) and item.get("independent") is True and item.get("blind") is True
                    and item.get("outcome") == "replicated" and _independent(item.get("evaluator"), authors)
                    and _receipt_ok(root, item.get("receipt"), gaps, "replication")]
    if not replications:
        return 3, gaps + ["L4 needs an independent blind replication by a non-author with its own receipt"]
    domains = {str(item.get("name")) for item in claim.get("domains") or []
               if isinstance(item, dict) and item.get("outcome") == "supported"
               and _commitment_ok(root, item.get("preregistration"), gaps, "domain", dated=True)}
    if len(domains) < 2:
        return 4, gaps + ["L5 needs two or more distinct preregistered domains supported"]
    predictions = [item for item in claim.get("external_predictions") or []
                   if isinstance(item, dict) and item.get("outcome") == "confirmed"
                   and _independent(item.get("data_owner"), authors)
                   and _commitment_ok(root, item.get("registration"), gaps, "prediction", dated=True)
                   and _receipt_ok(root, item.get("receipt"), gaps, "prediction")]
    if not predictions:
        return 5, gaps + ["L6 needs a registered prediction confirmed on third-party data"]
    uses = [item for item in claim.get("independent_uses") or []
            if isinstance(item, dict) and str(item.get("source", "")).startswith("https://")
            and _independent(item.get("user"), authors)]
    if not uses:
        return 6, gaps + ["L7 needs recorded independent use by a non-author with a public source"]
    return 7, gaps


def _audit_claim(claim: dict, root: Path, authors: set[str]) -> tuple[dict, list[str]]:
    """Audit one claim and return its row and violations."""
    identifier, status, violations = str(claim.get("id")), claim.get("status"), []
    if not isinstance(status, str) or status not in MINIMUM_LEVEL:
        return {}, [identifier + ": unknown status " + str(status)]
    if not str(claim.get("statement") or "").strip() or not str(claim.get("population") or "").strip():
        violations.append(identifier + ": statement and population are required")
    level, gaps = supported_level(claim, root, authors)
    declared = claim.get("declared_level")
    if type(declared) is not int or declared not in LEVELS:
        violations.append(identifier + ": declared_level must be an integer 0-7")
    elif declared > level:
        violations.append(f"{identifier}: declared L{declared} but artefacts support L{level}")
    if level < MINIMUM_LEVEL[status]:
        violations.append(f"{identifier}: status {status} needs at least L{MINIMUM_LEVEL[status]}")
    if status == "claim_not_supported" and type(declared) is int and declared > 2:
        violations.append(identifier + ": an unsupported claim cannot exceed L2")
    if status == "hypothesis_registered" and not _commitment_ok(
            root, claim.get("preregistration"), [], "preregistration", dated=False):
        violations.append(identifier + ": a registered hypothesis needs its published commitment")
    if status == "programme_stopped" and not str(claim.get("stop_reason") or "").strip():
        violations.append(identifier + ": a stopped line needs its registered stop reason")
    if not isinstance(claim.get("limits"), list) or not claim.get("limits"):
        violations.append(identifier + ": explicit limits are required at every level")
    return dict(id=identifier, status=status, declared_level=declared, supported_level=level,
                meaning=LEVELS[level], gaps=gaps), violations


def validate(ledger: dict, root: Path) -> dict:
    """Refuse declared levels and statuses that the registered artefacts do not justify."""
    root = Path(root).resolve()
    violations, rows, seen = [], [], set()
    claims = ledger.get("claims") if isinstance(ledger, dict) else None
    authors = ledger.get("authors") if isinstance(ledger, dict) else None
    if (not isinstance(ledger, dict) or ledger.get("schema") != 1 or not isinstance(claims, list) or not claims
            or not all(isinstance(claim, dict) for claim in claims) or not isinstance(authors, list)
            or not authors or not all(isinstance(name, str) and name.strip() for name in authors)):
        return dict(valid=False, claims=[],
                    violations=["Ledger schema 1 with named authors and a non-empty list of claim objects required"])
    author_names = {name.strip().casefold() for name in authors}
    for claim in claims:
        identifier = claim.get("id")
        if not isinstance(identifier, str) or not identifier.strip() or identifier in seen:
            violations.append("Missing or duplicate claim id: " + str(identifier))
            continue
        seen.add(identifier)
        try:
            row, problems = _audit_claim(claim, root, author_names)
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            row, problems = {}, [f"{identifier}: malformed claim ({type(error).__name__})"]
        violations += problems
        if row:
            rows.append(row)
    return dict(valid=not violations, violations=violations, claims=rows,
                scope="Artefact consistency only; independence, timestamps and merit need human review")


def main() -> None:
    """Audit a claims ledger; exit code 1 is a correct refusal of overstated evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate(read_json(args.ledger), args.root)
    write_json(args.output, report)
    print(json.dumps(dict(valid=report["valid"], violations=len(report["violations"])), indent=2))
    raise SystemExit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()
