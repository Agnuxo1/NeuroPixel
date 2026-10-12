"""Shared machinery for the A18.2 claim-review batches (same method as completion_review_claims_H.py).

A batch script declares claims (English statements bound to protocol and receipts), assigns one
classification to every frozen candidate line, and lists value checks. run_review() verifies full
coverage, checks every quoted number against the named receipts, and writes the review record,
the value-check record and the ledger fragment. No model is loaded or evaluated.
"""
from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/research/COMPLETION20261012"
AUTHORS = ["Francisco Angulo de Lafuente"]
NO_EXT = "No independent laboratory replication and no externally dated public preregistration"
ED_HEADER = "table header row: column labels, not an assertion"
ED_LINKS = "link list or reference definition: navigation only; linked artifacts are bound in the claims"
ED_FIGURE = "figure embed or caption: visual restatement of bound data rows"
ED_POINTER = "evidence-file pointer or reproduction command: navigation to artifacts bound in the claims"
ED_TITLE = "document title or section label carried as text: no assertion"
GOV = "project-process statement (next step, task ordering or closure governance), not a scientific or technical claim"
EDITORIAL = ("editorial", "governance")


def claim(cid, statement, population, status, level, limits, protocol=None, receipts=(), checks=(),
          stop_reason=None, context=(), study=""):
    return dict(id=cid, study=study, statement=statement, population=population, status=status,
                declared_level=level, limits=list(limits), protocol=protocol, receipt_paths=list(receipts),
                value_checks=list(checks), stop_reason=stop_reason, context_paths=list(context))


class Decisions:
    def __init__(self):
        self.table: dict[str, dict[int, tuple[str, object]]] = {}

    def put(self, source, lines, cls, target):
        lines = lines if isinstance(lines, (list, tuple, range)) else [lines]
        bucket = self.table.setdefault(source, {})
        for line in lines:
            assert line not in bucket, (source, line)
            bucket[line] = (cls, target)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


_cache: dict[str, tuple[list[float], str]] = {}
NUM = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _walk(value, out):
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        out.append(float(value))
    elif isinstance(value, dict):
        for item in value.values():
            _walk(item, out)
    elif isinstance(value, list):
        for item in value:
            _walk(item, out)
    elif isinstance(value, str):
        out.extend(float(x) for x in NUM.findall(value))


def _load(name: str, data: bytes, numbers: list[float], texts: list[str]):
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return
    texts.append(text)
    if name.endswith(".json"):
        try:
            _walk(json.loads(text), numbers)
            return
        except json.JSONDecodeError:
            pass
    numbers.extend(float(x) for x in NUM.findall(text))


def evidence(rel: str) -> tuple[list[float], str]:
    """All numbers and text inside a receipt file, directory or zip (JSON/CSV/text members only)."""
    if rel in _cache:
        return _cache[rel]
    path = ROOT / rel
    numbers: list[float] = []
    texts: list[str] = []
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    for file in files:
        if file.suffix == ".zip":
            with zipfile.ZipFile(file) as bundle:
                for name in bundle.namelist():
                    if name.endswith((".json", ".jsonl", ".log", ".txt", ".csv")):
                        _load(name, bundle.read(name), numbers, texts)
        elif (path.is_file() or file.suffix in (".json", ".jsonl", ".log", ".txt", ".csv")) and file.stat().st_size < 8_000_000:
            _load(file.name, file.read_bytes(), numbers, texts)
    _cache[rel] = (numbers, "\n".join(texts))
    return _cache[rel]


def number_ok(claimed: str, numbers: list[float]) -> bool:
    """True when some receipt number equals the quoted value at its printed precision (fraction or percent)."""
    value = float(claimed)
    decimals = len(claimed.split(".")[1]) if "." in claimed else 0
    tolerance = 0.5 * 10 ** (-decimals) + 1e-12
    return any(abs(scaled - value) <= tolerance for x in numbers for scaled in (x, x * 100.0, x / 100.0))


def _t95(values, sign):
    from scipy import stats
    n = len(values)
    m = sum(values) / n
    sd = (sum((v - m) ** 2 for v in values) / (n - 1)) ** 0.5
    return m + sign * stats.t.ppf(0.975, n - 1) * sd / n ** 0.5


CALC_NAMES = dict(mean=lambda xs: sum(xs) / len(xs), sum=sum, abs=abs,
                  t95lo=lambda xs: _t95(xs, -1), t95hi=lambda xs: _t95(xs, +1))


def calc_ok(body: str, rel: str) -> tuple[bool, dict]:
    """'<expected>[~tol]=<expression>': every literal in the expression must exist in the receipt at its printed
    precision (100 is a unit constant); the expression must then reproduce the quoted value."""
    head, expr = body.split("=", 1)
    expected, _, tol = head.partition("~")
    numbers = evidence(rel)[0]
    literals = [lit for lit in NUM.findall(expr) if lit not in ("100",)]
    missing = [lit for lit in literals if not number_ok(lit.lstrip("-"), numbers) and not number_ok(lit, numbers)]
    value = eval(expr, {"__builtins__": {}}, CALC_NAMES)  # literals and whitelisted helpers only
    tolerance = float(tol) if tol else 0.5 * 10 ** (-(len(expected.split(".")[1]) if "." in expected else 0)) + 1e-12
    ok = not missing and abs(value - float(expected)) <= tolerance
    return ok, dict(value=value, missing_literals=missing, tolerance=tolerance)


def single_check(spec: str, rel: str) -> tuple[bool, dict]:
    """text:, len: (rel 'path#key' list length), calc: or a plain number at its printed precision."""
    if spec.startswith("text:"):
        ok = spec[5:] in evidence(rel)[1]
        return ok, dict(check=spec, source=rel, ok=ok)
    if spec.startswith("len:"):
        path, _, key = rel.partition("#")
        data = json.loads((ROOT / path).read_text(encoding="utf-8"))
        for part in key.split("/"):
            data = data[part]
        ok = len(data) == int(spec[4:])
        return ok, dict(check=spec, source=rel, observed=len(data), ok=ok)
    if spec.startswith("calc:"):
        ok, detail = calc_ok(spec[5:], rel)
        return ok, dict(check=spec, source=rel, ok=ok, detail=detail)
    ok = number_ok(spec, evidence(rel)[0])
    return ok, dict(check=spec, source=rel, ok=ok)


def ledger_ids(*names: str) -> set:
    ids = set()
    for name in names:
        data = json.loads((ROOT / "docs/research/claims_ledger_parts" / name).read_text(encoding="utf-8"))
        ids |= {item["id"] for item in data["claims"]}
    return ids


def run_review(task: str, claims: list[dict], decisions: Decisions, discrepancies: list[dict],
               part_name: str, scope: str, recounts: dict | None = None, external_ids: set | None = None,
               restatements: list | None = None, unsourced: set | None = None) -> int:
    candidates = json.loads((OUT / "claim_candidates.json").read_text(encoding="utf-8"))["candidates"]
    batches = json.loads((OUT / "claim_review_batches.json").read_text(encoding="utf-8"))["batches"]
    batch = next(b["candidate_ids"] for b in batches if b["task"] == task)
    by_id = {cand["id"]: cand for cand in candidates}
    claim_ids = {item["id"] for item in claims}
    assert len(claim_ids) == len(claims), "duplicate claim ids"
    external_ids = set(external_ids or ())
    assert not claim_ids & external_ids, "claim id reused from another batch"
    known_ids = claim_ids | external_ids
    problems, rows, used = [], [], set()
    units: dict[str, list[str]] = {cid: [] for cid in claim_ids}
    for cid in batch:
        cand = by_id[cid]
        decision = decisions.table.get(cand["source_path"], {}).get(cand["source_line"])
        if decision is None:
            problems.append(f"unclassified candidate {cid} {cand['source_path']}:{cand['source_line']}")
            continue
        used.add((cand["source_path"], cand["source_line"]))
        cls, target = decision
        row = dict(candidate_id=cid, source_path=cand["source_path"], source_line=cand["source_line"],
                   source_sha256=cand["source_sha256"], classification=cls)
        if cls in EDITORIAL:
            row["exclusion_reason"] = target
        else:
            unknown = [t for t in target if t not in known_ids]
            if unknown:
                problems.append(f"{cid}: unknown claim ids {unknown}")
            row["claims"] = list(target)
            for t in target:
                units.setdefault(t, []).append(cid)
        rows.append(row)
    for src, lines in decisions.table.items():
        for line in lines:
            if (src, line) not in used:
                problems.append(f"decision without candidate {src}:{line}")
    for cid in claim_ids - set(unsourced or ()):
        if not any(row["classification"] in ("claim", "data_row") and cid in row.get("claims", []) for row in rows):
            problems.append(f"{cid}: no asserting source unit in this batch")

    recounts = recounts or {}
    checks_out = []
    for item in claims:
        results = []
        for spec, rel in item["value_checks"]:
            if spec.startswith("recount:"):
                observed = recounts[spec[8:]]
                try:
                    ok = number_ok(rel, [float(observed)])
                except (TypeError, ValueError):
                    ok = str(observed) == rel
                results.append(dict(check=spec, expected=rel, observed=observed, ok=ok))
            elif spec.startswith("calc:"):
                ok, detail = calc_ok(spec[5:], rel)
                results.append(dict(check=spec, source=rel, ok=ok, detail=detail))
            else:
                ok, result = single_check(spec, rel)
                results.append(result)
            if not ok:
                problems.append(f"{item['id']}: value check failed {spec} in {rel}")
        checks_out.append(dict(claim=item["id"], checks=results))
    restated = []
    for label, spec, rel in restatements or []:
        if spec.startswith("recount:"):
            observed = recounts[spec[8:]]
            ok = number_ok(rel, [float(observed)])
            result = dict(check=spec, expected=rel, observed=observed, ok=ok)
        else:
            ok, result = single_check(spec, rel)
        restated.append(dict(label=label, **result))
        if not ok:
            problems.append(f"restatement {label}: check failed {spec} in {rel}")
    tag = task.split(".")[-1]
    (OUT / f"value_checks_{tag}.json").write_text(
        json.dumps(dict(schema=1, task=task, recounts=recounts, claims=checks_out, restatements=restated), indent=2) + "\n",
        encoding="utf-8", newline="\n")

    ledger_claims = []
    for item in claims:
        for path in item["receipt_paths"] + ([item["protocol"]] if item["protocol"] else []) + item["context_paths"]:
            if not (ROOT / path).is_file():
                problems.append(f"{item['id']}: missing artifact {path}")
        if problems and any(p.startswith(item["id"] + ": missing artifact") for p in problems):
            continue
        entry = dict(id=item["id"], statement=item["statement"], population=item["population"],
                     status=item["status"], declared_level=item["declared_level"], limits=item["limits"],
                     receipts=[dict(path=p, sha256=sha256(ROOT / p)) for p in item["receipt_paths"]],
                     stop_reason=item["stop_reason"], review_batch=task, study=item["study"],
                     source_units=sorted(units.get(item["id"], [])))
        if item["protocol"]:
            entry["protocol"] = dict(path=item["protocol"], sha256=sha256(ROOT / item["protocol"]))
        if item["context_paths"]:
            entry["context_records"] = [dict(path=p, sha256=sha256(ROOT / p), role="summary_record_not_raw_evidence")
                                        for p in item["context_paths"]]
        ledger_claims.append(entry)

    classes: dict[str, int] = {}
    for row in rows:
        classes[row["classification"]] = classes.get(row["classification"], 0) + 1
    review = dict(schema=1, task=task, candidates=len(batch), classified=len(rows), classes=classes,
                  claims=len(claims), discrepancies=discrepancies, problems=problems, scope=scope, decisions=rows)
    (OUT / f"claim_review_{tag}.json").write_text(json.dumps(review, indent=2, ensure_ascii=False) + "\n",
                                                 encoding="utf-8", newline="\n")
    part = ROOT / "docs/research/claims_ledger_parts" / part_name
    part.parent.mkdir(parents=True, exist_ok=True)
    part.write_text(json.dumps(dict(schema=1, authors=AUTHORS, claims=ledger_claims), indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8", newline="\n")
    print(json.dumps(dict(task=task, candidates=len(batch), classified=len(rows), classes=classes,
                          claims=len(claims), recounts=recounts, problems=problems), indent=1))
    return 0 if not problems else 1
