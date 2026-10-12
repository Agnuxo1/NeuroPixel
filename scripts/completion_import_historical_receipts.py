"""Import historical receipts byte-for-byte from local validation checkouts into their documented paths.

No file is edited or regenerated. Each copy is authenticated against a hash stated by an
independent record (report text, closure receipt or transfer manifest) where one exists, and
the import manifest records the source, size and SHA-256 of every byte copied.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V6 = Path("D:/PROJECTS/NeuroPixel-Validation-20261006")
V7 = Path("D:/PROJECTS/NeuroPixel-Validation-20261007-continuation")
OUT = ROOT / "results/research/COMPLETION20261012/historical_receipt_import.json"
A06 = "results/research/06_A06_1_recovery/"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


closure = json.loads((V7 / A06 / "closure_receipt.json").read_text(encoding="utf-8"))["evidence_sha256"]
with zipfile.ZipFile(V6 / "coord/item4_evidence.zip") as bundle:
    transfer = {row["path"]: row["sha256"] for row in json.loads(bundle.read("transfer_manifest.json"))["files"]}

# (study, source, destination, expected sha256, authority for the expected hash, role)
PLAN = [
    ("04", V6 / "coord/item4_evidence.zip", "results/research/04_experiment/item4_evidence.zip", None,
     "internal transfer_manifest.json, verified separately entry by entry", "original_transfer_bundle"),
    ("04", V6 / "runs/item4_pilot_retry1/pilot_summary.json", "results/research/04_experiment/pilot/pilot_summary.json",
     transfer["pilot/pilot_summary.json"], "item4 transfer_manifest.json", "receipt"),
    ("04", V6 / "runs/item4_pilot_retry1/selection.json", "results/research/04_experiment/pilot/selection.json",
     "a62300cfccc8d01f9d2777042c20a2620ba0ebfbd24d1b2b0fe4a7ea3463673b",
     "docs/research/04_results.md line 57 and git commit a009f2a9", "receipt"),
    ("05", V6 / "coord/item5_recovery_receipt_20261007.json", "results/research/05_experiment/item5_recovery_receipt_20261007.json",
     None, "none; contemporaneous summary record, not raw evidence", "summary_record_not_raw_evidence"),
    ("05", V6 / "coord/item5_gpu_interrupted.zip", "results/research/05_experiment/item5_gpu_interrupted.zip",
     None, "none; interrupted GPU attempt export", "incident_record"),
    ("05", V6 / "coord/item5_queue_exit.json", "results/research/05_experiment/item5_queue_exit.json",
     None, "none; queue exit record", "incident_record"),
    ("06_A06_1", V7 / A06 / "original.zip", A06 + "original.zip", closure[A06 + "original.zip"],
     "closure_receipt.json and docs/research/06_A06_1_results.md line 10", "original_archive"),
    ("06_A06_1", V7 / A06 / "closure_receipt.json", A06 + "closure_receipt.json", None,
     "lists docs/research/06_A06_1_results.md sha256, compared below", "receipt"),
]
for name in ("audit_saved.json", "audit_intervals.json", "audit_replay.json", "numeric_diagnostic.json",
             "multiplicity_sensitivity.json"):
    PLAN.append(("06_A06_1", V7 / A06 / name, A06 + name, closure[A06 + name], "closure_receipt.json", "receipt"))
for name in ("ci_receipt.json", "regression_excerpt.txt"):
    PLAN.append(("06_A06_1", V7 / A06 / name, A06 + name, None, "none stated", "receipt"))
for name in ("audit_A06_1_saved.py", "audit_A06_1_intervals.py", "replay_A06_1_checkpoints.py",
             "diagnose_A06_1_numeric.py"):
    PLAN.append(("06_A06_1", V7 / "scripts" / name, "scripts/" + name, None, "none stated", "audit_source"))
for name in ("reconciliation_receipt.json", "source_comparison.json"):
    PLAN.append(("CONTINUATIONS_RECONCILED_20261008", V7 / "coord/recovery/reconciliation-20261008" / name,
                 "coord/recovery/reconciliation-20261008/" + name, None, "none stated", "receipt"))

report_sha = sha256(ROOT / "docs/research/06_A06_1_results.md")
assert report_sha == closure["docs/research/06_A06_1_results.md"], "A06.1 report differs from its closure receipt"

rows = []
for study, source, destination, expected, authority, role in PLAN:
    assert source.is_file(), source
    digest = sha256(source)
    if expected is not None:
        assert digest == expected, (source, digest, expected)
    target = ROOT / destination
    if target.exists():
        assert sha256(target) == digest, ("existing file differs", destination)
        action = "already_present_identical"
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        assert sha256(target) == digest
        action = "copied"
    rows.append(dict(study=study, source=str(source).replace("\\", "/"), destination=destination,
                     bytes=source.stat().st_size, sha256=digest, expected_sha256=expected,
                     authenticated=expected is not None, authority=authority, role=role, action=action))

record = dict(
    schema=1, created_utc=datetime.now(timezone.utc).isoformat(),
    scope=("Byte copies of historical receipts that existed only outside the publication tree. "
           "No regeneration, training, inference or editing. Study 05 raw 28-run archive, 05_analysis.json, "
           "05_independent_check.json and historical reassessment outputs were not found locally or on remote branches "
           "and are not substituted."),
    a06_1_report_matches_closure_receipt=True,
    not_imported=[
        {"path": A06 + "archive/", "reason": "extracted copy of original.zip; derivable from the authenticated original"},
        {"path": "results/research/04_experiment/pilot/* weights", "reason": "retained inside item4_evidence.zip; *.pt ignored by repository policy"},
    ],
    missing_everywhere=[
        "results/research/04_verification.json",
        "results/research/05_analysis.json (sha256 11fd91ee4ddb8155fc9127e5f6b826535654842190f048fecd23c3dd6f31e436)",
        "results/research/05_independent_check.json (sha256 adef6b11f29a0c00c58f1cb10a95453ba056eea5e711d8462179bcfafcff1453)",
        "results/research/05_experiment/ (215 files, executor commit 0a56621)",
        "results/research/05_historical/ and 05_historical_verification.json",
        "coord/recovery/TASK4_cloud_audit",
    ],
    files=rows,
)
OUT.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
print(json.dumps(dict(files=len(rows), authenticated=sum(r["authenticated"] for r in rows),
                      copied=sum(r["action"] == "copied" for r in rows)), indent=2))
