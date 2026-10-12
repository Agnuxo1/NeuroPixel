"""Verify every entry of the imported item-4 transfer bundle against its own transfer manifest.

Reads bytes only: no model is loaded and no example is generated.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "results/research/04_experiment/item4_evidence.zip"
OUT = ROOT / "results/research/COMPLETION20261012/item4_transfer_verification.json"

with zipfile.ZipFile(BUNDLE) as bundle:
    manifest = json.loads(bundle.read("transfer_manifest.json"))
    names = set(bundle.namelist()) - {"transfer_manifest.json"}
    rows, issues = [], []
    for entry in manifest["files"]:
        data = bundle.read(entry["path"]) if entry["path"] in names else None
        ok = (data is not None and len(data) == entry["bytes"]
              and hashlib.sha256(data).hexdigest() == entry["sha256"])
        rows.append(dict(path=entry["path"], ok=ok))
        if not ok:
            issues.append(entry["path"])
    unlisted = sorted(names - {entry["path"] for entry in manifest["files"]})
    for loose in ("pilot/pilot_summary.json", "pilot/selection.json"):
        imported = (ROOT / "results/research/04_experiment" / loose).read_bytes()
        if imported != bundle.read(loose):
            issues.append("imported copy differs: " + loose)

record = dict(bundle=str(BUNDLE.relative_to(ROOT)).replace("\\", "/"),
              bundle_sha256=hashlib.sha256(BUNDLE.read_bytes()).hexdigest(),
              manifest_entries=len(manifest["files"]), verified=sum(r["ok"] for r in rows),
              unlisted_entries=unlisted, issues=issues, status="verified" if not issues and not unlisted else "issues_found",
              scope="Internal byte consistency of the transferred item-4 evidence; not independent replication.")
OUT.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
print(json.dumps({k: record[k] for k in ("manifest_entries", "verified", "unlisted_entries", "status")}))
