"""Import documented-but-absent study dependencies byte-for-byte from the local continuation checkout.

Usage: python scripts/completion_import_dependencies.py <batch-label> <relative path> [<relative path> ...]

Each relative path is copied from the continuation checkout into the same relative path here.
Nothing existing is overwritten (identical files are accepted). A file counts as authenticated
when its SHA-256 appears in a tracked document of this repository that existed before the import.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("D:/PROJECTS/NeuroPixel-Validation-20261007-continuation")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tracked_text() -> str:
    names = subprocess.run(["git", "ls-files", "docs", "results", "coord"], cwd=ROOT, capture_output=True,
                           text=True, check=True).stdout.split()
    chunks = []
    for name in names:
        path = ROOT / name
        if path.suffix in (".md", ".json", ".csv", ".txt") and path.is_file() and path.stat().st_size < 4_000_000:
            chunks.append(path.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(chunks)


def main() -> None:
    label, rels = sys.argv[1], sys.argv[2:]
    known = tracked_text()
    rows = []
    for rel in rels:
        origin = SOURCE / rel
        assert origin.exists(), origin
        files = [origin] if origin.is_file() else sorted(p for p in origin.rglob("*") if p.is_file())
        for file in files:
            relative = file.relative_to(SOURCE).as_posix()
            digest = sha256(file)
            target = ROOT / relative
            if target.exists():
                assert sha256(target) == digest, ("existing file differs", relative)
                action = "already_present_identical"
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(file, target)
                assert sha256(target) == digest
                action = "copied"
            rows.append(dict(path=relative, bytes=file.stat().st_size, sha256=digest,
                             hash_cited_in_tracked_docs=digest in known, action=action))
    out = ROOT / f"results/research/COMPLETION20261012/dependency_import_{label}.json"
    assert not out.exists(), out
    out.write_text(json.dumps(dict(schema=1, batch=label, created_utc=datetime.now(timezone.utc).isoformat(),
                                   source_checkout=SOURCE.as_posix(), requested=rels,
                                   scope="Byte copies only; no regeneration, training or inference.",
                                   files=rows), indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(dict(files=len(rows), copied=sum(r["action"] == "copied" for r in rows),
                          hash_cited=sum(r["hash_cited_in_tracked_docs"] for r in rows),
                          bytes=sum(r["bytes"] for r in rows))))


if __name__ == "__main__":
    main()
