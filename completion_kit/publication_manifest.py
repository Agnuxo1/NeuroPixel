"""Authenticate every indexed original Git blob using its actual object identity."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .common import read_json, write_json, sha256_file


def git_blob_and_sha256(path: Path) -> tuple[str, str, int]:
    """Calculate the Git SHA-1 blob identity and SHA-256 with bounded memory."""
    size = path.stat().st_size
    git_digest = hashlib.sha1(f"blob {size}\0".encode("ascii"))
    raw_digest = hashlib.sha256()
    read = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            read += len(chunk)
            git_digest.update(chunk)
            raw_digest.update(chunk)
    if read != size or path.stat().st_size != size:
        raise ValueError("File changed during hashing: " + str(path))
    return git_digest.hexdigest(), raw_digest.hexdigest(), size


def verify_manifest(root: Path, manifest_path: Path, output: Path) -> dict:
    """Check all registered raw blobs, preserving a complete mismatch receipt."""
    if output.exists():
        raise FileExistsError("Audit receipt already exists")
    root = root.resolve()
    document = read_json(manifest_path)
    entries = document["raw_entries"]
    results, seen, issues = [], set(), []
    for entry in entries:
        relative = entry["path"]
        if relative in seen:
            raise ValueError("Repeated publication path: " + relative)
        seen.add(relative)
        path = (root / relative).resolve()
        record = dict(path=relative, expected_git_blob=entry["sha"],
                      declared_origin_commit=entry["origin_commit"])
        if (not path.is_relative_to(root) or not path.is_file()
                or entry["type"] != "blob" or entry["mode"] not in ("100644", "100755")):
            record.update(passed=False, error="Missing or invalid ordinary original file")
        else:
            actual, digest, size = git_blob_and_sha256(path)
            record.update(passed=actual == entry["sha"], actual_git_blob=actual, sha256=digest, bytes=size)
        if not record["passed"]:
            issues.append(relative)
        results.append(record)
        if len(results) % 100 == 0:
            print(json.dumps(dict(hashed=len(results), registered=len(entries), mismatches=len(issues))), flush=True)
    result = dict(status="verified_all_indexed_original_blobs" if not issues else "issues_found",
                  manifest_sha256=sha256_file(manifest_path), registered=len(entries),
                  verified=sum(item["passed"] for item in results), issues=issues, records=results,
                  origin_commit_ancestry_independently_verified=False,
                  scientific_replication=False, scope="Exact indexed raw-file identity only")
    write_json(output, result, exclusive=True)
    return result


def main() -> None:
    """Verify the entire publication index without loading neural checkpoints."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = args.manifest or args.root / "docs/research/publication_raw_manifest.json"
    result = verify_manifest(args.root, manifest, args.output)
    print(json.dumps({k: result[k] for k in ("status", "registered", "verified", "issues")}, indent=2))
    raise SystemExit(0 if not result["issues"] else 1)


if __name__ == "__main__":
    main()
