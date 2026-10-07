"""Recover explicitly pinned item-10 evidence as read-only workflow inputs.

The helper checks bytes but never parses dataset examples, opens TAR members,
loads tensors or imports a model. Detached checkouts live outside the source
workspace. Read-only describes the workflow contract, not external custody.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import tempfile
import time

import research_item9_worker as ops


def require(value, message):
    if not value:
        raise ValueError(message)


def validate_spec(spec):
    require(isinstance(spec, dict) and set(spec) == {"archive_commit", "path", "files_sha256"},
            "input spec must name exactly archive_commit, path and files_sha256")
    require(isinstance(spec["archive_commit"], str) and re.fullmatch(r"[0-9a-f]{40}", spec["archive_commit"]),
            "invalid archived input commit")
    path = Path(spec["path"])
    require(not path.is_absolute() and ".." not in path.parts and path.as_posix() == spec["path"]
            and spec["path"].startswith("results/research/10_cloud_runs/"), "input directory outside item-10 evidence")
    require(isinstance(spec["files_sha256"], dict) and spec["files_sha256"], "input file bindings missing")
    for name, digest in spec["files_sha256"].items():
        file = Path(name)
        require(not file.is_absolute() and ".." not in file.parts and file.as_posix() == name and name != ".",
                "invalid relative input file")
        require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest), "invalid input SHA-256")
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def check_files(spec, checkout, deadline):
    directory = checkout / spec["path"]
    require(directory.is_dir() and not directory.is_symlink(), "pinned input directory absent")
    require(ops.git("rev-parse", "HEAD", cwd=checkout, deadline=deadline) == spec["archive_commit"],
            "pinned input checkout moved")
    for name, digest in spec["files_sha256"].items():
        file = directory / name
        require(file.is_file() and not file.is_symlink() and file.resolve().is_relative_to(directory.resolve()),
                "pinned input file absent or outside its directory")
        require(ops.sha(file) == digest, "pinned input bytes differ: " + name)
    return directory


def recover_input(spec, output_root, identifier):
    """Return exact spec.path under a verified detached archive checkout."""
    fingerprint = validate_spec(spec)
    require(isinstance(identifier, str) and re.fullmatch(r"[a-z][a-z0-9_-]*", identifier), "invalid input identifier")
    output_root = Path(output_root)
    receipts = output_root / "input_recovery"
    receipts.mkdir(exist_ok=True)
    path = receipts / (identifier + ".json")
    deadline = time.monotonic() + 180
    ops.admit(deadline, minimum_seconds=60)
    if path.exists():
        receipt = json.loads(path.read_bytes())
        require(receipt["spec_sha256"] == fingerprint and receipt["input_spec"] == spec,
                "input identifier reused for different bytes")
        return check_files(spec, Path(receipt["checkout"]), deadline)
    base = Path(tempfile.mkdtemp(prefix="neuropixel-item10-pinned-input-"))
    checkout = base / "tree"
    receipt = {"schema_version": 1, "item": 10, "identifier": identifier,
               "started_at_utc": ops.now(), "input_spec": spec, "spec_sha256": fingerprint,
               "checkout": str(checkout), "status": "running",
               "scope": "Pinned byte verification and read-only workflow checkout; no example parsing or model execution."}
    ops.save(receipts / (identifier + "_started.json"), receipt, exclusive=True)
    try:
        ops.git("fetch", "--no-filter", "--no-tags", "origin", ops.RESULTS_BRANCH, deadline=deadline)
        # Only an observed ancestor of the dedicated results branch is an admitted input.
        ops.git("merge-base", "--is-ancestor", spec["archive_commit"], "FETCH_HEAD", deadline=deadline)
        ops.git("worktree", "add", "--detach", checkout, spec["archive_commit"], deadline=deadline)
        directory = check_files(spec, checkout, deadline)
        receipt.update(status="verified", completed_at_utc=ops.now(), directory=str(directory),
                       files_verified=len(spec["files_sha256"]))
        ops.save(path, receipt, exclusive=True)
        return directory
    except BaseException as error:
        receipt.update(status="failed", completed_at_utc=ops.now(),
                       error={"type": type(error).__name__, "message": str(error)})
        ops.save(receipts / (identifier + "_failed.json"), receipt, exclusive=True)
        raise
