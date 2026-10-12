"""Salted SHA-256 commit-reveal records for blind custody; time needs external proof."""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import canonical_bytes, read_json, sha256_file, write_json

DOMAIN = b"NeuroPixel-commitment-v1\x00"


def commitment_digest(value: Any, salt: bytes) -> str:
    """Bind a canonical JSON value to a secret salt under a fixed domain tag."""
    if not isinstance(salt, bytes) or not 16 <= len(salt) <= 255:
        raise ValueError("Between 16 and 255 salt bytes required")
    # The length prefix stops salt bytes from being reread as value bytes.
    return hashlib.sha256(DOMAIN + bytes([len(salt)]) + salt + canonical_bytes(value)).hexdigest()


def commit(value: Any, *, purpose: str, salt: bytes | None = None) -> tuple[dict, dict]:
    """Return a public record to publish now and a private opening to keep sealed."""
    if not purpose.strip():
        raise ValueError("Declare what the commitment fixes")
    salt = os.urandom(32) if salt is None else salt
    digest = commitment_digest(value, salt)
    public = dict(schema=1, algorithm="sha256(domain||len(salt)||salt||canonical_json)", commitment=digest,
                  purpose=purpose,
                  claimed_time_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  timestamp_evidence=None,
                  scope="Binding and hiding under SHA-256; the claimed time is not proof of time")
    private = dict(schema=1, commitment=digest, salt_hex=salt.hex(), value=value)
    return public, private


def verify_opening(public: dict, private: dict) -> bool:
    """Check that a revealed value and salt open the published commitment."""
    if not isinstance(public, dict) or not isinstance(private, dict):
        return False
    try:
        digest = commitment_digest(private["value"], bytes.fromhex(private["salt_hex"])).encode("ascii")
    except (KeyError, TypeError, ValueError):
        return False
    return (hmac.compare_digest(digest, str(public.get("commitment", "")).encode("utf-8"))
            and hmac.compare_digest(digest, str(private.get("commitment", "")).encode("utf-8")))


def directory_digest(root: Path) -> dict:
    """Identify every regular file of a sealed cohort; links, junctions and escapes are refused."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError("Directory required")
    entries = []
    # A parent sorts before its children, so a junction is refused before its contents are read.
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink() or path.is_junction() or not path.resolve().is_relative_to(root):
            raise ValueError("Links, junctions and escaping paths are not admitted in a sealed cohort: " + str(path))
        if path.is_file():
            entries.append(dict(path=path.relative_to(root).as_posix(), sha256=sha256_file(path),
                                bytes=path.stat().st_size))
    if not entries:
        raise ValueError("Empty cohort")
    return dict(schema=1, files=entries, digest=hashlib.sha256(canonical_bytes(entries)).hexdigest(),
                scope="File identities only; custody order needs published commitments and timestamps")


def main() -> None:
    """Commit, open or digest custody material without overwriting evidence."""
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("commit")
    make.add_argument("--value", type=Path, required=True)
    make.add_argument("--purpose", required=True)
    make.add_argument("--public", type=Path, required=True)
    make.add_argument("--private", type=Path, required=True)
    check = commands.add_parser("verify")
    check.add_argument("--public", type=Path, required=True)
    check.add_argument("--private", type=Path, required=True)
    digest = commands.add_parser("digest")
    digest.add_argument("--root", type=Path, required=True)
    digest.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "commit":
        public, private = commit(read_json(args.value), purpose=args.purpose)
        write_json(args.private, private, exclusive=True)
        write_json(args.public, public, exclusive=True)
        print(json.dumps(public, indent=2))
    elif args.command == "verify":
        valid = verify_opening(read_json(args.public), read_json(args.private))
        print(json.dumps(dict(valid=valid), indent=2))
        raise SystemExit(0 if valid else 1)
    else:
        record = directory_digest(args.root)
        write_json(args.output, record, exclusive=True)
        print(json.dumps(dict(files=len(record["files"]), digest=record["digest"]), indent=2))


if __name__ == "__main__":
    main()
