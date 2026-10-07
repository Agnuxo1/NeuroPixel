"""Export only hash-bound invented input fixtures; no unittest methods run.

The reviewed test module constructs its deterministic DEVELOPMENT_FIXTURES
constant using stdlib. This helper executes those bound module definitions,
not test methods or official data acquisition. No TEST payload is opened.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--test-sha256", required=True)
    parser.add_argument("--module-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.source_root.resolve()
    test = root / "tests" / "test_babi_qa.py"
    module = root / "neuropixel" / "research" / "babi_qa.py"
    expected = {"tests/test_babi_qa.py": args.test_sha256,
                "neuropixel/research/babi_qa.py": args.module_sha256}
    for name, value in expected.items():
        if not re.fullmatch(r"[0-9a-f]{64}", value) or digest(root / name) != value:
            raise ValueError("fixture source hash differs: " + name)
    memory = {line.split()[0].rstrip(":"): int(line.split()[1]) * 1024
              for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines()
              if len(line.split()) >= 2}
    if memory.get("MemAvailable", 0) < 8 * 1024**3:
        raise RuntimeError("available RAM is below8 GiB")
    if args.output.exists():
        raise FileExistsError("fixture receipt already exists")
    spec = importlib.util.spec_from_file_location("item10_fixture_exposure_source", test)
    imported = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imported)
    fixtures = imported.DEVELOPMENT_FIXTURES
    if not isinstance(fixtures, (list, tuple)) or not fixtures:
        raise ValueError("fixture ledger is empty or malformed")
    clean, normalized = [], []
    for item in fixtures:
        if not isinstance(item, dict) or set(item) != {"facts", "question"}:
            raise ValueError("fixture ledger contains non-input metadata")
        if not isinstance(item["facts"], list) or not item["facts"]:
            raise ValueError("fixture facts are malformed")
        if any(not isinstance(value, str) or not value.strip()
               for value in item["facts"] + [item["question"]]):
            raise ValueError("fixture raw text is empty or malformed")
        clean.append({"facts": list(item["facts"]), "question": item["question"]})
        tokenizer = lambda value: re.findall(r"\w+|[^\w\s]", value.lower(), flags=re.UNICODE)
        normalized.append({"facts": [tokenizer(fact) for fact in item["facts"]],
                           "question": tokenizer(item["question"])})
    for name, value in expected.items():
        if digest(root / name) != value:
            raise ValueError("fixture source changed during export")
    raw_keys = [hashlib.sha256(canonical(item)).hexdigest() for item in clean]
    normalized_keys = [hashlib.sha256(canonical(item)).hexdigest() for item in normalized]
    receipt = {
        "schema_version": 1, "item": 10, "status": "exported",
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": expected,
        "extractor_sha256": digest(Path(__file__).resolve()),
        "fixture_count": len(clean), "unique_raw_string_contexts": len(set(raw_keys)),
        "unique_normalized_contexts": len(set(normalized_keys)),
        "fixtures": clean, "normalized_input_sha256": normalized_keys,
        "fixture_sequence_sha256": hashlib.sha256(canonical(clean)).hexdigest(),
        "declared_scope": imported.DEVELOPMENT_FIXTURE_SCOPE,
        "test_methods_executed": False, "official_payloads_opened": False,
        "normalization": "lowercase_word_punctuation_v1",
        "limits": "Includes valid raw contexts used for OOV/overflow or unsupported-symbolic checks; encoder failures must be counted separately without truncation.",
        "admission_available_bytes": memory["MemAvailable"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(json.dumps(receipt, sort_keys=True, ensure_ascii=False,
                                indent=2, allow_nan=False).encode("utf-8") + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"status": "exported", "fixture_count": len(clean),
                      "output_sha256": digest(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
