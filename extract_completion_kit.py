"""Extract authenticated, complete source blocks from the NeuroPixel Markdown guide."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def extract(guide: Path, destination: Path) -> list[dict]:
    """Refuse path escapes, duplicate blocks, hashes that differ and conflicting files."""
    text = guide.read_text(encoding="utf-8")
    pattern = re.compile(r"^```(?:python|text|glsl|json) file=([A-Za-z0-9_./-]+)\n(.*?)^```[ \t]*$", re.M | re.S)
    blocks = pattern.findall(text)
    manifest_pattern = re.compile(r"<!-- CODE_MANIFEST\n(.*?)\nEND_CODE_MANIFEST -->", re.S)
    manifest_match = manifest_pattern.search(text)
    if not blocks or not manifest_match:
        raise ValueError("Complete embedded code and its manifest are required")
    expected = json.loads(manifest_match.group(1))
    destination = destination.resolve()
    seen, writes = set(), []
    for relative, source in blocks:
        if relative in seen:
            raise ValueError("Duplicate source block: " + relative)
        seen.add(relative)
        path = (destination / relative).resolve()
        if not path.is_relative_to(destination) or ".." in Path(relative).parts:
            raise ValueError("Path outside destination: " + relative)
        raw = source.encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest()
        if expected.get(relative) != digest:
            raise ValueError("Embedded code identity differs: " + relative)
        if path.exists() and path.read_bytes() != raw:
            raise FileExistsError("Different existing source; review before replacement: " + str(path))
        writes.append((path, raw, dict(path=relative, sha256=digest, bytes=len(raw))))
    if seen != set(expected):
        raise ValueError("Embedded manifest lists missing or extra code blocks")
    for path, raw, _ in writes:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(raw)
    return [item[2] for item in writes]


def main() -> None:
    """Materialize the self-contained guide without overwriting conflicting code."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--guide", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(dict(status="extracted_and_authenticated", files=extract(args.guide, args.destination)), indent=2))


if __name__ == "__main__":
    main()
