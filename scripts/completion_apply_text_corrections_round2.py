"""A18.3 round 2: publish the omitted VIS07 transformation results found by the independent verifier.

Each target file must still have the bytes left by round 1 (or its frozen A18.1 bytes if round 1
did not touch it). Pre-round bytes are archived under docs/archive/completion_20261012/A18_round2/.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROUND1 = ROOT / "results/research/COMPLETION20261012/A18_text_corrections.json"
FROZEN = ROOT / "docs/archive/completion_20261012/A18_sources"
ARCHIVE = ROOT / "docs/archive/completion_20261012/A18_round2"
OUT = ROOT / "results/research/COMPLETION20261012/A18_text_corrections_round2.json"
NOTE = "Evidence-level note (A18.3, 2026-10-13): "
TRANSFORM = ("the same frozen protocol also scored two registered test-time transformations: at half input intensity "
             "NeuroPixel averaged 88.737% versus 67.056% for the CNN (+21.681 pp), but with a one-column right shift and crop "
             "it averaged 55.826% versus 85.120% (-29.293 pp). NeuroPixel was far less robust to the shift "
             "(claim D-VIS07-05 in docs/research/claims_ledger.json).")

CORRECTIONS = [
    ("docs/research/VIS07_results.md", "## Profundidad y reproducción",
     "> " + NOTE + TRANSFORM + "\n\n## Profundidad y reproducción", "D-VIS07-05",
     "Registered transformation conditions were executed but not reported."),
    ("README.md", "near nominal parameter counts, differing functions/costs. |",
     "near nominal parameter counts, differing functions/costs. Under a one-column shift NeuroPixel fell to **55.83%** versus "
     "**85.12%** for the CNN; at half intensity it kept **88.74%** versus **67.06%** ([claims ledger](docs/research/claims_ledger.json), D-VIS07-05). |",
     "D-VIS07-05", "The headline row omitted the registered robustness conditions, one of them strongly negative."),
    ("docs/papers/NeuroPixel_manuscript_20261009.md", "The public corpus is not a blind custody evaluation.",
     "The public corpus is not a blind custody evaluation. Two registered test-time transformations were also scored: at half "
     "input intensity NeuroPixel averaged 88.737% versus 67.056% for the CNN, but with a one-column right shift and crop it "
     "averaged 55.826% versus 85.120%; NeuroPixel was far less robust to the shift.",
     "D-VIS07-05", "Registered transformation conditions were executed but not reported."),
]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    assert not OUT.exists(), OUT
    after_round1 = {f["path"]: f["sha256_after"] for f in json.loads(ROUND1.read_text(encoding="utf-8"))["files"]}
    files = sorted({c[0] for c in CORRECTIONS})
    texts, before = {}, {}
    for name in files:
        data = (ROOT / name).read_bytes()
        expected = after_round1.get(name) or sha((FROZEN / name).read_bytes())
        assert sha(data) == expected, "unexpected current bytes: " + name
        target = ARCHIVE / name
        target.parent.mkdir(parents=True, exist_ok=True)
        assert not target.exists(), target
        shutil.copyfile(ROOT / name, target)
        before[name], texts[name] = sha(data), data.decode("utf-8")
    log = []
    for name, old, new, claim_id, reason in CORRECTIONS:
        assert texts[name].count(old) == 1, (name, old[:60], texts[name].count(old))
        line = texts[name][: texts[name].index(old)].count("\n") + 1
        texts[name] = texts[name].replace(old, new, 1)
        log.append(dict(file=name, line=line, claim=claim_id, before=old, after=new, reason=reason))
    files_out = []
    for name in files:
        data = texts[name].encode("utf-8")
        (ROOT / name).write_bytes(data)
        files_out.append(dict(path=name, sha256_before=before[name], sha256_after=sha(data)))
    OUT.write_text(json.dumps(dict(schema=1, created_utc=datetime.now(timezone.utc).isoformat(),
                                   archived_pre_round_bytes=str(ARCHIVE.relative_to(ROOT)).replace("\\", "/"),
                                   files=files_out, corrections=log), indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8", newline="\n")
    print(json.dumps(dict(files=len(files), corrections=len(log))))


if __name__ == "__main__":
    main()
