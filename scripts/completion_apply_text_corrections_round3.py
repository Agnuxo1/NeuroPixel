"""A18.3 round 3: README negatives row references the VIS07 shift weakness (verifier follow-up)."""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT / "results/research/COMPLETION20261012/A18_text_corrections_round2.json"
ARCHIVE = ROOT / "docs/archive/completion_20261012/A18_round3"
OUT = ROOT / "results/research/COMPLETION20261012/A18_text_corrections_round3.json"
NAME = "README.md"
OLD = "| Yes. OriginalH1 remains unsupported; DEV04 original precision target failed; long continuations are not stable. |"
NEW = ("| Yes. OriginalH1 remains unsupported; DEV04 original precision target failed; long continuations are not stable; "
       "VIS07 accuracy collapses under a one-column shift (55.83% versus 85.12% for the CNN). |")


def main() -> None:
    assert not OUT.exists(), OUT
    expected = next(f["sha256_after"] for f in json.loads(PREVIOUS.read_text(encoding="utf-8"))["files"] if f["path"] == NAME)
    data = (ROOT / NAME).read_bytes()
    assert hashlib.sha256(data).hexdigest() == expected, "README differs from the round-2 result"
    (ARCHIVE).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / NAME, ARCHIVE / NAME)
    text = data.decode("utf-8")
    assert text.count(OLD) == 1
    line = text[: text.index(OLD)].count("\n") + 1
    new = text.replace(OLD, NEW, 1).encode("utf-8")
    (ROOT / NAME).write_bytes(new)
    OUT.write_text(json.dumps(dict(schema=1, created_utc=datetime.now(timezone.utc).isoformat(),
                                   archived_pre_round_bytes=str(ARCHIVE.relative_to(ROOT)).replace("\\", "/"),
                                   files=[dict(path=NAME, sha256_before=expected, sha256_after=hashlib.sha256(new).hexdigest())],
                                   corrections=[dict(file=NAME, line=line, claim="D-VIS07-05", before=OLD, after=NEW,
                                                     reason="Negatives row did not mention the VIS07 shift weakness")]),
                              indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(dict(files=1, corrections=1, line=line)))


if __name__ == "__main__":
    main()
