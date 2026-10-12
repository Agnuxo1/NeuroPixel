"""G3 audits: structural-family overlap of the kit generator, and the registered NEWREL train/DEV build."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import completion_g3_splits as g3  # noqa: E402
from completion_kit.datasets import make_dataset  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path("D:/PROJECTS/.cognition/neuropixel-completion-20261012/tmp/G3_kit_audit")
OUT = ROOT / "results/research/COMPLETION20261012/G3_kit_generator_family_audit.json"

if SCRATCH.exists():
    shutil.rmtree(SCRATCH)
manifest = make_dataset(SCRATCH, seed=20261012)
audit = g3.audit_family_overlap({name: SCRATCH / (name + ".npz") for name in ("train", "dev", "test_base", "test_long")})
record = dict(generator="completion_kit.datasets.make_dataset (default seed 20261012, 96/24/24 graphs)",
              kit_grouping=manifest["grouping"], panels=audit,
              interpretation=("test_base and test_long share graphs by design (paired diagnostics). Overlap counts between "
                              "train/dev/test families measure leakage under the structural-family unit; exact-identity "
                              "grouping alone does not guarantee zero."),
              scratch_dataset_not_retained=str(SCRATCH).replace("\\", "/"))
OUT.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
print(json.dumps(audit, indent=1))
