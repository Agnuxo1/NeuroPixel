"""A18.3: apply the archived text corrections found by the A18.2 claim review.

Each correction is an exact, unique string replacement on a public document whose current bytes
must still equal the frozen A18.1 copy (archived under docs/archive/completion_20261012/A18_sources/).
Results and receipts are never edited. A machine-readable log records file hashes before and after.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/archive/completion_20261012/A18_sources"
OUT = ROOT / "results/research/COMPLETION20261012/A18_text_corrections.json"
NOTE = "Evidence-level note (A18.3, 2026-10-13): "

CORRECTIONS = [
    ("docs/research/06_A06_1_results.md", "| +0,659 | [−0,272; +1,590] |", "| +0,659 | [−0,271; +1,590] |", "H06-04",
     "Receipt audit_saved.json gives -0.27145 pp, which rounds to -0.271; no retained file supports -0.272."),
    ("docs/research/04_results.md", "- `results/research/04_verification.json`: independent artifact audit.",
     "- `results/research/04_verification.json`: independent artifact audit. " + NOTE + "this file is not retained; the "
     "60-file transfer bundle `results/research/04_experiment/item4_evidence.zip` was re-verified against its own manifest.",
     "H04-04", "The cited verifier output is missing; the statement now names the evidence that exists."),
    ("docs/research/05_results.md", "**Report status: 28 final evaluations and both artifact checks verified. H1: `not_supported`. Updated 2026-10-07.**",
     "**Report status: 28 final evaluations and both artifact checks verified. H1: `not_supported`. Updated 2026-10-07.**\n\n"
     "> " + NOTE + "the analysis, independent recount, run tables, CSV, diagnostics, replication records and archive linked in "
     "this report are not present in this repository, on the project's local disks or on the remote branches. The executor "
     "archive (commit 0a56621) must be recovered by the owner; until then these claims are registered at level L0 "
     "(`blocked_external`) in `docs/research/claims_ledger.json`.",
     "H05-01..H05-09", "Verification statements refer to bytes that are not retained; level stated explicitly."),
    ("docs/research/VIS07_results.md", "T8 es el punto entrenado.",
     "> " + NOTE + "the RMS column above is not reproducible from retained receipts; the retained 128-row depth diagnostic "
     "gives mean state RMS 0.543 (T0), 149.3 (T16) and 134,843 (T32). Accuracy values are reproduced exactly.\n\nT8 es el punto entrenado.",
     "D-VIS07-02", "RMS source not retained; the retained diagnostic values are stated."),
    ("docs/research/OPT03_ROLE_results.md", "La admisión real del reanudador registró8,6749GiB",
     "La admisión real del reanudador registró8,6749GiB (" + NOTE + "value not present in the retained review files)",
     "D-ROLE-02", "Admission value not found in retained files."),
    ("docs/research/RENDER08_results.md", "| Workload | CUDA eager ms | CUDA Graph ms | Render vector ms | Mejor CUDA / render |",
     "| Workload | CUDA eager ms | CUDA Graph ms | Render escalar ms | Mejor CUDA / render |", "G-R08-01",
     "RENDER08 has one (scalar) renderer method; the column label said 'vector'."),
    ("docs/research/RENDER09_results.md", "otras configuraciones no superanCUDA y las mejoras pequeñas pueden depender delruido.",
     "otras dos configuraciones superan CUDA solo marginalmente (1,039 en C16/grid8 y 1,072 en RGB8), las cuatro restantes no "
     "lo superan, y las mejoras pequeñas pueden depender del ruido.", "G-R09-01",
     "Two other workloads have ratios above 1; the sentence said none did."),
    ("README.md", "3.14–12.25×", "3.14–12.24×", "G-R08-01", "Receipt medians give 3.138-12.244x."),
    ("docs/papers/NeuroPixel_manuscript_20261009.md", "3.14–12.25×", "3.14–12.24×", "G-R08-01", "Receipt medians give 3.138-12.244x."),
    ("docs/research/REPRODUCE_20261009.md",
     "RENDER11 retained100/105 original energy blocks after RAM8 stopped it; no complete comparative energy or wall-energy claim follows.",
     "RENDER11 originally retained 103/105 energy blocks after the RAM floor stopped it; its saved recovery is admitted below; "
     "no wall-energy claim follows.", "G-R11-02", "Stale count: the original receipt stores 103 rows."),
    ("README.md", "- Frozen plans, original checkpoints, datasets, raw predictions/states, failed attempts and hash manifests under `results/research/`.",
     "- Frozen plans, original checkpoints, datasets, raw predictions/states, failed attempts and hash manifests under "
     "`results/research/`, except the item-5 CPU archive and the historical reassessment outputs, which are unavailable; large "
     "recovery archives remain on local disks outside the Git tree. Evidence levels: "
     "[claims ledger](docs/research/claims_ledger.json).", "P-AVAIL-01", "Availability was overstated."),
    ("docs/papers/NeuroPixel_manuscript_20261009.md",
     "Frozen plans, source commits, weights, datasets, predictions, raw states, receipts, Git object identities and SHA256 manifests are supplied.",
     "Frozen plans, source commits, weights, datasets, predictions, raw states, receipts, Git object identities and SHA256 manifests "
     "are supplied, except the item-5 CPU archive and the historical reassessment outputs, which are unavailable.",
     "P-AVAIL-01", "Availability was overstated."),
    ("docs/research/RESULTS_INDEX_20261009.md", "All raw archives and their original Gitblobs are included/indexed by the publication manifest.",
     "All raw archives retained by the project and their original Gitblobs are included/indexed by the publication manifest; the "
     "item-5 CPU archive and historical reassessment outputs are unavailable.", "P-AVAIL-01", "Availability was overstated."),
    ("docs/research/REPRODUCE_20261009.md", "The publication manifest maps all raw archives to their original Gitblobs",
     "The publication manifest maps all retained raw archives (not the unavailable item-5 CPU archive) to their original Gitblobs",
     "P-AVAIL-01", "Availability was overstated."),
    ("README.md", "## What is demonstrated", "## What has been measured (internal evidence, levels L1–L2)", "P-ARCH-01",
     "The table reports internal L1-L2 measurements, not demonstrated general capabilities."),
    ("docs/papers/NeuroPixel_manuscript_20261009.md", "The implemented neural field demonstrates bounded fidelity,",
     "Within internal evidence levels L1–L2, the implemented neural field shows bounded fidelity,", "P-GPU-01",
     "Wording aligned with the registered evidence level."),
    ("README.md", "*Capability figure from the pre-scientific README. Memory and rest-state behaviour are not established by the current evidence; see the manuscript for the bounded claims.*",
     "*Capability figure from the pre-scientific README (illustrative, level L0, not registered evidence). Memory, rest-state and "
     "growth behaviour are not established by the current evidence; see the manuscript and the claims ledger for the bounded claims.*",
     "P-ARCH-01", "Pre-scientific figure labelled with its evidence level."),
    ("README.md", "*Scaling comparison. The original hypothesis H1 was not supported; see the results index.*",
     "*Historical retrospective scaling comparison (level L0: its reassessment outputs are not retained). The prospective test of "
     "the original hypothesis H1 was not supported (-37.0 pp versus the fixed Transformer); see the results index.*",
     "H05-05", "Historical figure labelled with its evidence level and the prospective negative result."),
]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    assert not OUT.exists(), OUT
    files = sorted({c[0] for c in CORRECTIONS})
    texts, before = {}, {}
    for name in files:
        data = (ROOT / name).read_bytes()
        frozen = (ARCHIVE / name).read_bytes()
        assert data == frozen, "current bytes differ from the frozen archived copy: " + name
        before[name] = sha(data)
        texts[name] = data.decode("utf-8")
    log = []
    for name, old, new, claim_id, reason in CORRECTIONS:
        assert texts[name].count(old) == 1, (name, old[:60], texts[name].count(old))
        line = texts[name][: texts[name].index(old)].count("\n") + 1
        texts[name] = texts[name].replace(old, new, 1)
        log.append(dict(file=name, line=line, claim=claim_id, before=old, after=new, reason=reason))
    after = {}
    for name in files:
        data = texts[name].encode("utf-8")
        (ROOT / name).write_bytes(data)
        after[name] = sha(data)
    OUT.write_text(json.dumps(dict(schema=1, created_utc=datetime.now(timezone.utc).isoformat(),
                                   archived_originals=str(ARCHIVE.relative_to(ROOT)).replace("\\", "/"),
                                   files=[dict(path=n, sha256_before=before[n], sha256_after=after[n]) for n in files],
                                   corrections=log), indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(dict(files=len(files), corrections=len(log)), indent=1))


if __name__ == "__main__":
    main()
