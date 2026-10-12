"""G17.1: end-to-end commit-reveal custody rehearsal with dummy material (no real models, no real TEST).

Roles are played locally and labelled as a rehearsal: an "evaluator" commits a TEST specification,
the "team" seals a dummy cohort and commits its digest, the evaluator reveals, anyone verifies,
regenerates the TEST and scores the cohort whose digest matches. Tampering cases must fail.
Every custody step runs the real CLI (python -m completion_kit.commitments).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import completion_g3_splits as g3  # noqa: E402
from completion_kit.commitments import directory_digest  # noqa: E402

PUBLIC = ROOT / "results/research/G17_REHEARSAL_20261013"
SEALED = Path("E:/Proyecto-concursos/NeuroPixel-sealed/G17_rehearsal_20261013")
SCRATCH = Path("D:/PROJECTS/.cognition/neuropixel-completion-20261012/tmp/G17_rehearsal")
PY = sys.executable
steps: list[dict] = []


def cli(*args: str, expect: int) -> dict:
    result = subprocess.run([PY, "-m", "completion_kit.commitments", *args], cwd=ROOT, capture_output=True, text=True)
    row = dict(command="commitments " + " ".join(Path(a).name if os.sep in a or "/" in a else a for a in args),
               exit_code=result.returncode, expected=expect, ok=result.returncode == expect,
               stdout=result.stdout.strip()[-400:])
    steps.append(row)
    if not row["ok"]:
        raise SystemExit(json.dumps(row, indent=2))
    return row


def main() -> None:
    for folder in (PUBLIC, SEALED, SCRATCH):
        assert not folder.exists(), folder
        folder.mkdir(parents=True)
    newrel = json.loads((ROOT / "results/research/NEWREL20261013/manifest.json").read_text(encoding="utf-8"))
    excluded = json.loads((ROOT / "results/research/NEWREL20261013/excluded_families.json").read_text(encoding="utf-8"))
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()

    # 1. Evaluator fixes the TEST specification (secret seed) and publishes only the commitment.
    spec = dict(rehearsal=True, generator="scripts/completion_g3_splits.py:make_family_test", generator_commit=git,
                seed=int.from_bytes(os.urandom(6), "big"), families=6, variants=1, examples_per_variant=4,
                excluded_families_sha256=newrel["excluded_families"]["sha256"])
    (SEALED / "evaluator_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    cli("commit", "--value", str(SEALED / "evaluator_spec.json"), "--purpose", "REHEARSAL: TEST specification fixed by evaluator",
        "--public", str(PUBLIC / "evaluator_commitment_public.json"), "--private", str(SEALED / "evaluator_opening.json"), expect=0)

    # 2. Team seals a dummy cohort (three fake predictors) and commits the cohort digest afterwards.
    cohort = SCRATCH / "cohort"
    cohort.mkdir()
    for k in range(3):
        (cohort / f"dummy_model_{k}.json").write_text(json.dumps(dict(rehearsal=True, rule="answer=(start+%d)%%12" % k)), encoding="utf-8")
    cli("digest", "--root", str(cohort), "--output", str(PUBLIC / "cohort_digest.json"), expect=0)
    cli("commit", "--value", str(PUBLIC / "cohort_digest.json"), "--purpose", "REHEARSAL: sealed dummy cohort before TEST reveal",
        "--public", str(PUBLIC / "cohort_commitment_public.json"), "--private", str(SEALED / "cohort_opening.json"), expect=0)

    # 3. Reveal: both openings verify against their public commitments.
    cli("verify", "--public", str(PUBLIC / "evaluator_commitment_public.json"), "--private", str(SEALED / "evaluator_opening.json"), expect=0)
    cli("verify", "--public", str(PUBLIC / "cohort_commitment_public.json"), "--private", str(SEALED / "cohort_opening.json"), expect=0)

    # 4. Tampering must fail: changed seed, changed salt.
    opening = json.loads((SEALED / "evaluator_opening.json").read_text(encoding="utf-8"))
    forged = dict(opening, value=dict(opening["value"], seed=opening["value"]["seed"] + 1))
    (SCRATCH / "forged_seed.json").write_text(json.dumps(forged), encoding="utf-8")
    cli("verify", "--public", str(PUBLIC / "evaluator_commitment_public.json"), "--private", str(SCRATCH / "forged_seed.json"), expect=1)
    forged = dict(opening, salt_hex=("00" if opening["salt_hex"][:2] != "00" else "11") + opening["salt_hex"][2:])
    (SCRATCH / "forged_salt.json").write_text(json.dumps(forged), encoding="utf-8")
    cli("verify", "--public", str(PUBLIC / "evaluator_commitment_public.json"), "--private", str(SCRATCH / "forged_salt.json"), expect=1)

    # 5. Anyone regenerates TEST from the revealed specification; it excludes every development family.
    revealed = opening["value"]
    test = g3.make_family_test(SCRATCH / "test", seed=revealed["seed"], families=revealed["families"], variants=revealed["variants"],
                               examples_per_variant=revealed["examples_per_variant"], excluded=excluded)
    regenerated = g3.make_family_test(SCRATCH / "test_again", seed=revealed["seed"], families=revealed["families"],
                                      variants=revealed["variants"], examples_per_variant=revealed["examples_per_variant"],
                                      excluded=excluded)
    deterministic = all(test["panels"][k]["sha256"] == regenerated["panels"][k]["sha256"] for k in test["panels"])
    steps.append(dict(command="regenerate TEST twice from the revealed spec", ok=deterministic and test["overlap_with_excluded"] == 0,
                      overlap_with_excluded=test["overlap_with_excluded"], deterministic=deterministic))

    # 6. Score only a cohort whose current digest equals the committed digest; a modified cohort is refused.
    committed = json.loads((SEALED / "cohort_opening.json").read_text(encoding="utf-8"))["value"]["digest"]
    matches = directory_digest(cohort)["digest"] == committed
    data = np.load(SCRATCH / "test" / "test_base.npz")
    scores = {}
    for k in range(3):
        start = data["inputs"][:, :12].reshape(len(data["inputs"]), 12, -1).max(axis=2)  # entity channels at the start cell
        guesses = (np.argmax(start, axis=1) + k) % 12
        scores[f"dummy_model_{k}"] = float((guesses == data["targets"]).mean())
    (cohort / "dummy_model_0.json").write_text("tampered after sealing", encoding="utf-8")
    refused = directory_digest(cohort)["digest"] != committed
    steps.append(dict(command="score sealed dummy cohort; modify a file; recheck digest", ok=matches and refused,
                      cohort_digest_matched_before_scoring=matches, tampered_cohort_refused=refused, dummy_scores=scores))

    receipt = dict(schema=1, created_utc=datetime.now(timezone.utc).isoformat(), rehearsal=True, real_data=False,
                   all_steps_ok=all(s["ok"] for s in steps), steps=steps,
                   public_records=sorted(p.name for p in PUBLIC.iterdir()),
                   sealed_openings_location=str(SEALED).replace("\\", "/"),
                   limits=["Roles played locally: not independent custody",
                           "claimed_time_utc is not proof of time; external timestamps (owner approval) are required for real use",
                           "Dummy predictors and dummy TEST; no scientific result"])
    (PUBLIC / "rehearsal_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(dict(all_steps_ok=receipt["all_steps_ok"], steps=len(steps)), indent=1))
    if not receipt["all_steps_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
