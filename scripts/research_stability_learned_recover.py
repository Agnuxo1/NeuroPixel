"""Recover the fixed item15 learned-input archive from existing Git objects.

Read-only source/object access. Only declared regular blobs are copied into the
owned result directory, before any weights or arrays are deserialized.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import research_item9_worker as ops

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_COMMIT = "b64d0e2ba222df76caf72bc9870c7602873e7236"
ORIGINAL_SOURCE = "83fe135e301f76bc0c74e30c66bb18e067ca5959"
ORIGINAL_PATH = "results/research/09_cloud_runs/37593731891-1-study"

def require(ok, message):
    if not ok:
        raise ValueError(message)

def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def relpath(name):
    p = Path(name)
    require(not p.is_absolute() and ".." not in p.parts and p.as_posix() == name,
            "unsafe relative path")
    return p

def recover(args):
    started = time.monotonic()
    deadline = started + 120
    args.output.mkdir(parents=True, exist_ok=False)
    state = {"schema_version": 1, "item": 15, "phase": "learned_input_recovery",
             "source_commit": os.environ.get("GITHUB_SHA"), "status": "running",
             "started_at_utc": ops.now(), "recovered": []}
    try:
        plan = json.loads(args.plan.read_bytes())
        require(plan["item"] == 15 and plan["status"] == "frozen", "unfrozen plan")
        spec = plan["inputs"]["learned"]
        require(spec["archive_commit"] == ORIGINAL_COMMIT
                and spec["source_commit"] == ORIGINAL_SOURCE
                and spec["path"] == ORIGINAL_PATH, "different original archive")
        mapping, sizes = spec["files_sha256"], spec["files_bytes"]
        expected = {"archive_manifest.json", "study/development_data/validation.npz"}
        expected.update("study/runs/neuropixel_seed" + str(seed) + "/" + name
                        for seed in range(40,45) for name in
                        ("checkpoint.pt", "validation_predictions.npz", "configuration.json", "run.json"))
        require(set(mapping) == set(sizes) == expected and len(expected) == 22,
                "different historical input inventory")
        require(sum(sizes.values()) == 1777007, "historical input byte count differs")
        bindings = plan["implementation_sha256"]
        require({p: digest(ROOT / relpath(p)) for p in bindings} == bindings, "source binding differs")
        ops.admit(deadline, minimum_seconds=20)
        require(all(os.environ.get(k) == "1" for k in ops.THREAD_ENV), "thread policy")
        require(ops.git("rev-parse", "HEAD", deadline=deadline) == os.environ["GITHUB_SHA"], "source HEAD moved")
        require(ops.git("rev-parse", ORIGINAL_COMMIT + "^{commit}", deadline=deadline) == ORIGINAL_COMMIT,
                "original archive object missing")
        # Worker Archive has already fetched the isolated results history.
        # No external download or ref mutation is required to read these objects.
        require(ops.git("show", "-s", "--format=%H", ORIGINAL_COMMIT, deadline=deadline) == ORIGINAL_COMMIT,
                "archive identity differs")
        sequence = ["archive_manifest.json"] + sorted(expected - {"archive_manifest.json"})
        manifest = None
        for name in sequence:
            ops.admit(deadline, minimum_seconds=5)
            path = args.output / relpath(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            original = ORIGINAL_PATH + "/" + name
            object_name = ORIGINAL_COMMIT + ":" + original
            listing = ops.git("ls-tree", ORIGINAL_COMMIT, "--", original, deadline=deadline)
            require(listing.startswith("100644 blob ") and listing.endswith("\t" + original),
                    "historical input is not an ordinary regular blob: " + name)
            require(ops.git("cat-file", "-t", object_name, deadline=deadline) == "blob",
                    "historical input is not a blob")
            require(int(ops.git("cat-file", "-s", object_name, deadline=deadline)) == sizes[name],
                    "historical Git blob size differs")
            if name != "archive_manifest.json":
                require(manifest is not None and manifest["files"][name] ==
                        {"bytes": sizes[name], "sha256": mapping[name]},
                        "original manifest does not bind this input")
            with path.open("xb") as out:
                completed = subprocess.run(["git", "-c", "pack.threads=1", "-c", "index.threads=1",
                        "show", object_name], cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out,
                        stderr=subprocess.PIPE, check=False,
                        timeout=min(60, ops.remaining(deadline)), env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
            require(completed.returncode == 0, "Git object read failed: " + name)
            require(path.stat().st_size == sizes[name] and digest(path) == mapping[name],
                    "historical input checksum differs: " + name)
            state["recovered"].append({"path": name, "bytes": sizes[name], "sha256": mapping[name],
                                      "archive_commit": ORIGINAL_COMMIT})
            if name == "archive_manifest.json":
                manifest = json.loads(path.read_bytes())
                require(manifest["source_commit"] == ORIGINAL_SOURCE
                        and manifest["item"] == 9 and manifest["final"] is True
                        and manifest["snapshot_kind"] == "final"
                        and manifest["attempt_status"] == "completed", "original archive not complete")
            ops.save(args.output / "recovery_status.json", state)
        require({p: digest(ROOT / relpath(p)) for p in bindings} == bindings, "source changed")
        require(ops.git("rev-parse", "HEAD", deadline=deadline) == os.environ["GITHUB_SHA"], "source HEAD changed")
        state.update(status="verified", inputs=spec, plan_sha256=digest(args.plan),
                     binary_files_deserialized=0, network_downloads=0, reference_mutations=0)
    except BaseException as error:
        state.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
    state.update(completed_at_utc=ops.now(), wall_seconds=time.monotonic()-started,
                 final_resource_observation=ops.resource_sample("item15_learned_input_recovery_finish"))
    if state["final_resource_observation"]["available_ram_gib"] < 8:
        state.update(status="failed", final_ram_floor_violation=True)
    ops.save(args.output / "recovery_status.json", state)
    print(json.dumps({"status": state["status"], "recovered_files": len(state["recovered"])}), flush=True)
    return int(state["status"] != "verified")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return recover(parser.parse_args())

if __name__ == "__main__":
    raise SystemExit(main())
