"""Item-10 archive/Git audit and fresh-process scientific recount; no Torch imports.

Inputs are explicitly pinned full run roots. The wrapper hashes binary files but
only returns the declared acquisition TRAIN member from TAR. In preflight it
never opens a TEST payload. Git ancestry is recorded workflow evidence, not
independent blind custody. The parent worker supervises this process group.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile

import research_item9_worker as ops
from research_item10_inputs import recover_input

ROOT = Path(__file__).resolve().parents[1]
TRAIN_MEMBER = "tasks_1-20_v1-2/en/qa4_two-arg-relations_train.txt"
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")


def read_json(path):
    def reject(value):
        raise ValueError("nonfinite JSON constant: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)


def safe_relative(name):
    value = PurePosixPath(name)
    if not isinstance(name, str) or not name or value.is_absolute() or ".." in value.parts or value.as_posix() != name or name == ".":
        raise ValueError("noncanonical relative artifact path")
    return value


def hashes(stream, size=None):
    digest, git_digest, total = hashlib.sha256(), hashlib.sha1(), 0
    if size is not None:
        git_digest.update(("blob " + str(size) + "\0").encode("ascii"))
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
        git_digest.update(block)
        total += len(block)
    return {"bytes": total, "sha256": digest.hexdigest(), "git_blob": git_digest.hexdigest() if size is not None else None}


class Review:
    def __init__(self, output, deadline):
        self.output, self.deadline, self.sources = output, deadline, {}
        self.report = {"schema_version": 1, "item": 10, "status": "running",
                       "started_at_utc": ops.now(), "checks": [], "issues": [],
                       "archives": {}, "source_checkouts": {}, "resource_samples": [],
                       "scope": "Exact Git/archive byte checks; no model, tensor deserialization or independent custody."}
        self.source_head = None
        self.persist()

    def check(self, condition, label, detail=None):
        self.report["checks"].append({"check": label, "passed": bool(condition), "detail": detail})
        if not condition:
            raise ValueError(label)

    def persist(self):
        self.report["check_count"] = len(self.report["checks"])
        self.report["failed_checks"] = sum(not row["passed"] for row in self.report["checks"])
        ops.save(self.output / "offline_archive_audit.json", self.report)

    def boundary(self, label):
        self.report["resource_samples"].append({"label": label, **ops.admit(self.deadline, minimum_seconds=15)})

    def source(self, commit):
        self.check(bool(HEX40.fullmatch(commit)), "source_commit_format", commit)
        if commit in self.sources:
            return self.sources[commit]
        self.boundary("recover_source")
        if self.source_head is None:
            ops.git("fetch", "--no-filter", "--no-tags", "origin", ops.SOURCE_BRANCH, deadline=self.deadline)
            self.source_head = ops.git("rev-parse", "FETCH_HEAD", deadline=self.deadline)
        ops.git("merge-base", "--is-ancestor", commit, self.source_head, deadline=self.deadline)
        checkout = Path(tempfile.mkdtemp(prefix="neuropixel-item10-audit-source-")) / "tree"
        ops.git("worktree", "add", "--detach", checkout, commit, deadline=self.deadline)
        self.check(ops.git("rev-parse", "HEAD", cwd=checkout, deadline=self.deadline) == commit,
                   "source_detached_HEAD", commit)
        inventory = {}
        listing = ops.git("ls-tree", "-r", "-z", commit, cwd=checkout, deadline=self.deadline)
        for entry in filter(None, listing.split("\0")):
            metadata, name = entry.split("\t", 1)
            mode, kind, blob = metadata.split()
            safe_relative(name)
            self.check(kind == "blob" and mode in ("100644", "100755"), "regular_Git_source", name)
            path = checkout / name
            self.check(path.is_file() and not path.is_symlink(), "source_file_present", name)
            with path.open("rb") as stream:
                identity = hashes(stream, path.stat().st_size)
            self.check(identity["git_blob"] == blob, "source_Git_blob_identity", name)
            inventory[name] = {**identity, "mode": mode}
            ops.remaining(self.deadline)
        self.check(bool(inventory), "nonempty_source_inventory", commit)
        self.sources[commit] = (checkout, inventory)
        self.report["source_checkouts"][commit] = {"checkout": str(checkout), "observed_source_branch_head": self.source_head,
            "ancestor_verified": True, "files": inventory}
        self.persist()
        return checkout, inventory

    def zip_source(self, root, commit, label):
        checkout, inventory = self.source(commit)
        found = {}
        with zipfile.ZipFile(root / "source.zip") as archive:
            for entry in archive.infolist():
                if entry.is_dir():
                    continue
                safe_relative(entry.filename)
                mode = entry.external_attr >> 16
                self.check(stat.S_IFMT(mode) in (0, stat.S_IFREG) and entry.filename not in found,
                           "regular_unique_ZIP_entry", label + ":" + entry.filename)
                self.check(entry.filename in inventory, "ZIP_entry_in_Git_source", label + ":" + entry.filename)
                with archive.open(entry) as stream:
                    identity = hashes(stream)
                expected = inventory[entry.filename]
                self.check(identity["bytes"] == entry.file_size == expected["bytes"]
                           and identity["sha256"] == expected["sha256"],
                           "ZIP_file_exact_source_bytes", label + ":" + entry.filename)
                found[entry.filename] = {"bytes": identity["bytes"], "sha256": identity["sha256"]}
                ops.remaining(self.deadline)
        self.check(set(found) == set(inventory), "ZIP_full_source_inventory", label)
        return checkout

    def archive(self, label, spec, phase, interim=False):
        self.boundary("recover_" + label)
        ops.admit(self.deadline, minimum_seconds=180)
        root = Path(recover_input(spec, self.output, label)).resolve()
        actual, git_blobs = {}, {}
        for path in root.rglob("*"):
            self.check(not path.is_symlink(), "archive_no_symlinks", label + ":" + path.relative_to(root).as_posix())
            if path.is_file():
                name = path.relative_to(root).as_posix()
                safe_relative(name)
                with path.open("rb") as stream:
                    value = hashes(stream, path.stat().st_size)
                actual[name] = {"bytes": value["bytes"], "sha256": value["sha256"]}
                git_blobs[name] = value["git_blob"]
                ops.remaining(self.deadline)
        self.check(set(actual) == set(spec["files_sha256"]), "full_input_spec_inventory", label)
        self.check(all(actual[name]["sha256"] == value for name, value in spec["files_sha256"].items()),
                   "full_input_spec_hashes", label)
        manifest_name = "interim_manifest.json" if interim else "archive_manifest.json"
        manifest = read_json(root / manifest_name)
        expected_kind = "interim_partial" if interim else "final"
        self.check(manifest["schema_version"] == 1 and manifest["item"] == 10
                   and manifest["snapshot_kind"] == expected_kind and manifest["final"] is (not interim),
                   "manifest_kind", label)
        self.check(set(actual) == set(manifest["files"]) | {manifest_name}, "manifest_exact_inventory", label)
        for name, value in manifest["files"].items():
            safe_relative(name)
            self.check(value == actual[name], "manifest_file_identity", label + ":" + name)
        worker = read_json(root / "worker_status.json")
        self.check(manifest["run_key"] == root.name == worker["run_key"]
                   and manifest["source_commit"] == worker["source_commit"] and worker["phase"] == phase,
                   "archive_source_run_phase", label)
        statuses = ("running",) if interim else ("completed", "failed")
        self.check(manifest["attempt_status"] == worker["status"] and worker["status"] in statuses,
                   "archive_attempt_status_recorded", label)
        # A failed-but-complete negative preflight is not censored here. Its
        # completeness and scientific admission are decided by the recount.
        if phase == "acquisition":
            self.check(worker["status"] == "completed", "acquisition_completed", label)
        commit = spec["archive_commit"]
        listing = ops.git("ls-tree", "-r", "-z", commit, "--", spec["path"], deadline=self.deadline)
        prefix, tracked = spec["path"] + "/", set()
        for entry in filter(None, listing.split("\0")):
            metadata, name = entry.split("\t", 1)
            mode, kind, blob = metadata.split()
            self.check(name.startswith(prefix) and mode in ("100644", "100755") and kind == "blob",
                       "regular_archive_Git_entry", label + ":" + name)
            relative = name[len(prefix):]
            tracked.add(relative)
            self.check(git_blobs.get(relative) == blob, "archive_actual_Git_blob", label + ":" + relative)
        self.check(tracked == set(actual), "archive_Git_exact_file_set", label)
        source_root = self.zip_source(root, manifest["source_commit"], label)
        plan_path = source_root / ("docs/research/10_" + phase + "_plan.json")
        self.check((root / "execution_plan.json").read_bytes() == plan_path.read_bytes(),
                   "archived_execution_plan_exact_source", label)
        info = {"spec": spec, "directory": str(root), "files": actual, "manifest": manifest,
                "phase": phase, "worker_status": worker["status"], "source_root": str(source_root),
                "git_blob_sha1": git_blobs}
        self.report["archives"][label] = info
        self.persist()
        return root, info

    def train_payload(self, acquisition):
        directory = acquisition / "data"
        exposure = read_json(directory / "selected_source.json")
        safe_relative(exposure["archive_filename"])
        archive = directory / exposure["archive_filename"]
        member_info = exposure["train_member"]
        self.check(ops.sha(archive) == exposure["archive_sha256"] == member_info["archive_sha256"],
                   "acquisition_compressed_archive_identity")
        self.check(member_info["member"] == TRAIN_MEMBER, "only_original_TRAIN_member")
        cap = read_json(acquisition / "execution_plan.json")["acquisition"]["train_member_byte_cap"]
        found, total, count = None, 0, 0
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar:
                total += member.size
                count += 1
                self.check(0 <= member.size and total <= 1024**3 and count <= 5000, "bounded_TAR_header_inventory")
                if member.name == TRAIN_MEMBER:
                    self.check(found is None and member.isfile() and 0 < member.size <= cap, "unique_regular_TRAIN_member")
                    with tar.extractfile(member) as stream:
                        found = hashes(stream)
                    self.check(found["bytes"] == member.size, "TRAIN_member_length")
                ops.remaining(self.deadline)
        self.check(found is not None, "TRAIN_member_present")
        self.check(found["bytes"] == member_info["bytes"] and found["sha256"] == member_info["sha256"]
                   == ops.sha(directory / "selected_qa4_train.txt"), "reextracted_TRAIN_matches_saved_bytes")
        safe_relative(member_info["train_filename"])
        self.check(found["sha256"] == ops.sha(directory / member_info["train_filename"]), "source_specific_TRAIN_copy")
        self.report["train_reextraction"] = {"member": TRAIN_MEMBER, **found,
            "test_payload_extracted": False, "scope": "Other TAR headers traversed; only TRAIN returned as a payload."}

    def gate(self, target, target_info, prior, prior_info):
        target_spec, prior_spec = target_info["spec"], prior_info["spec"]
        self.check(target_spec["path"] == prior_spec["path"], "gate_same_run_path")
        ops.git("merge-base", "--is-ancestor", prior_spec["archive_commit"], target_spec["archive_commit"],
                deadline=self.deadline)
        self.check(prior_spec["archive_commit"] != target_spec["archive_commit"], "gate_strict_ancestor")
        self.check(prior_info["manifest"]["source_commit"] == target_info["manifest"]["source_commit"],
                   "gate_same_scientific_source")
        names = set(prior_info["files"])
        forbidden = ("final_data/", "final_predictions/")
        self.check(not any(name.startswith(forbidden) or name in
                   ("final_access.json", "final_metrics.json", "final_control_predictions.json") for name in names),
                   "gate_has_no_final_data_predictions_or_consumed_access")
        manifest = read_json(target / "training_manifest.json")
        self.check(ops.sha(target / "training_manifest.json") == ops.sha(prior / "training_manifest.json"),
                   "training_manifest_identical_G_and_F")
        self.check(len(manifest["runs"]) == 10 and len(set(manifest["ordered_run_ids"])) == 10
                   and [x["run_id"] for x in manifest["runs"]] == manifest["ordered_run_ids"],
                   "ten_training_runs_before_final")
        for item in manifest["runs"]:
            run_prefix = "study/runs/" + item["run_id"] + "/"
            before = {name: value for name, value in prior_info["files"].items() if name.startswith(run_prefix)}
            after = {name: value for name, value in target_info["files"].items() if name.startswith(run_prefix)}
            self.check(bool(before) and before == after, "complete_training_subtree_identical_G_and_F", item["run_id"])
            run = read_json(target / item["summary"]["path"])
            references = [item["summary"], item["checkpoint"], run["config_artifact"], run["minibatch_indices"],
                          run["training_log"], run["validation"]["predictions"], run["train_probe"]["predictions"]]
            for reference in references:
                name = reference["path"]
                safe_relative(name)
                value = {"bytes": reference["bytes"], "sha256": reference["sha256"]}
                self.check(name.startswith(run_prefix) and prior_info["files"].get(name) == value
                           and target_info["files"].get(name) == value, "training_reference_present_G_and_F", name)
        receipt = read_json(target / "pre_final_archive_receipt.json")
        self.check(receipt["schema_version"] == 1 and receipt["item"] == 10
                   and receipt["training_archive_commit"] == prior_spec["archive_commit"]
                   and receipt["training_manifest_sha256"] == ops.sha(prior / "training_manifest.json")
                   and receipt["source_commit"] == target_info["manifest"]["source_commit"]
                   and receipt["run_key"] == target.name, "actual_G_matches_final_gate_receipt")
        self.report["gate_ancestry"] = {"G": prior_spec["archive_commit"], "F": target_spec["archive_commit"],
                                       "ancestor_verified": True, "claim": "Actual Git lineage; not external custody."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["audit"], required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    plan, started = read_json(args.plan), time.monotonic()
    worker = read_json(output / "worker_status.json")
    elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(worker["started_at_utc"])).total_seconds()
    budget = min(plan["limits"]["worker_seconds"] - elapsed - 180,
                 float(os.environ["ITEM10_JOB_DEADLINE_UNIX"]) - time.time() - 180,
                 next(s["seconds"] for s in plan["stages"] if s["name"] == "audit"))
    review = Review(output, started + budget)
    state = {"schema_version": 1, "item": 10, "status": "running", "started_at_utc": ops.now(),
             "plan_sha256": ops.sha(args.plan), "wrapper_sha256": ops.sha(__file__), "budget_seconds": budget}
    ops.save(output / "offline_started.json", state, exclusive=True)
    try:
        review.check(math.isfinite(budget) and budget > 60, "finite_budget_after_archive_reserve")
        review.check(plan["item"] == 10 and plan["phase"] == "audit" and plan["status"] == "frozen"
                     and plan["limits"]["archive_reserve_seconds"] == 180, "frozen_operational_audit_plan")
        review.check(plan["runtime"]["threads"] == 1 and all(os.environ.get(k) == "1" for k in ops.THREAD_ENV),
                     "one_thread_before_child")
        review.check(platform.python_version() == plan["runtime"]["python"]
                     and all(importlib.metadata.version(k) == v for k, v in plan["runtime"]["packages"].items())
                     and "torch" not in plan["runtime"]["packages"], "declared_no_Torch_audit_runtime")
        spec = plan["audit"]
        phase = spec["scientific_phase"]
        required = {"target", "acquisition"} | ({"preflight", "gate"} if phase == "study" else set())
        review.check(phase in ("preflight", "study") and set(spec["inputs"]) == required, "audit_input_inventory")
        recovered = {}
        for label in ("target", "acquisition", "preflight", "gate"):
            if label in required:
                expected_phase = {"target": phase, "acquisition": "acquisition", "preflight": "preflight", "gate": "study"}[label]
                recovered[label] = review.archive(label, spec["inputs"][label], expected_phase, interim=(label == "gate"))
        target, target_info = recovered["target"]
        acquisition, _ = recovered["acquisition"]
        source, _ = review.source(spec["scientific_source_commit"])
        review.check(target_info["manifest"]["source_commit"] == spec["scientific_source_commit"],
                     "target_bound_scientific_source")
        safe_relative(spec["scientific_plan_path"])
        scientific_plan = source / spec["scientific_plan_path"]
        review.check(spec["scientific_plan_path"] == "docs/research/10_" + phase + "_plan.json"
                     and ops.sha(scientific_plan) == spec["scientific_plan_sha256"]
                     and scientific_plan.read_bytes() == (target / "execution_plan.json").read_bytes(),
                     "exact_frozen_scientific_plan")
        scientific = read_json(scientific_plan)
        for label, scientific_key, suffix in (("acquisition", "development", "/data"),
                                               ("preflight", "preflight", "")):
            if label == "preflight" and phase != "study":
                continue
            actual_spec = spec["inputs"][label]
            expected_input = scientific["inputs"][scientific_key]
            review.check(actual_spec["archive_commit"] == expected_input["archive_commit"]
                         and actual_spec["path"] + suffix == expected_input["path"],
                         "scientific_input_actual_archive_binding", label)
            prefix = "data/" if suffix else ""
            review.check(all(actual_spec["files_sha256"].get(prefix + name) == value
                             for name, value in expected_input["files_sha256"].items()),
                         "scientific_input_fullroot_hash_binding", label)
        auditor = ROOT / "scripts/research_babi_audit.py"
        review.check(ops.sha(auditor) == plan["implementation_sha256"]["scripts/research_babi_audit.py"]
                     == scientific["implementation_sha256"]["scripts/research_babi_audit.py"], "prospectively_bound_auditor_source")
        review.train_payload(acquisition)
        if phase == "study":
            review.gate(target, target_info, *recovered["gate"])
        command = [sys.executable, str(auditor), "--phase", phase, "--input", str(target),
                   "--development", str(acquisition / "data"), "--plan", str(scientific_plan),
                   "--source-root", str(source), "--output-dir", str(output / "analysis")]
        if phase == "study":
            command += ["--preflight", str(recovered["preflight"][0])]
        review.report.update(status="verified", completed_at_utc=ops.now())
        review.persist()
        review.boundary("before_fresh_scientific_process")
        child = {"command": command, "started_at_utc": ops.now(), "status": "running",
                 "process_scope": "Fresh child in the parent's supervised process group; no new session."}
        ops.save(output / "offline_subprocess.json", child)
        child_started = time.monotonic()
        process = None
        try:
            with (output / "offline_scientific_stdout.log").open("xb") as stdout, \
                 (output / "offline_scientific_stderr.log").open("xb") as stderr:
                process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stdout,
                                           stderr=stderr, env=dict(os.environ))
                code = process.wait(timeout=max(1, review.deadline - time.monotonic() - 15))
            child.update(returncode=code, status="completed" if code == 0 else "failed")
        except BaseException as error:
            child.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
            raise
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            child.update(completed_at_utc=ops.now(), wall_seconds=time.monotonic() - child_started,
                         returncode=process.poll() if process is not None else None)
            ops.save(output / "offline_subprocess.json", child)
        audit_path = output / "analysis/scientific_audit.json"
        report = read_json(audit_path)
        review.check(code == 0 and report["status"] == "verified" and not report["issues"], "scientific_recount_verified")
        for label, (directory, info) in recovered.items():
            review.check(all((directory / name).stat().st_size == value["bytes"]
                             and ops.sha(directory / name) == value["sha256"] for name, value in info["files"].items()),
                         "input_unchanged_after_child", label)
        for commit, (checkout, inventory) in review.sources.items():
            review.check(ops.git("rev-parse", "HEAD", cwd=checkout, deadline=review.deadline) == commit
                         and not ops.git("status", "--porcelain", "--untracked-files=all",
                                         cwd=checkout, deadline=review.deadline), "source_unchanged_after_child", commit)
        state.update(status="completed", scientific_report={"path": "analysis/scientific_audit.json",
                     "bytes": audit_path.stat().st_size, "sha256": ops.sha(audit_path)})
        return 0
    except BaseException as error:
        state.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        review.report["issues"].append(state["error"])
        review.report["status"] = "failed"
        raise
    finally:
        state.update(completed_at_utc=ops.now(), wall_seconds=time.monotonic() - started)
        review.report["completed_at_utc"] = ops.now()
        review.persist()
        ops.save(output / "offline_status.json", state, exclusive=True)


if __name__ == "__main__":
    raise SystemExit(main())
