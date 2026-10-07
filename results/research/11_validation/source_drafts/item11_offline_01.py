"""Item-11 exact archive/Git audit and fresh-process saved-array recounts.

No Torch, checkpoint deserialization, model inference, training or dataset
generation. Inputs are pinned complete run roots. The original item-9 transport
is reused under item-11 owned-process supervision and input recovery. Git
lineage and recorded chronology do not establish independent blind custody.
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
import tempfile
import time
import zipfile

import research_item9_worker as ops
from research_item11_inputs import recover_input

ROOT = Path(__file__).resolve().parents[1]
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
        self.report = {"schema_version": 1, "item": 11, "status": "running",
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
        checkout = Path(tempfile.mkdtemp(prefix="neuropixel-item11-audit-source-")) / "tree"
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
        self.check(manifest["schema_version"] == 1 and manifest["item"] == 11
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
        # An incomplete/failed operational attempt remains readable evidence;
        # a verified scientific recount still requires its complete inventory.
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
        plan_path = source_root / ("docs/research/11_" + phase + "_plan.json")
        self.check((root / "execution_plan.json").read_bytes() == plan_path.read_bytes(),
                   "archived_execution_plan_exact_source", label)
        info = {"spec": spec, "directory": str(root), "files": actual, "manifest": manifest,
                "phase": phase, "worker_status": worker["status"], "source_root": str(source_root),
                "git_blob_sha1": git_blobs}
        self.report["archives"][label] = info
        self.persist()
        return root, info

    def gate(self, target, target_info, prior, prior_info):
        target_spec, prior_spec = target_info["spec"], prior_info["spec"]
        self.check(target_spec["path"] == prior_spec["path"], "gate_same_run_path")
        ops.git("merge-base", "--is-ancestor", prior_spec["archive_commit"], target_spec["archive_commit"],
                deadline=self.deadline)
        self.check(prior_spec["archive_commit"] != target_spec["archive_commit"], "gate_strict_ancestor")
        self.check(prior_info["manifest"]["source_commit"] == target_info["manifest"]["source_commit"],
                   "gate_same_scientific_source")
        names = set(prior_info["files"])
        forbidden_files = {"final_inventory.json", "final_access.json", "final_metrics.json",
                           "pre_final_archive_receipt.json"}
        self.check(not (names & forbidden_files)
                   and not any(name.startswith(("final_predictions/", "final_data/")) for name in names),
                   "gate_has_no_final_inventory_predictions_access_or_gate_receipt")
        manifest = read_json(target / "training_manifest.json")
        self.check(ops.sha(target / "training_manifest.json") == ops.sha(prior / "training_manifest.json"),
                   "training_manifest_identical_G_and_F")
        self.check(manifest["schema_version"] == 1 and manifest["item"] == 11
                   and manifest["status"] == "completed" and manifest["run_key"] == target.name
                   and manifest["source_commit"] == target_info["manifest"]["source_commit"],
                   "training_manifest_completed_identity")
        self.check(len(manifest["runs"]) == 10 and len(set(manifest["ordered_run_ids"])) == 10
                   and [x["run_id"] for x in manifest["runs"]] == manifest["ordered_run_ids"],
                   "ten_training_runs_before_final")
        for item in manifest["runs"]:
            self.check(isinstance(item["run_id"], str) and re.fullmatch(r"train_(neuropixel|gru)_s7[0-4]", item["run_id"]),
                       "declared_training_run_id", item["run_id"])
            run_prefix = "study/runs/" + item["run_id"] + "/"
            before = {name: value for name, value in prior_info["files"].items() if name.startswith(run_prefix)}
            after = {name: value for name, value in target_info["files"].items() if name.startswith(run_prefix)}
            self.check(bool(before) and before == after, "complete_training_subtree_identical_G_and_F", item["run_id"])
            safe_relative(item["summary"]["path"])
            self.check(item["summary"]["path"].startswith(run_prefix), "summary_in_own_run_subtree", item["run_id"])
            run = read_json(target / item["summary"]["path"])
            self.check(run["run_id"] == item["run_id"] and run["status"] == "completed"
                       and run["completed_updates"] == 2048 and run["config"] == item["config"]
                       and run["config_sha256"] == item["config_sha256"]
                       and run["checkpoint"] == item["checkpoint"], "completed_run_matches_manifest", item["run_id"])
            self.check(set(run["development"]) == {"normal_d2", "normal_d3", "normal_d4"},
                       "exact_development_inventory_before_final", item["run_id"])
            references = [item["summary"], item["checkpoint"], run["config_artifact"],
                          run["minibatch_indices"], run["training_delays"], run["training_log"],
                          run["nonpersistent_buffers"]]
            for key in ("normal_d2", "normal_d3", "normal_d4"):
                references.extend([run["development"][key]["predictions"], run["development"][key]["metadata"]])
            self.check(len(references) == 13 and len({ref["path"] for ref in references}) == 13,
                       "thirteen_distinct_run_references", item["run_id"])
            for reference in references:
                name = reference["path"]
                safe_relative(name)
                value = {"bytes": reference["bytes"], "sha256": reference["sha256"]}
                self.check(name.startswith(run_prefix) and prior_info["files"].get(name) == value
                           and target_info["files"].get(name) == value, "training_reference_present_G_and_F", name)
        receipt = read_json(target / "pre_final_archive_receipt.json")
        self.check(receipt["schema_version"] == 1 and receipt["item"] == 11
                   and receipt["training_archive_commit"] == prior_spec["archive_commit"]
                   and receipt["training_manifest_sha256"] == ops.sha(prior / "training_manifest.json")
                   and receipt["source_commit"] == target_info["manifest"]["source_commit"]
                   and receipt["run_key"] == target.name, "actual_G_matches_final_gate_receipt")
        prior_worker = read_json(prior / "worker_status.json")
        final_worker = read_json(target / "worker_status.json")
        self.check([s["name"] for s in prior_worker["stages"]] == ["contract_tests", "train"]
                   and all(s["status"] == "completed" and s["return_code"] == 0 for s in prior_worker["stages"]),
                   "gate_archived_after_both_completed_stages")
        self.check([s["name"] for s in final_worker["stages"]] == ["contract_tests", "train", "final"]
                   and all(s["status"] == "completed" and s["return_code"] == 0 for s in final_worker["stages"]),
                   "final_archive_has_three_completed_stages")
        access, metrics = read_json(target / "final_access.json"), read_json(target / "final_metrics.json")
        moments = {
            "training_manifest": manifest["created_at_utc"],
            "train_completed": prior_worker["stages"][-1]["completed_at_utc"],
            "G_snapshot": prior_info["manifest"]["created_at_utc"],
            "gate_receipt": receipt["at_utc"],
            "final_child_started": final_worker["stages"][-1]["started_at_utc"],
            "final_access": access["access_at_utc"],
            "final_metrics_completed": metrics["completed_at_utc"],
            "F_snapshot": target_info["manifest"]["created_at_utc"],
        }
        parsed = []
        for name, value in moments.items():
            instant = datetime.fromisoformat(value)
            self.check(instant.tzinfo is not None and instant.utcoffset() is not None,
                       "aware_recorded_timestamp", name)
            parsed.append(instant)
        self.check(all(a <= b for a, b in zip(parsed, parsed[1:])), "recorded_G_to_F_event_order", moments)
        self.report["gate_ancestry"] = {"G": prior_spec["archive_commit"], "F": target_spec["archive_commit"],
            "ancestor_verified": True, "recorded_chronology": moments,
            "claim": "Actual Git lineage and internally recorded chronology; no independent push-time or external-custody claim."}

    def child(self, label, command, report_relative):
        self.boundary("before_fresh_" + label + "_process")
        record = {"command": command, "started_at_utc": ops.now(), "status": "running",
                  "process_scope": "Fresh child in the parent's supervised process group; no new session."}
        record_path = self.output / ("offline_" + label + "_subprocess.json")
        self.check(not record_path.exists(), "exclusive_subprocess_record", label)
        ops.save(record_path, record, exclusive=True)
        started, process, code = time.monotonic(), None, None
        try:
            with (self.output / ("offline_" + label + "_stdout.log")).open("xb") as stdout, \
                 (self.output / ("offline_" + label + "_stderr.log")).open("xb") as stderr:
                process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stdout,
                                           stderr=stderr, env=dict(os.environ))
                code = process.wait(timeout=max(1, self.deadline - time.monotonic() - 15))
            record.update(returncode=code, status="completed" if code == 0 else "failed")
        except BaseException as error:
            record.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
            raise
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            record.update(completed_at_utc=ops.now(), wall_seconds=time.monotonic() - started,
                          returncode=process.poll() if process is not None else None)
            ops.save(record_path, record)
        audit_path = self.output / report_relative
        report = read_json(audit_path)
        self.check(code == 0 and report["status"] == "verified" and not report["issues"],
                   "saved_array_recount_verified", label)
        return {"path": report_relative, "bytes": audit_path.stat().st_size, "sha256": ops.sha(audit_path)}


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
                 float(os.environ["ITEM11_JOB_DEADLINE_UNIX"]) - time.time() - 180,
                 next(s["seconds"] for s in plan["stages"] if s["name"] == "audit"))
    review = Review(output, started + budget)
    state = {"schema_version": 1, "item": 11, "status": "running", "started_at_utc": ops.now(),
             "plan_sha256": ops.sha(args.plan), "wrapper_sha256": ops.sha(__file__), "budget_seconds": budget}
    ops.save(output / "offline_started.json", state, exclusive=True)
    try:
        review.check(math.isfinite(budget) and budget > 60, "finite_budget_after_archive_reserve")
        review.check(plan["item"] == 11 and plan["phase"] == "audit" and plan["status"] == "frozen"
                     and plan["limits"]["archive_reserve_seconds"] == 180, "frozen_operational_audit_plan")
        review.check(plan["runtime"]["threads"] == 1 and all(os.environ.get(k) == "1" for k in ops.THREAD_ENV),
                     "one_thread_before_child")
        review.check(platform.python_version() == plan["runtime"]["python"]
                     and all(importlib.metadata.version(k) == v for k, v in plan["runtime"]["packages"].items())
                     and "torch" not in plan["runtime"]["packages"], "declared_no_Torch_audit_runtime")
        spec = plan["audit"]
        phase = spec["scientific_phase"]
        populations = {"probe": {"target"}, "preflight": {"target", "probe"},
                       "study": {"target", "preflight", "gate", "probe"}}
        review.check(phase in populations and set(spec["inputs"]) == populations[phase], "audit_input_inventory")
        recovered = {}
        for label in ("target", "probe", "preflight", "gate"):
            if label in populations[phase]:
                expected_phase = {"target": phase, "probe": "probe", "preflight": "preflight", "gate": "study"}[label]
                recovered[label] = review.archive(label, spec["inputs"][label], expected_phase, interim=(label == "gate"))
        target, target_info = recovered["target"]
        source, _ = review.source(spec["scientific_source_commit"])
        review.check(target_info["manifest"]["source_commit"] == spec["scientific_source_commit"],
                     "target_bound_scientific_source")
        safe_relative(spec["scientific_plan_path"])
        scientific_plan = source / spec["scientific_plan_path"]
        review.check(spec["scientific_plan_path"] == "docs/research/11_" + phase + "_plan.json"
                     and ops.sha(scientific_plan) == spec["scientific_plan_sha256"]
                     and scientific_plan.read_bytes() == (target / "execution_plan.json").read_bytes(),
                     "exact_frozen_scientific_plan")
        scientific = read_json(scientific_plan)
        probe, probe_info = recovered["target"] if phase == "probe" else recovered["probe"]
        probe_source = Path(probe_info["source_root"])
        probe_plan_path = probe_source / "docs/research/11_probe_plan.json"
        probe_plan = read_json(probe_plan_path)
        probe_auditor = ROOT / "scripts/research_memory_probe_audit.py"
        review.check(ops.sha(probe_auditor) == plan["implementation_sha256"]["scripts/research_memory_probe_audit.py"],
                     "operationally_bound_probe_recount_source")
        reports = {}
        if phase != "probe":
            review.check(spec["inputs"]["probe"] == scientific["inputs"]["probe"],
                         "scientific_probe_complete_archive_binding")
            for name in ("neuropixel/model.py", "neuropixel/phase3.py", "neuropixel/research/stream_memory.py"):
                review.check(probe_plan["implementation_sha256"][name] == scientific["implementation_sha256"][name],
                             "probe_and_learned_study_shared_mechanism_bytes", name)
            auditor = ROOT / "scripts/research_memory_audit.py"
            review.check(ops.sha(auditor) == plan["implementation_sha256"]["scripts/research_memory_audit.py"]
                         == scientific["implementation_sha256"]["scripts/research_memory_audit.py"],
                         "prospectively_bound_learned_auditor_source")
        if phase == "study":
            actual_spec, expected_input = spec["inputs"]["preflight"], scientific["inputs"]["preflight"]
            review.check(actual_spec["archive_commit"] == expected_input["archive_commit"]
                         and actual_spec["path"] == expected_input["path"],
                         "scientific_preflight_actual_archive_binding")
            review.check(all(actual_spec["files_sha256"].get(name) == value
                             for name, value in expected_input["files_sha256"].items()),
                         "scientific_preflight_fullroot_hash_binding")
            prior, prior_info = recovered["preflight"]
            prior_source = Path(prior_info["source_root"])
            prior_plan_path = prior_source / "docs/research/11_preflight_plan.json"
            prior_plan = read_json(prior_plan_path)
            review.check(ops.sha(auditor) == prior_plan["implementation_sha256"]["scripts/research_memory_audit.py"],
                         "same_auditor_bound_in_preflight_and_study")
            review.gate(target, target_info, *recovered["gate"])
        review.report["archive_checks_completed_at_utc"] = ops.now()
        review.persist()
        probe_command = [sys.executable, str(probe_auditor), "--input", str(probe),
                         "--plan", str(probe_plan_path), "--source-root", str(probe_source),
                         "--output-dir", str(output / "probe_analysis")]
        reports["probe"] = review.child("probe", probe_command, "probe_analysis/probe_audit.json")
        if phase != "probe":
            command = [sys.executable, str(auditor), "--phase", phase, "--input", str(target),
                       "--plan", str(scientific_plan), "--source-root", str(source),
                       "--output-dir", str(output / "analysis")]
            if phase == "study":
                command += ["--preflight", str(prior), "--preflight-plan", str(prior_plan_path),
                            "--preflight-source-root", str(prior_source)]
            reports[phase] = review.child("scientific", command, "analysis/scientific_audit.json")
        for label, (directory, info) in recovered.items():
            after_names = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
            review.check(after_names == set(info["files"])
                         and all((directory / name).stat().st_size == value["bytes"]
                                 and ops.sha(directory / name) == value["sha256"] for name, value in info["files"].items()),
                         "input_unchanged_after_children", label)
        for commit, (checkout, inventory) in review.sources.items():
            review.check(ops.git("rev-parse", "HEAD", cwd=checkout, deadline=review.deadline) == commit
                         and not ops.git("status", "--porcelain", "--untracked-files=all",
                                         cwd=checkout, deadline=review.deadline), "source_unchanged_after_children", commit)
        review.report.update(status="verified", completed_at_utc=ops.now(), scientific_reports=reports)
        state.update(status="completed", scientific_reports=reports)
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
