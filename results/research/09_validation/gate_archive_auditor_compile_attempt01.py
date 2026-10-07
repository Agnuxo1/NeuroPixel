"""Read-only verification of the item-9 remote-archive gate from fetched Git objects.

The caller supplies independently recovered final and remote-head commit IDs.
No fetch, checkout, model, checkpoint deserialization or scientific recount occurs.
An exclusive JSON receipt preserves encountered issues and dependency limitations.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import traceback


RESULTS_BRANCH = "research/scientific-validation-2026-10-07-cloud-results"
FAMILIES = ("neuropixel", "relative_transformer")


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def commit_id(value):
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise argparse.ArgumentTypeError("commit IDs must be exactly 40 lowercase hexadecimal characters")
    return value


def relative_name(value):
    if not isinstance(value, str):
        raise ValueError("artifact path must be a string")
    path = PurePosixPath(value)
    if (not value or not path.parts or path.is_absolute() or ".." in path.parts or "." in path.parts
            or path.as_posix() != value or "\x00" in value):
        raise ValueError("artifact path must be canonical and relative")
    return value


def instant(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("timezone-aware recorded timestamp required")
    return result


def json_pairs(values):
    result = {}
    for key, value in values:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def nonfinite(value):
    raise ValueError("nonfinite JSON value: " + value)


def finite_float(value):
    result = float(value)
    if not math.isfinite(result):
        nonfinite(value)
    return result


def decode(payload):
    return json.loads(payload, object_pairs_hook=json_pairs, parse_constant=nonfinite, parse_float=finite_float)


class Audit:
    def __init__(self, args):
        self.args = args
        self.raw = args.input.resolve()
        self.repo = None
        self.prefix = None
        self.gate_commit = None
        self.manifests = {}
        self.report = {
            "schema_version": 1, "item": 9, "status": "running", "started_at_utc": now(),
            "input": str(self.raw), "source_commit": args.source_commit,
            "final_archive_commit": args.archive_commit, "provided_remote_head": args.remote_head,
            "checks": {}, "issues": [], "dependency_limits": [], "git_object_inputs": {},
            "model_or_rng_execution": False, "checkpoint_deserialization": False,
            "network_access": False,
            "limits": [
                "The caller must independently recover and identify the supplied final archive and remote branch head; this helper does not contact the server.",
                "Git ancestry authenticates content and structural ordering. Commit dates and worker timestamps do not independently attest when a push reached a remote server.",
                "The frozen worker checks push success and ls-remote equality before writing the receipt. Saved logs can corroborate its recorded sequence, not provide external blinding or custody.",
                "Absence from the intermediate Git tree does not prove absence from unrecorded files, machines or prior human access.",
                "Intermediate manifests describe the completed copy, not an atomic snapshot of every live file. The gate snapshot follows completed training; its referenced artifacts are checked directly.",
                "This same-team artifact audit neither establishes external replication nor changes the scientific protocol or earlier H1."]}

    def check(self, name, condition, detail=None):
        if name in self.report["checks"]:
            raise ValueError("duplicate audit check: " + name)
        passed = bool(condition)
        self.report["checks"][name] = passed
        if not passed:
            self.report["issues"].append({"check": name, "detail": detail})
        return passed

    def section(self, name, function):
        try:
            function()
        except Exception as error:
            self.report["issues"].append({"section": name, "type": type(error).__name__,
                "message": str(error), "traceback": traceback.format_exc(limit=6)})
            self.report["dependency_limits"].append(
                f"Section {name} could not complete; no success is inferred for its remaining dependent checks.")

    def git(self, *arguments, cwd=None, allowed=(0,)):
        result = subprocess.run(
            ["git", "--no-replace-objects", "--no-optional-locks", *arguments],
            cwd=cwd or self.repo, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=60,
            env=dict(os.environ, GIT_NO_LAZY_FETCH="1", GIT_TERMINAL_PROMPT="0"))
        if result.returncode not in allowed:
            raise RuntimeError(f"read-only Git command failed ({result.returncode}): "
                               + result.stderr.decode(errors="replace")[:2000])
        return result.returncode, result.stdout

    def blob(self, revision, relative):
        relative_name(relative)
        path = self.prefix + "/" + relative
        _, payload = self.git("cat-file", "blob", revision + ":" + path)
        self.report["git_object_inputs"][revision + ":" + path] = {
            "bytes": len(payload), "sha256": digest(payload)}
        return payload

    def json(self, revision, relative):
        return decode(self.blob(revision, relative))

    def tree(self, revision):
        _, payload = self.git("ls-tree", "-r", "-z", revision, "--", self.prefix + "/")
        result = {}
        for entry in payload.split(b"\x00"):
            if not entry:
                continue
            metadata, path = entry.split(b"\t", 1)
            mode, kind, identity = metadata.decode().split()
            path = path.decode("utf-8")
            if not path.startswith(self.prefix + "/"):
                raise ValueError("Git tree entry outside requested run")
            name = relative_name(path[len(self.prefix)+1:])
            self.check(f"regular_git_file:{revision}:{name}",
                       mode in ("100644", "100755") and kind == "blob")
            if name in result:
                raise ValueError("duplicate Git tree path")
            result[name] = {"mode": mode, "kind": kind, "git_blob": identity}
        return result

    def identity(self, revision, name, descriptor, label):
        if (not isinstance(descriptor, dict) or set(descriptor) != {"bytes", "sha256"}
                or type(descriptor["bytes"]) is not int or descriptor["bytes"] < 0
                or not isinstance(descriptor["sha256"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", descriptor["sha256"])):
            raise ValueError("invalid bytes/SHA descriptor: " + label)
        payload = self.blob(revision, name)
        self.check(label, len(payload) == descriptor["bytes"] and digest(payload) == descriptor["sha256"])
        return payload

    def study_artifact(self, descriptor, label):
        if not isinstance(descriptor, dict) or set(descriptor) != {"path", "bytes", "sha256"}:
            raise ValueError("invalid artifact descriptor: " + label)
        name = "study/" + relative_name(descriptor["path"])
        identity = {key: descriptor[key] for key in ("bytes", "sha256")}
        old = self.identity(self.gate_commit, name, identity, label + ":intermediate")
        new = self.identity(self.args.archive_commit, name, identity, label + ":final")
        self.check(label + ":unchanged_bytes", old == new)
        for revision in (self.gate_commit, self.args.archive_commit):
            self.check(label + ":manifest:" + revision,
                       self.manifests[revision]["files"].get(name) == identity)
        return old

    def foundation(self):
        if not self.raw.is_dir():
            raise ValueError("--input must be the recovered complete raw directory")
        _, root = self.git("rev-parse", "--show-toplevel", cwd=self.raw)
        self.repo = Path(root.decode().strip()).resolve()
        self.prefix = relative_name(self.raw.relative_to(self.repo).as_posix())
        if not re.fullmatch(r"results/research/09_cloud_runs/\d+-\d+-study", self.prefix):
            raise ValueError("input is not an item-9 study run/attempt directory")
        self.report.update(repository=str(self.repo), raw_git_prefix=self.prefix)
        for label, revision in (("source", self.args.source_commit),
                                ("final", self.args.archive_commit), ("remote_head", self.args.remote_head)):
            _, kind = self.git("cat-file", "-t", revision)
            self.check("commit_object:" + label, kind.strip() == b"commit")
        _, head = self.git("rev-parse", "HEAD")
        self.check("recovered_checkout_head", head.decode().strip() == self.args.archive_commit)
        code, _ = self.git("merge-base", "--is-ancestor", self.args.archive_commit,
                           self.args.remote_head, allowed=(0, 1))
        self.check("final_is_ancestor_of_provided_remote_head", code == 0)
        self.final_manifest = self.json(self.args.archive_commit, "archive_manifest.json")
        self.manifests[self.args.archive_commit] = self.final_manifest
        m = self.final_manifest
        self.check("complete_final_manifest", m["schema_version"] == 1 and m["item"] == 9
                   and m["final"] is True and m["snapshot_kind"] == "final"
                   and m["attempt_status"] == "completed" and m["source_commit"] == self.args.source_commit
                   and m["run_key"] == self.raw.name)
        self.final_tree = self.tree(self.args.archive_commit)
        self.check("final_manifest_inventory", set(self.final_tree) == set(m["files"]) | {"archive_manifest.json"})
        self.receipt = self.json(self.args.archive_commit, "study/gate_archive_receipt.json")
        self.identity(self.args.archive_commit, "study/gate_archive_receipt.json",
                      m["files"]["study/gate_archive_receipt.json"], "final_receipt_manifest_hash")
        r = self.receipt
        self.check("receipt_schema", set(r) == {"schema_version", "item", "source_commit", "archive_commit",
            "results_branch", "final_inventory_sha256", "archived_at_utc"}
            and r["schema_version"] == 1 and r["item"] == 9
            and r["source_commit"] == self.args.source_commit and r["results_branch"] == RESULTS_BRANCH)
        self.gate_commit = commit_id(r["archive_commit"])
        self.report["intermediate_archive_commit"] = self.gate_commit
        _, kind = self.git("cat-file", "-t", self.gate_commit)
        self.check("intermediate_commit_object", kind.strip() == b"commit")
        self.check("intermediate_differs_from_final", self.gate_commit != self.args.archive_commit)
        code, _ = self.git("merge-base", "--is-ancestor", self.gate_commit,
                           self.args.archive_commit, allowed=(0, 1))
        self.check("intermediate_is_ancestor_of_final", code == 0)
        _, chain = self.git("rev-list", "--ancestry-path", "--reverse",
                             self.gate_commit + ".." + self.args.archive_commit)
        self.report["descendant_path_commits"] = chain.decode().splitlines()
        self.report["commit_metadata"] = {}
        for revision in (self.gate_commit, self.args.archive_commit, self.args.remote_head):
            _, value = self.git("show", "-s", "--format=%H%n%P%n%aI%n%cI", revision)
            self.report["commit_metadata"][revision] = value.decode().splitlines()

    def intermediate(self):
        if self.gate_commit is None:
            raise ValueError("intermediate commit unavailable after foundation checks")
        m = self.json(self.gate_commit, "interim_manifest.json")
        self.manifests[self.gate_commit] = m
        self.check("intermediate_manifest_schema", m["schema_version"] == 1 and m["item"] == 9
                   and m["source_commit"] == self.args.source_commit and m["run_key"] == self.raw.name
                   and m["snapshot_kind"] == "interim_partial" and m["final"] is False
                   and m["attempt_status"] == "running")
        tree = self.tree(self.gate_commit)
        self.check("intermediate_full_inventory", set(tree) == set(m["files"]) | {"interim_manifest.json"})
        for name, descriptor in m["files"].items():
            self.section("intermediate_file:" + name,
                         lambda name=name, descriptor=descriptor: self.identity(
                             self.gate_commit, relative_name(name), descriptor, "intermediate_hash:" + name))
        forbidden = []
        for name in tree:
            for component in PurePosixPath(name).parts:
                if ((component.startswith("final_") or component == "final.log")
                        and name != "study/final_inventory.json") or component == "gate_archive_receipt.json":
                    forbidden.append(name)
                    break
        self.check("no_final_or_receipt_in_intermediate", not forbidden, forbidden)
        self.check("no_final_manifest_in_intermediate", "archive_manifest.json" not in tree)
        self.report["intermediate_inventory"] = {"manifested_files": len(m["files"]),
            "manifested_bytes": sum(x["bytes"] for x in m["files"].values()),
            "forbidden_paths_found": forbidden, "git_regular_files": len(tree)}

    def references(self):
        g, f = self.gate_commit, self.args.archive_commit
        payload = self.blob(g, "study/final_inventory.json")
        self.check("inventory_receipt_hash", digest(payload) == self.receipt["final_inventory_sha256"])
        self.check("inventory_unchanged_in_final", payload == self.blob(f, "study/final_inventory.json"))
        for revision in (g, f):
            self.identity(revision, "study/final_inventory.json",
                          self.manifests[revision]["files"]["study/final_inventory.json"],
                          "inventory_manifest_hash:" + revision)
        self.inventory = inventory = decode(payload)
        self.plan = plan = self.json(g, "execution_plan.json")
        recipe = self.json(g, "experiment_recipe.json")
        self.check("source_plan_recipe_unchanged", self.blob(g, "execution_plan.json") == self.blob(f, "execution_plan.json")
                   and self.blob(g, "experiment_recipe.json") == self.blob(f, "experiment_recipe.json")
                   and self.blob(g, "source.zip") == self.blob(f, "source.zip"))
        context = inventory["context"]
        self.check("inventory_source_plan_recipe", context["source_commit"] == self.args.source_commit
                   and context["plan_sha256"] == digest(self.blob(g, "execution_plan.json"))
                   and context["recipe_sha256"] == plan["recipe_sha256"] == digest(self.blob(g, "experiment_recipe.json")))
        self.check("study_plan_phase", plan["item"] == 9 and plan["phase"] == "study" and plan["status"] == "frozen")
        rates = plan["pilot_evidence"]["selected_learning_rates"]
        self.check("exact_recipe_population", set(rates) == set(FAMILIES)
                   and recipe["training"]["primary_initializations"] == [40, 41, 42, 43, 44]
                   and recipe["training"]["primary_updates"] == 4096
                   and rates == {"neuropixel": 0.001, "relative_transformer": 0.003})
        expected = [{"run_id": f"{family}_seed{seed}", "family": family, "initialization": seed,
                     "learning_rate": rates[family], "updates": 4096}
                    for seed in range(40, 45) for family in FAMILIES]
        self.check("plan_declared_inventory", plan["expected_primary_runs"] == expected)
        self.check("complete_inventory", inventory["schema_version"] == 1 and inventory["item"] == 9
                   and len(inventory["entries"]) == 10 and inventory["selected_learning_rates"] == rates
                   and inventory["final_data_generated"] is False and inventory["final_recipe"] == recipe["data"])
        index = decode(self.study_artifact(inventory["primary_index"], "primary_index"))
        names = [x["run_id"] for x in expected]
        self.check("completed_primary_index", inventory["primary_index"]["path"] == "primary_runs.json"
                   and index["status"] == "completed" and index["runs"] == names
                   and index["statuses"] == dict.fromkeys(names, "completed"))
        self.report["primary_references"] = []
        for number, (entry, spec) in enumerate(zip(inventory["entries"], expected)):
            def verify_entry(entry=entry, spec=spec, number=number):
                label = spec["run_id"]
                self.check("entry:" + label, set(entry) == set(spec) | {"summary", "checkpoint"}
                           and all(entry[key] == value for key, value in spec.items()))
                self.check("entry_paths:" + label,
                           entry["summary"]["path"] == f"runs/{label}/run.json"
                           and entry["checkpoint"]["path"] == f"runs/{label}/checkpoint.pt")
                summary = decode(self.study_artifact(entry["summary"], label + ":summary"))
                self.study_artifact(entry["checkpoint"], label + ":checkpoint")
                self.check("completed_summary:" + label, summary["status"] == "completed"
                           and summary["phase"] == "primary" and summary["context"] == context
                           and summary["updates_completed"] == 4096
                           and all(summary[key] == spec[key] for key in ("run_id", "family", "initialization", "learning_rate"))
                           and summary["artifacts"]["checkpoint"] == entry["checkpoint"])
                artifacts = summary["artifacts"]
                self.check("summary_artifact_population:" + label,
                           set(artifacts) == {"checkpoint", "training_log", "probe_predictions", "validation_predictions"})
                for key, reference in artifacts.items():
                    if key != "checkpoint":
                        self.study_artifact(reference, label + ":" + key)
                self.report["primary_references"].append({**spec, "summary": entry["summary"],
                    "checkpoint": entry["checkpoint"], "all_nested_artifacts_checked": True})
            self.section("primary_entry_" + str(number), verify_entry)
        self.check("ten_reference_sections_completed", len(self.report["primary_references"]) == 10)
        source = self.json(g, "source_before.json")
        self.check("intermediate_source_guard", source["verified"] is True
                   and source["source_commit"] == self.args.source_commit
                   and source["plan_sha256"] == context["plan_sha256"]
                   and source["implementation_sha256"] == plan["implementation_sha256"])

    def chronology(self):
        g, f = self.gate_commit, self.args.archive_commit
        train_status = self.json(g, "train_status.json")
        self.check("train_status_unchanged", self.blob(g, "train_status.json") == self.blob(f, "train_status.json"))
        worker_g, worker_f = self.json(g, "worker_status.json"), self.json(f, "worker_status.json")
        self.check("worker_source_run_identity", all(x["source_commit"] == self.args.source_commit
                   and x["phase"] == "study" and x["run_key"] == self.raw.name
                   and x["results_branch"] == RESULTS_BRANCH for x in (worker_g, worker_f)))
        self.check("intermediate_training_complete", [x["name"] for x in worker_g["stages"]] == ["contract_tests", "train"]
                   and all(x["status"] == "completed" and x["return_code"] == 0 for x in worker_g["stages"])
        self.check("final_worker_complete", worker_f["status"] == "completed"
                   and [x["name"] for x in worker_f["stages"]] == ["contract_tests", "train", "final"]
                   and all(x["status"] == "completed" and x["return_code"] == 0 for x in worker_f["stages"])
        access = self.json(f, "study/final_access.json")
        self.check("consumed_access_links", access["schema_version"] == 1 and access["item"] == 9
                   and access["source_commit"] == self.args.source_commit
                   and access["final_inventory_sha256"] == self.receipt["final_inventory_sha256"]
                   and access["consumed_before_final_generation"] is True and access["retry_allowed"] is False)
        descriptor = access["archive_receipt"]
        self.check("access_receipt_path", descriptor["path"] == "gate_archive_receipt.json")
        self.identity(f, "study/gate_archive_receipt.json", {k: descriptor[k] for k in ("bytes", "sha256")}, "access_receipt_identity")
        final_manifest = self.json(f, "study/final_data/manifest.json")
        self.check("final_dataset_links_consumed_access", final_manifest["created_after_access_sha256"]
                   == digest(self.blob(f, "study/final_access.json")))
        final_status = self.json(f, "final_status.json")
        train_start, train_end = self.json(g, "study/train_started.json"), self.json(g, "study/train_completed.json")
        final_start, final_end = self.json(f, "study/final_started.json"), self.json(f, "study/final_completed.json")
        self.check("phase_completion_records", train_end["status"] == final_end["status"] == "completed"
                   and train_end["primary_runs"] == 10 and train_end["final_performance_data_generated"] is False
                   and final_end["checkpoints_evaluated"] == 10)
        sequence = [
            ("source_freeze", self.plan["freeze_utc"]),
            ("train_worker_start", train_status["started_at_utc"]),
            ("train_scientific_start", train_start["at_utc"]),
            ("final_inventory_created", self.inventory["created_at_utc"]),
            ("train_scientific_complete", train_end["at_utc"]),
            ("train_worker_complete", train_status["completed_at_utc"]),
            ("intermediate_copy_manifest_created", self.manifests[g]["created_at_utc"]),
            ("gate_archive_receipt_written", self.receipt["archived_at_utc"]),
            ("final_worker_start", final_status["started_at_utc"]),
            ("final_scientific_start", final_start["at_utc"]),
            ("final_access_consumed", access["claimed_at_utc"]),
            ("final_scientific_complete", final_end["at_utc"]),
            ("final_worker_complete", final_status["completed_at_utc"]),
            ("source_after", self.json(f, "source_after.json")["at_utc"]),
            ("worker_complete", worker_f["completed_at_utc"]),
            ("final_copy_manifest_created", self.final_manifest["created_at_utc"])]
        for (left, a), (right, b) in zip(sequence, sequence[1:]):
            self.check("chronology:" + left + "->" + right, instant(a) <= instant(b))
        self.report["recorded_chronology"] = [{"event": name, "at_utc": value} for name, value in sequence]

    def job_log(self):
        if self.args.job_log is None:
            self.report["job_log"] = {"provided": False, "scope": "No external job-log corroboration requested."}
            return
        path = self.args.job_log.resolve()
        payload = path.read_bytes()
        text = payload.decode("utf-8-sig")
        archived, final_heartbeats = [], []
        decoder = json.JSONDecoder(object_pairs_hook=json_pairs, parse_constant=nonfinite, parse_float=finite_float)
        for line_number, line in enumerate(text.splitlines(), 1):
            position = line.find("{")
            if position < 0:
                continue
            try:
                event, _ = decoder.raw_decode(line[position:])
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            record = {"line": line_number, "prefix": line[:position], "event": event}
            if event.get("event") == "evidence_archived" and event.get("archive_commit") == self.gate_commit:
                archived.append(record)
            elif event.get("event") == "resource_heartbeat" and event.get("phase") == "final":
                final_heartbeats.append(record)
        self.report["job_log"] = {"provided": True, "path": str(path), "bytes": len(payload),
            "sha256": digest(payload), "intermediate_archive_events": archived,
            "final_heartbeat_events": final_heartbeats,
            "scope": "The frozen worker emits no final-stage-start stdout event. Log order can establish archive event before the first observed final heartbeat; start ordering also compares recorded event UTC with the saved final_status timestamp. Prefix timestamps are retained without assuming their origin."}
        self.check("job_log_one_matching_archive_event", len(archived) == 1)
        self.check("job_log_final_heartbeat_present", bool(final_heartbeats))
        if len(archived) != 1:
            self.report["dependency_limits"].append("No unique matching archive event; its log chronology cannot be authenticated.")
            return
        row = archived[0]
        event = row["event"]
        self.check("job_log_archive_event_metadata", event["snapshot_kind"] == "interim_partial"
                   and event["files"] == len(self.manifests[self.gate_commit]["files"]))
        stamp = instant(event["at_utc"])
        self.check("job_log_event_between_copy_and_receipt",
                   instant(self.manifests[self.gate_commit]["created_at_utc"]) <= stamp
                   <= instant(self.receipt["archived_at_utc"]))
        final_status = self.json(self.args.archive_commit, "final_status.json")
        self.check("job_log_event_timestamp_before_recorded_final_start", stamp <= instant(final_status["started_at_utc"]))
        if final_heartbeats:
            self.check("job_log_archive_line_before_final_heartbeats", all(row["line"] < x["line"] for x in final_heartbeats))
            self.check("job_log_final_heartbeats_in_recorded_stage", all(
                instant(final_status["started_at_utc"]) <= instant(x["event"]["at_utc"])
                <= instant(final_status["completed_at_utc"]) for x in final_heartbeats))

    def run(self):
        memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        available = int(memory["MemAvailable"].split()[0]) / 1024**2
        self.report["available_ram_gib_at_admission"] = available
        if not self.check("available_ram_at_least_8gib", available >= 8):
            self.report["dependency_limits"].append("No Git object inspection performed because RAM admission failed.")
            return
        self.section("foundation", self.foundation)
        self.section("intermediate_manifest", self.intermediate)
        self.section("gate_references", self.references)
        self.section("recorded_chronology", self.chronology)
        self.section("optional_job_log", self.job_log)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Recovered complete study raw directory")
    parser.add_argument("--source-commit", type=commit_id, required=True)
    parser.add_argument("--archive-commit", type=commit_id, required=True)
    parser.add_argument("--remote-head", type=commit_id, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--job-log", type=Path, help="Optional complete UTF-8 worker-step/Actions text log, not a ZIP")
    args = parser.parse_args()
    raw, output = args.input.resolve(), args.output.resolve()
    if output.exists() or output == raw or raw in output.parents:
        parser.error("output must be new and outside the raw archive")
    audit = Audit(args)
    audit.section("audit", audit.run)
    audit.report.update(status="failed" if audit.report["issues"] else "verified",
                        completed_at_utc=now(), auditor_sha256=digest(Path(__file__).read_bytes()),
                        checks_count=len(audit.report["checks"]),
                        failed_checks=sum(not value for value in audit.report["checks"].values()))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(audit.report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({"status": audit.report["status"], "checks": audit.report["checks_count"],
                      "issues": len(audit.report["issues"]), "output": str(output),
                      "sha256": digest(output.read_bytes())}))
    return int(bool(audit.report["issues"]))


if __name__ == "__main__":
    raise SystemExit(main())
