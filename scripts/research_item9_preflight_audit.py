"""Offline recount of the frozen item-9 preflight, without Torch or simulation.

Reads a completed downloaded raw directory and an exact frozen source checkout.
No project modules, model weights, pickle payloads or random generators execute.
The exclusive JSON receipt is written outside the input directory, including on
failure. This auditor is a later artifact, not part of the execution freeze.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import statistics
import subprocess
import sys
import traceback
import xml.etree.ElementTree as ET
import zipfile

EXPECTED_SOURCE = "e63764ccb924ab23d65c2deb94b477387d102c82"
EXPECTED_TREE = "2c4de6ea03dbc916ab2d252dd42fc96c76178588"
PLAN_SHA = "7802ee68ae8ce19bf17e6135815e0c796216696e05c8d322347a314a2c88df04"
RECIPE_SHA = "7450b1ede40542c5ff940842e42eb2725ac4420946baad6bac920e779a214b97"
TEST_FILES = ["tests/test_complex_binding_data.py", "tests/test_complex_binding_protocol.py"]
FAMILIES = ("neuropixel", "relative_transformer")
PARAMETERS = {"neuropixel": 29856, "relative_transformer": 30157}
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")
ATOL, RTOL = 1e-10, 1e-12


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def pretty(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def compact(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def compact_hash(value):
    return hashlib.sha256(compact(value)).hexdigest()


def stamp(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return result


def memory():
    entries = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(entries["MemAvailable"].split()[0]) * 1024 / 2**30
    if available < 8:
        raise RuntimeError("offline audit requires at least8GiB available RAM")
    return {"at_utc": now(), "available_ram_gib": available}


def safe_path(root, relative):
    name = PurePosixPath(relative)
    if name.is_absolute() or ".." in name.parts or str(name) != relative:
        raise ValueError("artifact path must be canonical and relative")
    path = root.joinpath(*name.parts)
    path.resolve().relative_to(root.resolve())
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing/nonregular artifact: {relative}")
    return path


class Audit:
    def __init__(self, source, raw):
        self.source, self.raw, self.study = source, raw, raw / "study"
        self.report = {"schema_version": 1, "item": 9, "phase": "preflight", "status": "running",
                       "started_at_utc": now(), "expected_source_commit": EXPECTED_SOURCE,
                       "input": str(raw), "source_root": str(source), "checks": 0,
                       "input_sha256": {}, "issues": [], "sections": {},
                       "arithmetic_tolerance": {"absolute": ATOL, "relative": RTOL},
                       "models_or_rng_executed": False, "checkpoint_deserialization": False}

    def check(self, condition, message):
        self.report["checks"] += 1
        if not bool(condition):
            raise ValueError(message)

    def note_file(self, path):
        value = sha(path)
        self.report["input_sha256"][str(path)] = value
        return value

    def read(self, path):
        self.note_file(path)
        return json.loads(path.read_bytes())

    def equal(self, observed, expected, label):
        if isinstance(expected, dict):
            self.check(isinstance(observed, dict) and set(observed) == set(expected), label + " fields")
            for key in expected:
                self.equal(observed[key], expected[key], label + "." + key)
        elif isinstance(expected, list):
            self.check(isinstance(observed, list) and len(observed) == len(expected), label + " length")
            for i, (a, b) in enumerate(zip(observed, expected)):
                self.equal(a, b, f"{label}[{i}]")
        elif type(expected) is float:
            self.check(type(observed) in (float, int) and math.isfinite(observed)
                       and math.isclose(observed, expected, abs_tol=ATOL, rel_tol=RTOL), label)
        else:
            self.check(observed == expected and (type(observed) is type(expected)
                       or expected is None), label)

    def artifact(self, row, root=None):
        root = self.study if root is None else root
        self.check(set(row) == {"path", "bytes", "sha256"}, "artifact fields")
        path = safe_path(root, row["path"])
        self.check(type(row["bytes"]) is int and path.stat().st_size == row["bytes"]
                   and self.note_file(path) == row["sha256"], "artifact bytes/hash: " + row["path"])
        return path

    def git(self, *args):
        return subprocess.check_output(["git", *args], cwd=self.source, stdin=subprocess.DEVNULL,
            env=dict(os.environ, GIT_NO_LAZY_FETCH="1", GIT_TERMINAL_PROMPT="0"), timeout=30)

    def archive_and_source(self):
        manifest = self.read(self.raw / "archive_manifest.json")
        self.check(manifest["item"] == 9 and manifest["final"] is True
                   and manifest["snapshot_kind"] == "final" and manifest["attempt_status"] == "completed",
                   "a completed final preflight archive is required")
        self.check(manifest["source_commit"] == EXPECTED_SOURCE and manifest["run_key"].endswith("-preflight"),
                   "raw source/phase identity")
        observed_files = {p.relative_to(self.raw).as_posix() for p in self.raw.rglob("*") if p.is_file()}
        self.check(observed_files == set(manifest["files"]) | {"archive_manifest.json"}, "exact raw file inventory")
        for name, row in manifest["files"].items():
            self.artifact({"path": name, **row}, self.raw)
        self.check(self.git("rev-parse", "HEAD").decode().strip() == EXPECTED_SOURCE, "exact frozen checkout HEAD")
        self.check(self.git("rev-parse", "HEAD^{tree}").decode().strip() == EXPECTED_TREE, "exact source tree")
        plan_path = self.raw / "execution_plan.json"
        self.check(self.note_file(plan_path) == PLAN_SHA, "preflight plan hash")
        plan = self.read(plan_path)
        recipe = self.read(self.raw / "experiment_recipe.json")
        self.check(self.note_file(self.raw / "experiment_recipe.json") == RECIPE_SHA == plan["recipe_sha256"],
                   "recipe hash")
        self.check(plan["phase"] == "preflight" and plan["status"] == "frozen"
                   and plan["expected_test_count"] == 34 and plan["test_inventory"] == TEST_FILES
                   and plan["final_performance_access_permitted"] is False
                   and plan["expected_scientific_runs"] == {"memorization": 2, "pilot": 4}
                   and len(plan["implementation_sha256"]) == 49, "exact frozen preflight contract")
        tracked = {}
        for entry in self.git("ls-tree", "-rz", "HEAD").split(b"\0"):
            if entry:
                metadata, name = entry.split(b"\t", 1)
                mode, kind, identity = metadata.decode().split()
                self.check(kind == "blob" and mode in ("100644", "100755"), "regular tracked source files")
                tracked[name.decode()] = identity
        with zipfile.ZipFile(self.raw / "source.zip") as archive:
            info = archive.infolist()
            self.check(len({x.filename for x in info}) == len(info), "no duplicate source ZIP entries")
            files = [x for x in info if not x.is_dir()]
            self.check({x.filename for x in files} == set(tracked), "source ZIP regular files equal Git inventory")
            self.check(archive.comment.decode().strip() == EXPECTED_SOURCE, "Git archive commit comment")
            for entry in files:
                data = archive.read(entry)
                blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
                self.check(blob == tracked[entry.filename], "source ZIP Git blob: " + entry.filename)
                self.check(data == safe_path(self.source, entry.filename).read_bytes(),
                           "source ZIP/worktree byte identity: " + entry.filename)
            self.check(archive.read("docs/research/09_preflight_plan.json") == plan_path.read_bytes(), "archived plan bytes")
            for name, digest in plan["implementation_sha256"].items():
                self.check(hashlib.sha256(archive.read(name)).hexdigest() == digest, "bound source: " + name)
        for label in ("before", "after"):
            source = self.read(self.raw / f"source_{label}.json")
            self.check(source["verified"] is True and source["source_commit"] == EXPECTED_SOURCE
                       and source["plan_sha256"] == PLAN_SHA and source["recipe_sha256"] == RECIPE_SHA
                       and source["implementation_sha256"] == plan["implementation_sha256"]
                       and source["all_bound_files_tracked"] is True and source["plan_matches_committed_blob"] is True
                       and not source["unexpected_untracked_files"] and not source["tracked_diff"], "source guard " + label)
        self.check(not any((self.study / x).exists() for x in
                   ("final_data", "final_access.json", "final_inventory.json", "gate_archive_receipt.json", "final_predictions")),
                   "no final access or final performance artifacts in preflight")
        self.report["sections"]["archive"] = {"manifested_files": len(manifest["files"]),
            "manifested_bytes": sum(x["bytes"] for x in manifest["files"].values()),
            "source_zip_files": len(tracked), "bound_source_files": 49,
            "source_commit": EXPECTED_SOURCE, "source_tree": EXPECTED_TREE,
            "manifest_sha256": self.note_file(self.raw / "archive_manifest.json"),
            "remote_git_provenance": "Not authenticated by this local helper; root verifies downloaded archive commit separately."}
        return plan, recipe

    def runtime_and_tests(self, plan, recipe):
        worker = self.read(self.raw / "worker_status.json")
        self.check(worker["status"] == "completed" and worker["phase"] == "preflight"
                   and worker["source_commit"] == EXPECTED_SOURCE
                   and worker["resource_policy"] == plan["resource_policy"]
                   and [s["name"] for s in worker["stages"]] == ["contract_tests", "preflight"], "worker/stage inventory")
        boundaries = []
        timestamps = [stamp(plan["freeze_utc"]), stamp(worker["started_at_utc"]),
                      stamp(self.read(self.raw / "source_before.json")["at_utc"])]
        resources = [worker["initial_resources"], worker["final_resources"]]
        for stage in worker["stages"]:
            name = stage["name"]
            saved = self.read(self.raw / f"{name}_status.json")
            self.check(saved == stage and stage["status"] == "completed" and stage["return_code"] == 0,
                       "zero complete stage: " + name)
            self.check(self.note_file(self.raw / f"{name}.log") == stage["log_sha256"], "stage stdout hash")
            self.check(0 < stage["wall_seconds_including_shutdown"] <= stage["stage_limit_seconds"] + 20,
                       "bounded stage timer")
            timestamps.extend([stamp(stage["started_at_utc"]), stamp(stage["completed_at_utc"])])
            env = self.read(self.raw / f"{name}_environment.json")
            self.check(not env["issues"] and not env["skipped_nodeids"] and len(env["records"]) == 2,
                       "runtime records and no unexpected skips")
            for row in env["records"]:
                self.check(row["python"] == recipe["runtime"]["python"] and row["versions"] == recipe["runtime"]["versions"]
                           and row["torch_threads"] == 2 and row["torch_interop_threads"] == 1
                           and row["torch_cuda_version"] is None
                           and all(row["thread_environment"][key] == "2" for key in THREAD_ENV), "exact CPU runtime")
            boundaries.extend(env["test_thread_boundaries"])
            monitor = self.read(self.raw / f"{name}_resource_monitor.json")
            self.check(monitor["failure"] is None and monitor["interval_seconds"] == 1
                       and len(monitor["samples"]) > 0, "successful independent resource monitor")
            resources.extend(monitor["samples"])
            self.check(monitor["minimum_sampled_available_ram_gib"] ==
                       min(x["available_ram_gib"] for x in monitor["samples"]), "resource minimum recount")
        timestamps.extend([stamp(self.read(self.raw / "source_after.json")["at_utc"]),
                           stamp(worker["completed_at_utc"]), stamp(self.read(self.raw / "archive_manifest.json")["created_at_utc"])])
        self.check(timestamps == sorted(timestamps), "freeze/source/stages/final archive chronology")
        self.check(all(math.isfinite(x["available_ram_gib"]) and x["available_ram_gib"] >= 8 for x in resources), "sampled RAM floor")
        self.check(0 < worker["worker_wall_seconds_before_archive"] <= worker["initial_budget_seconds"] <= 6600,
                   "worker budget and wall timer")
        collected = self.read(self.raw / "test_collection.json")
        expected_ids = []
        for name in TEST_FILES:
            tree = ast.parse((self.source / name).read_text())
            for cls in tree.body:
                if isinstance(cls, ast.ClassDef):
                    expected_ids.extend(f"{name}::{cls.name}::{method.name}" for method in cls.body
                                        if isinstance(method, ast.FunctionDef) and method.name.startswith("test_"))
        self.check(len(expected_ids) == 34 and collected["count"] == collected["expected_count"] == 34
                   and len(set(collected["nodeids"])) == 34 and set(collected["nodeids"]) == set(expected_ids)
                   and collected["files"] == TEST_FILES, "34 exact collected test IDs")
        self.check(len(boundaries) == 34 and {x["nodeid"] for x in boundaries} == set(expected_ids)
                   and all(x["before"] == x["after"] == [2, 1] for x in boundaries), "all test thread boundaries2/1")
        xml = ET.parse(self.raw / "tests.xml").getroot()
        cases = xml.findall(".//testcase")
        xml_names = [x.attrib["classname"] + "." + x.attrib["name"] for x in cases]
        expected_names = [x.split("::")[0][:-3].replace("/", ".") + "." + ".".join(x.split("::")[1:]) for x in expected_ids]
        self.check(len(cases) == 34 and sorted(xml_names) == sorted(expected_names)
                   and not xml.findall(".//failure") and not xml.findall(".//error") and not xml.findall(".//skipped"),
                   "34 passed XML parent cases, including both real model integration methods")
        self.report["sections"]["execution"] = {"tests": 34, "passed_case_elements": 34, "skips": 0,
            "junit_suite_attributes": [x.attrib for x in xml.findall("testsuite")],
            "test_nodeids": collected["nodeids"], "stages": worker["stages"],
            "worker_seconds_before_archive": worker["worker_wall_seconds_before_archive"],
            "ram_samples": len(resources), "minimum_sampled_available_ram_gib": min(x["available_ram_gib"] for x in resources),
            "limitations": "JUnit aggregate counters may include subtests; exact parent IDs/elements are counted. Samples are not continuous RAM/CPU or energy measurements."}

    def scenario(self, value):
        self.check(set(value) == {"schema_version", "group_id", "bag", "facts", "layout"}
                   and value["schema_version"] == 1, "scenario fields")
        bag = value["bag"]
        self.check(set(bag) == {"nouns", "verbs", "places"}, "bag categories")
        for name, count, allowed in (("nouns", 4, range(5, 17)), ("verbs", 2, range(17, 27)), ("places", 2, range(27, 35))):
            self.check(len(bag[name]) == count and bag[name] == sorted(set(bag[name]))
                       and all(type(x) is int and x in allowed for x in bag[name]), "canonical distinct fillers")
        self.check(compact_hash(bag) == value["group_id"], "canonical semantic group hash")
        self.check(len(value["facts"]) == len(value["layout"]) == 8, "eight facts/positions")
        pairs, rows, encountered = set(), set(), {"nouns": [], "verbs": [], "places": []}
        canvas = [[0] * 8 for _ in range(10)]
        for fact, slot in zip(value["facts"], value["layout"]):
            self.check(set(fact) == {"event", "role", "filler"} and set(slot) == {"row", "start"}, "fact/layout schema")
            e, r, f, row, col = fact["event"], fact["role"], fact["filler"], slot["row"], slot["start"]
            self.check(all(type(x) is int for x in (e, r, f, row, col)) and e in (0, 1) and r in range(4)
                       and row in range(9) and col in range(6) and row not in rows and (e, r) not in pairs,
                       "unique semantic keys and nonoverlapping positions")
            category = "nouns" if r in (0, 2) else "verbs" if r == 1 else "places"
            self.check(f in bag[category], "filler category")
            encountered[category].append(f)
            pairs.add((e, r)); rows.add(row)
            canvas[row][col:col+3] = [35 + e, 1 + r, f]
        self.check({k: sorted(v) for k, v in encountered.items()} == bag, "fact/bag multiplicities")
        return canvas

    def visible_truth(self, canvas):
        """Decode a visible record independently; do not consult scenario facts."""
        self.check(canvas.shape == (10, 8) and canvas.dtype.kind in "iu" and ((canvas >= 0) & (canvas < 37)).all(), "visible canvas contract")
        self.check((canvas[9, [0, 1, 2, 3, 4, 7]] == 0).all()
                   and int(canvas[9, 5]) in (35, 36) and int(canvas[9, 6]) in (1, 2, 3, 4), "explicit query/output")
        table, bags = {}, {"nouns": [], "verbs": [], "places": []}
        for line in canvas[:9]:
            cols = [i for i in range(8) if int(line[i])]
            if not cols:
                continue
            self.check(len(cols) == 3 and cols == [cols[0], cols[0]+1, cols[0]+2], "one visible adjacent triple per fact row")
            e, r, f = [int(line[c]) for c in cols]
            self.check(e in (35, 36) and r in (1, 2, 3, 4) and (e, r) not in table, "visible event/role uniqueness")
            category, allowed = ("nouns", range(5, 17)) if r in (1, 3) else ("verbs", range(17, 27)) if r == 2 else ("places", range(27, 35))
            self.check(f in allowed, "visible filler category")
            bags[category].append(f); table[e, r] = f
        self.check(len(table) == 8 and sorted(len(set(bags[k])) for k in bags) == [2, 2, 4], "all visible keys and distinct fillers")
        bag = {k: sorted(v) for k, v in bags.items()}
        event, role = int(canvas[9, 5]), int(canvas[9, 6])
        return table[event, role], event - 35, role - 1, compact_hash(bag), table

    def arrays(self, path):
        self.note_file(path)
        with np.load(path, allow_pickle=False) as archive:
            result = {name: archive[name] for name in archive.files}
        return result

    def content_digest(self, canvas, target):
        digest = hashlib.sha256()
        for name, values in (("canvas", canvas), ("target", target)):
            a = np.ascontiguousarray(values, dtype="<i8")
            header = pretty({"name": name, "shape": list(a.shape), "dtype": "<i8"})
            digest.update(len(header).to_bytes(8, "little")); digest.update(header); digest.update(a.tobytes())
        return digest.hexdigest()

    def development(self, recipe):
        folder = self.study / "development_data"
        manifest = self.read(folder / "manifest.json")
        for row in manifest["files"].values():
            self.artifact(row)
        ledger = self.read(folder / "fixture_exposure.json")
        frozen_ledger = self.read(self.source / "results/research/09_validation/complex_binding_fixture_exposure.json")
        self.check(ledger == frozen_ledger and manifest["excluded_fixture_groups"] == ledger["excluded_groups"], "frozen fixture exposure ledger")
        all_candidates = {ledger["manual_group_id"]}
        all_constructed = {ledger["manual_group_id"]}
        for run in ledger["generation_runs"]:
            all_candidates.update(run["candidate_group_ids"]); all_constructed.update(run["constructed_group_ids"])
        self.check(sorted(all_candidates) == ledger["excluded_groups"]
                   and sorted(all_constructed) == ledger["constructed_group_ids"]
                   and all_constructed <= all_candidates, "exposure union from saved fixture transcript; no RNG replay")
        populations, panels, support = {}, {}, {}
        cfg = recipe["data"]
        for name, count in (("train", cfg["train_bags"]), ("validation", cfg["validation_bags"]),
                            ("probe", cfg["probe_bags"]), ("memorization", cfg["memorization_bags"])):
            memory()
            p = folder / f"{name}_scenarios.json.gz"
            values = json.loads(gzip.decompress(p.read_bytes()))
            self.check(len(values) == count == manifest["counts"][name], "scenario count: " + name)
            self.check(manifest["identities"][p.name] == self.note_file(p), "scenario identity")
            ids = [s["group_id"] for s in values]
            self.check(len(set(ids)) == count and not set(ids) & all_candidates, "unique unexposed groups: " + name)
            allowed = range(70, 85) if name == "validation" else range(70)
            self.check(all(int(identity, 16) % 100 in allowed for identity in ids), "hash split: " + name)
            base_canvases = [self.scenario(s) for s in values]
            per_role, per_er = {str(r): Counter() for r in range(4)}, {f"{e}:{r}": Counter() for e in range(2) for r in range(4)}
            for s in values:
                for fact in s["facts"]:
                    per_role[str(fact["role"])][str(fact["filler"])] += 1
                    per_er[f"{fact['event']}:{fact['role']}"][str(fact["filler"])] += 1
            support[name] = {"per_role": {k: dict(v) for k, v in per_role.items()}, "per_event_role": {k: dict(v) for k, v in per_er.items()}}
            self.check(support[name] == manifest["support"][name], "dataset support counts")
            populations[name] = values
            if name == "train":
                continue
            a = self.arrays(folder / f"{name}.npz")
            keys = {"canvas", "target", "base_target", "role", "query_event", "base_query_event", "group_index",
                    "condition_index", "changed_gold", "group_id", "record_id", "pair_id"}
            self.check(set(a) == keys and a["canvas"].shape == (count*8, 10, 8)
                       and all(x.shape == (count*8,) for k, x in a.items() if k != "canvas"), "base panel array shape")
            self.check(all(a[k].dtype.kind in "iu" for k in keys - {"changed_gold", "group_id", "record_id", "pair_id"})
                       and a["changed_gold"].dtype == np.bool_
                       and all(a[k].dtype.kind == "U" for k in ("group_id", "record_id", "pair_id")), "panel array types")
            self.check(manifest["identities"][f"{name}.npz"] == self.content_digest(a["canvas"], a["target"]), "canonical dataset content hash")
            for i in range(count*8):
                gi, q = divmod(i, 8); event, role = divmod(q, 4)
                gold, visible_event, visible_role, group, _ = self.visible_truth(a["canvas"][i])
                expected_canvas = [line.copy() for line in base_canvases[gi]]
                expected_canvas[9][5:7] = [35+event, 1+role]
                self.check(np.array_equal(a["canvas"][i], expected_canvas), "base rendering corresponds to saved scenario")
                self.check(int(a["target"][i]) == int(a["base_target"][i]) == gold
                           and int(a["role"][i]) == visible_role == role
                           and int(a["query_event"][i]) == int(a["base_query_event"][i]) == visible_event == event
                           and int(a["group_index"][i]) == gi and int(a["condition_index"][i]) == 0
                           and not bool(a["changed_gold"][i]) and str(a["group_id"][i]) == group == ids[gi], "visible gold and query/group indices")
                sid = compact_hash(values[gi])
                pair = compact_hash({"scenario_id": sid, "base_query_event": event, "query_role": role})
                record = compact_hash({"pair_id": pair, "condition": "base", "canvas": expected_canvas})
                self.check(str(a["pair_id"][i]) == pair and str(a["record_id"][i]) == record, "record and pair identities")
            panels[name] = a
        train_ids = [s["group_id"] for s in populations["train"]]
        self.check(not set(train_ids) & {s["group_id"] for s in populations["validation"]}, "train/validation disjoint groups")
        for name in ("probe", "memorization"):
            self.check([s["group_id"] for s in populations[name]] == train_ids[:len(populations[name])], "training-derived probe/memorization group prefix")
        majority = {role: min((int(k) for k in counts), key=lambda t: (-counts[str(t)], t))
                    for role, counts in support["train"]["per_role"].items()}
        self.check(majority == manifest["train_only_role_majority"] and manifest["final_performance_data_generated"] is False,
                   "training-only majority and no final data")
        controls = self.arrays(self.study / "development_controls.npz")
        a = panels["validation"]
        expected = {"symbolic": np.ones(len(a["target"])), "role_only": np.full(len(a["target"]), .5),
                    "event_category": np.where(np.isin(a["role"], [0, 2]), .5, 1.),
                    "bag_category": np.where(np.isin(a["role"], [0, 2]), .25, .5),
                    "train_role_majority": np.asarray([majority[str(int(r))] == int(t) for r, t in zip(a["role"], a["target"])], dtype=float)}
        self.check(set(controls) == set(expected) and all(np.array_equal(controls[k], v) for k, v in expected.items()), "exact input-control expectations and training-only majority")
        self.check(manifest["identities"]["fixture_exposure.json"] == self.note_file(folder / "fixture_exposure.json"), "fixture ledger identity")
        self.report["sections"]["development"] = {"groups": manifest["counts"], "panel_rows": {k: len(v["target"]) for k, v in panels.items()},
            "excluded_fixture_candidates": len(all_candidates), "fixture_groups_constructed": len(all_constructed),
            "identities": manifest["identities"], "majority_from_train": majority,
            "control_expected_accuracy": {k: float(v.mean()) for k, v in expected.items()},
            "scope": "Saved scenes and panel tensors recounted; neither generation RNG nor historical exposure outside the declared ledger is reconstructed."}
        return panels, manifest

    def metrics(self, arrays, dataset):
        n = len(dataset["target"])
        self.check(set(arrays) == {"logits", "pred", "nll", "target", "role", "group_index", "record_id"}, "prediction archive keys")
        self.check(arrays["logits"].shape == (n, 37) and arrays["logits"].dtype == np.float32
                   and np.isfinite(arrays["logits"]).all(), "finite saved float32 logits")
        for key in ("target", "role", "group_index", "record_id"):
            self.check(np.array_equal(arrays[key], dataset[key]), "prediction/dataset alignment: " + key)
        self.check(arrays["pred"].dtype.kind in "iu"
                   and np.array_equal(arrays["pred"], arrays["logits"].argmax(1)), "saved integer argmax")
        logits = arrays["logits"].astype(np.float64)
        maximum = logits.max(1)
        nll = maximum + np.log(np.exp(logits-maximum[:, None]).sum(1)) - logits[np.arange(n), dataset["target"]]
        self.check(arrays["nll"].shape == (n,) and arrays["nll"].dtype == np.float64
                   and np.isfinite(arrays["nll"]).all() and (arrays["nll"] >= 0).all()
                   and np.allclose(arrays["nll"], nll, atol=ATOL, rtol=RTOL), "independent float64 logsumexp NLL")
        correct = arrays["pred"] == dataset["target"]
        roles = [float(correct[dataset["role"] == r].mean()) for r in range(4)]
        return {"n": n, "groups": n//8, "correct": int(correct.sum()), "global": float(correct.mean()),
                "per_role": roles, "binding": (roles[0]+roles[2])/2, "macro": sum(roles)/4,
                "cross_entropy": float(arrays["nll"].mean()), "all_eight": float(correct.reshape(-1, 8).all(1).mean())}

    def checkpoint_container(self, path):
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            self.check(len(set(names)) == len(names), "checkpoint ZIP no duplicates")
            by_tail = {name.split("/", 1)[-1]: name for name in names}
            self.check({"data.pkl", "version", "byteorder"} <= set(by_tail), "Torch-save container metadata")
            return {"sha256": self.note_file(path), "bytes": path.stat().st_size,
                    "container_version": z.read(by_tail["version"]).decode().strip(),
                    "byteorder": z.read(by_tail["byteorder"]).decode().strip(),
                    "pickle_sha256": hashlib.sha256(z.read(by_tail["data.pkl"])).hexdigest(),
                    "storage_entries": sum(key.startswith("data/") for key in by_tail),
                    "tensor_values_and_embedded_configuration_deserialized": False}

    def scientific_runs(self, recipe, panels, manifest):
        cfg = recipe["training"]
        context = self.read(self.study / "context.json")
        self.check(context["source_commit"] == EXPECTED_SOURCE and context["plan_sha256"] == PLAN_SHA
                   and context["recipe_sha256"] == RECIPE_SHA and context["development_identities"] == manifest["identities"], "scientific context")
        self.check(self.artifact(context["development_manifest"]) == self.study / "development_data/manifest.json", "bound development manifest")
        start = self.read(self.study / "preflight_started.json")
        self.check(start["context"] == {k: context[k] for k in ("source_commit", "plan_sha256", "recipe_sha256")}, "scientific initial source context")
        rt = start["runtime"]
        self.check(rt["python"] == recipe["runtime"]["python"] and rt["versions"] == recipe["runtime"]["versions"]
                   and rt["device"] == "cpu" and rt["torch_cuda_version"] is None
                   and rt["intra_threads"] == 2 and rt["inter_threads"] == 1
                   and rt["deterministic_algorithms"] is True, "scientific deterministic CPU runtime")
        index = self.read(self.study / "preflight_runs.json")
        expected = [(f"memorization_{f}", "memorization", f, cfg["memorization_initialization"], cfg["memorization_lr"], cfg["memorization_max_updates"]) for f in FAMILIES]
        expected += [(f"pilot_{f}_lr{rate:g}".replace(".", "p"), "pilot", f, cfg["pilot_initialization"], rate, cfg["pilot_updates"]) for f in FAMILIES for rate in cfg["pilot_learning_rates"]]
        self.check(set(p.name for p in (self.study / "runs").iterdir()) == {x[0] for x in expected}, "exact two memorization/four pilot directories")
        self.check([x["run_id"] for x in index["memorization"]+index["pilots"]] == [x[0] for x in expected], "complete ordered preflight index")
        recounted, streams, timings = [], {}, {}
        for spec, copied in zip(expected, index["memorization"]+index["pilots"]):
            memory()
            name, phase, family, seed, rate, budget = spec
            folder = self.study / "runs" / name
            row = self.read(folder / "run.json")
            self.check(row == copied and row["run_id"] == name and row["status"] == "completed" and row["phase"] == phase
                       and row["family"] == family and row["initialization"] == seed and row["learning_rate"] == rate
                       and row["context"] == context and not (folder / "failure.json").exists(), "successful run identity/context: " + name)
            memorize = phase == "memorization"
            configuration = {"family": family, "model": recipe["models"][family], "vocab": 37, "height": 10, "width": 8,
                "out_pos": [9, 7], "trainable_parameters": PARAMETERS[family], "initialization": seed, "learning_rate": rate,
                "optimizer": "AdamW", "weight_decay": cfg["weight_decay"], "gradient_clip": cfg["gradient_clip"],
                "batch_size": cfg["batch_size"], "loss": cfg["loss"], "scheduled_updates": budget, "memorization": memorize}
            self.equal(row["configuration"], configuration, "declared configuration")
            self.equal(self.read(folder / "configuration.json"), configuration, "configuration artifact")
            started = self.read(folder / "started.json")
            self.check(started["context"] == context and started["configuration"] == row["configuration"], "run start identity")
            for artifact in row["artifacts"].values():
                self.artifact(artifact)
            log_path = folder / "training.jsonl"
            lines = [json.loads(line) for line in log_path.read_text().splitlines()]
            self.check(len(lines) == row["updates_completed"] and 0 < len(lines) <= budget
                       and (memorize or len(lines) == budget), "scheduled/completed update count")
            aggregate = hashlib.sha256()
            previous_elapsed, inputs, deltas = 0.0, [], []
            for i, entry in enumerate(lines, 1):
                self.check(entry["step"] == i and type(entry["loss"]) in (int, float) and math.isfinite(entry["loss"]) and entry["loss"] >= 0
                           and math.isfinite(entry["gradient_norm_before_clip"]) and entry["gradient_norm_before_clip"] >= 0
                           and math.isfinite(entry["elapsed_seconds"]) and entry["elapsed_seconds"] > previous_elapsed,
                           "finite sequential update record")
                digest = bytes.fromhex(entry["input_sha256"])
                self.check(len(digest) == 32, "training batch digest width")
                aggregate.update(digest)
                chosen = entry["training_group_indices"]
                if memorize:
                    self.check(chosen == list(range(len(panels["memorization"]["target"])))
                               and entry["input_sha256"] == self.content_digest(panels["memorization"]["canvas"], panels["memorization"]["target"]), "memorization fixed examples/digest")
                else:
                    self.check(len(chosen) == cfg["batch_size"] and all(type(x) is int and 0 <= x < recipe["data"]["train_bags"] for x in chosen), "pilot train-group indices")
                inputs.append((entry["input_sha256"], chosen))
                if i > 1:
                    deltas.append(entry["elapsed_seconds"] - previous_elapsed)
                previous_elapsed = entry["elapsed_seconds"]
            self.check(aggregate.hexdigest() == row["training_input_sequence_sha256"], "aggregate training input stream hash")
            self.check(math.isfinite(row["wall_seconds"]) and row["wall_seconds"] >= previous_elapsed
                       and len(row["resource_samples"]) >= 2
                       and all(math.isfinite(x["available_ram_gib"]) and x["available_ram_gib"] >= 8 for x in row["resource_samples"]), "run timers/resources")
            resource_times = [stamp(x["at_utc"]) for x in row["resource_samples"]]
            self.check(resource_times == sorted(resource_times)
                       and stamp(start["at_utc"]) <= resource_times[0] <= stamp(started["at_utc"]) <= resource_times[-1], "run resource/start chronology")
            metrics = {}
            for endpoint in (("memorization",) if memorize else ("probe", "validation")):
                a = self.arrays(self.artifact(row["artifacts"][endpoint + "_predictions"]))
                metrics[endpoint] = self.metrics(a, panels[endpoint])
                self.equal(row[endpoint], metrics[endpoint], "recounted " + name + "/" + endpoint)
            if memorize:
                streak, reached = 0, False
                checks = row["memorization_checks"]
                self.check([x["step"] for x in checks] == list(range(cfg["memorization_check_every"], len(lines)+1, cfg["memorization_check_every"])), "complete scheduled memorization probes")
                for probe in checks:
                    score = probe["metrics"]
                    self.check(math.isfinite(score["global"]) and math.isfinite(score["cross_entropy"])
                               and 0 <= score["global"] <= 1 and score["cross_entropy"] >= 0, "finite memorization diagnostics")
                    passed = score["global"] >= cfg["memorization_min_accuracy"] and score["cross_entropy"] <= cfg["memorization_max_ce"]
                    streak = streak + 1 if passed else 0
                    self.check(probe["consecutive"] == streak and not reached, "memorization consecutive rule/no postcriterion continuation")
                    reached = streak >= cfg["memorization_consecutive_checks"]
                self.check(row["memorization_criterion_reached"] is reached and (reached or len(lines) == budget), "memorization fixed rule/maximum budget")
                self.equal(checks[-1]["metrics"], metrics["memorization"], "last scheduled memorization probe vs final saved prediction")
            else:
                self.check(row["memorization_checks"] == [] and row["memorization_criterion_reached"] is None
                           and row["probe_below_95_percent_binding"] is (metrics["probe"]["binding"] < .95), "pilot diagnostic flags")
                streams[name] = inputs
            timings[name] = {"updates": len(lines), "last_training_elapsed_seconds": previous_elapsed,
                "run_wall_seconds": row["wall_seconds"], "post_training_seconds": row["wall_seconds"]-previous_elapsed,
                "median_update_delta_seconds_excluding_first": statistics.median(deltas),
                "mean_update_delta_seconds_excluding_first": statistics.mean(deltas)}
            recounted.append({"run_id": name, "phase": phase, "family": family, "initialization": seed,
                "learning_rate": rate, "configuration": configuration, "updates_completed": len(lines),
                "metrics": metrics, "memorization_criterion_reached": row["memorization_criterion_reached"],
                "checkpoint_container": self.checkpoint_container(self.artifact(row["artifacts"]["checkpoint"])),
                "training_input_sequence_sha256": row["training_input_sequence_sha256"], "timing": timings[name]})
        first_stream = next(iter(streams.values()))
        self.check(all(value == first_stream for value in streams.values()), "four pilot runs share exactly the logged per-update input/group sequence")
        selection = self.read(self.study / "pilot_selection.json")
        ranking, selected = {}, {}
        for family in FAMILIES:
            rows = [row for row in recounted if row["phase"] == "pilot" and row["family"] == family]
            rows.sort(key=lambda x: (-x["metrics"]["validation"]["binding"], x["metrics"]["validation"]["cross_entropy"], x["learning_rate"]))
            ranking[family] = [{"run_id": x["run_id"], "binding": x["metrics"]["validation"]["binding"],
                                "cross_entropy": x["metrics"]["validation"]["cross_entropy"], "learning_rate": x["learning_rate"]} for x in rows]
            selected[family] = rows[0]["learning_rate"]
        self.check(selection["context"] == context and selection["selected_learning_rates"] == selected
                   and selection["rule"] == cfg["lr_selection"] and selection["development_identities"] == manifest["identities"]
                   and selection["pilot_runs"] == [x["run_id"] for x in index["pilots"]]
                   and selection["final_performance_data_generated"] is False, "validation-only pilot selection recount")
        self.check(self.artifact(selection["pilot_records"]) == self.study / "preflight_runs.json", "selection binds complete six-run record")
        self.check(selection["memorization_flags"] == {x["family"]: not x["memorization_criterion_reached"] for x in index["memorization"]}, "memorization flags retained separately from pilot selection")
        complete = self.read(self.study / "preflight_completed.json")
        self.check(complete["status"] == "completed" and complete["memorization_completed"] == 2 and complete["pilot_completed"] == 4
                   and complete["selected_learning_rates"] == selected and complete["final_performance_data_generated"] is False
                   and stamp(start["at_utc"]) <= stamp(selection["selected_at_utc"]) <= stamp(complete["at_utc"]), "scientific stage completion/selection chronology")
        self.check(all(stamp(row["resource_samples"][-1]["at_utc"]) <= stamp(selection["selected_at_utc"])
                       for row in index["memorization"] + index["pilots"])
                   and math.isfinite(complete["wall_seconds"]) and complete["wall_seconds"] > 0
                   and all(math.isfinite(x["resource"]["available_ram_gib"]) and x["resource"]["available_ram_gib"] >= 8
                           for x in (start, complete)), "all training completes before selection, scientific timers/resources")
        self.report["sections"]["scientific_runs"] = recounted
        self.report["sections"]["selection"] = {"ranking": ranking, "selected_learning_rates": selected,
            "selection_sha256": self.note_file(self.study / "pilot_selection.json"),
            "preflight_runs_sha256": self.note_file(self.study / "preflight_runs.json"),
            "recipe_sha256": RECIPE_SHA, "preflight_source_commit": EXPECTED_SOURCE,
            "development_identities": manifest["identities"],
            "handoff": "Root must authenticate the raw archive Git identity and preserve the complete selection/preflight_runs objects by these hashes before the Stage-B plan freezes."}
        projected = []
        for family in FAMILIES:
            winner = ranking[family][0]["run_id"]
            t = timings[winner]
            count, updates = len(cfg["primary_initializations"]), cfg["primary_updates"]
            projected.append({"family": family, "timing_reference_run": winner, "training_runs": count,
                "updates_each": updates, "median_delta_projection_seconds": count*updates*t["median_update_delta_seconds_excluding_first"],
                "mean_delta_projection_seconds": count*updates*t["mean_update_delta_seconds_excluding_first"],
                "repeated_measured_post_training_seconds": count*t["post_training_seconds"]})
        self.report["sections"]["stage_b_timing_projection"] = {"by_family": projected,
            "total_training_update_seconds_from_medians": sum(x["median_delta_projection_seconds"] for x in projected),
            "total_training_update_seconds_from_means": sum(x["mean_delta_projection_seconds"] for x in projected),
            "repeated_pilot_post_training_seconds": sum(x["repeated_measured_post_training_seconds"] for x in projected),
            "planned_worker_seconds": 13800, "planned_execution_seconds_before_archive_reserve": 13620,
            "limitations": "Extrapolation, not a measured Stage-B duration or lower/upper bound. Delta times include generation, learning and intervening log/progress work. Excludes first-update/setup cost, final122880prediction rows, tests, data preparation, live archival and hardware/load changes. Post-training pilot overhead is a separate observed scope, not an isolated evaluation benchmark. No budget or scientific setting is selected here."}

    def run(self):
        self.report["resource_observations"] = [memory()]
        plan, recipe = self.archive_and_source()
        self.runtime_and_tests(plan, recipe)
        global np
        import numpy as np
        self.report["auditor_runtime"] = {"python": sys.version, "numpy": np.__version__,
                                          "thread_environment": {k: os.environ[k] for k in THREAD_ENV}}
        panels, manifest = self.development(recipe)
        self.scientific_runs(recipe, panels, manifest)
        self.report["resource_observations"].append(memory())
        self.report["status"] = "verified"
        self.report["limitations"] = [
            "Independent implementation within the same assistant team, not external replication or independent custody.",
            "No RNG was replayed. Log hashes establish agreement of recorded input streams, not independent reconstruction of every training canvas, initialization or firing mask.",
            "Checkpoint bytes and serialization-container metadata are verified without loading pickle or tensors; embedded weights/configuration and numerical reload equivalence are not re-executed.",
            "Intermediate memorization criteria are recounted from saved metrics; only the final saved logits are independently scored.",
            "Timing projections do not authorize changes to the frozen scientific recipe or guarantee completion of the next stage.",
            "No final performance population, result or item10 is created or accessed by this audit."]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw, source, output = args.input.resolve(), args.source_root.resolve(), args.output.resolve()
    if output == raw or raw in output.parents or output.exists():
        parser.error("output must be new and outside the raw archive")
    for key in THREAD_ENV:
        os.environ[key] = "1"
    audit = Audit(source, raw)
    try:
        audit.run()
    except Exception as exc:
        audit.report.update(status="failed", failed_at_utc=now())
        audit.report["issues"].append({"type": type(exc).__name__, "message": str(exc),
                                      "traceback": traceback.format_exc(limit=8)})
    audit.report.update(completed_at_utc=now(), auditor_sha256=sha(Path(__file__)))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(audit.report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({"status": audit.report["status"], "checks": audit.report["checks"],
                      "issues": len(audit.report["issues"]), "output": str(output), "sha256": sha(output)}))
    return 0 if audit.report["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
