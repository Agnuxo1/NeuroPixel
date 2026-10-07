"""Export exact CPU split memberships, without sampling examples or loading models.

Run in the pinned Torch 2.6.0+cpu environment. The separate standard-library
auditor reconstructs partition membership from these recorded permutations.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (0, 1, 2, 101, 202)
SOURCE_HASHES = {
    "neuropixel/task.py": "e088d35937040f9924264b227e70e0ba9aaf23ede1b1c6bc9f8d6c78821a156f",
    "neuropixel/research/data.py": "44cdf2b9ff492fe9eab115c8092c61c6d38bda5388babaf148380b89f2a7f94b",
    "neuropixel/research/growth_ablation.py": "25f3880569eb10168bb0979cfc1d400c65e1615b1e31f23ab37446df2e47aaa5",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def resource_sample():
    """Admit Linux CPU-only membership work using available system RAM."""
    fields = {key: int(value.split()[0]) * 1024 for key, value in
              (line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())}
    available = fields["MemAvailable"] / 2**30
    if available < 8:
        raise RuntimeError(f"available RAM {available:.9f} GiB is below 8 GiB")
    return {"at_utc": utc_now(), "available_ram_gib": available,
            "provider": "/proc/meminfo:MemAvailable", "active_numerical_threads_limit": 1}


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, record):
    data = (json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)
    if path.read_bytes() != data:
        raise OSError("membership export readback differs")


def export_memberships(source_root, output):
    if output.exists():
        raise FileExistsError("refusing to overwrite a membership export")
    started, resources = utc_now(), [resource_sample()]
    actual = {name: file_hash(source_root / name) for name in SOURCE_HASHES}
    if actual != SOURCE_HASHES:
        raise RuntimeError("historical task/research-data/growth source hashes differ")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    import torch

    if torch.__version__ != "2.6.0+cpu":
        raise RuntimeError("the membership export requires exactly Torch 2.6.0+cpu")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("one-thread runtime admission failed")
    resources.append(resource_sample())
    sys.path.insert(0, str(source_root))
    import neuropixel.task as original
    import neuropixel.research.data as research
    for module, relative in ((original, "neuropixel/task.py"), (research, "neuropixel/research/data.py")):
        if Path(module.__file__).resolve() != source_root / relative:
            raise RuntimeError("a task module came from outside the declared source checkout")
    seeds = []
    with torch.device("cpu"):
        for seed in SEEDS:
            resources.append(resource_sample())
            permutation = torch.randperm(1320, generator=torch.Generator(device="cpu").manual_seed(seed),
                                         device="cpu", dtype=torch.int64).tolist()
            old = original.RoleTask(8, 8, heldout_frac=0.2, seed=seed)
            new = research.ResearchRoleTask(8, 8, seed=seed)
            seeds.append({"seed": seed, "permutation": permutation,
                          "direct_role_task": {split: [list(t) for t in getattr(old, split + "_triples")]
                                               for split in ("train", "test")},
                          "direct_research_role_task": {split: [list(t) for t in getattr(new, split + "_triples")]
                                                        for split in ("train", "validation", "test")}})
            del old, new
        partitions = []
        source_pools = seeds[0]["direct_research_role_task"]
        for topic in range(3):
            resources.append(resource_sample())
            allowed = set(range(topic * 4, topic * 4 + 4))
            pools = {split: [triple for triple in values if triple[0] in allowed and triple[2] in allowed]
                     for split, values in source_pools.items()}
            payload = json.dumps(pools, sort_keys=True, separators=(",", ":")).encode()
            partitions.append({"topic": topic, "split_seed": 0, "triples": pools,
                               "counts": {split: len(values) for split, values in pools.items()},
                               "triples_sha256": hashlib.sha256(payload).hexdigest()})
    if any(name in sys.modules for name in ("neuropixel.model", "neuropixel.research.models", "neuropixel.research.experiment")):
        raise RuntimeError("model/experiment modules unexpectedly imported")
    resources.append(resource_sample())
    if {name: file_hash(source_root / name) for name in SOURCE_HASHES} != actual:
        raise RuntimeError("source changed during membership export")
    head = subprocess.check_output(["git", "--no-optional-locks", "rev-parse", "HEAD"], cwd=source_root,
                                   stdin=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=15,
                                   env=dict(os.environ, GIT_OPTIONAL_LOCKS="0")).decode().strip()
    record = {
        "schema_version": 1, "item": 7, "status": "membership_exported", "started_at_utc": started,
        "completed_at_utc": utc_now(), "exporter_sha256": file_hash(Path(__file__)),
        "source": {"git_commit": head, "sha256": actual},
        "runtime": {"torch": torch.__version__, "python": platform.python_version(), "platform": platform.platform(),
                    "device": "cpu", "torch_threads": torch.get_num_threads(),
                    "torch_interop_threads": torch.get_num_interop_threads()},
        "vocabulary": {"nouns": original.NOUNS, "verbs": original.VERBS, "places": original.PLACES, "roles": original.ROLES},
        "generator_definition": {"function": "torch.randperm", "n": 1320, "device": "cpu", "dtype": "torch.int64",
                                 "generator": "new CPU torch.Generator().manual_seed(seed) for each permutation"},
        "seeds": seeds, "growth_partitions": partitions, "resource_samples": resources,
        "growth_partition_method": "Filter direct ResearchRoleTask(seed0) pools by noun groups 0..3/4..7/8..11; growth module not imported.",
        "task_sample_called": False, "models_imported": False, "optimizer_updates": 0,
        "scope": [
            "These are eligible triple pools reconstructed now under a pinned runtime, not observed historical minibatches.",
            "RoleTask and ResearchRoleTask were independently instantiated for membership only; no sample/frozen_dataset/model method was called.",
            "Only historical seeds 0/1/2 and prospective seeds 0/101/202 describe declared study revisions. Extra direct constructor outputs support consistency checks, not additional experiments.",
            "The independent auditor must check slicing, set membership and the archived item-6 growth partitions before making exposure-pool claims."
        ],
    }
    write_new(output, record)
    print(json.dumps({"status": record["status"], "seeds": list(SEEDS), "output": str(output),
                      "sha256": file_hash(output), "minimum_recorded_available_ram_gib": min(r["available_ram_gib"] for r in resources)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    args = parser.parse_args()
    export_memberships(args.source_root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
