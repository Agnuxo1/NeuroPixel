"""Read-only source, environment and development-data admission for item 6.

No optimizer is constructed and no test examples or final predictions are opened.
This is an operational check in a new CPU environment, not a scientific replicate.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
THREADS = 2
MINIMUM_RAM_GIB = 8.0


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import psutil

    available = psutil.virtual_memory().available / 2**30
    if available < MINIMUM_RAM_GIB:
        raise RuntimeError("RAM admission failed before importing Torch")
    manifest_path = ROOT / "docs/research/06_recovery_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest["available_historical_sha256"]
    verified = {}
    for relative, wanted in {**expected, **manifest["verified_document_sha256"]}.items():
        actual = digest(ROOT / relative)
        if actual != wanted:
            raise RuntimeError(f"historical source/document hash mismatch: {relative}")
        verified[relative] = actual
    for relative in manifest["unavailable_historical_sha256"]:
        if (ROOT / relative).exists():
            raise RuntimeError(f"unavailable historical file unexpectedly exists: {relative}")
    if (ROOT / "source_snapshot.json").exists():
        raise RuntimeError("a stale root source snapshot would conceal actual Git HEAD")
    packages = {name: importlib.metadata.version(name)
                for name in ("torch", "numpy", "scipy", "pytest", "psutil", "Pillow")}
    required = {"torch": "2.6.0+cpu", "numpy": "2.2.6", "scipy": "1.15.1",
                "pytest": "9.1.1", "psutil": "7.2.2", "Pillow": "12.3.0"}
    if packages != required or platform.python_version() != "3.12.8":
        raise RuntimeError(f"declared CPU software differs: {packages}, Python {platform.python_version()}")
    import torch
    torch.set_num_interop_threads(1)
    from neuropixel.research.data import ResearchRoleTask, frozen_dataset
    from neuropixel.research.experiment import (
        configure_runtime, environment_record, resource_sample, tensor_digest,
    )
    from neuropixel.research.models import model_config
    device = configure_runtime("cpu", THREADS)
    task = ResearchRoleTask(8, 8, seed=0)
    development_hashes = {}
    for split, n, seed, wanted in [
        ("validation", 2048, 61001, "44873b31d17df696ceef3ff7c5fd2f1878921305d19623443780a443be364e0a"),
        ("train", 512, 61006, "f74bb2ae12e99b579eeea485bf6c262805a73f3c086a94421478466fc166b564"),
    ]:
        values = frozen_dataset(task, split, n, seed)
        actual = tensor_digest(*values)
        if actual != wanted:
            raise RuntimeError(f"development tensor identity changed under this environment: {split}")
        development_hashes[split] = actual
    expected_parameters = {"neuropixel": 29824, "standard_nca": 30384,
                           "convgru": 27835, "relative_transformer": 29547}
    parameters = {family: model_config(family)["parameters"] for family in expected_parameters}
    if parameters != expected_parameters:
        raise RuntimeError(f"historical model parameter counts changed: {parameters}")
    resource = resource_sample(device, MINIMUM_RAM_GIB)
    record = {
        "schema_version": 1, "item": 6, "status": "preflight_passed",
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "recovery_manifest_sha256": digest(manifest_path),
        "verified_sha256": verified,
        "unavailable_historical_sha256": manifest["unavailable_historical_sha256"],
        "environment": environment_record(device),
        "validated_package_versions": packages,
        "resources": resource,
        "initial_available_ram_gib": available,
        "cpu_count_logical": psutil.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "parameter_counts": parameters,
        "development_tensor_sha256": development_hashes,
        "optimizer_updates": 0,
        "final_test_accessed": False,
    }
    print(json.dumps(record, indent=2, sort_keys=True, allow_nan=False), flush=True)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as stream:
            stream.write("# Item 6: source/environment preflight\n\n")
            stream.write("Seven historical source files and both recovered protocol/report anchors verified. ")
            stream.write("No optimizer updates or final-data access.\n\n")
            fence = chr(96) * 3
            stream.write(fence + "json\n" + json.dumps(record, indent=2, sort_keys=True) + "\n" + fence + "\n")


if __name__ == "__main__":
    main()
