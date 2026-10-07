"""Check both published item-6 inventories without importing a numeric runtime."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    core = load("item6_inventory_core", "scripts/research_ablations.py")
    growth = load("item6_inventory_growth", "scripts/research_growth_ablation.py")
    protocol = core.load_protocol()
    observed_core = core.inventory(protocol)
    expected_core = json.loads((ROOT / "docs/research/06_core_inventory.json").read_text(encoding="utf-8"))
    if observed_core != expected_core:
        raise RuntimeError("published core inventory does not match its actual constructor")
    observed_growth = growth.operations.growth_inventory()
    expected_growth = json.loads((ROOT / "docs/research/06_growth_inventory.json").read_text(encoding="utf-8"))
    if observed_growth != expected_growth:
        raise RuntimeError("published growth inventory does not match its actual constructor")
    if "torch" in sys.modules or "numpy" in sys.modules:
        raise RuntimeError("inventory construction imported a numerical runtime")
    print(json.dumps({
        "status": "verified", "core_training_runs": len(observed_core["planned_runs"]),
        "core_evaluation_cases": len(observed_core["planned_evaluations"]),
        "core_optimizer_updates": sum(row["updates"] for row in observed_core["planned_runs"]),
        "growth_trajectories": observed_growth["planned_trajectories"],
        "growth_training_stages": observed_growth["planned_training_stages"],
        "growth_optimizer_updates": observed_growth["planned_optimizer_updates"],
        "growth_gate_fits": observed_growth["planned_gate_fits"],
        "growth_routing_evaluations": observed_growth["planned_routing_evaluations"],
        "source": core.source_record(), "numerical_runtime_imported": False,
        "final_test_accessed": False,
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
