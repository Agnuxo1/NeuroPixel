"""Generate an evidence-based English report from current admitted published data."""
from __future__ import annotations

import argparse
from pathlib import Path

from .common import read_json, sha256_file
from .energy import merge_receipts, summarize
from .statistics import paired_interval


def render(root: Path, recovery_path: Path | None = None) -> str:
    """Use actual rows and scopes; never infer an unmeasured capability."""
    original_path = root / "results/research/RENDER11_energy/original_20261009A/receipt.json"
    original = read_json(original_path)
    plan_path = root / "docs/research/RENDER11_energy_plan.json"
    plan = read_json(plan_path)
    if sha256_file(plan_path) != original["plan_sha256"]:
        raise ValueError("Energy plan identity differs")
    recovery = read_json(recovery_path) if recovery_path else None
    energy = merge_receipts(original, recovery, plan)
    visual = read_json(root / "results/research/SPRINT_review/VIS07_summary.json")
    timing = read_json(root / "results/research/RENDER10_benchmark/recovered_20261009B/independent_summary.json")
    if (timing["rows"], timing["gates"], timing["old_rows_repeated"]) != (245, 77, 0):
        raise ValueError("Complete registered RENDER10 cohort required")
    rows = [r for r in visual["rows"] if r["condition"] == "base"]
    seeds = sorted({r["seed"] for r in rows})
    by_seed = {}
    for seed in seeds:
        by_seed[seed] = {}
        for family in ("nca", "cnn"):
            found = [r for r in rows if r["seed"] == seed and r["family"] == family]
            if len(found) != 1:
                raise ValueError("Incomplete saved visual pair")
            by_seed[seed][family] = found[0]
    differences = [100*(pair["nca"]["correct"]-pair["cnn"]["correct"])/pair["nca"]["n"] for pair in by_seed.values()]
    interval = paired_interval(differences)
    nca = sum(100*p["nca"]["correct"]/p["nca"]["n"] for p in by_seed.values())/len(seeds)
    cnn = sum(100*p["cnn"]["correct"]/p["cnn"]["n"] for p in by_seed.values())/len(seeds)
    lines = ["# NeuroPixel: bounded visual utility and graphics execution", "",
             "Francisco Angulo de Lafuente. Evidence reconciliation report, 12 October 2026.", "",
             "## Abstract", "",
             f"Five archived paired fits on UCI optical digits give mean accuracy {nca:.4f}% for NeuroPixel and {cnn:.4f}% for the CNN. "
             f"The mean paired difference is {interval['mean']:.4f} percentage points, with exploratory Student interval "
             f"[{interval['interval'][0]:.4f}, {interval['interval'][1]:.4f}]. "
             "The completed RENDER10 cohort contains 245 registered timing rows and 77 output gates. "
             "These results establish bounded task utility and implementation checks; independent external replication remains pending.", "",
             "## Methods and statistical scope", "",
             "The saved VIS07 recipe uses five fresh paired initializations, DEV-only selection, a fixed forty-epoch budget and the official 1,797-image test. "
             "The models have similar nominal parameter counts but different functional capacity, receptive geometry and operation costs. "
             "The interval concerns training variation on one fixed test population, not general population or writer uncertainty. "
             "Graphics timing is FP32 on one RTX 3090, with resident inputs, sixteen updates and complete decoding, compared with CUDA eager and CUDA Graph. "
             "Recovered timing sessions and technical repeats are not independent scientific replications.", "",
             "## Visual outcomes", "", "| Seed | NeuroPixel correct | CNN correct | Test examples |", "|---|---:|---:|---:|"]
    for seed, pair in by_seed.items():
        lines.append(f"| {seed} | {pair['nca']['correct']} | {pair['cnn']['correct']} | {pair['nca']['n']} |")
    lines += ["", "## Graphics outcomes", "", "| Workload | Best CUDA ms | Compute ms | Best CUDA / compute |", "|---|---:|---:|---:|"]
    for case in timing["table"]:
        best = min(case["median_ms"]["cuda_eager"], case["median_ms"]["cuda_graph"])
        compute = case["median_ms"]["render"]
        lines.append(f"| {case['case']} | {best:.6f} | {compute:.6f} | {best/compute:.6f} |")
    lines += ["", "## Energy admission", "",
              f"The original receipt contains {len(original['rows'])} rows. The currently supplied evidence contains "
              f"{energy['unique_rows']}/{energy['expected_rows']} unique registered rows. "
              "The endpoint concerns whole-board counters, including background consumption; CPU, RAM, PSU and wall energy are excluded."]
    if energy["complete"]:
        measured = summarize(original, recovery, plan)
        lines += ["", "| Workload | Method | Median gross J/call | Technical minimum | Technical maximum |", "|---|---|---:|---:|---:|"]
        for row in measured["table"]:
            lines.append(f"| {row['case']} | {row['method']} | {row['median_j']:.6f} | {row['minimum_j']:.6f} | {row['maximum_j']:.6f} |")
        lines += ["", "This is a descriptive complete device cohort across interrupted sessions; it is not a universal efficiency claim."]
    else:
        lines += ["", "Missing keys: " + "; ".join(str(key) for key in energy["missing"]) + ". No complete comparative energy claim is admitted."]
    lines += ["", "## Negative findings and open capabilities", "",
              "The original relational H1 remains unsupported under its registered recipe. A06.1 component attribution used two initializations; "
              "general component benefits are unestablished. DEV04 failed its original precision target. "
              "The visual model is trained at eight updates; longer continuations reduce accuracy and increase state magnitude. "
              "Historical memory, growth and repair audits do not establish competent continual learning or autonomous learned repair. "
              "Rendering a learned state does not identify a unique semantic causal mechanism.", "",
              "## Mathematical extensions", "",
              "The completion kit proposes nonnormal-Jacobian diagnostics motivated by the Crouzeix result reported in openai/math family 325, "
              "and exact finite Boolean-noise controls corresponding to family 119. Their applicability requires the stated hypotheses. "
              "The supplied leaky tanh reference field uses a separate classical contraction argument. "
              "Its exact-weight certificate concerns ideal real arithmetic and fixed input; it does not prove useful indefinite memory, "
              "all-time GPU roundoff control or an empirical advantage of the new mathematics. Formal proof rebuilding remains a distinct gate.", "",
              "## Reproduction and independent assessment", "",
              "Use the frozen per-study source commits, original arrays, weights and receipts. Retain failed executions, numerical discrepancies and negative hypotheses. "
              "Publish the complete new cohort before drawing an architectural conclusion. Request external reproduction with an independently supplied final task. "
              "Repository publication and internal replay are not peer review or outside replication.", "",
              "## Evidence locations", "",
              "- `docs/research/VIS07_results.md` and `results/research/SPRINT_review/VIS07_summary.json`.",
              "- `docs/research/RENDER10_results.md` and `results/research/RENDER10_benchmark/recovered_20261009B/independent_summary.json`.",
              "- `docs/research/06_A06_1_results.md`, `DEV04_results.md` and `TASK4_historical_capabilities_audit_20261009.md`.",
              "- `results/research/RENDER11_energy/` and the original frozen energy plan.", "",
              "This automatically generated report reconciles saved numbers. The final submission also requires the full methods, references, "
              "scope-specific mathematical derivations and independently reviewed limitations described in the completion guide.", ""]
    return "\n".join(lines)


def main() -> None:
    """Write a new report path; never overwrite the historical manuscript."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--recovery", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    text = render(args.root, args.recovery)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


if __name__ == "__main__":
    main()
