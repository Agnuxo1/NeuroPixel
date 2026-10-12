"""A18.2.G: semantic review of the graphics-backend timing and device-energy claim candidates (RENDER08-11).

Every numeric table cell in the four reports is recounted from the retained receipt rows
(median over technical blocks); no benchmark is re-executed.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict

from completion_claim_review_lib import ED_HEADER, NO_EXT, ROOT, Decisions, claim, run_review

R = "docs/research/"
RR = "results/research/"
POP = ("One NVIDIA RTX 3090, FP32 with TF32 off, resident seeding, sixteen updates plus full decoding/scanner; "
       "seven registered workloads (C16/C48 channels on 8, 32 and 128 grids, RGB8 retina C16)")
TECH = "Technical blocks on one GPU are not scientific replicas; descriptive point comparisons without confidence intervals"
TIME_SCOPE = "Median host milliseconds per call; setup, compilation, transfers and training are outside the primary timing"
ENERGY_UNKNOWN = ("Per-inference energy of short calls is unknown: counter zeros or staleness are not zero energy; "
                  "NVML measures the GPU only, not wall, CPU or RAM")
RECEIPT = {
    "R08": RR + "RENDER08_benchmark/original_20261009A/receipt.json",
    "R09": RR + "RENDER09_benchmark/original_20261009A/receipt.json",
    "R10": RR + "RENDER10_benchmark/recovered_20261009B/receipt.json",
    "R11": RR + "RENDER11_energy/recovered_20261009B/receipt.json",
}
# report column order -> receipt method (None marks the reported ratio column)
COLUMNS = {
    "R08": ["cuda_eager", "cuda_graph", "render", None],
    "R09": ["cuda_eager", "cuda_graph", "render", "scalar_render", None],
    "R10": ["cuda_eager", "cuda_graph", "scalar_render", "vector_render", "render", None],
}
RATIO_TARGET = {"R08": "render", "R09": "render", "R10": "render"}
TABLE_LINES = {"R08": range(5, 12), "R09": range(5, 12), "R10": range(7, 14)}


def recount() -> tuple[dict, list]:
    out, checks = {}, defaultdict(list)
    medians = {}
    for tag in ("R08", "R09", "R10"):
        rows = json.loads((ROOT / RECEIPT[tag]).read_text(encoding="utf-8"))["rows"]
        groups = defaultdict(list)
        for row in rows:
            groups[(row["case"], row["method"])].append(1000.0 * row["host_seconds"] / row["repetitions"])
        medians[tag] = {key: statistics.median(values) for key, values in groups.items()}
        out[f"{tag}|rows"] = len(rows)
        lines = (ROOT / f"{R}RENDER{tag[1:]}_results.md").read_text(encoding="utf-8").splitlines()
        for number in TABLE_LINES[tag]:
            cells = [c.strip() for c in lines[number - 1].strip().strip("|").split("|")]
            case = cells[0]
            for method, cell in zip(COLUMNS[tag], cells[1:]):
                if method is None:
                    best = min(medians[tag][(case, "cuda_eager")], medians[tag][(case, "cuda_graph")])
                    value = best / medians[tag][(case, RATIO_TARGET[tag])]
                    key = f"{tag}|{case}|ratio"
                else:
                    value, key = medians[tag][(case, method)], f"{tag}|{case}|{method}"
                out[key] = value
                checks[tag].append(("recount:" + key, cell.replace("×", "")))
    rows = json.loads((ROOT / RECEIPT["R11"]).read_text(encoding="utf-8"))["rows"]
    out["R11|rows"] = len(rows)
    out["R11|unique_keys"] = len({(r["case"], r["block"], r["method"]) for r in rows})
    groups = defaultdict(list)
    for row in rows:
        groups[(row["case"], row["method"])].append(row["gpu_gross_j_per_call"])
    lines = (ROOT / f"{R}RENDER11_results.md").read_text(encoding="utf-8").splitlines()
    for number in range(11, 46):
        case, method, med, low, high = [c.strip() for c in lines[number - 1].strip().strip("|").split("|")]
        values = groups[(case, method)]
        assert len(values) == 3, (case, method, len(values))
        for name, value, cell in (("median", statistics.median(values), med), ("min", min(values), low),
                                  ("max", max(values), high)):
            out[f"R11|{case}|{method}|{name}"] = value
            checks["R11"].append((f"recount:R11|{case}|{method}|{name}", cell))
    lower = higher = 0
    for case in sorted({c for c, _ in groups}):
        best_cuda = min(statistics.median(groups[(case, m)]) for m in ("cuda_eager", "cuda_graph"))
        best_render = min(statistics.median(groups[(case, m)]) for m in ("vector_render", "render"))
        lower += best_render < best_cuda
        higher += best_render > best_cuda
    out["R11|render_lower_workloads"], out["R11|render_higher_workloads"] = lower, higher
    original = json.loads((ROOT / RR / "RENDER10_benchmark/original_20261009A/receipt.json").read_text(encoding="utf-8"))["rows"]
    out["R10|original_rows"] = len(original)
    return out, checks


RECOUNTS, TABLE_CHECKS = recount()

CLAIMS = [
    claim("G-R08-01", "On one RTX 3090 (FP32, 16 updates plus full scanner, resident inputs), the RENDER08 fragment renderer "
          "was slower than the best measured CUDA comparator (eager or CUDA Graph) in all seven registered workloads, with "
          "best-CUDA/render ratios of 0.082-0.319 (median host milliseconds over seven technical blocks).",
          POP, "experiment_complete", 2, [TECH, TIME_SCOPE, ENERGY_UNKNOWN, NO_EXT,
          "The report column labelled 'Render vector ms' holds the single (scalar) RENDER08 renderer"],
          R + "RENDER08_benchmark_plan.json", [RECEIPT["R08"]],
          TABLE_CHECKS["R08"] + [("recount:R08|rows", "147")], study="RENDER08"),
    claim("G-R09-01", "In RENDER09 the vectorized fragment renderer was faster than the scalar renderer in all seven workloads; "
          "against the best CUDA comparator its point ratio was 1.536 on C16/grid128, 1.039 on C16/grid8 and 1.072 on the "
          "RGB8 retina (small and possibly noise-sensitive), and below 1 in the other four workloads.",
          POP, "experiment_complete", 2, [TECH, TIME_SCOPE, ENERGY_UNKNOWN, NO_EXT,
          "No general advantage, graphics training or energy efficiency is established"],
          R + "RENDER09_benchmark_plan.json", [RECEIPT["R09"]],
          TABLE_CHECKS["R09"] + [("recount:R09|rows", "196")], study="RENDER09"),
    claim("G-R10-01", "In RENDER10 (OpenGL 4.6), all 245 registered rows and 77 output gates are complete; the compute path "
          "was faster than the best CUDA comparator in six of seven point comparisons (1.062-2.084x, largest on C16/grid128) "
          "and slower on C48/grid128 (0.524x).",
          POP, "experiment_complete", 2, [TECH, TIME_SCOPE, NO_EXT,
          "Measurement sessions were interrupted and resumed; descriptive technical measurements, not independent replications",
          "CUDA eager and CUDA Graph remain strong comparators; no universal advantage is claimed"],
          R + "RENDER10_benchmark_plan.json",
          [RECEIPT["R10"], RR + "RENDER10_benchmark/recovered_20261009B/independent_summary.json"],
          TABLE_CHECKS["R10"] + [("recount:R10|rows", "245")], study="RENDER10"),
    claim("G-R10-02", "The RENDER10 recovery preserved the original 143 rows exactly and measured only the missing keys, with "
          "identical registered input arrays and unchanged source, completing 245 rows.",
          POP, "engineering_complete", 1, ["The interruption separates measurement sessions and qualifies interpretation", NO_EXT],
          None, [RR + "RENDER10_benchmark/original_20261009A/receipt.json", RECEIPT["R10"],
                 RR + "RENDER10_benchmark/recovered_20261009B/independent_summary.json"],
          [("recount:R10|original_rows", "143"), ("recount:R10|rows", "245")], study="RENDER10"),
    claim("G-R11-01", "Sustained GPU-board energy (whole-board NVML counters, gross joules per call, median of three technical "
          "blocks; complete 105-row registered cohort): the best render-family method used less board energy per call than "
          "the best CUDA method in four of seven workloads (C16 grid8/32/128 and the RGB8 retina) and more in three "
          "(C48 grid8/32/128); for example 0.485 J (vector render) versus 0.641 J (CUDA eager) on C16/grid128, and "
          "2.554 J (vector render) versus 1.818 J (CUDA eager) on C48/grid128.",
          POP + "; blocks of at least three active seconds with idle observations before and after", "experiment_complete", 2,
          ["GPU board energy only, including background consumption; CPU, RAM, PSU and whole-system wall energy unmeasured",
           "Three technical blocks across interrupted sessions are not independent scientific replicates",
           "Idle-adjusted estimates, including negative values, are kept unclipped in the receipt; baselines are descriptive",
           "Does not establish universal efficiency or energy savings", NO_EXT],
          R + "RENDER11_energy_plan.json",
          [RECEIPT["R11"], RR + "COMPLETION20261012/energy_complete_summary.json",
           RR + "COMPLETION20261012/energy_recovery_admission.json"],
          TABLE_CHECKS["R11"] + [("recount:R11|render_lower_workloads", "4"), ("recount:R11|render_higher_workloads", "3"),
                                 ("recount:R11|rows", "105"), ("recount:R11|unique_keys", "105")], study="RENDER11"),
    claim("G-R11-02", "The original RENDER11 execution (receipt sha256 883fc6be...) preserved 103 of 105 registered rows and "
          "all 77 output gates before the unchanged 8 GiB RAM floor stopped it; the earlier recovery (receipt sha256 "
          "bc19c7b3..., Git archive 87c60227) was admitted with 105 unique keys, the 103 original rows identical and zero "
          "repeated closed measurements, and no physical measurement occurred during admission.",
          POP, "engineering_complete", 1, ["Admission proves byte and key consistency only", NO_EXT], None,
          [RR + "RENDER11_energy/original_20261009A/receipt.json", RECEIPT["R11"],
           RR + "COMPLETION20261012/energy_recovery_admission.json"],
          [("text:883fc6be14663f112d6fdf36f4ef112e35215bddfa7f9699d3c37a85559719a2", RR + "COMPLETION20261012/energy_recovery_admission.json"),
           ("text:bc19c7b3bff67b3fc8fe8544188f63aaf342a7cfaa435f3ef3e89e397ff1039f", RR + "COMPLETION20261012/energy_recovery_admission.json"),
           ("103", RR + "COMPLETION20261012/energy_recovery_admission.json")], study="RENDER11"),
]

D = Decisions()
for tag, cid in (("08", "G-R08-01"), ("09", "G-R09-01")):
    F = f"{R}RENDER{tag}_results.md"
    D.put(F, 3, "editorial", ED_HEADER); D.put(F, range(5, 12), "data_row", [cid]); D.put(F, 13, "method", [cid])
    D.put(F, 15, "claim", [cid]); D.put(F, 17, "limit", [cid])
F = R + "RENDER10_results.md"
D.put(F, 3, "claim", ["G-R10-01", "G-R10-02"]); D.put(F, 5, "editorial", ED_HEADER); D.put(F, range(7, 14), "data_row", ["G-R10-01"])
D.put(F, 15, "claim", ["G-R10-01"]); D.put(F, 17, "limit", ["G-R10-01"]); D.put(F, 19, "claim", ["G-R11-02", "G-R11-01"])
F = R + "RENDER11_results.md"
D.put(F, 3, "claim", ["G-R11-02"]); D.put(F, [5, 7], "provenance", ["G-R11-02"]); D.put(F, 9, "editorial", ED_HEADER)
D.put(F, range(11, 46), "data_row", ["G-R11-01"]); D.put(F, 47, "limit", ["G-R11-01"])

DISCREPANCIES = [
    dict(source=R + "RENDER08_results.md", line=3, claim="G-R08-01", report_text="column 'Render vector ms'",
         receipt=RECEIPT["R08"], receipt_field="rows[*].method == 'render'",
         finding="RENDER08 has a single renderer method whose medians (11.76-44.17 ms) match the RENDER09 scalar renderer; "
                 "line 15 calls it the scalar renderer. The column label needs an archived correction in A18.3."),
    dict(source=R + "RENDER09_results.md", line=15, claim="G-R09-01", report_text="otras configuraciones no superan CUDA",
         receipt=RECEIPT["R09"], receipt_field="median host ms ratios",
         finding="Two other workloads have point ratios above 1 (C16/grid8 1.0389, RGB8 retina 1.0720); the sentence should say "
                 "they exceed CUDA only marginally and may reflect noise. The ledger states the ratios."),
]

if __name__ == "__main__":
    sys.exit(run_review("A18.2.G", CLAIMS, D, DISCREPANCIES, "A18.2.G.json",
                        "Semantic review of batch A18.2.G; every table cell recounted from receipt rows; board energy kept "
                        "separate from wall energy and technical repeats separate from replicas.", recounts=RECOUNTS))
