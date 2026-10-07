"""Plot saved item-7 null-selection summaries without drawing new samples.

The stored normal intervals describe Monte Carlo precision of the mean. They
are copied directly, converted to percent, and are not architecture uncertainty.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resource_sample():
    """Enforce the available-memory floor using the current Linux observation."""
    memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(memory["MemAvailable"].split()[0]) * 1024 / 2**30
    if available < 8:
        raise RuntimeError("Plotting requires at least 8 GiB available RAM")
    return {"at_utc": datetime.now(timezone.utc).isoformat(),
            "available_ram_gib": available, "provider": "/proc/meminfo:MemAvailable"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    started = datetime.now(timezone.utc).isoformat()
    samples = [resource_sample()]
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    outputs = {kind: args.output_prefix.with_suffix("." + kind) for kind in ("png", "pdf")}
    receipt = args.output_prefix.with_suffix(".receipt.json")
    for path in [*outputs.values(), receipt]:
        if path.exists():
            raise FileExistsError(f"Preserve existing output: {path}")
    analysis_sha = digest(args.analysis)
    data = json.loads(args.analysis.read_text())
    if (data["item"] != 7 or data["schema_version"] != 1 or data["status"] != "completed"
            or data["h1_eligible"] is not False):
        raise ValueError("Expected a completed item-7 illustrative analysis")
    recipe = data["recipe"]
    if (recipe["search_sizes"] != [1, 4, 16, 64] or recipe["true_accuracy"] != .5
            or recipe["replicates"] != 10000 or recipe["examples_per_partition"] != 256
            or data["replicate_count"] != 10000):
        raise ValueError("Saved analysis differs from the declared plotting scope")
    summaries = data["summaries"]
    if [row["search_size"] for row in summaries] != [1, 4, 16, 64]:
        raise ValueError("Missing, duplicated, or reordered search conditions")
    rows = []
    for source in summaries:
        row = {"search_size": source["search_size"],
               "analytic_development_mean_percent": 100 * source["analytic"]["development_mean"]}
        for field in ("development", "final"):
            saved = source["descriptive"][field]
            mean = saved["mean"]
            interval = saved["mean_monte_carlo_95_normal_interval"]
            if (len(interval) != 2 or not all(math.isfinite(v) for v in [mean, *interval])
                    or not 0 <= interval[0] <= mean <= interval[1] <= 1):
                raise ValueError("Invalid stored mean or Monte Carlo interval")
            row[field + "_mean_percent"] = 100 * mean
            row[field + "_95_mc_interval_percent"] = [100 * value for value in interval]
        if not math.isfinite(row["analytic_development_mean_percent"]):
            raise ValueError("Invalid analytic expectation")
        rows.append(row)

    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
    samples.append(resource_sample())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, (main_ax, zoom_ax) = plt.subplots(1, 2, figsize=(8.5, 4.5),
                                          gridspec_kw={"width_ratios": [1.65, 1]})
    fig.subplots_adjust(left=.085, right=.975, bottom=.32, top=.76, wspace=.32)
    fig.suptitle("Selection raises DEV scores under a 50% null", x=.085, ha="left", y=.965,
                 fontsize=14, fontweight="bold")
    fig.text(.085, .895, "10,000 independent replicates; 256 examples per partition; all candidates have true accuracy 50%.",
             ha="left", fontsize=9)
    xs = [row["search_size"] for row in rows]

    def draw_observed(ax, field, color, marker, label):
        means = [row[field + "_mean_percent"] for row in rows]
        intervals = [row[field + "_95_mc_interval_percent"] for row in rows]
        errors = [[mean - bounds[0] for mean, bounds in zip(means, intervals)],
                  [bounds[1] - mean for mean, bounds in zip(means, intervals)]]
        return ax.errorbar(xs, means, yerr=errors, color=color, marker=marker, markersize=4,
                           linewidth=1.4, elinewidth=1.1, capsize=4, capthick=1.1,
                           label=label, zorder=3)

    analytic, = main_ax.plot(xs, [row["analytic_development_mean_percent"] for row in rows],
                            color="#222222", linestyle="--", linewidth=1.2,
                            marker="D", markerfacecolor="white", markersize=4,
                            label="Analytic E[chosen DEV]", zorder=4)
    dev = draw_observed(main_ax, "development", "#C15E16", "o", "Chosen DEV mean")
    final = draw_observed(main_ax, "final", "#166C96", "s", "Independent FINAL mean")
    null = main_ax.axhline(50, color="#6A6A6A", linestyle=":", linewidth=1.2,
                           label="Stipulated true accuracy: 50%", zorder=1)
    draw_observed(zoom_ax, "final", "#166C96", "s", "Independent FINAL mean")
    zoom_ax.axhline(50, color="#6A6A6A", linestyle=":", linewidth=1.2, zorder=1)
    for ax in (main_ax, zoom_ax):
        ax.set_xscale("log", base=4)
        ax.set_xticks(xs, [str(x) for x in xs])
        ax.set_xlim(.8, 80)
        ax.set_xlabel("Candidates compared on DEV, M\n(log scale)")
        ax.grid(axis="y", color="#DDDDDD", linewidth=.6)
        ax.set_axisbelow(True)
    main_ax.set_title("Chosen scores and analytic expectation", loc="left", fontsize=10)
    main_ax.set_ylabel("Mean accuracy (%)")
    main_ax.set_ylim(49.5, 58.8)
    zoom_ax.set_title("FINAL: expanded vertical scale", loc="left", fontsize=10)
    zoom_ax.set_ylabel("Mean FINAL accuracy (%)")
    zoom_ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    bounds = [value for row in rows for value in row["final_95_mc_interval_percent"]]
    zoom_ax.set_ylim(min(bounds) - .025, max(bounds) + .025)
    fig.legend(handles=[dev, final, analytic, null], loc="lower left", bbox_to_anchor=(.075, .065),
               ncol=2, frameon=False, fontsize=8.5, columnspacing=2)
    fig.text(.085, .031, "Bars: 95% Monte Carlo precision of the mean (normal approximation), not architecture uncertainty.",
             fontsize=8.5, ha="left")
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    saved_at = datetime.fromisoformat(data["completed_at_utc"])
    for kind, path in outputs.items():
        metadata = ({"Software": "research_plot_holdout_selection.py"} if kind == "png" else
                    {"Creator": "research_plot_holdout_selection.py", "CreationDate": saved_at, "ModDate": saved_at})
        with path.open("xb") as stream:
            fig.savefig(stream, format=kind, dpi=240, facecolor="white", metadata=metadata)
            stream.flush()
            os.fsync(stream.fileno())
    plt.close(fig)
    samples.append(resource_sample())
    if digest(args.analysis) != analysis_sha:
        raise RuntimeError("Input changed while plotting; preserve outputs for inspection")
    record = {"schema_version": 1, "item": 7, "status": "rendered_from_saved_analysis",
              "started_at_utc": started, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "input": {"path": os.path.relpath(args.analysis, ROOT), "sha256": analysis_sha},
              "script_sha256": digest(Path(__file__)),
              "figures": {kind: {"path": os.path.relpath(path, ROOT), "bytes": path.stat().st_size,
                                  "sha256": digest(path)} for kind, path in outputs.items()},
              "runtime": {"python": platform.python_version(), "matplotlib": matplotlib.__version__,
                          "platform": platform.platform(), "numerical_threads": 1,
                          "torch_imported": "torch" in sys.modules},
              "resource_samples": samples, "plotted_values_percent": rows,
              "interval_source": "descriptive.{development,final}.mean_monte_carlo_95_normal_interval",
              "transform": "Multiply saved means, interval endpoints and analytic expectation by 100; no recomputation or clipping.",
              "scope": "Presentation only; no RNG calls, simulations, model evaluation, or new scientific contrasts.",
              "limitations": [data["mean_interval_scope"], data["dependence"], data["scientific_scope"],
                              "RAM observations are sampled available memory, not continuous minima or process peak RSS."]}
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"status": record["status"], "receipt": str(receipt), "sha256": digest(receipt)}))


if __name__ == "__main__":
    main()
