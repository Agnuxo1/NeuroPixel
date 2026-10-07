"""Render two item-9 figures from a complete VERIFIED saved-analysis JSON.

Presentation only: no model, generator, resampling, confidence-interval fitting,
or hypothesis test is imported or executed. Every plotted estimate/interval is
read from the frozen auditor's report. The analytic control lines are marginal
binding expectations, not fitted neural baselines or joint-success probabilities.
The t interval concerns five paired training realizations conditional on the
fixed dataset/selected recipe; it is not total architecture uncertainty or H1.

The auditor does not store a GitHub run ID, so --run-id is a required explicit
annotation, distinguished from verified JSON provenance in the render receipt.
Refuse an existing output directory. Failed renders retain any partial outputs.
The resulting figures still require a visual review before publication.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import re


THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")
for _variable in THREAD_VARIABLES:
    os.environ[_variable] = "1"

AUDITOR_SHA256 = "b12ac084aed181b1a22aa96f1b28a1c467d55587e205e0de7f18bb7e32b80e05"
FAMILIES = ("neuropixel", "relative_transformer")
FAMILY_LABELS = {"neuropixel": "NeuroPixel", "relative_transformer": "RelativeTransformer"}
COLORS = {"neuropixel": "#1263A0", "relative_transformer": "#C04B26"}
SEEDS = (40, 41, 42, 43, 44)
CONDITIONS = ("base", "swap_queried_agent_patient", "swap_other_agent_patient",
              "relabel_events", "query_switch", "layout_permutation")
CONDITION_LABELS = ("Base", "Queried event:\nagent/patient swap", "Other event:\nagent/patient swap",
                    "Relabel all\nevents + query", "Switch queried\nevent", "Permute\nlayout")
MARKERS = ("o", "s", "^", "P", "X")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def memory_sample(phase):
    fields = {line.split(":")[0]: int(line.split()[1])
              for line in Path("/proc/meminfo").read_text().splitlines()}
    row = {"at_utc": now(), "phase": phase, "available_ram_gib": fields["MemAvailable"]/1024**2}
    require(row["available_ram_gib"] >= 8, f"at least8GiB available RAM is required at {phase}")
    return row


def _object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError("nonfinite JSON value: " + value)


def number(value, label, *, accuracy=False):
    require(type(value) in (int, float) and math.isfinite(value), "invalid finite number: " + label)
    if accuracy:
        require(0 <= value <= 1, "accuracy outside[0,1]: " + label)
    return float(value)


def hash_string(value, digits, label):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{"+str(digits)+r"}", value),
            "invalid provenance hash: " + label)
    return value


def extract(report):
    """Validate the plotted subset and return saved values without new estimates."""
    require(report.get("schema_version") == 1 and report.get("item") == 9
            and report.get("status") == "verified" and report.get("issues") == [],
            "a verified item-9 report with zero issues is required")
    require(report.get("analysis_source_sha256") == AUDITOR_SHA256,
            "report was not produced by the frozen approved auditor")
    gate = report["gate"]
    require(gate["verified_local_hash_links"] is True and gate["chronology_verified"] is True,
            "report lacks verified gate/provenance links")
    hash_string(gate["source_commit"], 40, "source_commit")
    hash_string(gate["archive_commit_claim"], 40, "archive_commit_claim")
    hash_string(report["recipe_sha256"], 64, "recipe_sha256")
    dataset = report["final_dataset"]
    require(dataset["groups"] == 256 and dataset["nominal_rows"] == 12288,
            "figure scope requires the frozen256-bag/six-condition final panel")
    expected_ids = {f"{family}_seed{seed}" for family in FAMILIES for seed in SEEDS}
    require(set(report["final_recount"]) == expected_ids, "missing or unexpected model/seed arm")
    require(set(report["family_training_realizations"]) == set(FAMILIES), "family summary inventory differs")
    binding, means = {}, {}
    for family in FAMILIES:
        summaries = report["family_training_realizations"][family]
        require(set(summaries) == set(CONDITIONS), "family condition summary inventory differs")
        binding[family], means[family] = {}, {}
        for condition in CONDITIONS:
            values = []
            for seed in SEEDS:
                row = report["final_recount"][f"{family}_seed{seed}"]
                require(row["family"] == family and row["seed"] == seed and set(row["metrics"]) == set(CONDITIONS),
                        "model/seed/condition identity differs")
                point = row["metrics"][condition]
                require(point["n"] == 2048 and point["groups"] == 256, "condition denominator differs")
                values.append(number(point["binding"], f"{family}/{seed}/{condition}", accuracy=True))
            saved = summaries[condition]["binding"]
            require(saved["n"] == 5 and saved["values"] == values, "saved five-seed summary values differ")
            binding[family][condition] = values
            means[family][condition] = number(saved["mean"], family+"/"+condition+" mean", accuracy=True)
    paired = report["primary_paired_training_realizations"]
    require(paired["seeds"] == list(SEEDS) and paired["n"] == 5 and paired["df"] == 4,
            "the paired interval requires exactly five seeds and df4")
    require(paired["left"] == binding[FAMILIES[0]]["base"] and paired["right"] == binding[FAMILIES[1]]["base"],
            "paired interval endpoints differ from saved base binding")
    require(isinstance(paired["differences"], list) and len(paired["differences"]) == 5,
            "paired difference vector is incomplete")
    differences = [number(value, "paired difference") for value in paired["differences"]]
    # This checks sign/alignment, not a new estimate or uncertainty calculation.
    require(all(value == left-right for value, left, right in zip(differences, paired["left"], paired["right"])),
            "paired difference sign/alignment differs")
    mean = number(paired["mean"], "saved paired mean")
    require(isinstance(paired["t95"], list) and len(paired["t95"]) == 2, "saved t interval is malformed")
    low, high = [number(value, "saved t95 endpoint") for value in paired["t95"]]
    require(low <= mean <= high, "saved t interval does not contain its mean")
    controls = dataset["controls"]
    require(controls["bag_category"]["binding"] == .25
            and controls["role_only"]["binding"] == .5
            and controls["event_category"]["binding"] == .5,
            "saved analytic binding controls differ from the declared reference lines")
    return {"binding": binding, "means": means, "differences": differences, "paired_mean": mean,
            "t95": [low, high], "analytic_binding_lines": {"bag_category": .25, "role_only_and_event_category": .5},
            "source_commit": gate["source_commit"], "recipe_sha256": report["recipe_sha256"],
            "inventory_archive_commit_claim": gate["archive_commit_claim"]}


def provenance_line(values, run_id, input_hash):
    return (f"Run {run_id}  |  source {values['source_commit'][:12]}  |  recipe {values['recipe_sha256'][:12]}"
            f"  |  verified analysis {input_hash[:12]}")


def binding_figure(plt, values, provenance):
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(14, 8.5))
    fig.subplots_adjust(left=.07, right=.985, bottom=.31, top=.85)
    fig.suptitle("Two-event role binding across the six fixed conditions", fontsize=17, x=.07, ha="left", y=.97)
    fig.text(.07, .915, "Five paired training realizations per family; every point is shown. Solid lines connect saved means.", fontsize=11)
    ax.axhline(25, color="#767676", linestyle=(0, (5, 4)), linewidth=1.3, zorder=1)
    ax.axhline(50, color="#525252", linestyle=(0, (2, 3)), linewidth=1.3, zorder=1)
    for family, family_offset in zip(FAMILIES, (-.19, .19)):
        x = [index+family_offset for index in range(6)]
        ax.plot(x, [100*values["means"][family][condition] for condition in CONDITIONS],
                color=COLORS[family], linewidth=2.5, zorder=2)
        # Fixed horizontal spacing prevents equal scores from hiding seeds.
        for index, (seed, marker, seed_offset) in enumerate(zip(SEEDS, MARKERS, (-.14, -.07, 0, .07, .14))):
            ax.scatter([value+seed_offset for value in x],
                       [100*values["binding"][family][condition][index] for condition in CONDITIONS],
                       color=COLORS[family], marker=marker, s=62, edgecolor="white", linewidth=.65,
                       zorder=4, label=f"Seed {seed}" if family == FAMILIES[0] else None)
    ax.set_xticks(range(6), CONDITION_LABELS)
    ax.tick_params(axis="x", labelsize=10, pad=10)
    ax.set_xlim(-.6, 5.6)
    ax.set_ylim(-2, 102)
    ax.set_yticks(range(0, 101, 10))
    ax.set_ylabel("Macro agent/patient accuracy (%)", fontsize=12)
    ax.grid(axis="y", color="#DADADA", linewidth=.65, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    lines = [Line2D([0], [0], color=COLORS[family], linewidth=2.5, label=FAMILY_LABELS[family]+" mean") for family in FAMILIES]
    lines += [Line2D([0], [0], color="#767676", linestyle=(0, (5, 4)), linewidth=1.3,
                     label="25%: analytic bag/category expectation"),
              Line2D([0], [0], color="#525252", linestyle=(0, (2, 3)), linewidth=1.3,
                     label="50%: analytic role-only or event/category expectation")]
    fig.legend(handles=lines, loc="lower center", bbox_to_anchor=(.51, .158), ncol=2, frameon=False, fontsize=10)
    seed_handles = [Line2D([0], [0], color="#555555", linestyle="none", marker=marker, markersize=7,
                           label=f"Seed {seed}") for seed, marker in zip(SEEDS, MARKERS)]
    fig.legend(handles=seed_handles, loc="lower center", bbox_to_anchor=(.51, .106), ncol=5, frameon=False, fontsize=9.5)
    fig.text(.07, .075, "Analytic lines are untrained input-only marginal expectations, not fitted neural baselines. No uncertainty intervals are drawn here.", fontsize=9)
    fig.text(.07, .047, "Seeds vary initialization and input/firing streams. Conditions share 256 bags and correlated query pairs; they are not independent replications.", fontsize=9)
    fig.text(.07, .017, provenance, fontsize=8.5, color="#444444")
    return fig


def difference_figure(plt, values, provenance):
    fig, ax = plt.subplots(figsize=(11.5, 7))
    fig.subplots_adjust(left=.22, right=.95, bottom=.29, top=.82)
    fig.suptitle("Base binding: paired NeuroPixel − Transformer differences", fontsize=16, x=.065, ha="left", y=.97)
    fig.text(.065, .91, "Five saved differences and their saved mean with a two-sided 95% t interval (df = 4).", fontsize=11)
    deltas = [100*value for value in values["differences"]]
    mean, low, high = 100*values["paired_mean"], 100*values["t95"][0], 100*values["t95"][1]
    y = list(range(5, 0, -1))
    ax.axvline(0, color="#777777", linestyle=(0, (4, 4)), linewidth=1.1, zorder=0)
    for row, delta, marker in zip(y, deltas, MARKERS):
        ax.scatter([delta], [row], marker=marker, color=COLORS["neuropixel"], edgecolor="white", s=88, zorder=3)
    ax.hlines(-.2, low, high, color="#202020", linewidth=2.5, zorder=2)
    ax.vlines([low, high], -.32, -.08, color="#202020", linewidth=1.7, zorder=2)
    ax.scatter([mean], [-.2], marker="D", color="#202020", s=88, zorder=4)
    lower, upper = min(0, low, *deltas), max(0, high, *deltas)
    margin = max(2, .12*(upper-lower))
    ax.set_xlim(lower-margin, upper+margin)  # Keep the original, unclipped t interval.
    ax.set_ylim(-.7, 5.6)
    ax.set_yticks(y+[-.2], [f"Seed {seed}" for seed in SEEDS]+["Mean + saved t95"])
    ax.set_xlabel("NeuroPixel − RelativeTransformer binding (percentage points)", fontsize=11)
    ax.grid(axis="x", color="#DADADA", linewidth=.65, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.text(.065, .178, f"Saved mean: {mean:+.2f} pp   |   saved t95: [{low:+.2f}, {high:+.2f}] pp   |   n = 5 paired training realizations", fontsize=10.5)
    fig.text(.065, .125, "The interval concerns training realizations conditional on this fixed dataset, selected rates and recipe; it is not total architecture uncertainty.", fontsize=9)
    fig.text(.065, .083, "No interval was recomputed for this figure. Negative values favor the Transformer; positive values favor NeuroPixel. Historical H1 is unchanged.", fontsize=9)
    fig.text(.065, .035, provenance, fontsize=8.5, color="#444444")
    return fig


def save_new(path, payload):
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Final verified JSON from the frozen item-9 auditor")
    parser.add_argument("--output-dir", type=Path, required=True, help="A new directory; existing directories are refused")
    parser.add_argument("--run-id", required=True, help="Explicit GitHub run annotation; the analysis JSON does not contain it")
    args = parser.parse_args()
    require(re.fullmatch(r"[0-9]+", args.run_id), "run ID annotation must be numeric")
    require(not args.output_dir.exists(), "refusing to overwrite an existing output directory")
    samples = [memory_sample("before_input")]
    payload = args.input.read_bytes()
    input_hash = sha_bytes(payload)
    report = json.loads(payload, object_pairs_hook=_object, parse_constant=_nonfinite)
    values = extract(report)
    samples.append(memory_sample("before_matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "svg.fonttype": "none", "svg.hashsalt": input_hash, "axes.unicode_minus": True})
    args.output_dir.mkdir(parents=True, exist_ok=False)
    receipt = {"schema_version": 1, "item": 9, "status": "rendering", "started_at_utc": now(),
        "input": {"path": str(args.input.resolve()), "sha256": input_hash, "bytes": len(payload), "status": "verified"},
        "helper_sha256": sha_bytes(Path(__file__).read_bytes()), "frozen_auditor_sha256": AUDITOR_SHA256,
        "source_commit": values["source_commit"], "recipe_sha256": values["recipe_sha256"],
        "inventory_archive_commit_claim": values["inventory_archive_commit_claim"],
        "run_annotation": {"value": args.run_id, "provenance": "explicit caller annotation; not stored in the analyzer JSON"},
        "plotted_saved_values": values, "numeric_derivations": "Multiplication by100 for percent/percentage-point axes only; no newly estimated mean or interval.",
        "runtime": {"python": platform.python_version(), "matplotlib": matplotlib.__version__,
                    "numpy": importlib.metadata.version("numpy"), "thread_environment": {key: os.environ[key] for key in THREAD_VARIABLES}},
        "resource_samples": samples, "outputs": {}, "visual_review_pending": True,
        "limits": "Editorial rendering of saved outcomes; no model, data generation, RNG draws, uncertainty refit or scientific test."}
    try:
        provenance = provenance_line(values, args.run_id, input_hash)
        for stem, factory in (("binding_by_condition", binding_figure), ("base_paired_difference", difference_figure)):
            samples.append(memory_sample("before_"+stem))
            fig = factory(plt, values, provenance)
            try:
                for extension in ("png", "svg"):
                    buffer = io.BytesIO()
                    metadata = ({"Title": stem.replace("_", " "), "Description": provenance, "Date": None}
                                if extension == "svg" else {"Title": stem.replace("_", " "), "Description": provenance})
                    fig.savefig(buffer, format=extension, dpi=180, facecolor="white", metadata=metadata)
                    image = buffer.getvalue()
                    destination = args.output_dir/f"{stem}.{extension}"
                    save_new(destination, image)
                    receipt["outputs"][destination.name] = {"bytes": len(image), "sha256": sha_bytes(image)}
            finally:
                plt.close(fig)
        samples.append(memory_sample("after_figures"))
        receipt["status"] = "rendered"
    except Exception as error:
        receipt.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        receipt["completed_at_utc"] = now()
        receipt["minimum_sampled_available_ram_gib"] = min(row["available_ram_gib"] for row in samples)
        save_new(args.output_dir/"figure_receipt.json", (json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False)+"\n").encode())
    print(json.dumps({"status": receipt["status"], "output_dir": str(args.output_dir), "files": list(receipt["outputs"]),
                      "visual_review_pending": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
