"""Plot the two verified item-6 analysis reports without accessing study arrays.

This is a post-freeze editorial helper. It reads only the two supplied JSON
reports, preserves the prespecified condition order, and performs no inference,
recount, model selection, or confidence-interval calculation. It does not import
the scientific code or analyzers. Matplotlib is imported only after validation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
from itertools import product
import json
import math
from pathlib import Path
import platform
import re


SEEDS = (20, 21)
ROLES = ("AGENTE", "ACCION", "PACIENTE", "LUGAR")
ROUTERS = ("uniform", "random", "scanner", "learned")
# Order of first appearance in the frozen growth controller, not score order.
BANKS = ("snapshots", "duplicate_slots", "single_final",
         "adaptive_novelty", "adaptive_resonance")
TOPICS = ("0", "1", "2", "all")
BANK_LABELS = {"snapshots": "Snapshots [A, B, C]",
               "duplicate_slots": "Duplicate slots [A, B, B]",
               "single_final": "Final sequential expert C",
               "adaptive_novelty": "Adaptive novelty",
               "adaptive_resonance": "Adaptive resonance"}
TRAIN_DAMAGE = {"after_step": 8, "apply_probability": .5,
                "erase_probability": .3, "seed": 62001}
EVAL_DAMAGE = {"after_step": 8, "erase_probability": .3, "seed": 62002}
FILES = ("06_core_paired_scores.png", "06_core_paired_scores.pdf",
         "06_growth_binding.png", "06_growth_binding.pdf", "06_plot_manifest.json")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def no_nonfinite(value):
    if isinstance(value, dict):
        for item in value.values():
            no_nonfinite(item)
    elif isinstance(value, list):
        for item in value:
            no_nonfinite(item)
    elif isinstance(value, float):
        require(math.isfinite(value), "report contains a nonfinite numerical value")


def score(value, label):
    require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
            f"{label} must be a finite score in [0, 1]")
    return float(value)


def read_report(path, panel):
    data = path.read_bytes()

    def reject_constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")

    report = json.loads(data, parse_constant=reject_constant)
    require(isinstance(report, dict) and report.get("schema_version") == 1
            and report.get("item") == 6 and report.get("panel") == panel
            and report.get("status") == "verified" and report.get("issues") == [],
            f"{panel} input must be a completed verified item-6 report with zero issues")
    no_nonfinite(report)
    return report, {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest(),
                    "bytes": len(data)}


def expected_core():
    """Transcribe the frozen 26 training recipes and their 34 evaluation cases."""
    definitions = []
    for seed, tied, school, reinject in product(SEEDS, (False, True), (0., .3), (False, True)):
        definitions.append(("factorial", seed, tied, school, reinject, 16, 1024, None))
    for seed, depth in product(SEEDS, (1, 4)):
        definitions.append(("recurrence", seed, True, 0., True, depth, 1024, None))
    for seed in SEEDS:
        definitions.append(("damage", seed, True, 0., True, 16, 1024, dict(TRAIN_DAMAGE)))
    for seed, school in product(SEEDS, (0., .3)):
        definitions.append(("optimization", seed, True, school, True, 16, 8192, None))
    expected = []
    for panel, seed, tied, school, reinject, depth, updates, damage in definitions:
        school_id = str(float(school)).replace(".", "p")
        run_id = (f"{panel}_tie{int(tied)}_school{school_id}_reinj{int(reinject)}"
                  f"_T{depth}_damage{int(damage is not None)}_s0_i{seed}_u{updates}")
        config = {"run_id": run_id, "panel": panel, "phase": "item6_ablation",
                  "protocol_id": "NP-SCI-20261006-v1", "family": "neuropixel",
                  "vocab": 35, "height": 8, "width": 8, "split_seed": 0,
                  "init_seed": seed, "learning_rate": .003, "updates": updates,
                  "batch_size": 64, "steps": depth, "weight_decay": .0001,
                  "gradient_clip": 1, "school_weight": school,
                  "train_sample_seed": 9002, "update_random_seed": 9001,
                  "variant": {"tied": tied, "reinject": reinject, "freeze_pad": False},
                  "training_damage": damage}
        cases = [("clean", "clean", None, None)]
        reference = panel == "factorial" and tied and reinject and school == 0
        if reference:
            cases.extend((f"deploy_T{t}", "deployment_truncation", t, None) for t in (1, 4))
        if reference or panel == "damage":
            cases.append(("lesion", "fixed_lesion", None, dict(EVAL_DAMAGE)))
        for suffix, kind, override, lesion in cases:
            case = {"evaluation_id": run_id + "__" + suffix, "run_id": run_id,
                    "kind": kind, "steps_override": override, "damage": lesion}
            expected.append((config, case))
    return expected


def condition_label(config, case):
    panel = {"factorial": "Factorial", "recurrence": "Trained depth",
             "damage": "Training damage", "optimization": "Longer budget"}[config["panel"]]
    variant = config["variant"]
    first = (f"{panel} | tie {int(variant['tied'])} | school {config['school_weight']:g}"
             f" | reinj {int(variant['reinject'])}")
    effective = case["steps_override"] or config["steps"]
    training = "damage" if config["training_damage"] is not None else "clean"
    evaluation = "lesion" if case["damage"] is not None else "clean"
    second = (f"train T{config['steps']}, {config['updates']:,} updates, {training}"
              f" | eval T{effective}, {evaluation}")
    return first + "\n" + second


def validate_core(report):
    require(report.get("selection") == "none"
            and report.get("H1") == "not_reconsidered_in_exploratory_item6", "core scope differs")
    exposure = report["exposure"]
    require(exposure["training_runs"] == 26 and exposure["evaluation_cases"] == 34
            and exposure["optimizer_updates"] == 55296 and exposure["minibatch_examples"] == 3538944,
            "core completed inventory/exposure differs")
    rows, expected = report["evaluations"], expected_core()
    require(len(rows) == len(expected) == 34, "core requires all 34 evaluation rows")
    groups, seen, run_ids = {}, set(), set()
    for row, (config, case) in zip(rows, expected):
        expected_config = dict(config, controller_source=report["controller_source"])
        require(row.get("status") == "completed" and row["config"] == expected_config
                and row["evaluation_case"] == case, "core order or frozen recipe differs")
        require(row["source"] == report["source"]
                and row["controller_source"] == report["controller_source"], "core row provenance differs")
        ident, seed = case["evaluation_id"], config["init_seed"]
        require(ident not in seen, "duplicate core evaluation")
        seen.add(ident)
        run_ids.add(config["run_id"])
        metric = row["recounted_metrics"]
        require(metric["n"] == 4096 and set(metric["per_role"]) == set(ROLES), "core metric scope differs")
        for role in ROLES:
            score(metric["per_role"][role]["accuracy"], f"core {role} accuracy")
        values = {"accuracy": score(metric["accuracy"], "core accuracy"),
                  "binding": score(metric["macro_agent_patient_accuracy"], "core binding")}
        name = ident.replace(f"_i{seed}_", "_iPAIRED_")
        group = groups.setdefault(name, {"label": condition_label(config, case), "scores": {}})
        require(group["label"] == condition_label(config, case) and seed not in group["scores"],
                "core condition is not paired unambiguously")
        group["scores"][seed] = values
    require(len(run_ids) == 26 and len(groups) == 17
            and all(set(group["scores"]) == set(SEEDS) for group in groups.values()),
            "core needs exactly 17 complete paired conditions")
    return list(groups.values()), list(groups)


def growth_order():
    return [(bank, router) for bank in BANKS
            for router in (("uniform",) if bank == "single_final" else ROUTERS)]


def validate_growth(report):
    require(report.get("h1_eligible") is False, "growth cannot reopen H1")
    require(report["verified_counts"] == {"trajectories": 6, "training_stages": 18,
            "gate_fits": 8, "routing_evaluations": 34, "preallocation_checks": 2},
            "growth completed inventory differs")
    order = growth_order()
    require(len(order) == 17 and len(report["rows"]) == 136, "growth row inventory differs")
    expected = {(seed, bank, router, topic) for seed in SEEDS for bank, router in order for topic in TOPICS}
    records = {}
    for row in report["rows"]:
        key = (row["seed"], row["kind"], row["router"], row["topic"])
        require(key in expected and key not in records, "missing/duplicate/unexpected growth row")
        require(row["bank_id"] == f"{row['kind']}_i{row['seed']}"
                and row["n"] == (3072 if row["topic"] == "all" else 1024),
                "growth row identity or sample count differs")
        score(row["accuracy"], "growth accuracy")
        score(row["macro_all_roles"], "growth macro accuracy")
        records[key] = score(row["binding"], "growth binding")
    require(set(records) == expected, "growth requires both seeds and every topic/router")
    matrices = {seed: [[records[(seed, bank, router, topic)] for topic in TOPICS]
                       for bank, router in order] for seed in SEEDS}
    labels = [f"{BANK_LABELS[bank]} / {router}" for bank, router in order]
    return matrices, labels, order


def attempt_ids(value):
    normalized = str(value).replace("\\", "/")
    return set(re.findall(r"(?:^|/)(?:runs/item6|06_cloud_runs|06_root_review)/(\d+-\d+)(?:/|$)", normalized))


def provenance(core, growth, inputs):
    require(core["source"] == growth["source"]
            and core["controller_source"] == growth["controller_source"], "panel source/freeze differs")
    head = core["source"]["git_commit"]
    require(isinstance(head, str) and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", head), "invalid execution HEAD")
    require(growth["execution_head"] == head, "growth execution HEAD differs")
    core_ids = attempt_ids(inputs["core"]["path"])
    for row in core["evaluations"]:
        core_ids.update(attempt_ids(row["predictions"]["path"]))
    growth_ids = attempt_ids(inputs["growth"]["path"])
    require(len(core_ids) <= 1 and len(growth_ids) <= 1, "ambiguous run attempt identity")
    require(not core_ids or not growth_ids or core_ids == growth_ids, "panel run attempt paths differ")
    core_attempt = next(iter(core_ids), None)
    growth_attempt = next(iter(growth_ids), None)
    return {"execution_head": head, "core_attempt": core_attempt, "growth_attempt": growth_attempt,
            "run_identity_scope": "Core may identify the attempt in its recorded prediction paths. The growth report has no run-ID field; its attempt is identified from the supplied run/archive/root-review path, not inferred from scores.",
            "reports": inputs}


def footer(figure, identity):
    core_run = identity["core_attempt"] or "not encoded in report/path"
    growth_run = identity["growth_attempt"] or "not encoded in report/path"
    lines = [f"Frozen execution source: {identity['execution_head']}",
             f"Run attempt: core {core_run}; growth {growth_run}",
             "Core report SHA-256: " + identity["reports"]["core"]["sha256"],
             "Growth report SHA-256: " + identity["reports"]["growth"]["sha256"],
             "n=2 initializations, one split; exploratory. No CI drawn: see verified score/contrast CSVs for full, untruncated intervals.",
             "Post-freeze presentation only. Prespecified order; no ranking, outcome selection or new inference."]
    figure.text(.02, .017, "\n".join(lines), ha="left", va="bottom", fontsize=7.2,
                linespacing=1.4, color="#343434")


def core_figure(plt, groups, identity):
    figure, axes = plt.subplots(1, 2, figsize=(17, 13.5), sharey=True)
    figure.subplots_adjust(left=.35, right=.975, bottom=.17, top=.88, wspace=.12)
    styles = {20: ("o", "#1465a8", -.10), 21: ("s", "#b34b12", .10)}
    for axis, metric, title in zip(axes, ("accuracy", "binding"),
                                   ("Global accuracy", "Binding: mean agent/patient accuracy")):
        for index, group in enumerate(groups):
            if index % 2 == 0:
                axis.axhspan(index - .45, index + .45, color="#f3f5f7", zorder=0)
            axis.plot([group["scores"][seed][metric] for seed in SEEDS],
                      [index + styles[seed][2] for seed in SEEDS], color="#a9b2bb", linewidth=1, zorder=2)
        for seed, (marker, color, offset) in styles.items():
            axis.scatter([group["scores"][seed][metric] for group in groups],
                         [i + offset for i in range(len(groups))], s=39, marker=marker,
                         facecolor=color, edgecolor="white", linewidth=.6,
                         label=f"Initialization {seed}", zorder=3)
        axis.set_xlim(0, 1)
        axis.set_xticks([0, .25, .5, .75, 1])
        axis.set_ylim(len(groups) - .5, -.5)
        axis.set_yticks(range(len(groups)))
        axis.set_title(title, fontsize=12, pad=13)
        axis.set_xlabel("Accuracy (0–1)", fontsize=10)
        axis.grid(axis="x", color="#dce1e5", linewidth=.7)
        axis.tick_params(axis="y", length=0, pad=10)
        for side in ("top", "right", "left"):
            axis.spines[side].set_visible(False)
    axes[0].set_yticklabels([group["label"] for group in groups], fontsize=8.3, linespacing=1.3)
    axes[1].tick_params(labelleft=False)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(.66, .946), ncol=2,
                  frameon=False, fontsize=10)
    figure.suptitle("Item 6 · Core: all 17 paired conditions", fontsize=18, y=.982, fontweight="bold")
    figure.text(.5, .955, "tie = dictionary tying; reinj = input reinjection; school = auxiliary-loss weight",
                ha="center", fontsize=9, color="#454545")
    footer(figure, identity)
    return figure


def growth_figure(plt, matrices, labels, identity):
    figure = plt.figure(figsize=(14.5, 11.5))
    grid = figure.add_gridspec(1, 3, left=.28, right=.94, bottom=.19, top=.875,
                              width_ratios=(1, 1, .055), wspace=.12)
    axes = [figure.add_subplot(grid[0, i]) for i in range(2)]
    cmap = plt.get_cmap("viridis")
    for axis, seed in zip(axes, SEEDS):
        matrix = matrices[seed]
        image = axis.imshow(matrix, vmin=0, vmax=1, cmap=cmap, aspect="auto", interpolation="nearest")
        axis.set_title(f"Initialization {seed}", fontsize=13, pad=15)
        axis.set_xticks(range(4), ("A", "B", "C", "All"))
        axis.set_xlabel("Topic", fontsize=11)
        axis.set_yticks(range(len(labels)), labels if seed == 20 else [""] * len(labels))
        axis.tick_params(axis="y", length=0, labelsize=9, pad=8)
        axis.set_xticks([i - .5 for i in range(5)], minor=True)
        axis.set_yticks([i - .5 for i in range(len(labels) + 1)], minor=True)
        axis.grid(which="minor", color="white", linewidth=.5)
        axis.tick_params(which="minor", bottom=False, left=False)
        for boundary in (3.5, 7.5, 8.5, 12.5):
            axis.axhline(boundary, color="#172331", linewidth=1.1)
        for row, values in enumerate(matrix):
            for column, value in enumerate(values):
                red, green, blue, _ = cmap(value)
                luminance = .2126 * red + .7152 * green + .0722 * blue
                axis.text(column, row, f"{value:.3f}", ha="center", va="center", fontsize=9,
                          color="black" if luminance > .48 else "white")
    colorbar = figure.colorbar(image, cax=figure.add_subplot(grid[0, 2]))
    colorbar.set_ticks([0, .25, .5, .75, 1])
    colorbar.set_label("Binding accuracy (0–1)", fontsize=10)
    figure.suptitle("Item 6 · Growth: all 17 bank/router conditions", fontsize=17, y=.98, fontweight="bold")
    figure.text(.5, .942, "Mean agent/patient accuracy · identical color scale · annotated values rounded to three decimals",
                ha="center", fontsize=9, color="#454545")
    footer(figure, identity)
    return figure


def save_new_figure(figure, path, identity):
    extension = path.suffix.removeprefix(".")
    metadata = {"Title": path.stem.replace("_", " "),
                "Creator": "NeuroPixel item-6 post-freeze editorial plot helper"}
    if extension == "pdf":
        metadata["Subject"] = json.dumps(identity, sort_keys=True)
    else:
        metadata["Description"] = json.dumps(identity, sort_keys=True)
    buffer = io.BytesIO()
    figure.savefig(buffer, format=extension, dpi=180, facecolor="white", metadata=metadata)
    payload = buffer.getvalue()
    with path.open("xb") as stream:
        stream.write(payload)
    return {"file": path.name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--growth", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = [args.output_dir / name for name in FILES]
    require(not any(path.exists() for path in paths), "refusing to overwrite an existing plot or manifest")
    core, core_input = read_report(args.core, "core")
    growth, growth_input = read_report(args.growth, "growth")
    groups, core_order = validate_core(core)
    matrices, labels, bank_order = validate_growth(growth)
    identity = provenance(core, growth, {"core": core_input, "growth": growth_input})
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for figure, destinations in ((core_figure(plt, groups, identity), paths[:2]),
                                 (growth_figure(plt, matrices, labels, identity), paths[2:4])):
        try:
            for path in destinations:
                outputs.append(save_new_figure(figure, path, identity))
        finally:
            plt.close(figure)
    manifest = {"schema_version": 1, "item": 6, "kind": "post_freeze_editorial_figures",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(), **identity,
                "plot_helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "presentation_environment": {"python": platform.python_version(),
                                             "matplotlib": matplotlib.__version__},
                "outputs": outputs, "core_condition_order": core_order,
                "growth_condition_order": [{"bank": bank, "router": router} for bank, router in bank_order],
                "growth_topic_order": list(TOPICS), "seeds": list(SEEDS), "score_limits": [0, 1],
                "confidence_intervals_drawn": False, "ranking_or_selection": False,
                "scope": "Both JSON reports required verified/zero issues. Structural checks only; no source, prediction array, model, or scientific analyzer was executed."}
    with paths[4].open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "plotted", "core_conditions": 17, "growth_conditions_per_seed": 17,
                      "output_dir": str(args.output_dir), "files": [path.name for path in paths]}))


if __name__ == "__main__":
    main()
