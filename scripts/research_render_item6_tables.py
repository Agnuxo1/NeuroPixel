"""Render an editorial appendix from the two completed, verified item-6 reports.

This post-freeze helper uses only the standard library. It does not import the
analyzers, read prediction arrays, recount examples, or execute scientific code.
Run only after both independent reports have completed successfully.
"""
from __future__ import annotations

import argparse
import hashlib
from itertools import combinations, product
import json
import math
import os
from pathlib import Path
import statistics
from urllib.parse import quote


SEEDS = (20, 21)
ROUTERS = ("uniform", "random", "scanner", "learned")
BANKS = ("single_final", "snapshots", "duplicate_slots", "adaptive_novelty", "adaptive_resonance")
TOPICS = ("all", "0", "1", "2")
MEASURES = ("accuracy", "macro_all_roles", "macro_agent_patient_accuracy", "cross_entropy")
T95_DF1 = 1 / math.tan(math.pi * 0.025)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    require(type(value) in (int, float) and math.isfinite(value), "expected a finite numerical value")
    return value


def number(value):
    return format(finite(value), ".12g")


def pair(values):
    require(len(values) == 2, "a displayed pair requires two values")
    return " / ".join(number(value) for value in values)


def interval(values):
    require(len(values) == 2 and finite(values[0]) <= finite(values[1]), "invalid interval or range")
    return "[" + ", ".join(number(value) for value in values) + "]"


def derived_summary(values):
    """Presentation-only summary of two stored seed scores, never new examples."""
    require(len(values) == 2, "exactly two initializations are required")
    values = [finite(value) for value in values]
    mean, sd = statistics.mean(values), statistics.stdev(values)
    half = T95_DF1 * sd / math.sqrt(2)
    return [number(mean), number(sd), interval([min(values), max(values)]),
            interval([mean - half, mean + half])]


def saved_core_summary(row):
    require(row["n"] == 2 and row["df"] == 1 and row["seeds"] == list(SEEDS), "core summary seed scope differs")
    pair(row["values"])
    return [number(row["mean"]), number(row["sample_sd"]), interval(row["range"]), interval(row["t_interval_95"])]


def saved_growth_summary(row):
    require(row["n_initializations"] == 2 and set(row["raw_seed_effects"]) == {"20", "21"},
            "growth summary seed scope differs")
    values = [finite(row["raw_seed_effects"][str(seed)]) for seed in SEEDS]
    # The frozen growth analyzer exports both values but no explicit range.
    return [number(row["mean"]), number(row["sample_standard_deviation"]),
            interval([min(values), max(values)]), interval(row["student_t_95_df1_unclipped"])]


def table(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    result = ["| " + " | ".join(map(cell, headers)) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        require(len(row) == len(headers), "table width differs")
        result.append("| " + " | ".join(map(cell, row)) + " |")
    return "\n".join(result) + "\n"


def read_report(path, panel):
    def reject(value):
        raise ValueError(f"nonfinite JSON constant: {value}")
    def unique_object(items):
        result = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result
    data = path.read_bytes()
    report = json.loads(data, parse_constant=reject, object_pairs_hook=unique_object)
    require(report.get("item") == 6 and report.get("panel") == panel
            and report.get("status") == "verified" and report.get("issues") == [],
            f"{panel}: only a verified, issue-free item-6 report may be rendered")
    return report, hashlib.sha256(data).hexdigest()


def grouped(rows, key, seed):
    groups = {}
    for row in rows:
        ident, initialization = key(row), seed(row)
        require(type(initialization) is int and initialization in SEEDS, "unexpected initialization")
        group = groups.setdefault(ident, {})
        require(initialization not in group, f"duplicate seed arm: {ident}")
        group[initialization] = row
    require(all(set(group) == set(SEEDS) for group in groups.values()), "a paired arm is missing")
    return groups


def condition(config):
    require(config["split_seed"] == 0, "unexpected split")
    return (config["panel"], int(config["variant"]["tied"]), config["school_weight"],
            int(config["variant"]["reinject"]), config["steps"], config["updates"])


def condition_label(key):
    panel, tied, school, reinject, steps, updates = key
    return f"{panel}: A{tied} B{int(school > 0)} C{reinject}; train T{steps}; {updates} updates"


def core_inventory():
    training = [("factorial", a, b, c, 16, 1024) for a, b, c in product((0, 1), (0.0, 0.3), (0, 1))]
    training += [("recurrence", 1, 0.0, 1, steps, 1024) for steps in (1, 4)]
    training += [("damage", 1, 0.0, 1, 16, 1024)]
    training += [("optimization", 1, b, 1, 16, 8192) for b in (0.0, 0.3)]
    evaluation = [(key, "clean", None) for key in training]
    baseline = ("factorial", 1, 0.0, 1, 16, 1024)
    evaluation += [(baseline, "deployment_truncation", steps) for steps in (1, 4)]
    evaluation += [(key, "fixed_lesion", None) for key in (baseline, ("damage", 1, 0.0, 1, 16, 1024))]
    return training, evaluation


def core_contrast_names():
    names, result = ("tying", "school", "reinjection"), set()
    for size in (1, 2, 3):
        for axes in combinations(range(3), size):
            label = "_by_".join(names[index] for index in axes)
            result.add(label)
            remaining = [index for index in range(3) if index not in axes]
            if remaining:
                for bits in product((0, 1), repeat=len(remaining)):
                    result.add(label + "__" + "_".join(f"{names[index]}{bit}" for index, bit in zip(remaining, bits)))
    for depth in (1, 4):
        result.add(f"trained_T{depth}_minus_trained_T16")
        result.add(f"same_T16_weights_eval_T{depth}_minus_eval_T16")
    for value in (0, 1):
        result.add(f"damage_training_minus_clean_training__eval_lesion{value}")
        result.add(f"eval_lesion_minus_clean__training_damage{value}")
        result.add(f"updates8192_minus1024__school{value}")
    result.update(("training_damage_by_eval_lesion", "budget_by_school", "school03_minus0__updates8192"))
    return result


def growth_comparisons():
    names = [f"{kind}:scanner_minus_{router}" for kind in BANKS[1:] for router in ("uniform", "random", "learned")]
    for router in ROUTERS:
        names += [f"snapshots_{router}_minus_single_final_uniform", f"snapshots_minus_duplicate_slots:{router}"]
        names += [f"{policy}_minus_fixed_snapshots:{router}" for policy in BANKS[3:]]
    return names


def core_sections(report):
    training_order, evaluation_order = core_inventory()
    training = grouped(report["training_observables"], lambda row: condition(row["config"]), lambda row: row["config"]["init_seed"])
    evaluations = grouped(report["evaluations"],
                          lambda row: (condition(row["config"]), row["evaluation_case"]["kind"], row["evaluation_case"]["steps_override"]),
                          lambda row: row["config"]["init_seed"])
    require(set(training) == set(training_order) and set(evaluations) == set(evaluation_order), "core arms differ from 13 training/17 evaluation pairs")
    summaries = {}
    for row in report["paired_score_summaries"]:
        key = row["evaluation"], row["metric"]
        require(key not in summaries, "duplicate core score summary")
        summaries[key] = row
    score_rows, expected_summaries = [], set()
    for key in evaluation_order:
        rows = [evaluations[key][seed] for seed in SEEDS]
        names = [row["evaluation_case"]["evaluation_id"].replace(f"_i{seed}_", "_iPAIRED_") for row, seed in zip(rows, SEEDS)]
        require(names[0] == names[1], "core evaluation identifiers do not pair")
        expected_summaries.update((names[0], metric) for metric in ("accuracy", "macro_agent_patient_accuracy", "cross_entropy"))
        summary = summaries[names[0], "macro_agent_patient_accuracy"]
        values = [row["recounted_metrics"]["macro_agent_patient_accuracy"] for row in rows]
        require(summary["values"] == values, "core saved summary values differ from its score rows")
        label = condition_label(key[0]) + "; " + key[1] + (f" T{key[2]}" if key[2] is not None else "")
        score_rows.append([label, pair([row["recounted_metrics"]["accuracy"] for row in rows]), pair(values),
                           *saved_core_summary(summary), pair([row["recounted_metrics"]["cross_entropy"] for row in rows])])
    require(set(summaries) == expected_summaries, "core score-summary inventory differs")
    contrasts = report["contrasts"]
    require(len(contrasts) == 38 and {row["contrast"] for row in contrasts} == core_contrast_names(), "core contrast inventory differs")
    contrast_rows = [[row["panel"], row["contrast"], pair(row["values"]), *saved_core_summary(row)] for row in contrasts]
    diagnostic_rows = []
    for key in training_order:
        rows = [training[key][seed] for seed in SEEDS]
        require(all(type(row["optimization_budget_limited"]) is bool for row in rows), "invalid probe flag")
        require(all(row["curve"] and row["curve"][-1]["update"] == key[-1] for row in rows), "last training diagnostic is missing")
        diagnostic_rows.append([
            condition_label(key), pair([row["parameter_count"] for row in rows]),
            pair([row["train_probe_reported"]["macro_agent_patient_accuracy"] for row in rows]),
            pair([row["validation_recounted"]["macro_agent_patient_accuracy"] for row in rows]),
            " / ".join(str(row["optimization_budget_limited"]).lower() for row in rows),
            pair([row["curve"][-1]["mean_training_loss_since_last_log"] for row in rows]),
            pair([row["curve"][-1]["last_answer_loss"] for row in rows]),
            pair([row["training_seconds"] for row in rows]),
        ])
    return [
        "## Core: all 17 paired evaluation conditions\n\nA=tying, B=school (B0=0, B1=0.3), C=reinjection. Binding summaries below are copied from the verified analyzer.\n\n"
        + table(["Condition", "Global 20 / 21", "Binding 20 / 21", "Binding mean", "SD", "Range", "t95, df1", "CE 20 / 21"], score_rows),
        "## Core: all 38 binding contrasts\n\nSigns, contrast scale, SD, range and intervals are those of the verified analyzer.\n\n"
        + table(["Panel", "Contrast", "Delta 20 / 21", "Mean", "SD", "Range", "t95, df1"], contrast_rows),
        "## Core: training diagnostics for all 13 paired conditions\n\nProbe counts are reported by the trainer and checked for consistency; probe prediction arrays were not saved. The budget-limited flag denotes probe binding below 0.95. The last objective is a 128-update-window mean; answer loss is the last update only. Their difference is not an estimate of school loss. Times cover recorded expert training, not isolated gate fitting.\n\n"
        + table(["Condition", "Parameters 20 / 21", "Probe binding 20 / 21", "Validation binding 20 / 21", "Budget-limited 20 / 21", "Last window objective 20 / 21", "Last answer CE 20 / 21", "Training seconds 20 / 21"], diagnostic_rows),
    ]


def growth_sections(report):
    require(report["verified_counts"] == {"trajectories": 6, "training_stages": 18, "gate_fits": 8,
                                          "routing_evaluations": 34, "preallocation_checks": 2}, "growth completion counts differ")
    combinations_order = [(kind, router, topic) for kind in BANKS
                          for router in (("uniform",) if kind == "single_final" else ROUTERS) for topic in TOPICS]
    scores = grouped(report["rows"], lambda row: (row["kind"], row["router"], row["topic"]), lambda row: row["seed"])
    require(set(scores) == set(combinations_order), "growth bank/router/topic arms are missing or extra")
    score_rows = []
    for key in combinations_order:
        rows = [scores[key][seed] for seed in SEEDS]
        require(all(row["n"] == (3072 if key[2] == "all" else 1024) for row in rows), "growth example count differs")
        values = [row["binding"] for row in rows]
        score_rows.append([key[0], key[1], {"all": "pooled", "0": "A", "1": "B", "2": "C"}[key[2]],
                           pair([row["expert_slots"] for row in rows]), pair(values), *derived_summary(values)])
    decisions = report["decisions"]
    expected = {(policy, seed, stage) for policy in ("fixed_sequential", "adaptive_novelty", "adaptive_resonance")
                for seed in SEEDS for stage in range(3)}
    require(len(decisions) == 18 and {(row["policy"], row["seed"], row["stage"]) for row in decisions} == expected,
            "growth stage decisions are missing or duplicated")
    decision_rows, margin_rows = [], []
    for row in sorted(decisions, key=lambda row: (row["policy"], row["seed"], row["stage"])):
        decision, margin = row["decision"], row["threshold_margin"]
        require(type(row["experts_after"]) is int and 1 <= row["experts_after"] <= 3, "invalid retained expert count")
        decision_rows.append([row["policy"], row["seed"], "ABC"[row["stage"]], decision["action"],
                              "none" if decision["parent"] is None else decision["parent"], decision["target_index"],
                              row["experts_after"], "none" if row["tau"] is None else number(row["tau"])])
        if row["stage"] and row["policy"] != "fixed_sequential":
            require(margin is not None, "adaptive decision margin is missing")
            margin_rows.append([row["policy"], row["seed"], "ABC"[row["stage"]],
                                number(margin["selected_expert_raw_score"]), number(margin["selected_expert_rounded_score"]),
                                number(margin["threshold"]), number(margin["raw_creation_margin"]),
                                number(margin["rounded_creation_margin"]),
                                str(margin["rounding_changes_comparison_for_selected_expert"]).lower()])
        else:
            require(margin is None, "unexpected creation margin")
    contrasts = report["contrasts"]
    comparison_order = growth_comparisons()
    index = {(row["contrast"], row["topic"], row["measure"]): row for row in contrasts}
    require(len(contrasts) == len(index) == 448 and set(index) == set(product(comparison_order, TOPICS, MEASURES)),
            "the complete 448-row growth contrast inventory is required")
    contrast_rows = []
    for name in comparison_order:
        row = index[name, "all", "macro_agent_patient_accuracy"]
        values = [row["raw_seed_effects"][str(seed)] for seed in SEEDS]
        contrast_rows.append([name, pair(values), *saved_growth_summary(row)])
    return [
        "## Growth: all 17 bank/router pairs across four topic groups\n\nThere are 68 paired rows. Pooled means the fixed concatenation of A/B/C, 1,024 examples each. Binding summaries here are derived editorially from the two saved binding scores, with n=2, sample SD, range and the same unclipped t95 formula. Slots and topics do not enlarge the replication count.\n\n"
        + table(["Bank", "Router", "Topic", "Slots 20 / 21", "Binding 20 / 21", "Mean", "SD", "Range", "t95, df1"], score_rows),
        "## Growth: all 18 stage decisions and retained expert counts\n\nIndices are zero-based. K is the retained bank size after that stage; the fixed trajectory retains snapshots. These stage observations are not independent training replications. The two preallocation checks are construction identities, not extra seeds.\n\n"
        + table(["Policy", "Seed", "Stage", "Action", "Parent", "Target slot", "K after stage", "Resonance tau"], decision_rows)
        + "\n### Adaptive creation margins\n\nMargins are copied from the analyzer, not recomputed. A strictly positive rounded margin means creation; the sign convention already reverses for resonance. Numerical display rounding never drives decisions. Full score vectors and tie/lineage information remain in the source JSON and its artifact links.\n\n"
        + table(["Policy", "Seed", "Stage", "Selected raw", "Selected round4", "Threshold", "Raw margin", "Rounded margin", "Rounding changes comparison"], margin_rows),
        "## Growth: all 28 pooled binding contrasts\n\nDifferences use the analyzer's left-minus-right convention. Saved means, sample SDs and t95 intervals are retained; ranges are derived only as min/max of its two saved seed differences. The complete growth contrast CSV contains 448 rows: 28 comparisons x 4 topic groups x 4 metrics. Negative cross-entropy differences favor the left arm, unlike accuracy differences.\n\n"
        + table(["Contrast", "Delta 20 / 21", "Mean", "SD", "Range", "t95, df1"], contrast_rows),
    ]


def render(core_path, growth_path, output):
    require(not output.exists(), "refusing to overwrite the appendix")
    core, core_hash = read_report(core_path, "core")
    growth, growth_hash = read_report(growth_path, "growth")
    require(core["source"] == growth["source"] and core["controller_source"] == growth["controller_source"],
            "the two reports do not describe the same frozen study source")
    def link(path, label):
        relative = Path(os.path.relpath(path, output.parent)).as_posix()
        return f"[{label}]({quote(relative, safe='/')})"
    sources = [
        [link(core_path, "Core verified JSON"), core_hash],
        [link(growth_path, "Growth verified JSON"), growth_hash],
        ["Editorial renderer SHA-256", hashlib.sha256(Path(__file__).read_bytes()).hexdigest()],
    ]
    sections = [
        "# Item 6: complete presentation tables\n\n"
        "Post-freeze editorial rendering of two completed independent reports. This appendix performs no model evaluation, example recount, selection, or new scientific analysis. Status checks and inventory checks are presentation safeguards; they do not repeat either artifact audit.\n\n"
        "All accuracy/binding values are fractions, CE is mean negative log likelihood in nats, and durations are seconds. Paired columns always show seed20 / seed21. The inferential unit is the initialization: n=2 on split0, df=1. The t95 formula is mean +/- cot(pi*0.025)*sample_SD/sqrt(2), without clipping. Display uses up to 12 significant digits; exact stored precision remains in the linked JSON. No rounded display value drives a decision.\n\n"
        + table(["Evidence", "SHA-256"], sources)
        + "\nComplete machine-readable score tables: "
        + ", ".join(link(path, path.name) for path in
                     (core_path.parent / "core_scores.csv", core_path.parent / "core_contrasts.csv",
                      growth_path.parent / "06_growth_runs.csv", growth_path.parent / "06_growth_contrasts.csv"))
        + ". These CSV links identify the analyzers' companion outputs; the renderer reads only the two JSON files. Per-role scores, other metrics, diagnostic arrays and provenance references are retained there, not replaced by these binding-focused tables.\n",
        *core_sections(core), *growth_sections(growth),
        "## Scope and unavailable measurements\n\nAll comparisons remain exploratory; H1 is not reconsidered. Two seeds do not establish generalization across training randomness or datasets. Saved example-bootstrap intervals concern their individual checkpoints and are not substituted for seed-level intervals. Adaptive contrasts condition on realized K and lineage; ABC versus ABB matches nominal slots, not unique functional capacity. Router fitting adds supervised optimization. Neither exclusive gate-fit elapsed time nor peak process RSS was measured. Available-RAM minima are minima of recorded samples, not continuous minima. No FLOPs, energy or physical inference-latency claim is introduced.\n",
    ]
    text = "\n".join(sections)
    require(hashlib.sha256(core_path.read_bytes()).hexdigest() == core_hash
            and hashlib.sha256(growth_path.read_bytes()).hexdigest() == growth_hash, "an input report changed during rendering")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True, help="Verified core_analysis.json")
    parser.add_argument("--growth", type=Path, required=True, help="Verified 06_growth_analysis.json")
    parser.add_argument("--output", type=Path, required=True, help="New Markdown appendix; never overwritten")
    args = parser.parse_args()
    try:
        render(args.core.resolve(), args.growth.resolve(), args.output.resolve())
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        parser.exit(1, f"Rendering rejected: {type(error).__name__}: {error}\n")
    print(f"Wrote presentation appendix: {args.output}")


if __name__ == "__main__":
    main()
