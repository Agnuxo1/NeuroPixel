"""G18: programme-level statistics registry.

Writes docs/research/program_registry.json (ordered confirmatory hypotheses with their alpha and
status, the online error-control rule fixed in advance, equivalence-margin policy) and
results/research/COMPLETION20261012/G18_sensitivity.json (hierarchical bootstraps for nested
designs and specification curves for the central results). Sensitivities never replace the
registered primary analyses. Inputs are retained receipts only.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from completion_kit.inference_plus import hierarchical_bootstrap, online_bonferroni, specification_curve  # noqa: E402
from completion_kit.statistics import paired_interval  # noqa: E402

RR = ROOT / "results/research"
SEED = 20261013


def vis07_counts() -> dict:
    with zipfile.ZipFile(RR / "VIS07MAIN_cloud/37908785698/raw.zip") as bundle:
        scores = json.loads(bundle.read("result.json"))["scores"]
    out = {}
    for condition in ("base", "half_intensity", "shift_right_crop"):
        for family in ("nca", "cnn"):
            rows = sorted((r for r in scores if r["family"] == family and r["condition"] == condition), key=lambda r: r["seed"])
            out[(condition, family)] = [(r["correct"], r["n"]) for r in rows]
    return out


def t_interval(diff: list[float]) -> tuple[float, float, float]:
    n, mean = len(diff), statistics.mean(diff)
    half = stats.t.ppf(0.975, n - 1) * statistics.stdev(diff) / math.sqrt(n)
    return mean, mean - half, mean + half


def seed_bootstrap(diff: list[float], replicates: int = 20000) -> tuple[float, float, float]:
    rng = np.random.default_rng(SEED)
    values = np.asarray(diff)
    means = values[rng.integers(0, len(values), size=(replicates, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def pooled_binomial(nca: list[tuple[int, int]], cnn: list[tuple[int, int]]) -> tuple[float, float, float]:
    a = sum(c for c, _ in nca) / sum(n for _, n in nca)
    b = sum(c for c, _ in cnn) / sum(n for _, n in cnn)
    se = math.sqrt(a * (1 - a) / sum(n for _, n in nca) + b * (1 - b) / sum(n for _, n in cnn))
    return 100 * (a - b), 100 * (a - b - 1.96 * se), 100 * (a - b + 1.96 * se)


def vis07_curves(counts: dict) -> dict:
    curves = {}
    for condition in ("base", "half_intensity", "shift_right_crop"):
        nca, cnn = counts[(condition, "nca")], counts[(condition, "cnn")]
        diff = [100 * (a / n - b / m) for (a, n), (b, m) in zip(nca, cnn)]
        rows = []
        for name, (effect, low, high) in (
                ("paired t95 over five seeds (registered primary for base)", t_interval(diff)),
                ("seed bootstrap percentile, 20000 resamples", seed_bootstrap(diff)),
                ("pooled binomial over examples (ignores training variance; anti-conservative)", pooled_binomial(nca, cnn)),
                ("median paired difference with observed range", (float(np.median(diff)), min(diff), max(diff)))):
            rows.append(dict(specification=name, effect=effect, low=low, high=high))
        sign = stats.binomtest(sum(d > 0 for d in diff), len(diff), 0.5)
        curve = specification_curve(rows)
        curve["paired_differences_pp"] = diff
        curve["exact_sign_test_two_sided_p"] = float(sign.pvalue)
        curves[condition] = curve
    return curves


def nested_development() -> dict:
    dev04 = json.loads((RR / "DEV04_review/37845945011/analysis/summary.json").read_text(encoding="utf-8"))
    groups = {}
    for pair in dev04["pairs"]:
        groups.setdefault(str(pair["partition_seed"]), []).append(pair["difference_pp"])
    glob = {}
    with (RR / "GLOB05_review/37887762153/analysis/cases.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            glob.setdefault(row["partition_seed"], []).append(100 * (float(row["attention_joint"]) - float(row["active_global_joint"])))
    out = {}
    for name, grouped, registered in (("DEV04 attention minus local", groups, dev04["primary"]),
                                      ("GLOB05 attention minus active global", glob,
                                       json.loads((RR / "GLOB05_review/37887762153/analysis/summary.json").read_text(encoding="utf-8"))
                                       ["effects"]["attention_minus_active_global"])):
        boot = hierarchical_bootstrap(grouped, replicates=20000, seed=SEED)
        out[name] = dict(design="initializations nested in three partition policies", hierarchical_bootstrap=boot,
                         registered_primary=dict(mean_pp=registered["mean_pp"], t95_pp=registered["t95_pp"], df=2),
                         note="Descriptive sensitivity; the registered df=2 analysis of policy means stays primary")
    return out


def main() -> None:
    counts = vis07_counts()
    stream = online_bonferroni([1.0], alpha=0.05)  # thresholds only; p=1.0 placeholder shows the first threshold
    first = [6 / (math.pi**2 * j**2) * 0.05 for j in range(1, 6)]
    registry = dict(
        schema=1, created_utc=datetime.now(timezone.utc).isoformat(),
        error_control=dict(rule="online_bonferroni (FWER, valid under arbitrary dependence)", total_alpha=0.05,
                           gamma="6/(pi^2 j^2), j = position of the confirmatory hypothesis in this registry",
                           first_thresholds=first, kit_check=stream["thresholds"][0],
                           scope="Applies to every confirmatory hypothesis registered after 2026-10-13; order fixed by registration time",
                           references=["Ioannidis 2005", "Javanmard and Montanari 2018"]),
        equivalence_policy=dict(rule=("Any claim of equivalence or non-inferiority requires a margin in percentage points registered "
                                      "before the data and a TOST or non-inferiority test (tost_paired / noninferiority_paired); a "
                                      "non-significant difference never establishes equivalence (Schuirmann 1987; Lakens 2017)"),
                                current_equality_claims=0,
                                check="No statement in docs/research/claims_ledger.json asserts equivalence; limits state the opposite where relevant"),
        hypotheses=[
            dict(position=0, id="H1", protocol="NP-SCI-20261006-v1", registered="2026-10-06 (before this registry)",
                 decision_rule="five primary pairs; mean NeuroPixel binding >= 0.90, mean paired advantage >= 0.05 and two-sided t95 lower bound > 0",
                 alpha=0.05, alpha_source="own protocol; outside the online stream", status="not_supported",
                 evidence_level="L0 blocked_external (raw archive unavailable; see A18.2.H.ext)"),
        ],
        exploratory_studies=["A06.1", "OPT03 (D, MODE, ROLE, QTRAIN, QTRAIN-U16)", "READ03", "DEV04", "GLOB05", "SCN06", "VIS07",
                             "RENDER08-11"],
        exploratory_note="These studies used exploratory, unadjusted intervals; they spend no alpha and support no confirmatory claim",
        next_position=1,
        sensitivity="results/research/COMPLETION20261012/G18_sensitivity.json",
    )
    sensitivity = dict(schema=1, seed=SEED, central_results=dict(VIS07=vis07_curves(counts)),
                       nested_designs=nested_development(),
                       budget_note=("Search budgets: VIS07 used no hyperparameter search (fixed recipe, five seeds); DEV04/GLOB05/READ03 "
                                    "selected one recipe after earlier development phases on exposed pools (Dodge et al. 2019)"))
    (ROOT / "docs/research/program_registry.json").write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8", newline="\n")
    (RR / "COMPLETION20261012/G18_sensitivity.json").write_text(json.dumps(sensitivity, indent=2) + "\n", encoding="utf-8", newline="\n")
    summary = {c: dict(median=round(v["median_effect"], 3), sign_p=round(v["exact_sign_test_two_sided_p"], 4),
                       range=[round(min(r["low"] for r in v["curve"]), 3), round(max(r["high"] for r in v["curve"]), 3)])
               for c, v in sensitivity["central_results"]["VIS07"].items()}
    nested = {k: dict(estimate=round(v["hierarchical_bootstrap"]["estimate"], 3),
                      interval=[round(x, 3) for x in v["hierarchical_bootstrap"]["interval"]]) for k, v in sensitivity["nested_designs"].items()}
    print(json.dumps(dict(vis07=summary, nested=nested), indent=1))


if __name__ == "__main__":
    main()
