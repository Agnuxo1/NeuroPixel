"""Analytically recount two saved PAD fixtures using only the standard library.

No project imports, Torch, model execution, sampling, test run or optimizer is
used. The formulae apply only to the two hash-pinned, explicitly weighted cases.
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
import subprocess
import sys


SOURCE_COMMIT = "d01e20625068a2ec19bf25554e106c1fc2cc4420"
SOURCE_SHA256 = {
    "tests/test_padding_invariant.py": "3c35b37c807b69492c3ac6e76a89c2d859fe08cbd6687250445dadb3388db830",
    "results/research/08_validation/padding_model_before.py": "564045fff799185e15b921873225d344014a4d4bcae6df6deacaf42f6e4bd701",
    "docs/research/08_cloud_validation_plan.json": "46b7f19063325b78e593ea53fbbc9bf4372a3f81e7d503a8ac58ee301c15455b",
}
INPUT_SHA256 = "3f542c4c96898475510e5f67520645a316ab729e9e7c569acd5d7042f40fea62"
FLOAT32_EPSILON = 2.0 ** -23
RELATIVE_TOLERANCE = 16 * FLOAT32_EPSILON


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def ram_sample():
    line = next(line for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:"))
    return {"at_utc": now(), "available_ram_gib": int(line.split()[1]) * 1024 / 2**30,
            "provider": "/proc/meminfo:MemAvailable"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def probabilities_and_loss(logits, target):
    offset = max(logits)
    exponentials = [math.exp(value - offset) for value in logits]
    total = math.fsum(exponentials)
    return [value / total for value in exponentials], offset + math.log(total) - logits[target]


def derive():
    """Differentiate the manually simplified two-dimensional fixture by hand."""
    answer_p, answer_loss = probabilities_and_loss([-10000.0, 2.0, -2.0, 1.0], 2)
    answer_g = 2 * math.fsum([1.0, answer_p[1], -answer_p[2], answer_p[3] / 2])
    lens_p, lens_loss = probabilities_and_loss([0.0, 1.0, -1.0, .5], 1)
    result = {}
    for route, probabilities, loss, gradient in (
            ("answer_reinjection", answer_p, answer_loss, answer_g),
            ("lens_readout", lens_p, lens_loss, lens_p[0])):
        result[route] = {"probabilities": probabilities, "loss": loss,
                         "pad_before": [0.0] * 4,
                         "pad_gradient": [gradient, 0.0, 0.0, 0.0],
                         "pad_after_sgd": [-.1 * gradient, 0.0, 0.0, 0.0]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve the existing analytical receipt")
    record = {"schema_version": 1, "item": 8, "started_at_utc": now(),
              "auditor_sha256": sha(Path(__file__)), "issues": [], "comparisons": [],
              "resource_samples": [], "source_sha256": {}, "expected_source_commit": SOURCE_COMMIT,
              "scope": "Post-run analytical audit of two saved fixed fixtures; not another experiment or an external independent replication.",
              "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                          "arithmetic": "Python binary64 floats and math; no project imports",
                          "active_numerical_threads": 1},
              "tolerance": {"float32_epsilon": FLOAT32_EPSILON, "rtol": RELATIVE_TOLERANCE,
                            "atol": 0.0, "zero_expected_values_require_exact_zero": True,
                            "rationale": "Six nonzero outputs follow short well-conditioned chains: three/four-term softmax and log, two positive ReLU updates, a weighted derivative sum and one scalar SGD multiplication. Sixteen float32 epsilons allow accumulated rounding and float32 exp/log/backward implementation differences relative to the binary64 algebra. This is an explicit comparison envelope, not a universal kernel error bound; no tolerance is allowed on the twenty structurally zero entries."},
              "formulas": {
                  "common": "E rows are (x,0,0,0),(1,0,0,0),(-1,0,0,0),(.5,0,0,0); initially x=0. All unlisted parameters are zero; SGD learning rate is .1.",
                  "answer_reinjection": "At the PAD output cell: seed=0, delta_s0=ReLU(1+x), two steps give u=2(1+x). Effective logits=(-10000,u,-u,u/2), target index2. At x=0, L=4+log(1+exp(-4)+exp(-1)); dL/dx=2*(1+p1-p2+p3/2). The masked PAD output logit has zero derivative, so this PAD gradient comes through identity reinjection.",
                  "lens_readout": "Four identical positions have read-vector (1,0,0,0), logits=(x,1,-1,.5), target index1. Mean CE is L=log(1+exp(1)+exp(-1)+exp(.5))-1; dL/dx=p0=1/(1+exp(1)+exp(-1)+exp(.5)). Averaging cancels the four identical contributions. This route reaches PAD through dictionary readout.",
                  "update": "For both routes, the PAD-row gradient is (g,0,0,0) and one SGD step gives (-.1*g,0,0,0). Grounding masks are false and the preserved dictionary does not project PAD back to zero.",
                  "masked_class_precision": "The answer formula neglects only the exp(-10002) term, below 10^-4000. It underflows to zero in both binary64 and float32, vastly below the stated envelope."}}
    try:
        record["resource_samples"].append(ram_sample())
        require(record["resource_samples"][-1]["available_ram_gib"] >= 8, "RAM admission below 8 GiB")
        commit = subprocess.check_output(["git", "--no-optional-locks", "rev-parse", "HEAD"],
                                         cwd=args.source_root, text=True, timeout=15).strip()
        record["observed_source_commit"] = commit
        require(commit == SOURCE_COMMIT, "source commit differs from the audited fixture")
        for name, expected in SOURCE_SHA256.items():
            actual = sha(args.source_root / name)
            record["source_sha256"][name] = actual
            require(actual == expected, "source hash differs: " + name)
        record["input"] = {"sha256": sha(args.input), "bytes": args.input.stat().st_size,
                           "name": args.input.name, "archive_run": "37584119610-1"}
        require(record["input"]["sha256"] == INPUT_SHA256, "saved fixture hash differs")
        observed = json.loads(args.input.read_text())
        require(observed["status"] == "legacy_defect_reproduced" and observed["fixture_only"] is True
                and observed["device"] == "cpu" and observed["torch"] == "2.6.0+cpu"
                and observed["optimizer_steps_per_route"] == 1
                and observed["source_sha256"] == SOURCE_SHA256["results/research/08_validation/padding_model_before.py"],
                "saved fixture metadata differs")
        require([row["route"] for row in observed["routes"]] == ["answer_reinjection", "lens_readout"],
                "route inventory differs")
        expected = derive()
        record["analytical_values"] = expected
        for row in observed["routes"]:
            route = row["route"]
            require(row["zero_pad_contract_violated"] is True, "saved defect flag differs")
            for field in ("loss", "pad_before", "pad_gradient", "pad_after_sgd"):
                values = [row[field]] if field == "loss" else row[field]
                reference = [expected[route][field]] if field == "loss" else expected[route][field]
                require(isinstance(values, list) and len(values) == len(reference), "value shape differs")
                for index, (actual, target) in enumerate(zip(values, reference)):
                    require(type(actual) in (int, float) and math.isfinite(actual), "nonfinite or invalid saved value")
                    error = abs(actual - target)
                    limit = RELATIVE_TOLERANCE * abs(target)
                    ulp = 2.0 ** (math.floor(math.log2(abs(target))) - 23) if target else None
                    comparison = {"route": route, "field": field, "component": index,
                                  "saved_float32_value": actual, "analytical_binary64_value": target,
                                  "absolute_error": error, "allowed_absolute_error": limit,
                                  "relative_error": error / abs(target) if target else None,
                                  "error_in_float32_ulps_at_expected_value": error / ulp if ulp else None,
                                  "passed": error <= limit}
                    record["comparisons"].append(comparison)
                    if not comparison["passed"]:
                        record["issues"].append(f"numerical mismatch: {route}/{field}/{index}")
        require(len(record["comparisons"]) == 26, "comparison inventory differs")
        record["maximum_absolute_error"] = max(row["absolute_error"] for row in record["comparisons"])
        record["maximum_nonzero_relative_error"] = max(row["relative_error"] or 0 for row in record["comparisons"])
        record["maximum_error_float32_ulps"] = max(row["error_in_float32_ulps_at_expected_value"] or 0
                                                  for row in record["comparisons"])
        record["resource_samples"].append(ram_sample())
        require(record["resource_samples"][-1]["available_ram_gib"] >= 8, "RAM below 8 GiB after recount")
        require(sha(args.input) == INPUT_SHA256 and all(sha(args.source_root / p) == h for p, h in SOURCE_SHA256.items()),
                "input/source changed during audit")
    except Exception as exc:
        record["issues"].append(type(exc).__name__ + ": " + str(exc))
    record.update(status="verified" if not record["issues"] else "failed", completed_at_utc=now(),
                  project_or_numerical_libraries_imported=any(k in sys.modules for k in ("torch", "numpy", "neuropixel")),
                  limitations=["Only two explicitly weighted legacy PAD routes are checked; no model performance, historical study result or general gradient theorem is inferred.",
                               "The correction's full regression suite is separate evidence. This audit checks the saved legacy defect numbers, not a new corrected-model execution.",
                               "Available RAM observations are sampled, not a continuous minimum or peak RSS."])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"status": record["status"], "issues": record["issues"],
                      "comparisons": len(record["comparisons"]), "sha256": sha(args.output)}))
    return int(bool(record["issues"]))


if __name__ == "__main__":
    raise SystemExit(main())
