"""MATH119 executor: exact finite checks of the one-bit Courtade-Kumar bound (frozen plan docs/research/MATH119_plan.json)."""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from completion_kit.dynamics import binary_entropy, boolean_noise_information  # noqa: E402

PLAN = ROOT / "docs/research/MATH119_plan.json"
PLAN_SHA256 = "2424396c0ccf022b0cec7b8bdfb4eea74dd5257134057b21557213acb5fa8d2a"
OUT = ROOT / "results/research/MATH119"


def channel(n: int, p: float) -> np.ndarray:
    size = 1 << n
    xs = np.arange(size)
    distance = np.array([bin(int(v)).count("1") for v in range(size)])[xs[:, None] ^ xs[None, :]]
    return p ** distance * (1 - p) ** (n - distance)


def information(table: np.ndarray, kernel: np.ndarray) -> tuple[float, float, float]:
    """I(f;Y), total probability and max deviation of P(Y) from uniform, vectorized."""
    size = len(table)
    joint = np.stack([kernel[table == label].sum(axis=0) / size for label in (0, 1)])
    py, pf = joint.sum(axis=0), joint.sum(axis=1)
    mask = joint > 0
    denominator = pf[:, None] * py[None, :]
    value = float(np.sum(joint[mask] * np.log2(joint[mask] / denominator[mask])))
    return value, float(joint.sum()), float(np.abs(py - 1 / size).max())


def eca_table(rule: int, n: int, steps: int) -> np.ndarray:
    xs = np.arange(1 << n)
    state = ((xs[:, None] >> np.arange(n)) & 1).astype(np.int64)
    for _ in range(steps):
        left, right = np.roll(state, 1, axis=1), np.roll(state, -1, axis=1)
        state = (rule >> (4 * left + 2 * state + right)) & 1
    return state[:, 0].astype(np.int64)


def depends_on(table: np.ndarray, n: int) -> int:
    xs = np.arange(1 << n)
    return sum(int(np.any(table[xs] != table[xs ^ (1 << bit)])) for bit in range(n))


def main() -> None:
    data = PLAN.read_bytes()
    assert hashlib.sha256(data).hexdigest() == PLAN_SHA256, "plan differs from the frozen protocol"
    plan = json.loads(data)
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.time()
    ps = plan["p_dev"] + plan["p_test"]
    kernels = {(n, p): channel(n, p) for n in plan["n"] for p in ps}
    cases, violations, max_mass_error, max_uniform_error = 0, [], 0.0, 0.0
    controls, readouts = [], {}
    for n in plan["n"]:
        xs = np.arange(1 << n)
        bit = lambda k: (xs >> k) & 1  # noqa: E731
        named = {"dictator": bit(0), "anti_dictator": 1 - bit(0), "constant": np.zeros(1 << n, dtype=np.int64),
                 "parity_2": (bit(0) ^ bit(1)), "parity_n": np.bitwise_xor.reduce(np.stack([bit(k) for k in range(n)]), axis=0)}
        if n % 2:
            named["majority"] = (np.stack([bit(k) for k in range(n)]).sum(axis=0) > n // 2).astype(np.int64)
        for p in ps:
            ceiling = 1 - binary_entropy(p)
            closed = {"dictator": ceiling, "anti_dictator": ceiling, "constant": 0.0,
                      "parity_2": 1 - binary_entropy((1 - (1 - 2 * p) ** 2) / 2),
                      "parity_n": 1 - binary_entropy((1 - (1 - 2 * p) ** n) / 2)}
            for name, table in named.items():
                value, mass, uniform = information(table, kernels[(n, p)])
                cases += 1
                max_mass_error, max_uniform_error = max(max_mass_error, abs(mass - 1)), max(max_uniform_error, uniform)
                if value > ceiling + 1e-12:
                    violations.append([name, n, p, value, ceiling])
                controls.append(dict(function=name, n=n, p=p, information=value, ceiling=ceiling,
                                     closed_form=closed.get(name), deviation=None if name not in closed else abs(value - closed[name])))
            for rule in range(256):
                for steps in (1, 2, 3):
                    table = eca_table(rule, n, steps)
                    value, mass, uniform = information(table, kernels[(n, p)])
                    cases += 1
                    max_mass_error, max_uniform_error = max(max_mass_error, abs(mass - 1)), max(max_uniform_error, uniform)
                    if value > ceiling + 1e-12:
                        violations.append([f"eca{rule}_T{steps}", n, p, value, ceiling])
                    readouts[(rule, steps, n, p)] = value
    # Selection on DEV p at n = 8 among readouts depending on at least two bits.
    eligible = []
    for rule in range(256):
        for steps in (1, 2, 3):
            if depends_on(eca_table(rule, 8, steps), 8) >= 2:
                eff = np.mean([readouts[(rule, steps, 8, p)] / (1 - binary_entropy(p)) for p in plan["p_dev"]])
                eligible.append((-eff, rule, steps))
    eligible.sort()
    best_eff, best_rule, best_steps = -eligible[0][0], eligible[0][1], eligible[0][2]
    table = eca_table(best_rule, 8, best_steps)
    selected = dict(rule=best_rule, steps=best_steps, dev_mean_efficiency=best_eff, depends_on_bits=depends_on(table, 8),
                    output_bias=float(table.mean()),
                    test=[dict(p=p, information=readouts[(best_rule, best_steps, 8, p)], ceiling=1 - binary_entropy(p),
                               efficiency=readouts[(best_rule, best_steps, 8, p)] / (1 - binary_entropy(p))) for p in plan["p_test"]])
    # E5: agreement with the kit's reference implementation on 300 registered random cases.
    rng = np.random.default_rng(20261013)
    agreement = 0.0
    for _ in range(300):
        n = int(rng.choice(plan["n"]))
        p = float(rng.choice(ps))
        table = rng.integers(0, 2, size=1 << n)
        if table.min() == table.max():
            table[0] ^= 1
        kit = boolean_noise_information(table, p)["information_bits"]
        agreement = max(agreement, abs(kit - information(table, kernels[(n, p)])[0]))
    deviation = max(c["deviation"] for c in controls if c["deviation"] is not None)
    result = dict(schema=1, plan_sha256=PLAN_SHA256, cases=cases, eligible_readouts_n8=len(eligible),
                  E1_violations=violations, E2_max_closed_form_deviation=deviation, E3_selected=selected,
                  E4=dict(max_joint_mass_error=max_mass_error, max_Y_uniformity_error=max_uniform_error),
                  E5_max_kit_disagreement=agreement, controls=controls, elapsed_seconds=time.time() - started,
                  passed=dict(E1=not violations, E2=deviation <= 1e-12, E4=max_mass_error <= 1e-12 and max_uniform_error <= 1e-12,
                              E5=agreement <= 1e-12),
                  scope="Finite exact checks of examples; not a proof of the general theorem")
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(dict(cases=cases, violations=len(violations), passed=result["passed"],
                          selected=dict(rule=best_rule, steps=best_steps, dev_eff=round(best_eff, 4),
                                        test=[round(t["efficiency"], 4) for t in selected["test"]]),
                          seconds=round(result["elapsed_seconds"], 1)), indent=1))


if __name__ == "__main__":
    main()
