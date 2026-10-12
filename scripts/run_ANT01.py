"""ANT01 executor: global versus locally coupled Hopfield memory and repair at equal state size.

Runs exactly the frozen plan docs/research/ANT01_plan.json (its SHA-256 is checked first) on CPU
with NumPy only, and writes every grid cell, negatives included. Grid cells are enumerated in the
order pattern_family, K, f, sigma (each in the plan's listed order).
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np  # noqa: E402
from scipy import ndimage, stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs/research/ANT01_plan.json"
PLAN_SHA256 = "38436579c892e3f2fcd216d6bce32e814f49034ea526b9b2d6cd63f9d0f655b6"
OUT = ROOT / "results/research/ANT01"
SIDE, N, SWEEPS = 8, 64, 50
MODELS = ("global_hopfield", "local_r1", "local_r2")


def neighbourhood_mask(radius: int | None) -> np.ndarray:
    coords = np.array([(i // SIDE, i % SIDE) for i in range(N)])
    if radius is None:
        mask = np.ones((N, N), dtype=bool)
    else:
        diff = np.abs(coords[:, None, :] - coords[None, :, :]).max(axis=2)
        mask = diff <= radius
    np.fill_diagonal(mask, False)
    return mask


MASKS = {"global_hopfield": neighbourhood_mask(None), "local_r1": neighbourhood_mask(1), "local_r2": neighbourhood_mask(2)}


def patterns(rng: np.random.Generator, family: str, k: int) -> np.ndarray:
    if family == "rand":
        return rng.choice(np.array([-1.0, 1.0]), size=(k, N))
    field = rng.normal(size=(k, SIDE, SIDE))
    out = np.empty((k, N))
    for index in range(k):
        smooth = ndimage.gaussian_filter(field[index], sigma=1.5, mode="reflect").ravel()
        out[index] = np.where(smooth > np.median(smooth), 1.0, -1.0)
    return out


def run_dynamics(weights: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, int]:
    """Asynchronous raster sweeps over a batch; weights (B,N,N), state (B,N)."""
    state = state.copy()
    for sweep in range(1, SWEEPS + 1):
        changed = np.zeros(state.shape[0], dtype=bool)
        for i in range(N):
            h = np.einsum("bj,bj->b", weights[:, i, :], state)
            new = np.where(h > 0, 1.0, np.where(h < 0, -1.0, np.where(state[:, i] == 0, 1.0, state[:, i])))
            changed |= new != state[:, i]
            state[:, i] = new
        if not changed.any():
            return state, sweep
    return state, SWEEPS


def summarize(values: np.ndarray) -> dict:
    mean = float(values.mean())
    if len(values) > 1 and values.std(ddof=1) > 0:
        half = float(stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / math.sqrt(len(values)))
    else:
        half = 0.0
    return dict(mean=mean, t95=[mean - half, mean + half])


def main() -> None:
    data = PLAN.read_bytes()
    assert hashlib.sha256(data).hexdigest() == PLAN_SHA256, "plan differs from the frozen protocol"
    plan = json.loads(data)
    OUT.mkdir(parents=True, exist_ok=False)
    grid = plan["grid"]
    instances, cues = plan["replication"]["memory_instances_per_cell"], plan["replication"]["cues_per_instance"]
    rows, started = [], time.time()
    cells = list(itertools.product(grid["pattern_family"], grid["K"], grid["f"], grid["sigma"]))
    for cell_index, (family, k, f, sigma) in enumerate(cells):
        pats, cue_states, targets, visibles = [], [], [], []
        for instance in range(instances):
            rng = np.random.default_rng(20261013 + 1000 * cell_index + instance)
            stored = patterns(rng, family, k)
            pats.append(stored)
            for _ in range(cues):
                target = int(rng.integers(k))
                erased = rng.choice(N, size=int(round(f * N)), replace=False)
                cue = stored[target].copy()
                cue[erased] = 0.0
                visible = np.ones(N, dtype=bool)
                visible[erased] = False
                cue[visible] += sigma * rng.normal(size=int(visible.sum()))
                cue_states.append(cue)
                targets.append(target)
                visibles.append(visible)
        stored_all = np.stack(pats)                                   # (I, K, N)
        cue_all = np.stack(cue_states)                                # (I*C, N)
        target_all = np.array(targets)
        target_patterns = stored_all[np.repeat(np.arange(instances), cues), target_all]
        initial = np.where(np.stack(visibles), np.sign(cue_all), 0.0)
        record = dict(cell=cell_index, pattern_family=family, K=k, f=f, sigma=sigma, models={})
        per_instance = {}
        for model in MODELS:
            hebb = np.einsum("ikn,ikm->inm", stored_all, stored_all) / N * MASKS[model]
            final, sweeps = run_dynamics(np.repeat(hebb, cues, axis=0), initial)
            exact = (final == target_patterns).all(axis=1).reshape(instances, cues).mean(axis=1)
            overlap = ((final * target_patterns).sum(axis=1) / N).reshape(instances, cues).mean(axis=1)
            hamming = (final[:, None, :] != stored_all[np.repeat(np.arange(instances), cues)]).sum(axis=2)
            others = hamming.copy()
            others[np.arange(len(target_all)), target_all] = N + 1
            basin = (hamming[np.arange(len(target_all)), target_all] < others.min(axis=1)).reshape(instances, cues).mean(axis=1)
            start = stored_all.reshape(instances * k, N)
            fixed, _ = run_dynamics(np.repeat(hebb, k, axis=0), start)
            storage = (fixed == start).all(axis=1).reshape(instances, k).mean(axis=1)
            per_instance[model] = dict(E1=exact, E2=overlap)
            record["models"][model] = dict(E1_exact_repair=summarize(exact), E2_final_overlap=summarize(overlap),
                                           E3_storage=summarize(storage), E4_basin_identification=summarize(basin),
                                           max_sweeps_used=int(sweeps))
        visible_all = np.stack(visibles)
        scores = np.einsum("bn,bkn->bk", np.where(visible_all, cue_all, 0.0),
                           stored_all[np.repeat(np.arange(instances), cues)])
        decoded = scores.argmax(axis=1)
        dec_exact = (decoded == target_all).reshape(instances, cues).mean(axis=1)
        record["models"]["explicit_decoder"] = dict(E1_exact_repair=summarize(dec_exact),
                                                    E4_basin_identification=summarize(dec_exact),
                                                    tie_rule="lowest index among maximal overlaps")
        per_instance["explicit_decoder"] = dict(E1=dec_exact)
        record["paired_differences"] = {
            f"global_minus_{local}_{endpoint}": summarize(per_instance["global_hopfield"][endpoint] - per_instance[local][endpoint])
            for local in ("local_r1", "local_r2") for endpoint in ("E1", "E2")}
        means = {m: record["models"][m]["E1_exact_repair"]["mean"] for m in record["models"]}
        record["PR1_holds"] = None if family != "rand" or k < 2 else bool(
            means["global_hopfield"] >= means["local_r1"] and means["global_hopfield"] >= means["local_r2"])
        record["PR2_holds"] = bool(all(means["explicit_decoder"] >= means[m] for m in MODELS))
        rows.append(record)
    pr1 = [r["PR1_holds"] for r in rows if r["PR1_holds"] is not None]
    pr2 = [r["PR2_holds"] for r in rows]
    result = dict(schema=1, plan="docs/research/ANT01_plan.json", plan_sha256=PLAN_SHA256, cells=rows,
                  predictions=dict(PR1=dict(cells=len(pr1), held=sum(pr1),
                                            violations=[(r["pattern_family"], r["K"], r["f"], r["sigma"]) for r in rows if r["PR1_holds"] is False]),
                                   PR2=dict(cells=len(pr2), held=sum(pr2),
                                            violations=[(r["pattern_family"], r["K"], r["f"], r["sigma"]) for r in rows if not r["PR2_holds"]])),
                  elapsed_seconds=time.time() - started, software=dict(numpy=np.__version__, python=sys.version.split()[0]))
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(dict(cells=len(rows), PR1=result["predictions"]["PR1"]["held"], PR1_of=len(pr1),
                          PR2=result["predictions"]["PR2"]["held"], PR2_of=len(pr2), seconds=round(result["elapsed_seconds"], 1))))


if __name__ == "__main__":
    main()
