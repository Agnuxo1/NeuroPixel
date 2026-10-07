"""Bounded item13 protocol witnesses, label census and representation isomorphism.

Literal baselines and activated but untrained tiny models establish mechanisms,
not new semantic learning or trained distractor robustness. No checkpoint load.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import research_item9_worker as ops

def require(ok, message):
    if not ok:
        raise ValueError(message)

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def solve(canvas):
    """Independent of generator metadata: find four unique immediate role pairs."""
    query = canvas[7][6]
    require(query in (1, 2, 3, 4) and canvas[7][7] == 0, "invalid query/output")
    pairs = {}
    for row in canvas[:7]:
        for col, token in enumerate(row):
            if token in (1, 2, 3, 4):
                require(token not in pairs and col < 7 and row[col + 1] >= 5,
                        "invalid or repeated role pair")
                pairs[token] = row[col + 1]
    require(set(pairs) == {1, 2, 3, 4}, "role inventory differs")
    return pairs[query]

def execute(args):
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    state = {"schema_version": 1, "item": 13, "status": "running",
             "started_at_utc": ops.now(), "source_commit": os.environ.get("GITHUB_SHA"),
             "plan_sha256": digest(args.plan), "checks": []}
    def check(name, ok, **extra):
        state["checks"].append({"name": name, "passed": bool(ok), **extra})
        require(ok, name)
    try:
        plan = json.loads(args.plan.read_bytes())
        require(plan["item"] == 13 and plan["status"] == "frozen", "wrong plan")
        ops.admit(started + 120, minimum_seconds=10)
        require(all(os.environ.get(k) == "1" for k in ops.THREAD_ENV), "threads differ")
        import torch
        from neuropixel.model import NeuroPixel, TinyTransformer
        from neuropixel.task import RoleTask
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.manual_seed(130002)
        conditions = ("base", "new_agent", "new_patient", "two_new",
                      "irrelevant_old", "irrelevant_new")
        examples = []
        for scene in range(32):
            a = 5 + scene % 12
            p = 5 + (scene % 12 + 1 + scene // 12) % 12
            fillers = [a, 17 + scene % 10, p, 27 + scene % 8]
            rows = [(scene + j) % 7 for j in range(4)]
            cols = [(3 * scene + 2 * j) % 7 for j in range(4)]
            for query in range(1, 5):
                base = [[0] * 8 for _ in range(8)]
                for j in range(4):
                    base[rows[j]][cols[j]] = j + 1
                    base[rows[j]][cols[j] + 1] = fillers[j]
                base[7][6] = query
                blank = next((r, c) for r in range(7) for c in range(8) if base[r][c] == 0)
                for condition in conditions:
                    canvas = copy.deepcopy(base)
                    if condition in ("new_agent", "two_new"):
                        canvas[rows[0]][cols[0] + 1] = 35
                    if condition in ("new_patient", "two_new"):
                        canvas[rows[2]][cols[2] + 1] = 36 if condition == "two_new" else 35
                    if condition in ("irrelevant_old", "irrelevant_new"):
                        canvas[blank[0]][blank[1]] = a if condition == "irrelevant_old" else 35
                    expected = fillers[query - 1]
                    if condition in ("new_agent", "two_new") and query == 1:
                        expected = 35
                    if condition in ("new_patient", "two_new") and query == 3:
                        expected = 36 if condition == "two_new" else 35
                    parsed = solve(canvas)
                    require(parsed == expected, "independent role parser differs from transformation gold")
                    visible = any(35 in row for row in canvas)
                    gate = 35 if visible else solve(canvas)
                    examples.append({"scene": scene, "query": query, "condition": condition,
                                     "canvas": canvas, "gold": expected, "base_gold": fillers[query - 1],
                                     "constant_35": 35, "presence_gate_with_symbolic_fallback": gate,
                                     "symbolic": parsed, "added_unpaired_cell": list(blank)
                                     if condition.startswith("irrelevant_") else None})
        rows_out = []
        for condition in conditions:
            for subset in ("all", "binding", "new_target", "old_target"):
                values = [e for e in examples if e["condition"] == condition and
                          (subset == "all" or subset == "binding" and e["query"] in (1, 3)
                           or subset == "new_target" and e["gold"] >= 35
                           or subset == "old_target" and e["gold"] < 35)]
                rows_out.append({"condition": condition, "subset": subset, "n": len(values),
                                 **{name + "_correct": sum(e[name] == e["gold"] for e in values)
                                    for name in ("constant_35", "presence_gate_with_symbolic_fallback", "symbolic")}})
        state["label_census"] = {"scenes": 32, "queries_per_scene": 4,
                                 "conditions": list(conditions), "examples": examples, "counts": rows_out,
                                 "scope": "Constructed controls and literal baseline witnesses; no trained-model scores."}
        check("all_768_gold_labels_match_role_parser", len(examples) == 768 and
              all(e["gold"] == e["symbolic"] for e in examples))
        check("unpaired_distractors_preserve_every_gold", all(
              e["gold"] == e["base_gold"] for e in examples if e["condition"].startswith("irrelevant_")))
        check("constant_new_solves_single_new_positive_only_panels", all(
              e["constant_35"] == e["gold"] for e in examples if
              e["condition"] in ("new_agent", "new_patient") and e["gold"] == 35))
        check("presence_shortcut_fails_all_unpaired_new_cases", all(
              e["presence_gate_with_symbolic_fallback"] != e["gold"] for e in examples
              if e["condition"] == "irrelevant_new"))

        task = RoleTask(8, 8, seed=0)
        original = {"train": [list(t) for t in task.train_triples],
                    "test": [list(t) for t in task.test_triples]}
        projections = []
        for role in ("agent", "patient", "both"):
            projected = {}
            for split, triples in original.items():
                projected[split] = sorted({(35 if role in ("agent", "both") else a,
                                           b, 36 if role == "both" else
                                           35 if role == "patient" else p)
                                          for a, b, p in triples})
            overlap = sorted(set(projected["train"]) & set(projected["test"]))
            projections.append({"replacement": role,
                                "train_unique": len(projected["train"]), "test_unique": len(projected["test"]),
                                "shared_unique": len(overlap), "shared_keys": [list(t) for t in overlap]})
        check("original_1320_triples_disjoint", len(original["train"]) == 1056 and
              len(original["test"]) == 264 and not
              (set(map(tuple, original["train"])) & set(map(tuple, original["test"]))))
        state["transformed_split_census"] = {"original": original, "projections": projections,
              "scope": "Exhaustive triple projections of the current legacy split recipe. Place, layout and actual historical five-shot draws are not identical-example assertions."}

        softmax = []
        fixtures = [([[-10000., 1., 2., -1.], [-10000., -2., .5, 1.5]],
                     [[0.], [3.]], [2, 3]),
                    ([[-10000., 1., 2., -1.], [-10000., -2., .5, 1.5]],
                     [[10., -4.], [3., 2.]], [2, 3])]
        for old_rows, appended, target_rows in fixtures:
            old = torch.tensor(old_rows, dtype=torch.float64)
            added = torch.tensor(appended, dtype=torch.float64)
            new = torch.cat((old, added), dim=1)
            targets = torch.tensor(target_rows)
            old_lp, new_lp = old.log_softmax(-1), new.log_softmax(-1)
            log_ratio = torch.logsumexp(new, -1) - torch.logsumexp(old, -1)
            observed = -new_lp[torch.arange(2), targets] + old_lp[torch.arange(2), targets]
            restored = new_lp[:, :4] - torch.logsumexp(new_lp[:, :4], -1, keepdim=True)
            torch.testing.assert_close(observed, log_ratio, rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(restored, old_lp, rtol=1e-12, atol=1e-12)
            softmax.append({"old_logits": old.tolist(), "new_logits": new.tolist(),
                            "targets": targets.tolist(), "old_log_probabilities": old_lp.tolist(),
                            "new_log_probabilities": new_lp.tolist(),
                            "observed_nll_increase": observed.tolist(),
                            "log_normalizer_increase": log_ratio.tolist(),
                            "new_class_mass": new.softmax(-1)[:, 4:].sum(-1).tolist(),
                            "old_argmax": old.argmax(-1).tolist(), "new_argmax": new.argmax(-1).tolist()})
        state["softmax_fixtures"] = softmax
        check("literal_softmax_identity", True, fixtures=2, rows=4, dtype="float64")

        canvas = torch.tensor([[[1, 5, 0], [3, 6, 0], [0, 1, 0]],
                               [[1, 7, 0], [3, 8, 0], [0, 3, 0]]])
        permutation = torch.tensor([0, 1, 2, 3, 4, 6, 5, 8, 7])
        transformations = []
        for family in ("neuropixel", "transformer"):
            torch.manual_seed(130003)
            if family == "neuropixel":
                rgb = torch.arange(27, dtype=torch.float32).reshape(9, 3) / 100
                mask = torch.zeros(9, dtype=torch.bool)
                mask[5] = True
                base = NeuroPixel(9, (2, 2), c_id=4, c=8, hidden=12, steps=2,
                                  fire_rate=.5, grounded=(rgb, mask)).double().eval()
                with torch.no_grad():
                    base.f2.weight.fill_(.005)
                    base.f2.bias.fill_(.02)
                    base.read.bias.copy_(torch.tensor([.05, .1, .15, .2], dtype=torch.float64))
            else:
                base = TinyTransformer(9, 3, 3, (2, 2), d=8, layers=1, heads=2, ff=12).double().eval()
            renamed = copy.deepcopy(base)
            with torch.no_grad():
                if family == "neuropixel":
                    renamed.embed.weight[permutation] = base.embed.weight
                    renamed.g_rgb[permutation] = base.g_rgb
                    renamed.g_mask[permutation] = base.g_mask
                else:
                    renamed.tok.weight[permutation] = base.tok.weight
                    renamed.head.weight[permutation] = base.head.weight
                    renamed.head.bias[permutation] = base.head.bias
                old_output = base(canvas)
                new_output = renamed(permutation[canvas])
            aligned = new_output["logits"][:, permutation]
            torch.testing.assert_close(aligned, old_output["logits"], rtol=1e-12, atol=1e-12)
            if family == "neuropixel":
                torch.testing.assert_close(new_output["state"], old_output["state"], rtol=0, atol=0)
            transformations.append({"family": family, "permutation_old_to_new": permutation.tolist(),
                "old_canvas": canvas.tolist(), "new_canvas": permutation[canvas].tolist(),
                "old_logits": old_output["logits"].tolist(), "new_logits": new_output["logits"].tolist(),
                "max_aligned_logit_difference": float((aligned - old_output["logits"]).abs().max()),
                "old_state": old_output["state"].tolist() if "state" in old_output else None,
                "new_state": new_output["state"].tolist() if "state" in new_output else None})
        state["joint_renaming"] = {"seed_per_family": 130003, "records": transformations,
            "scope": "Constructed change of ID coordinates with correspondingly permuted model tables. Untrained fixture; NP residual layer manually activated; no new semantics."}
        check("consistent_renaming_preserves_aligned_logits", True, families=2)
        check("neuropixel_consistent_renaming_preserves_state", True)
        state.update(status="verified", model_training_executed=False,
                     torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
                     limitations=["Literal shortcuts do not identify the rule a historical trained model used.",
                                  "Split projection overlap is not proof of exact historical support/test-example overlap.",
                                  "Unpaired distractor labels are verified; no learned resistance is claimed.",
                                  "Joint token-and-table renaming is an isomorphism, not an unchanged-model learning test."])
    except BaseException as error:
        state.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
    state.update(completed_at_utc=ops.now(), wall_seconds=time.monotonic() - started,
                 resource=ops.resource_sample("item13_native_finish"))
    if state["resource"]["available_ram_gib"] < 8:
        state.update(status="failed", ram_floor_violation=True)
    ops.save(args.output / "report.json", state, exclusive=True)
    print(json.dumps({"status": state["status"], "checks": len(state["checks"])}), flush=True)
    return int(state["status"] != "verified")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return execute(parser.parse_args())

if __name__ == "__main__":
    raise SystemExit(main())
