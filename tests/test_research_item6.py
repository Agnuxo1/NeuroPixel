import json
from pathlib import Path

import numpy as np
import torch

from neuropixel.research.ablations import (
    EVAL_DAMAGE, core_evaluation_cases, core_training_configs,
    evaluation_keep_mask, training_metadata,
)
from neuropixel.research.growth import (
    TOPICS, TopicRoleTask, balanced_topic_dataset, fit_gate, visible_features,
)

ROOT = Path(__file__).resolve().parents[1]

def protocol():
    return json.loads((ROOT / "docs/research/protocol.json").read_text(encoding="utf-8"))

def test_item6_core_inventory_is_exact():
    configs = core_training_configs(protocol())
    cases = core_evaluation_cases(configs)
    assert len(configs) == 26
    assert len({x["run_id"] for x in configs}) == 26
    assert len(cases) == 34
    assert len({x["evaluation_id"] for x in cases}) == 34
    panels = {}
    for row in configs:
        panels[row["panel"]] = panels.get(row["panel"], 0) + 1
    assert panels == {"factorial": 16, "recurrence": 4, "damage": 2, "optimization": 4}

def test_tying_parameter_difference_and_school_times():
    configs = core_training_configs(protocol())
    tied = next(x for x in configs if x["panel"] == "factorial"
                and x["variant"]["tied"] and x["school_weight"] == 0.0
                and x["variant"]["reinject"] and x["init_seed"] == 20)
    untied = next(x for x in configs if x["panel"] == "factorial"
                  and not x["variant"]["tied"] and x["school_weight"] == 0.0
                  and x["variant"]["reinject"] and x["init_seed"] == 20)
    school = next(x for x in configs if x["panel"] == "factorial"
                  and x["variant"]["tied"] and x["school_weight"] == 0.3
                  and x["variant"]["reinject"] and x["init_seed"] == 20)
    assert training_metadata(tied)["parameters"] == 29824
    assert training_metadata(untied)["parameters"] == 30384
    assert training_metadata(school)["school_state_times"] == [3, 7, 11, 15]

def test_evaluation_damage_mask_is_deterministic():
    a = evaluation_keep_mask(64, 8, 8, EVAL_DAMAGE)
    b = evaluation_keep_mask(64, 8, 8, EVAL_DAMAGE)
    assert a.dtype == torch.bool
    assert torch.equal(a, b)
    erased = 1.0 - float(a.float().mean())
    assert 0.20 < erased < 0.40

def test_topic_partitions_only_contain_in_topic_agent_patient():
    base_counts = {"train": 0, "validation": 0, "test": 0}
    seen = {split: set() for split in base_counts}
    for topic, allowed_values in enumerate(TOPICS):
        task = TopicRoleTask(topic, seed=0)
        allowed = set(allowed_values)
        for split, triples in (("train", task.train_triples),
                               ("validation", task.validation_triples),
                               ("test", task.test_triples)):
            assert triples
            for triple in triples:
                assert int(triple[0]) in allowed
                assert int(triple[2]) in allowed
                assert tuple(map(int, triple)) not in seen[split]
                seen[split].add(tuple(map(int, triple)))
            base_counts[split] += len(triples)
    assert all(value > 0 for value in base_counts.values())

def test_balanced_topic_data_and_visible_features():
    task = TopicRoleTask(0, seed=0)
    canvas, target, roles = balanced_topic_dataset(task, "train", 512, 62011)
    assert canvas.shape == (512, 8, 8)
    assert target.shape == roles.shape == (512,)
    assert torch.bincount(roles, minlength=4).tolist() == [128, 128, 128, 128]
    features = visible_features(canvas, task.role_ids)
    assert features.shape == (512, 40)
    assert torch.all(features[:, 0] == 0)  # PAD/empty is not a visible token feature.
    assert torch.all(features[:, -1] == 1)
    assert torch.all(features[:, 35:39].sum(1) == 1)

def test_learned_gate_reduces_declared_objective():
    task = TopicRoleTask(0, seed=0)
    canvas, target, _ = balanced_topic_dataset(task, "train", 32, 62011)
    features = visible_features(canvas, task.role_ids)
    rng = np.random.default_rng(7)
    raw = rng.random((3, 32, 35))
    raw /= raw.sum(-1, keepdims=True)
    # Give expert 0 a weak target advantage so full-batch descent has a signal.
    for i, y in enumerate(target.tolist()):
        raw[0, i, y] += 0.2
        raw[0, i] /= raw[0, i].sum()
    fit = fit_gate(features, raw, target, steps=32)
    assert fit["weight"].shape == (40, 3)
    assert np.isfinite(fit["weight"]).all()
    assert fit["final_loss"] < fit["initial_loss"]
