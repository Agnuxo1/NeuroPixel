"""Controlled expert-bank growth diagnostics with lazy numerical/model imports.

Historical growth adds copied NeuroPixel models, not spatial cells. The frozen
inventory distinguishes a sequential snapshot bank from two original adaptive
heuristics. Routing functions consume visible inputs and saved expert outputs;
targets enter only gate fitting or scoring. No function silently selects settings
from final outcomes.
"""
from __future__ import annotations

import hashlib
import json
import math

TOPICS = ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11))
POLICIES = ("fixed_sequential", "adaptive_novelty", "adaptive_resonance")
ROUTERS = ("uniform", "random", "scanner", "learned")
ROLES = ("AGENTE", "ACCION", "PACIENTE", "LUGAR")
SEEDS = (20, 21)
EXPERT_PARAMETERS = 29824
FEATURES = 40


def growth_inventory():
    """Return the exact prospective panel without importing NumPy or Torch."""
    trajectories = [
        {"trajectory_id": f"{policy}_i{seed}", "policy": policy, "init_seed": seed,
         "split_seed": 0, "topics": [0, 1, 2], "stage_updates": 512,
         "batch_size": 64, "learning_rate": 0.003, "weight_decay": 0.0001,
         "gradient_clip": 1.0, "school_weight": 0.3, "steps": 16,
         "tying": True, "reinjection": True, "optimizer_reset_each_stage": True}
        for seed in SEEDS for policy in POLICIES
    ]
    return {
        "schema_version": 1, "panel": "growth", "analysis": "exploratory",
        "planned_trajectories": 6, "planned_training_stages": 18,
        "planned_optimizer_updates": 9216, "planned_gate_fits": 8,
        "planned_routing_evaluations": 34, "planned_preallocation_checks": 2,
        "trajectories": trajectories,
        "topics": [list(topic) for topic in TOPICS],
        "partition_rule": "filter original research pools; never reshuffle or rebalance triple membership",
        "diagnostic_role_balance": "natural sampling; no forced role balance",
        "diagnostic_recount_atol": 1e-6,
        "stages": [
            {"topic": stage, "train_sample_seed": 9002 + 100 * stage,
             "firing_seed": 9001 + 100 * stage, "diagnostic_n": 512,
             "diagnostic_seed": 62010 + 100 * stage,
             "gate_fit_n": 512, "gate_fit_seed": 62011 + 100 * stage,
             "test_n": 1024, "test_seed": 62012 + 100 * stage}
            for stage in range(3)
        ],
        "novelty": {"own_token_probability_threshold": 0.5, "creation_threshold": 0.1,
                    "aggregation": "occupied-cell fraction over the entire diagnostic batch",
                    "score_round_decimals": 4, "comparison": "strictly greater",
                    "parent": "argmin", "tie_break": "lowest existing expert index"},
        "resonance": {"aggregation": "mean of per-example occupied-cell mean probability",
                      "score_round_decimals": 4, "comparison": "strictly less",
                      "threshold": "0.75 times unrounded post-stage-A mean on the same stage-A diagnostic batch",
                      "parent": "argmax", "tie_break": "lowest existing expert index"},
        "maximum_experts": 3, "maximum_experts_reason": "at most one creation in each of three stages",
        "gate": {"features": "binary visible-token bag of 35 IDs, query one-hot of 4 roles, bias",
                 "feature_count": FEATURES, "initialization": "zeros",
                 "updates": 128, "learning_rate": 0.1, "l2": 0.0001,
                 "optimizer": "deterministic full-batch gradient descent",
                 "objective": "mean negative log likelihood of expert probability mixture plus 0.5*l2*sum(W**2), including bias",
                 "expert_weights_frozen": True},
        "routers": list(ROUTERS), "random_router_seed": 62013,
        "fixed_banks": {"single_final": [2], "snapshots": [0, 1, 2],
                        "duplicate_slots": [0, 1, 1], "preallocated_equivalent": [0, 1, 2]},
        "adaptive_bank": "all final experts retained by the respective heuristic",
        "parameter_accounting": "report slots, unique checkpoints, nominal expert parameters and learned gate parameters separately",
        "hard_selection_oracle_scope": "upper bound for hard expert selection only; not for probability mixtures",
        "h1_eligible": False,
    }


def growth_decision(policy, stage, novelty, resonance, tau=None):
    """Apply the original, separate creation rules to predeclared TRAIN scores.

    Scores are rounded before heuristic selection/comparison as in phase3.py.
    The resonance threshold is supplied unrounded and never adjusted here.
    Fixed sequential stages copy the latest snapshot. At most three stages are
    valid; the maximum expert count follows from that schedule, not a tuned cap.
    """
    if policy not in POLICIES or type(stage) is not int or not 0 <= stage < 3:
        raise ValueError("unknown policy or stage")
    if len(novelty) != len(resonance) or len(novelty) > stage:
        raise ValueError("expert-score inventory is inconsistent with the stage")
    for value in list(novelty) + list(resonance):
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("heuristic scores must be finite probabilities")
    count = len(novelty)
    if stage == 0:
        if count or tau is not None:
            raise ValueError("stage A must begin with an empty bank and no threshold")
        return {"action": "create", "parent": None, "target_index": 0,
                "scores_used": [], "threshold": None}
    if not count:
        raise ValueError("a later stage requires an existing trained expert")
    if policy == "fixed_sequential":
        if count != stage:
            raise ValueError("fixed snapshot lineage must contain one expert per prior stage")
        return {"action": "create", "parent": count - 1, "target_index": count,
                "scores_used": [], "threshold": None}
    if policy == "adaptive_novelty":
        scores = [round(value, 4) for value in novelty]
        parent = min(range(count), key=lambda index: scores[index])
        create = scores[parent] > 0.1
        threshold = 0.1
    else:
        if type(tau) not in (int, float) or not math.isfinite(tau) or not 0 <= tau <= 0.75:
            raise ValueError("resonance requires its unrounded fixed stage-A threshold")
        scores = [round(value, 4) for value in resonance]
        parent = max(range(count), key=lambda index: scores[index])
        create = scores[parent] < tau
        threshold = tau
    return {"action": "create" if create else "update", "parent": parent,
            "target_index": count if create else parent,
            "scores_used": scores, "threshold": threshold}


def make_topic_task(topic, split_seed=0):
    """Filter a fresh research task's three pools without editing frozen code."""
    if type(topic) is not int or not 0 <= topic < len(TOPICS) or split_seed != 0:
        raise ValueError("the frozen growth panel uses topics 0/1/2 and split 0")
    import torch
    from neuropixel.research.data import ResearchRoleTask

    task = ResearchRoleTask(8, 8, seed=split_seed)
    allowed = set(TOPICS[topic])
    for split, attribute in (("train", "train_triples"), ("validation", "validation_triples"),
                             ("test", "test_triples")):
        pool = tuple(triple for triple in getattr(task, attribute)
                     if triple[0] in allowed and triple[2] in allowed)
        if not pool:
            raise ValueError(f"topic {topic} has an empty {split} pool")
        setattr(task, attribute, pool)
        task._triples_cpu[split] = torch.tensor(pool, dtype=torch.long, device="cpu")
    task.topic_index = topic
    return task


def partition_manifest(task):
    """Record actual subset counts and membership; topic ratios need not be exact."""
    pools = {split: [list(triple) for triple in getattr(task, attribute)]
             for split, attribute in (("train", "train_triples"),
                                      ("validation", "validation_triples"), ("test", "test_triples"))}
    sets = {split: {tuple(triple) for triple in values} for split, values in pools.items()}
    if any(sets[left] & sets[right] for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("filtered partitions overlap")
    if len(set.union(*sets.values())) != 120:
        raise ValueError("a four-noun topic must cover exactly 120 distinct triples")
    payload = json.dumps(pools, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"topic": task.topic_index, "split_seed": task.split_seed,
            "counts": {split: len(values) for split, values in pools.items()},
            "triples_sha256": hashlib.sha256(payload).hexdigest(), "triples": pools}


def input_features(canvas, query_pos, role_token_ids):
    """Encode only visible inputs; no targets, topic ID or metadata are accepted."""
    import numpy as np

    canvas = np.asarray(canvas)
    role_token_ids = np.asarray(role_token_ids)
    if canvas.ndim != 3 or canvas.dtype.kind not in "iu" or len(canvas) == 0:
        raise ValueError("canvas must be a nonempty integer batch")
    if np.any((canvas < 0) | (canvas >= 35)):
        raise ValueError("visible token ID is outside the fixed vocabulary")
    if role_token_ids.shape != (4,) or len(set(map(int, role_token_ids))) != 4:
        raise ValueError("four distinct query role IDs are required")
    row, column = query_pos
    if not 0 <= row < canvas.shape[1] or not 0 <= column < canvas.shape[2]:
        raise ValueError("query position is outside the canvas")
    query = canvas[:, row, column, None] == role_token_ids[None, :]
    if not np.all(query.sum(axis=1) == 1):
        raise ValueError("each visible query must identify exactly one role")
    bag = np.zeros((len(canvas), 35), dtype=np.float64)
    bag[np.arange(len(canvas))[:, None], canvas.reshape(len(canvas), -1)] = 1.0
    bag[:, 0] = 0.0
    return np.concatenate((bag, query.astype(np.float64), np.ones((len(canvas), 1))), axis=1)


def _logsumexp(values, axis):
    import numpy as np

    maximum = np.max(values, axis=axis, keepdims=True)
    result = maximum + np.log(np.exp(values - maximum).sum(axis=axis, keepdims=True))
    return np.squeeze(result, axis=axis)


def log_probabilities(logits):
    """Convert finite saved expert logits to normalized float64 log probabilities."""
    import numpy as np

    values = np.asarray(logits, dtype=np.float64)
    if values.ndim < 2 or not np.all(np.isfinite(values)):
        raise ValueError("finite expert logits are required")
    return values - _logsumexp(values, axis=-1)[..., None]


def _expert_arrays(expert_log_probabilities):
    import numpy as np

    values = np.asarray(expert_log_probabilities, dtype=np.float64)
    if values.ndim != 3 or not 1 <= values.shape[0] <= 3 or not values.shape[1] or values.shape[2] < 2:
        raise ValueError("expert outputs must have shape E,N,V with one to three experts")
    if not np.all(np.isfinite(values)) or not np.allclose(_logsumexp(values, axis=-1), 0.0, atol=1e-5, rtol=0):
        raise ValueError("expert log probabilities must be finite and normalized")
    return values


def gate_objective(features, expert_log_probabilities, target, weights, l2=0.0001):
    """Return the prescribed mixture NLL, its gradient and unpenalized mean NLL."""
    import numpy as np

    experts = _expert_arrays(expert_log_probabilities)
    x, w = np.asarray(features, dtype=np.float64), np.asarray(weights, dtype=np.float64)
    y = np.asarray(target)
    e, n, vocab = experts.shape
    if x.ndim != 2 or x.shape[0] != n or w.shape != (x.shape[1], e):
        raise ValueError("gate feature/weight shapes do not match expert outputs")
    if y.shape != (n,) or y.dtype.kind not in "iu" or np.any((y < 0) | (y >= vocab)):
        raise ValueError("gate fitting requires in-range integer TRAIN targets")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(w)) or l2 < 0 or not math.isfinite(l2):
        raise ValueError("gate features, weights or regularization are invalid")
    logits = x @ w
    log_weights = logits - _logsumexp(logits, axis=1)[:, None]
    expert_target = experts[:, np.arange(n), y].T
    log_mixture_target = _logsumexp(log_weights + expert_target, axis=1)
    posterior = np.exp(log_weights + expert_target - log_mixture_target[:, None])
    gradient = x.T @ (np.exp(log_weights) - posterior) / n + l2 * w
    mean_nll = float(-log_mixture_target.mean())
    objective = mean_nll + 0.5 * l2 * float(np.square(w).sum())
    if not math.isfinite(objective) or not np.all(np.isfinite(gradient)):
        raise FloatingPointError("gate objective or gradient became nonfinite")
    return objective, gradient, mean_nll


def fit_gate(features, expert_log_probabilities, target):
    """Fit exactly 128 deterministic full-batch steps from zero, with frozen experts."""
    import numpy as np

    experts = _expert_arrays(expert_log_probabilities)
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != FEATURES:
        raise ValueError("the frozen gate requires exactly 40 visible-input features")
    weights = np.zeros((FEATURES, experts.shape[0]), dtype=np.float64)
    trace = []
    for update in range(129):
        objective, gradient, nll = gate_objective(x, experts, target, weights)
        if update % 32 == 0:
            trace.append({"update": update, "objective": objective, "cross_entropy": nll})
        if update < 128:
            weights -= 0.1 * gradient
    return {"weights": weights, "trace": trace, "updates": 128, "learning_rate": 0.1,
            "l2": 0.0001, "parameter_count": int(weights.size)}


def route_experts(expert_log_probabilities, policy, *, scanner=None, features=None,
                  weights=None, random_seed=62013):
    """Route from saved outputs without access to labels or topic metadata.

    Random/scanner are hard selectors. Uniform/learned mix probabilities and
    therefore need not obey an any-individual-expert-correct upper bound.
    Call once on the complete A/B/C example order, not once per inference batch.
    ``expert_pick`` is the executed index only for hard selectors. For mixtures
    it reports an argmax of routing weights (the first tied slot for uniform),
    not a single expert used for prediction. Summarize mixture allocation from
    ``routing_weights``; counting mixture ``expert_pick`` values is misleading.
    """
    import numpy as np

    experts = _expert_arrays(expert_log_probabilities)
    e, n, _ = experts.shape
    if policy not in ROUTERS:
        raise ValueError("unknown routing policy")
    if policy == "uniform":
        routing = np.full((n, e), 1.0 / e, dtype=np.float64)
        picks = np.zeros(n, dtype=np.int64)
    elif policy == "random":
        picks = np.random.default_rng(random_seed).integers(e, size=n, dtype=np.int64)
        routing = np.eye(e, dtype=np.float64)[picks]
    elif policy == "scanner":
        scores = np.asarray(scanner, dtype=np.float64)
        if scores.shape != (e, n) or not np.all(np.isfinite(scores)):
            raise ValueError("scanner requires one finite unrounded score per expert/example")
        picks = scores.argmax(axis=0).astype(np.int64)
        routing = np.eye(e, dtype=np.float64)[picks]
    else:
        x, w = np.asarray(features, dtype=np.float64), np.asarray(weights, dtype=np.float64)
        if x.shape != (n, FEATURES) or w.shape != (FEATURES, e) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(w)):
            raise ValueError("learned gate feature/weight shape or values differ")
        logits = x @ w
        routing = np.exp(logits - _logsumexp(logits, axis=1)[:, None])
        picks = routing.argmax(axis=1).astype(np.int64)
    if policy in ("random", "scanner"):
        mixed = experts[picks, np.arange(n), :]
    else:
        with np.errstate(divide="ignore"):
            log_routing = np.log(routing)
        mixed = _logsumexp(experts.transpose(1, 0, 2) + log_routing[:, :, None], axis=1)
    if not np.all(np.isfinite(mixed)):
        raise FloatingPointError("routed probabilities became nonfinite")
    return {"log_probabilities": mixed, "routing_weights": routing, "expert_pick": picks,
            "prediction": mixed.argmax(axis=1).astype(np.int64)}


def metrics_from_correct(correct, roles, topic):
    """Count global, role and topic outcomes without adding inferential claims."""
    import numpy as np

    correct, roles, topic = np.asarray(correct), np.asarray(roles), np.asarray(topic)
    if correct.ndim != 1 or correct.dtype != np.bool_ or roles.shape != correct.shape or topic.shape != correct.shape:
        raise ValueError("correctness, roles and topic arrays must align")
    if not len(correct) or not np.all(np.isin(roles, range(4))) or not np.all(np.isin(topic, range(3))):
        raise ValueError("scoring requires valid roles and topic IDs")

    def count(mask):
        rows = {}
        for index, name in enumerate(ROLES):
            selected = mask & (roles == index)
            n = int(selected.sum())
            if not n:
                raise ValueError("every scored topic/panel must contain every role")
            k = int(correct[selected].sum())
            rows[name] = {"n": n, "correct": k, "accuracy": k / n}
        n, k = int(mask.sum()), int(correct[mask].sum())
        return {"n": n, "correct": k, "accuracy": k / n, "per_role": rows,
                "macro_all_roles": sum(row["accuracy"] for row in rows.values()) / 4,
                "macro_agent_patient_accuracy": (rows["AGENTE"]["accuracy"] + rows["PACIENTE"]["accuracy"]) / 2}

    result = count(np.ones(len(correct), dtype=bool))
    result["per_topic"] = {str(index): count(topic == index) for index in range(3)}
    return result


def score_routing(routed, target, roles, topic):
    """Score only after routing; original labels never enter routing functions."""
    import numpy as np

    y = np.asarray(target)
    logp = np.asarray(routed["log_probabilities"])
    if y.shape != (len(logp),) or y.dtype.kind not in "iu" or np.any((y < 0) | (y >= logp.shape[1])):
        raise ValueError("evaluation targets are invalid")
    result = metrics_from_correct(routed["prediction"] == y, roles, topic)
    result["cross_entropy"] = float(-logp[np.arange(len(y)), y].mean())
    return result
