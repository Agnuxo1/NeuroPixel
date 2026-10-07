"""Independent, standard-library two-event binding grammar for item 9.

This module defines data and input-only controls, not an access-control gate.
The study controller must prohibit final generation until its frozen gate opens.
No historical task, model, numerical runtime, or global RNG is imported or used.
"""
from __future__ import annotations

import hashlib
import json
import math
import random


HEIGHT, WIDTH, VOCAB_SIZE = 10, 8, 37
ROLES = ("agent", "action", "patient", "place")
ROLE_IDS = (1, 2, 3, 4)
NOUN_IDS = tuple(range(5, 17))
VERB_IDS = tuple(range(17, 27))
PLACE_IDS = tuple(range(27, 35))
EVENT_IDS = (35, 36)
OUT_POS = (9, 7)
CONDITIONS = ("base", "swap_queried_agent_patient", "swap_other_agent_patient",
              "relabel_events", "query_switch", "layout_permutation")
CONTROLS = ("symbolic", "role_only", "event_category", "bag_category")
GROUP_UNIVERSE_SIZE = math.comb(12, 4) * math.comb(10, 2) * math.comb(8, 2)


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _digest(value):
    return hashlib.sha256(_bytes(value)).hexdigest()


def _index(value, limit, label):
    if type(value) is not int or not 0 <= value < limit:
        raise ValueError(f"{label} must be an integer in [0, {limit})")
    return value


def canonical_group(bag):
    """Return the sorted category bag; assignment, layout and queries are absent."""
    if not isinstance(bag, dict) or set(bag) != {"nouns", "verbs", "places"}:
        raise ValueError("bag must contain exactly nouns, verbs and places")
    result = {}
    for name, size, allowed in (("nouns", 4, NOUN_IDS), ("verbs", 2, VERB_IDS),
                                ("places", 2, PLACE_IDS)):
        values = bag[name]
        if (not isinstance(values, (list, tuple)) or len(values) != size
                or any(type(x) is not int or x not in allowed for x in values)
                or len(set(values)) != size):
            raise ValueError(f"{name} must contain {size} distinct category token IDs")
        result[name] = sorted(values)
    return result


def group_id(bag):
    """SHA-256 of compact, sorted-key canonical category-bag JSON."""
    return _digest(canonical_group(bag))


def split_for_group(bag):
    """Hash buckets 0..69/70..84/85..99; proportions are not exact counts."""
    bucket = int(group_id(bag), 16) % 100
    return "train" if bucket < 70 else "validation" if bucket < 85 else "final"


def _category(role):
    return NOUN_IDS if role in (0, 2) else VERB_IDS if role == 1 else PLACE_IDS


def _layout(rng):
    return [{"row": row, "start": rng.randrange(6)} for row in rng.sample(range(9), 8)]


def _scenario_from_bag(bag, rng):
    bag = canonical_group(bag)
    nouns, verbs, places = (rng.sample(bag[name], len(bag[name]))
                            for name in ("nouns", "verbs", "places"))
    facts = []
    for event in range(2):
        fillers = (nouns[2 * event], verbs[event], nouns[2 * event + 1], places[event])
        facts.extend({"event": event, "role": role, "filler": filler}
                     for role, filler in enumerate(fillers))
    return {"schema_version": 1, "group_id": group_id(bag), "bag": bag,
            "facts": facts, "layout": _layout(rng)}


def validate_scenario(scenario):
    """Validate the semantic bag, eight facts and nonoverlapping spatial slots."""
    required = {"schema_version", "group_id", "bag", "facts", "layout"}
    if not isinstance(scenario, dict) or set(scenario) != required or scenario["schema_version"] != 1:
        raise ValueError("scenario schema differs from version 1")
    bag = canonical_group(scenario["bag"])
    if scenario["group_id"] != group_id(bag):
        raise ValueError("scenario group_id differs from its category bag")
    facts, layout = scenario["facts"], scenario["layout"]
    if not isinstance(facts, list) or len(facts) != 8 or not isinstance(layout, list) or len(layout) != 8:
        raise ValueError("scenario requires eight facts and eight layout slots")
    keys, observed, rows = set(), {"nouns": [], "verbs": [], "places": []}, set()
    for fact, slot in zip(facts, layout):
        if not isinstance(fact, dict) or set(fact) != {"event", "role", "filler"}:
            raise ValueError("invalid fact fields")
        event, role = _index(fact["event"], 2, "event"), _index(fact["role"], 4, "role")
        if (event, role) in keys:
            raise ValueError("duplicate event/role fact")
        keys.add((event, role))
        filler = fact["filler"]
        if type(filler) is not int or filler not in _category(role):
            raise ValueError("fact filler does not match its role category")
        observed["nouns" if role in (0, 2) else "verbs" if role == 1 else "places"].append(filler)
        if not isinstance(slot, dict) or set(slot) != {"row", "start"}:
            raise ValueError("invalid layout fields")
        row = _index(slot["row"], 9, "fact row")
        _index(slot["start"], 6, "triple start")
        if row in rows:
            raise ValueError("each fact must occupy a distinct row")
        rows.add(row)
    if canonical_group(observed) != bag:
        raise ValueError("fact fillers differ from the declared bag")
    return scenario


def generate_scenarios(split, count, seed, *, max_attempts=None, excluded_groups=()):
    """Sample distinct group bags, then assignments/layouts, using a private RNG.

    The explicit rejection bound prevents hanging when a request is impossible
    or almost exhausts a partition. This is not a full-universe enumerator.
    Python's runtime/version must be recorded by the execution controller.
    """
    return _generate_scenarios(split, count, seed, max_attempts, excluded_groups, None)


def _generate_scenarios(split, count, seed, max_attempts, excluded_groups, trace):
    if split not in ("train", "validation", "final"):
        raise ValueError("split must be train, validation or final")
    if type(count) is not int or not 0 < count <= GROUP_UNIVERSE_SIZE:
        raise ValueError("count must be positive and no greater than the group universe")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if not isinstance(excluded_groups, (list, tuple, set, frozenset)):
        raise ValueError("excluded_groups must be a collection of SHA-256 strings")
    if any(not isinstance(value, str) or len(value) != 64
           or any(char not in "0123456789abcdef" for char in value) for value in excluded_groups):
        raise ValueError("excluded group IDs must be lowercase SHA-256 hex strings")
    excluded = set(excluded_groups)
    if max_attempts is None:
        max_attempts = max(1000, count * 100)
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    rng, seen, scenarios = random.Random(seed), set(), []
    for _ in range(max_attempts):
        bag = {"nouns": rng.sample(NOUN_IDS, 4), "verbs": rng.sample(VERB_IDS, 2),
               "places": rng.sample(PLACE_IDS, 2)}
        identity = group_id(bag)
        if trace is not None:
            trace["candidate_group_ids"].append(identity)
        if identity in excluded or identity in seen or split_for_group(bag) != split:
            continue
        seen.add(identity)
        scenarios.append(_scenario_from_bag(bag, rng))
        if trace is not None:
            trace["constructed_group_ids"].append(identity)
        if len(scenarios) == count:
            return scenarios
    raise RuntimeError(f"scenario rejection bound exhausted: {len(scenarios)}/{count} accepted")


def fixture_exposure_ledger():
    """Declare the small deterministic test inventory, including rejected bags.

    This function reconstructs fixture-only draws; it never uses study seeds.
    Excluding its union is conservative: many candidate bags were inspected only
    for membership and never rendered. It does not establish external blindness.
    """
    manual_bag = {"nouns": [5, 6, 7, 8], "verbs": [17, 18], "places": [27, 28]}
    specifications = [("train", 5, 910901, None), ("train", 7, 910910, None),
                      ("validation", 7, 910911, None), ("final", 7, 910912, None),
                      ("train", 2, 910930, 1)]
    runs, all_candidates, all_constructed = [], {group_id(manual_bag)}, {group_id(manual_bag)}
    first_group = None
    for split, count, seed, bound in specifications:
        trace = {"candidate_group_ids": [], "constructed_group_ids": []}
        status = "completed"
        try:
            scenarios = _generate_scenarios(split, count, seed, bound, (), trace)
            if first_group is None:
                first_group = scenarios[0]["group_id"]
        except RuntimeError:
            if bound != 1 or count != 2:
                raise
            status = "expected_rejection_bound_failure"
        runs.append({"split": split, "count": count, "seed": seed,
                     "max_attempts": bound, "excluded_groups": [], "status": status, **trace})
        all_candidates.update(trace["candidate_group_ids"])
        all_constructed.update(trace["constructed_group_ids"])
    # The exclusion regression changes RNG consumption after skipping one bag;
    # list its distinct candidate stream explicitly instead of assuming overlap.
    trace = {"candidate_group_ids": [], "constructed_group_ids": []}
    _generate_scenarios("train", 4, 910901, None, [first_group], trace)
    runs.append({"split": "train", "count": 4, "seed": 910901, "max_attempts": None,
                 "excluded_groups": [first_group], "status": "completed", **trace})
    all_candidates.update(trace["candidate_group_ids"])
    all_constructed.update(trace["constructed_group_ids"])
    return {"schema_version": 1, "scope": "item-9 deterministic unit fixtures, not performance populations",
            "manual_group": manual_bag, "manual_group_id": group_id(manual_bag),
            "generation_runs": runs, "randomization_only_seeds": [910920],
            "global_rng_independence_test_seed": 910902,
            "constructed_group_ids": sorted(all_constructed),
            "excluded_groups": sorted(all_candidates),
            "exclusion_scope": "all fixture candidate bags, including rejected membership-only draws"}


def randomize_scenario(scenario, rng):
    """Resample assignment and layout within the same bag, without mutating it."""
    validate_scenario(scenario)
    if not isinstance(rng, random.Random):
        raise ValueError("an explicit random.Random instance is required")
    return _scenario_from_bag(scenario["bag"], rng)


def _canvas(facts, layout, query_event, query_role):
    canvas = [[0] * WIDTH for _ in range(HEIGHT)]
    for fact, slot in zip(facts, layout):
        row, start = slot["row"], slot["start"]
        canvas[row][start:start + 3] = [EVENT_IDS[fact["event"]], ROLE_IDS[fact["role"]], fact["filler"]]
    canvas[9][5:7] = [EVENT_IDS[query_event], ROLE_IDS[query_role]]
    return canvas


def _parse_canvas(canvas):
    if (not isinstance(canvas, (list, tuple)) or len(canvas) != HEIGHT
            or any(not isinstance(row, (list, tuple)) or len(row) != WIDTH for row in canvas)):
        raise ValueError("canvas must have shape 10 by 8")
    if any(type(token) is not int or not 0 <= token < VOCAB_SIZE for row in canvas for token in row):
        raise ValueError("canvas tokens must be integer IDs in [0, 37)")
    query = canvas[9]
    if (any(query[c] != 0 for c in (0, 1, 2, 3, 4, 7))
            or query[5] not in EVENT_IDS or query[6] not in ROLE_IDS):
        raise ValueError("last row must contain only query event/role at columns 5/6 and PAD output")
    facts, rows = {}, 0
    observed = {"nouns": [], "verbs": [], "places": []}
    for row in canvas[:9]:
        occupied = [column for column, token in enumerate(row) if token != 0]
        if not occupied:
            continue
        rows += 1
        if len(occupied) != 3 or occupied != list(range(occupied[0], occupied[0] + 3)):
            raise ValueError("each occupied fact row must contain exactly one contiguous triple")
        event_token, role_token, filler = [row[column] for column in occupied]
        if event_token not in EVENT_IDS or role_token not in ROLE_IDS:
            raise ValueError("facts must have event, role, filler order")
        event, role = EVENT_IDS.index(event_token), ROLE_IDS.index(role_token)
        if filler not in _category(role):
            raise ValueError("visible filler does not match its role category")
        if (event, role) in facts:
            raise ValueError("duplicate visible event/role fact")
        facts[event, role] = filler
        observed["nouns" if role in (0, 2) else "verbs" if role == 1 else "places"].append(filler)
    if rows != 8 or set(facts) != {(event, role) for event in range(2) for role in range(4)}:
        raise ValueError("canvas must contain every event/role exactly once")
    canonical_group(observed)
    return facts, EVENT_IDS.index(query[5]), ROLE_IDS.index(query[6])


def symbolic_answer(canvas):
    """Read gold solely from visible triples and the visible event/role query."""
    facts, event, role = _parse_canvas(canvas)
    return facts[event, role]


def render_scenario(scenario, event_idx, role_idx, condition="base", layout_seed=None):
    """Render a paired intervention; preserve both anchor and effective query IDs.

    Layouts are shared across all eight queries. The layout intervention is also
    shared across queries, and is guaranteed to change the rendered fact grid.
    Identity across some other condition/query pairs is intentional dependence.
    """
    validate_scenario(scenario)
    event_idx, role_idx = _index(event_idx, 2, "query event"), _index(role_idx, 4, "query role")
    if condition not in CONDITIONS:
        raise ValueError("unknown condition")
    if layout_seed is not None and type(layout_seed) is not int:
        raise ValueError("layout_seed must be an integer or None")
    facts, layout = [dict(x) for x in scenario["facts"]], [dict(x) for x in scenario["layout"]]
    base_canvas = _canvas(facts, layout, event_idx, role_idx)
    query_event = event_idx
    if condition in ("swap_queried_agent_patient", "swap_other_agent_patient"):
        affected = event_idx if condition == "swap_queried_agent_patient" else 1 - event_idx
        by_key = {(fact["event"], fact["role"]): fact for fact in facts}
        agent, patient = by_key[affected, 0], by_key[affected, 2]
        agent["filler"], patient["filler"] = patient["filler"], agent["filler"]
    elif condition == "relabel_events":
        for fact in facts:
            fact["event"] = 1 - fact["event"]
        query_event = 1 - event_idx
    elif condition == "query_switch":
        query_event = 1 - event_idx
    elif condition == "layout_permutation":
        seed = int(_digest({"scenario": scenario, "purpose": "layout_permutation"}), 16) if layout_seed is None else layout_seed
        layout = _layout(random.Random(seed))
        if layout == scenario["layout"]:
            layout = layout[1:] + layout[:1]
    canvas = _canvas(facts, layout, query_event, role_idx)
    target, base_target = symbolic_answer(canvas), symbolic_answer(base_canvas)
    scenario_id = _digest(scenario)
    pair_id = _digest({"scenario_id": scenario_id, "base_query_event": event_idx, "query_role": role_idx})
    return {"record_id": _digest({"pair_id": pair_id, "condition": condition, "canvas": canvas}),
            "scenario_id": scenario_id, "pair_id": pair_id, "group_id": scenario["group_id"],
            "canvas": canvas, "target": target, "base_target": base_target,
            "base_query_event": event_idx, "query_event": query_event, "query_role": role_idx,
            "condition": condition, "changed_gold": target != base_target}


def control_weights(canvas, control):
    """Return an input-only probability distribution over visible filler tokens.

    role_only ignores event membership when selecting two same-role fillers.
    event_category ignores noun-role assignment within the queried event.
    bag_category ignores event and role binding and uses only filler category.
    Parsing validates the grammar; none of these selectors takes labels/metadata.
    """
    if control not in CONTROLS:
        raise ValueError("unknown control")
    facts, event, role = _parse_canvas(canvas)
    if control == "symbolic":
        candidates = [facts[event, role]]
    elif control == "role_only":
        candidates = [filler for (_, fact_role), filler in facts.items() if fact_role == role]
    else:
        candidates = [filler for (fact_event, _), filler in facts.items()
                      if filler in _category(role) and (control == "bag_category" or fact_event == event)]
    candidates = sorted(candidates)
    return {token: 1 / len(candidates) for token in candidates}
