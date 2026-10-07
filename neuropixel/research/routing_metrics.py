"""Descriptive summaries of hard routing assignments, independent of Torch.

Slot IDs are nominal labels. Their arithmetic mean is not a routing frequency,
chosen expert, prediction accuracy, or ownership-label agreement.
"""
from __future__ import annotations

from collections.abc import Mapping
import math


def _integer(value, name, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}; bool is invalid")
    return value


def _distribution(picks, n_slots):
    counts = [0] * n_slots
    for pick in picks:
        counts[pick] += 1
    n = len(picks)
    modes = [slot for slot, count in enumerate(counts) if count == max(counts)]
    return {"n": n, "counts": counts, "fractions": [count / n for count in counts],
            "modal_slot_ids": modes, "unique_modal_slot_id": modes[0] if len(modes) == 1 else None}


def summarize_routing(picks, n_slots, *, topics=None, correct_slot_by_topic=None,
                      argmax_scores=None):
    """Count one assigned slot per example; reject empty or invalid inputs.

    ``picks`` contains integer IDs in ``range(n_slots)``; unused slots remain
    explicit zeros. ``topics`` is optional aligned, genuinely known integer
    metadata. No topic labels or correct slots are inferred from assignments.

    ``correct_slot_by_topic`` is an optional caller-supplied ownership mapping.
    It must cover every observed topic; valid additional topics are allowed.
    Its metric is mapped-slot agreement, not prediction accuracy or an oracle.

    Optional ``argmax_scores`` is example-by-slot. Finite scores must reproduce
    each supplied pick using the first maximum in slot order, matching the
    original driver's argmax. Exact score ties are then counted; without scores
    their number is unknown. A modal-frequency tie never selects an arbitrary
    representative. Renaming a fixed assignment permutes its counts, whereas
    the first-maximum routing decision itself is order-sensitive at score ties.

    To pool unequal batches, concatenate their assignments and metadata. All
    fractions then use example counts, not an unweighted average of batch rates.
    """
    _integer(n_slots, "n_slots", minimum=1)
    picks = list(picks)
    if not picks:
        raise ValueError("routing summaries require at least one example")
    for pick in picks:
        _integer(pick, "slot ID")
        if pick >= n_slots:
            raise ValueError("slot ID is outside the declared slot inventory")
    if topics is not None:
        topics = list(topics)
        if len(topics) != len(picks):
            raise ValueError("topics must align with all routing assignments")
        for topic in topics:
            _integer(topic, "topic ID")
    mapping = None
    if correct_slot_by_topic is not None:
        if topics is None or not isinstance(correct_slot_by_topic, Mapping):
            raise ValueError("mapped-slot agreement requires topics and an explicit mapping")
        mapping = dict(correct_slot_by_topic)
        for topic, slot in mapping.items():
            _integer(topic, "mapped topic ID")
            _integer(slot, "mapped slot ID")
            if slot >= n_slots:
                raise ValueError("mapped slot ID is outside the declared inventory")
        if set(topics) - set(mapping):
            raise ValueError("ownership mapping is missing an observed topic")
    score_ties = None
    if argmax_scores is not None:
        scores = list(argmax_scores)
        if len(scores) != len(picks):
            raise ValueError("argmax_scores must align with all routing assignments")
        score_ties = []
        for pick, row in zip(picks, scores):
            row = list(row)
            if len(row) != n_slots:
                raise ValueError("every score row must match the slot inventory")
            if any(type(value) not in (int, float) or not math.isfinite(value) for value in row):
                raise ValueError("argmax scores must be finite numeric values, excluding bool")
            maximum = max(row)
            if row.index(maximum) != pick:
                raise ValueError("pick disagrees with first-maximum argmax of supplied scores")
            score_ties.append(sum(value == maximum for value in row) > 1)
    result = {"schema_version": 1, "kind": "hard_assignment_counts", "n_slots": n_slots,
              **_distribution(picks, n_slots), "by_topic": None,
              "argmax_ties": None, "mapped_slot_agreement": None}
    if score_ties is not None:
        tied = sum(score_ties)
        result["argmax_ties"] = {"count": tied, "n": len(picks), "fraction": tied / len(picks),
                                 "definition": "multiple exactly equal maximum scores",
                                 "tie_break": "first maximum in slot order"}
    if topics is not None:
        result["by_topic"] = []
        for topic in sorted(set(topics)):
            indices = [i for i, value in enumerate(topics) if value == topic]
            row = {"topic_id": topic, **_distribution([picks[i] for i in indices], n_slots)}
            if score_ties is not None:
                row["argmax_tie_count"] = sum(score_ties[i] for i in indices)
            if mapping is not None:
                matched = row["counts"][mapping[topic]]
                row["mapped_slot_agreement"] = {"mapped_slot_id": mapping[topic],
                                                "matching_count": matched, "n": row["n"],
                                                "fraction": matched / row["n"]}
            result["by_topic"].append(row)
    if mapping is not None:
        matched = sum(pick == mapping[topic] for pick, topic in zip(picks, topics))
        result["mapped_slot_agreement"] = {
            "mapping": [{"topic_id": topic, "slot_id": mapping[topic]} for topic in sorted(mapping)],
            "matching_count": matched, "n": len(picks), "fraction": matched / len(picks),
            "meaning": "Agreement with supplied ownership labels; not prediction accuracy or oracle optimality."}
    return result
