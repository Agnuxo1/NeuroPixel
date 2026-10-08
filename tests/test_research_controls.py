"""Check the controls against independent fixtures and real generator truth."""
import gzip
import inspect
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neuropixel.research.controls import (  # noqa: E402
    adjacent_exact, category_probabilities, category_seeded_draw, same_row_exact,
    swap_agent_patient,
)
from neuropixel.task import ROLES, Vocab  # noqa: E402
from scripts.research_controls import (  # noqa: E402
    _prediction_file, _wilson_interval, content_hash, evaluate_dataset, sample_dataset,
)


def manual_canvas(kind):
    """A hand-authored case with truth independent of either reference solver."""
    vocab = Vocab()
    h, w = (8, 8) if kind == "adjacent" else (12, 12)
    canvas = torch.zeros(4, h, w, dtype=torch.long)
    words = ["perro", "mira", "robot", "casa"]
    role_columns = [1, 2, 0, 3]
    filler_columns = [col + 1 for col in role_columns] if kind == "adjacent" else [9, 6, 10, 8]
    for row, (role, word, role_col, filler_col) in enumerate(zip(ROLES, words, role_columns, filler_columns)):
        canvas[:, row, role_col] = vocab.idx[role]
        canvas[:, row, filler_col] = vocab.idx[word]
    canvas[:, h - 1, w - 2] = torch.tensor(vocab.ids(ROLES))
    return vocab, canvas, torch.tensor(vocab.ids(words))


@pytest.mark.parametrize("kind", ["adjacent", "far"])
def test_independent_fixtures_and_wrong_schema(kind):
    vocab, canvas, target = manual_canvas(kind)
    exact = adjacent_exact if kind == "adjacent" else same_row_exact
    assert torch.equal(exact(canvas, vocab=vocab), target)
    probabilities = category_probabilities(canvas, vocab=vocab)
    assert torch.equal(probabilities.sum(1), torch.ones(4, dtype=torch.float64))
    assert torch.equal(probabilities.gather(1, target[:, None]).flatten(),
                       torch.tensor([0.5, 1.0, 0.5, 1.0], dtype=torch.float64))
    if kind == "far":
        with pytest.raises(ValueError, match="adjacent schema"):
            adjacent_exact(canvas, vocab=vocab)


@pytest.mark.parametrize("kind", ["adjacent", "far"])
@pytest.mark.parametrize("seed", [0, 101, 202])
@pytest.mark.parametrize("split", ["train", "test"])
def test_exact_references_solve_original_generator_schemas(kind, seed, split):
    task, canvas, target, roles, _ = sample_dataset(kind, seed, split, 61004, examples_per_role=16)
    exact = adjacent_exact if kind == "adjacent" else same_row_exact
    assert torch.equal(exact(canvas, query_pos=task.query_pos, vocab=task.v), target)
    assert torch.equal(torch.bincount(roles, minlength=4), torch.full((4,), 16))
    assert not set(task.train_triples).intersection(task.test_triples)
    assert (len(task.train_triples), len(task.test_triples)) == (1056, 264)


def test_predictor_interfaces_cannot_receive_targets_or_generator_metadata():
    _, canvas, target = manual_canvas("adjacent")
    for predictor in (adjacent_exact, same_row_exact, category_probabilities, category_seeded_draw):
        params = inspect.signature(predictor).parameters
        assert not {"target", "targets", "task", "meta", "metadata", "fillers"}.intersection(params)
        assert not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        kwargs = {"seed": 61005} if predictor is category_seeded_draw else {}
        with pytest.raises(TypeError):
            predictor(canvas, target=target, **kwargs)
        with pytest.raises(TypeError):
            predictor(canvas, meta={"fillers": target}, **kwargs)
        before = predictor(canvas, **kwargs)
        target.fill_(0)  # A scoring label mutation cannot change a predictor's input.
        assert torch.equal(predictor(canvas, **kwargs), before)


@pytest.mark.parametrize("kind", ["adjacent", "far"])
@pytest.mark.parametrize("split", ["train", "test"])
def test_paired_swap_invalidates_the_category_shortcut(kind, split):
    task, canvas, target, roles, fillers = sample_dataset(kind, 101, split, 61004, examples_per_role=64)
    schema = {"query_pos": task.query_pos, "vocab": task.v}
    exact = adjacent_exact if kind == "adjacent" else same_row_exact
    original = canvas.clone()
    swapped = swap_agent_patient(canvas, **schema)
    assert torch.equal(canvas, original)
    assert torch.equal(canvas.flatten(1).sort(1).values, swapped.flatten(1).sort(1).values)
    assert torch.equal(canvas[:, *task.query_pos], swapped[:, *task.query_pos])
    # The oracle comes from generator truth, independently of the solver under test.
    changed_target = fillers[:, [2, 1, 0, 3]].gather(1, roles[:, None]).flatten()
    assert torch.equal(exact(swapped, **schema), changed_target)
    nominal = (roles == 0) | (roles == 2)
    assert (changed_target[nominal] != target[nominal]).all()
    assert torch.equal(changed_target[~nominal], target[~nominal])
    assert torch.equal(category_probabilities(canvas, **schema), category_probabilities(swapped, **schema))
    pred = category_seeded_draw(canvas, seed=61005, **schema)
    changed_pred = category_seeded_draw(swapped, seed=61005, **schema)
    assert torch.equal(pred, changed_pred)
    a, b = pred[nominal] == target[nominal], changed_pred[nominal] == changed_target[nominal]
    assert torch.logical_xor(a, b).all()
    assert not (a & b).any()


def test_seeded_draw_and_dataset_hash_are_reproducible():
    first = sample_dataset("far", 202, "test", 61004, examples_per_role=32)
    second = sample_dataset("far", 202, "test", 61004, examples_per_role=32)
    changed = sample_dataset("far", 202, "test", 61006, examples_per_role=32)
    h1 = content_hash(canvas=first[1], target=first[2], query_role_indices=first[3])
    h2 = content_hash(canvas=second[1], target=second[2], query_role_indices=second[3])
    h3 = content_hash(canvas=changed[1], target=changed[2], query_role_indices=changed[3])
    assert h1 == h2 and h1 != h3
    state = torch.random.get_rng_state().clone()
    p1 = category_seeded_draw(first[1], seed=61005)
    assert torch.equal(torch.random.get_rng_state(), state)
    torch.manual_seed(123456)
    p2 = category_seeded_draw(second[1], seed=61005)
    assert torch.equal(p1, p2)


def test_per_role_wilson_interval_uses_binary_counts():
    lo, hi = _wilson_interval(50, 100)
    assert lo == pytest.approx(0.4038315303659956)
    assert hi == pytest.approx(0.5961684696340044)
    assert _wilson_interval(0, 1024)[0] == 0.0
    assert _wilson_interval(1024, 1024)[1] == 1.0
    with pytest.raises(ValueError):
        _wilson_interval(1, 0)


def test_prediction_artifact_is_deterministic_and_preserves_examples(tmp_path):
    rows = [{"example_id": "a", "target": 3, "prediction": 3},
            {"example_id": "b", "target": 4, "prediction": 1}]
    a, b = tmp_path / "a.gz", tmp_path / "b.gz"
    assert _prediction_file(a, iter(rows)) == _prediction_file(b, iter(rows))
    assert a.read_bytes() == b.read_bytes()
    with gzip.open(a, "rt", encoding="utf-8") as f:
        assert [json.loads(line) for line in f] == rows


def test_interrupted_artifact_write_preserves_previous_complete_destination(tmp_path):
    path = tmp_path / "prediction.jsonl.gz"
    _prediction_file(path, [{"example_id": "complete", "target": 3}])
    before = path.read_bytes()

    def interrupted_rows():
        yield {"example_id": "partial", "target": 4}
        raise RuntimeError("simulated interrupted export")

    with pytest.raises(RuntimeError, match="simulated interrupted export"):
        _prediction_file(path, interrupted_rows())
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["prediction.jsonl.gz"]


def test_runner_scores_truth_separately_and_records_counterfactual_partitions(tmp_path):
    spec = {"id": "test_small", "panel": "unit_test_only", "task_type": "far", "split_seed": 0,
            "split": "test", "sampling_seed": 61004, "examples_per_role": 16}
    result = evaluate_dataset(spec, 61005, 0.95, tmp_path)
    assert result["acceptance_passed"]
    assert result["exact"]["correct"] == result["exact"]["n"] == 64
    assert result["category_expected"]["expected_correct"] == 48
    assert result["category_expected"]["macro_agent_patient_accuracy"] == 0.5
    path = tmp_path / result["prediction_artifact"]["path"]
    with gzip.open(path, "rt", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    assert len(rows) == 64
    swaps = [row["agent_patient_swap"] for row in rows if "agent_patient_swap" in row]
    assert len(swaps) == 32
    for row in rows:
        assert row["triple_membership"] == "test"
        if "agent_patient_swap" in row:
            pair = row["agent_patient_swap"]
            assert pair["triple_membership"] in {"train", "test"}
            assert pair["target"] != row["target"]
            assert pair["category_seeded_prediction"] == row["category_seeded_prediction"]
    for role in ("AGENTE", "PACIENTE"):
        metrics = result["counterfactual"]["per_role"][role]
        assert sum(metrics["partition_transitions"].values()) == metrics["pairs"]
