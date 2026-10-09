"""Independent protocol checks for partitioning and data random-state isolation."""
from itertools import product
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neuropixel.research.data import ResearchRoleTask, frozen_dataset  # noqa: E402
from neuropixel.task import NOUNS, ROLES, VERBS, RoleTask  # noqa: E402


def protocol_partition(seed):
    """An independent literal construction of the declared vocabulary product."""
    triples = [(agent, verb, patient) for agent, verb, patient in product(range(12), range(10), range(12))
               if agent != patient]
    order = torch.randperm(1320, generator=torch.Generator(device="cpu").manual_seed(seed),
                           device="cpu").tolist()
    shuffled = tuple(triples[i] for i in order)
    return {"test": shuffled[:264], "validation": shuffled[264:396], "train": shuffled[396:]}


@pytest.mark.parametrize("seed", [0, 101, 202])
def test_exact_partition_membership_order_disjointness_and_coverage(seed):
    task = ResearchRoleTask(seed=seed)
    expected = protocol_partition(seed)
    actual = {"train": task.train_triples, "validation": task.validation_triples, "test": task.test_triples}
    assert actual == expected
    assert {k: len(v) for k, v in actual.items()} == {"train": 924, "validation": 132, "test": 264}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert not set(actual[left]).intersection(actual[right])
    assert len(set().union(*[set(v) for v in actual.values()])) == 1320
    original = RoleTask(seed=seed)
    assert task.test_triples == tuple(original.test_triples)
    assert task.validation_triples + task.train_triples == tuple(original.train_triples)
    assert (task.h, task.w, task.query_pos, task.out_pos) == (8, 8, (7, 6), (7, 7))
    assert task.v.tokens == original.v.tokens
    assert torch.equal(task.role_ids, original.role_ids)


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_sampled_membership_and_truth_are_valid_independently_of_cache(split):
    task = ResearchRoleTask(seed=101)
    canvas, target, metadata = task.sample(512, split, torch.Generator().manual_seed(9002), meta=True)
    noun_index = {token: i for i, token in enumerate(task.v.ids(NOUNS))}
    verb_index = {token: i for i, token in enumerate(task.v.ids(VERBS))}
    expected = set(protocol_partition(101)[split])
    for b, (agent, verb, patient, _) in enumerate(metadata["fillers"].tolist()):
        assert (noun_index[agent], verb_index[verb], noun_index[patient]) in expected
        query_token = int(canvas[b, *task.query_pos])
        query_index = task.v.ids(ROLES).index(query_token)
        assert target[b] == metadata["fillers"][b, query_index]
        for role in range(4):
            row, col = metadata["rows"][b, role], metadata["cols"][b, role]
            assert canvas[b, row, col] == task.role_ids[role]
            assert canvas[b, row, col + 1] == metadata["fillers"][b, role]
    assert ((canvas != 0).sum((1, 2)) == 9).all()
    assert all(t.device.type == "cpu" for t in (canvas, target, *metadata.values()))


def test_switching_split_call_order_never_reuses_another_pool():
    task = ResearchRoleTask(seed=202)
    first = {}
    for split in ("test", "train", "validation", "train", "test", "validation"):
        sample = task.sample(128, split, torch.Generator().manual_seed(61001), meta=True)
        if split in first:
            assert torch.equal(sample[0], first[split][0])
            assert torch.equal(sample[1], first[split][1])
        else:
            first[split] = sample
        fresh = ResearchRoleTask(seed=202).sample(128, split, torch.Generator().manual_seed(61001), meta=True)
        assert torch.equal(sample[0], fresh[0])
        assert torch.equal(sample[2]["fillers"], fresh[2]["fillers"])
    triple_sets = {split: set(tuple(row[:3]) for row in data[2]["fillers"].tolist())
                   for split, data in first.items()}
    assert not triple_sets["validation"].intersection(triple_sets["test"])
    assert not triple_sets["train"].intersection(triple_sets["test"])
    assert not triple_sets["train"].intersection(triple_sets["validation"])


@pytest.mark.parametrize("query_role", [None, 0, 1, 2, 3])
def test_original_test_cpu_samples_and_metadata_are_preserved(query_role):
    original, research = RoleTask(seed=0), ResearchRoleTask(seed=0)
    old = original.sample(64, "test", torch.Generator().manual_seed(5), query_role=query_role, meta=True)
    new = research.sample(64, "test", torch.Generator().manual_seed(5), query_role=query_role, meta=True)
    assert torch.equal(old[0], new[0]) and torch.equal(old[1], new[1])
    assert all(torch.equal(old[2][key], new[2][key]) for key in old[2])


@pytest.mark.parametrize("split,n,seed", [("validation", 2048, 61001), ("test", 4096, 61002),
                                         ("train", 512, 61006)])
def test_frozen_protocol_sizes_order_and_independence_from_initialization_rng(split, n, seed):
    task = ResearchRoleTask(seed=0)
    original_state = torch.random.get_rng_state().clone()
    try:
        torch.manual_seed(7)
        before = torch.random.get_rng_state().clone()
        first = frozen_dataset(task, split, n, seed)
        assert torch.equal(torch.random.get_rng_state(), before)
        torch.manual_seed(987654)
        torch.randn(1234)  # Consume arbitrary global randomness, as initialization would.
        before_second = torch.random.get_rng_state().clone()
        second = frozen_dataset(task, split, n, seed)
        assert torch.equal(torch.random.get_rng_state(), before_second)
        assert all(torch.equal(a, b) for a, b in zip(first, second))
        canvas, target, roles = first
        assert canvas.shape == (n, 8, 8) and target.shape == roles.shape == (n,)
        assert torch.equal(roles, torch.arange(4).repeat_interleave(n // 4))
        assert torch.equal(canvas[:, *task.query_pos], task.role_ids[roles])
    finally:
        torch.random.set_rng_state(original_state)


def test_training_stream_is_independent_and_advances_only_its_private_generator():
    task = ResearchRoleTask(seed=0)
    stream = torch.Generator(device="cpu").manual_seed(9002)
    state = torch.random.get_rng_state().clone()
    first = task.sample(64, "train", stream)
    second = task.sample(64, "train", stream)
    assert torch.equal(torch.random.get_rng_state(), state)
    assert not torch.equal(first[0], second[0])
    replay = torch.Generator(device="cpu").manual_seed(9002)
    for expected in (first, second):
        actual = task.sample(64, "train", replay)
        assert all(torch.equal(a, b) for a, b in zip(expected, actual))


@pytest.mark.parametrize("n", [0, -4, 1, 5, 4.0, True])
def test_frozen_dataset_rejects_nonpositive_or_unbalanced_sizes(n):
    with pytest.raises(ValueError, match="positive integer multiple of four"):
        frozen_dataset(ResearchRoleTask(), "validation", n, 61001)


def test_missing_rng_unknown_split_and_invalid_role_are_rejected():
    task = ResearchRoleTask()
    with pytest.raises(ValueError, match="explicit independent CPU"):
        task.sample(4)
    with pytest.raises(ValueError, match="split must"):
        task.sample(4, "val", torch.Generator().manual_seed(0))
    with pytest.raises(ValueError, match="query_role"):
        task.sample(4, "train", torch.Generator().manual_seed(0), query_role=4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable; CPU generation path checked separately")
def test_cpu_and_cuda_destinations_receive_identical_inputs_and_metadata():
    task = ResearchRoleTask()
    cpu = task.sample(64, "validation", torch.Generator().manual_seed(61001), "cpu", meta=True)
    cuda = task.sample(64, "validation", torch.Generator().manual_seed(61001), "cuda", meta=True)
    assert torch.equal(cpu[0], cuda[0].cpu()) and torch.equal(cpu[1], cuda[1].cpu())
    assert all(torch.equal(cpu[2][key], cuda[2][key].cpu()) for key in cpu[2])
    cpu_fixed = frozen_dataset(task, "test", 64, 61002, "cpu")
    cuda_fixed = frozen_dataset(task, "test", 64, 61002, "cuda")
    assert all(torch.equal(a, b.cpu()) for a, b in zip(cpu_fixed, cuda_fixed))
    with pytest.raises(ValueError, match="CPU torch.Generator"):
        task.sample(4, "train", torch.Generator(device="cuda").manual_seed(9002), "cuda")
