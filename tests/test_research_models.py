"""Behavioral checks for the item-4 neural references; no fitting experiment."""
from pathlib import Path
import sys

import pytest
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neuropixel.model import NeuroPixel, n_params  # noqa: E402
from neuropixel.research.models import (  # noqa: E402
    ResearchNCA, RelativeSelfAttention, RelativeTransformer, build_model,
    grid_coordinates, model_config, relative_position_index,
)
from neuropixel.task import RoleTask  # noqa: E402


FAMILIES = ("neuropixel", "standard_nca", "convgru", "relative_transformer")
COUNTS = {"neuropixel": 29824, "standard_nca": 30384,
          "convgru": 27835, "relative_transformer": 29547}


@pytest.fixture(scope="module", autouse=True)
def limited_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def input_batch(batch=2):
    task = RoleTask(8, 8, seed=0)
    canvas, target = task.sample(batch, generator=torch.Generator().manual_seed(912))
    return task, canvas, target


def nonzero_dynamics(model):
    # Zero-initialized updates would make an equivalence check largely vacuous.
    with torch.no_grad():
        model.f2.weight.normal_(0, 0.025)
        model.f2.bias.normal_(0, 0.01)


@pytest.mark.parametrize("camera_mode", ["none", "direct", "retina"])
def test_research_nca_matches_legacy_state_logits_activity_and_trace(camera_mode):
    torch.manual_seed(91)
    task, canvas, _ = input_batch()
    options = {"grounded": task.v.grounded(), "retina": camera_mode == "retina"}
    legacy = NeuroPixel(len(task.v), task.out_pos, steps=8, **options).eval()
    nonzero_dynamics(legacy)
    with torch.no_grad():
        legacy.embed.weight[0].fill_(0.125)  # Preserve a nonzero historical PAD row too.
    research = ResearchNCA(len(task.v), task.out_pos, steps=8, **options).eval()
    research.load_state_dict(legacy.state_dict(), strict=True)
    kwargs = {"trace": True, "lens_every": 4}
    if camera_mode != "none":
        camera = torch.zeros_like(canvas, dtype=torch.bool)
        camera[:, :2, :2] = True
        canvas = canvas.clone()
        canvas[camera] = 0
        kwargs.update(rgb=torch.rand(len(canvas), 3, 8, 8) * 2 - 1, cam=camera)
    mask = torch.ones(len(canvas), 1, 8, 8)
    mask[:, :, ::2, ::2] = 0
    kwargs["hook"] = lambda t, state: state * mask if t == 4 else state
    with torch.no_grad():
        expected = legacy(canvas, **kwargs)
        actual = research(canvas, **kwargs)
    for key in ("state", "logits", "activity", "frames", "lens"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert actual["frames"].shape == (len(canvas), 9, 48, 8, 8)


def test_lens_checkpoints_are_pre_update_not_post_update():
    torch.manual_seed(93)
    _, canvas, _ = input_batch()
    model = build_model("neuropixel", steps=8).eval()
    nonzero_dynamics(model)
    with torch.no_grad():
        result = model(canvas, trace=True, lens_every=4)
        for index, time in enumerate((3, 7)):
            expected = model.lens_logits(result["frames"][:, time])
            torch.testing.assert_close(result["lens"][:, index], expected, rtol=0, atol=0)
        after_update = model.lens_logits(result["frames"][:, 4])
    assert not torch.allclose(result["lens"][:, 0], after_update)


def test_no_reinjection_preserves_seed_and_removes_only_update_identity():
    torch.manual_seed(95)
    _, canvas, _ = input_batch()
    original = build_model("neuropixel", steps=3).eval()
    nonzero_dynamics(original)
    no_reinjection = build_model("neuropixel", steps=3, reinject=False).eval()
    no_reinjection.load_state_dict(original.state_dict())
    update_inputs = []
    handle = no_reinjection.f1.register_forward_pre_hook(
        lambda module, args: update_inputs.append(args[0].detach().clone()))
    with torch.no_grad():
        original_result = original(canvas, trace=True)
        ablated_result = no_reinjection(canvas, trace=True)
    handle.remove()
    torch.testing.assert_close(original_result["frames"][:, 0], ablated_result["frames"][:, 0])
    assert ablated_result["frames"][:, 0].abs().sum() > 0
    assert all(torch.count_nonzero(value[:, -16:]) == 0 for value in update_inputs)
    assert not torch.allclose(original_result["state"], ablated_result["state"])


def test_untied_dictionary_starts_as_exact_clone_with_separate_gradient_routes():
    torch.manual_seed(97)
    model = build_model("neuropixel", tied=False)
    torch.testing.assert_close(model.output_dictionary, model.dictionary(), rtol=0, atol=0)
    assert model.output_dictionary.data_ptr() != model.embed.weight.data_ptr()
    state = torch.randn(2, 48, 3, 3)
    model.lens_logits(state).square().mean().backward()
    assert model.output_dictionary.grad is not None and model.output_dictionary.grad.abs().sum() > 0
    assert model.embed.weight.grad is None
    model.zero_grad(set_to_none=True)
    model.dictionary().square().sum().backward()
    assert model.embed.weight.grad is not None and model.embed.weight.grad.abs().sum() > 0
    assert model.output_dictionary.grad is None
    before = model.dictionary().detach().clone()
    with torch.no_grad():
        model.output_dictionary[1].add_(2)
    torch.testing.assert_close(model.dictionary(), before, rtol=0, atol=0)


@pytest.mark.parametrize("tied", [False, True])
def test_explicit_pad_policy_is_independent_of_parent_dictionary(monkeypatch, tied):
    legacy_policy = build_model("neuropixel", tied=tied, freeze_pad=False)
    frozen_policy = build_model("neuropixel", tied=tied, freeze_pad=True)
    for model in (legacy_policy, frozen_policy):
        with torch.no_grad():
            model.embed.weight[0].fill_(7)
            if not tied:
                model.output_dictionary[0].fill_(9)
    # A subsequent implementation change to the parent must not alter this recipe.
    monkeypatch.setattr(NeuroPixel, "dictionary", lambda self: torch.zeros_like(self.embed.weight))
    torch.testing.assert_close(legacy_policy.dictionary()[0], torch.full((16,), 7.0))
    assert frozen_policy.dictionary()[0].count_nonzero() == 0
    assert frozen_policy.decoder_dictionary()[0].count_nonzero() == 0
    loss = frozen_policy.dictionary().sum() + frozen_policy.decoder_dictionary().sum()
    loss.backward()
    assert frozen_policy.embed.weight.grad[0].count_nonzero() == 0
    if not tied:
        assert frozen_policy.output_dictionary.grad[0].count_nonzero() == 0


@pytest.mark.parametrize("family", FAMILIES)
def test_all_families_have_finite_forward_and_backward(family):
    torch.manual_seed(99)
    _, canvas, target = input_batch()
    model = build_model(family, steps=4)
    result = model(canvas)
    assert result["logits"].shape == (len(canvas), 35)
    assert result["state"].shape[0] == len(canvas)
    for name in ("logits", "state", "activity"):
        assert torch.isfinite(result[name]).all()
    loss = F.cross_entropy(result["logits"], target)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(grad).all() for grad in gradients)
    assert any(grad.count_nonzero() > 0 for grad in gradients)
    assert torch.all(result["logits"][:, 0] == -1e4)


@pytest.mark.parametrize("family", FAMILIES)
def test_model_metadata_records_actual_count_without_advancing_rng(family):
    torch.manual_seed(101)
    before = torch.random.get_rng_state().clone()
    config = model_config(family)
    assert torch.equal(torch.random.get_rng_state(), before)
    assert config["parameters"] == COUNTS[family] == n_params(build_model(family))
    assert config["primary_supervision"] == "answer_cross_entropy_only"
    assert config["dropout"] == 0
    assert config["recurrent_steps"] == (None if family == "relative_transformer" else 16)


def test_model_metadata_forces_cpu_even_inside_another_default_device(monkeypatch):
    import neuropixel.research.models as models
    original_factory = models.build_model
    observed_devices = []

    def observed_factory(*args, **kwargs):
        result = original_factory(*args, **kwargs)
        observed_devices.extend(parameter.device.type for parameter in result.parameters())
        return result

    monkeypatch.setattr(models, "build_model", observed_factory)
    before = torch.random.get_rng_state().clone()
    # A meta default exercises device override without needing a GPU or its RNG.
    with torch.device("meta"):
        config = models.model_config("relative_transformer")
    assert observed_devices and set(observed_devices) == {"cpu"}
    assert config["parameters"] == COUNTS["relative_transformer"]
    assert torch.equal(torch.random.get_rng_state(), before)


def test_relative_index_has_explicit_row_column_and_key_minus_query_orientation():
    coordinates = grid_coordinates(3, 4)
    index = relative_position_index(3, 4)
    assert coordinates[6].tolist() == [1, 2]
    assert index[0, 6].item() == 26  # (dr=+1, dc=+2): (1+2)*7+(2+3).
    assert index[6, 0].item() == 8   # (dr=-1, dc=-2).
    assert index[0, 1].item() == 18  # One column to the right.
    assert index[0, 4].item() == 24  # One row down.
    assert torch.all(index.diag() == 17)
    permutation = torch.tensor([11, 0, 7, 2, 5, 4, 3, 9, 8, 6, 1, 10])
    permuted = relative_position_index(3, 4, coordinates[permutation])
    torch.testing.assert_close(permuted, index[permutation][:, permutation], rtol=0, atol=0)


def test_relative_encoder_permutation_preserves_coordinates_and_output_identity():
    torch.manual_seed(103)
    model = RelativeTransformer(35, 3, 4, d=16, heads=4, ff=32).eval()
    with torch.no_grad():
        for layer in model.layers:
            layer.attention.relative_bias.normal_(0, 0.3)
    tokens = torch.randint(1, 35, (2, 12))
    tokens[:, [0, 3, 11]] = 0
    permutation = torch.tensor([11, 0, 7, 2, 5, 4, 3, 9, 8, 6, 1, 10])
    coordinates = grid_coordinates(3, 4)
    with torch.no_grad():
        expected = model.encode_tokens(tokens)
        actual = model.encode_tokens(tokens[:, permutation], coordinates[permutation])
        wrong_coordinates = model.encode_tokens(tokens[:, permutation])
        ordinary = model(tokens.reshape(2, 3, 4))
    torch.testing.assert_close(actual, expected[:, permutation], rtol=1e-5, atol=1e-6)
    assert not torch.allclose(wrong_coordinates, expected[:, permutation])
    expected_logits = model.head(expected[:, 11])
    expected_logits[:, 0] = -1e4
    torch.testing.assert_close(ordinary["logits"], expected_logits)
    output_in_permutation = int((permutation == 11).nonzero()[0])
    permuted_logits = model.head(actual[:, output_in_permutation])
    permuted_logits[:, 0] = -1e4
    torch.testing.assert_close(ordinary["logits"], permuted_logits, rtol=1e-5, atol=1e-6)


def test_attention_masks_empty_keys_and_handles_all_empty_input_without_nan():
    torch.manual_seed(105)
    attention = RelativeSelfAttention(8, 2, 2, 3)
    x = torch.randn(2, 6, 8)
    valid = torch.tensor([[True, False, True, False, True, False], [False] * 6])
    altered = x.clone()
    altered[~valid] = torch.randn_like(altered[~valid]) * 100
    output = attention(x, valid)
    changed = attention(altered, valid)
    torch.testing.assert_close(output[valid], changed[valid], rtol=0, atol=0)
    assert torch.isfinite(output).all()
    output.square().sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in attention.parameters() if p.grad is not None)
    model = build_model("relative_transformer")
    result = model(torch.zeros(2, 8, 8, dtype=torch.long))
    F.cross_entropy(result["logits"], torch.tensor([5, 6])).backward()
    assert torch.isfinite(result["state"]).all()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@pytest.mark.parametrize("family", ["convgru", "relative_transformer"])
def test_non_nca_families_reject_unavailable_auxiliary_lens(family):
    _, canvas, _ = input_batch()
    with pytest.raises(ValueError, match="no dictionary lens"):
        build_model(family)(canvas, lens_every=4)
