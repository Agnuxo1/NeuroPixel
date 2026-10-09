"""Prospective item-6 NCA configurations and explicitly recorded interventions.

Inventory construction uses the standard library only. Torch is imported lazily
when a model or evaluation mask is actually requested. Historical scientific
modules are reused without editing their source.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from itertools import product
import math

_FACTORY_ACTIVE = False
TRAIN_DAMAGE = {
    "after_step": 8, "apply_probability": 0.5,
    "erase_probability": 0.3, "seed": 62001,
}
EVAL_DAMAGE = {"after_step": 8, "erase_probability": 0.3, "seed": 62002}


def _damage_spec(value, *, training, steps):
    if value is None:
        return None
    expected = {"after_step", "erase_probability", "seed"}
    if training:
        expected.add("apply_probability")
    if set(value) != expected:
        raise ValueError(f"damage fields must be exactly {sorted(expected)}")
    value = deepcopy(value)
    if type(value["after_step"]) is not int or not 1 <= value["after_step"] < steps:
        raise ValueError("damage must leave at least one subsequent recurrent update")
    if type(value["seed"]) is not int or not 0 <= value["seed"] < 2**63:
        raise ValueError("damage requires an explicit nonnegative CPU RNG seed")
    for name in expected & {"apply_probability", "erase_probability"}:
        if not isinstance(value[name], (int, float)) or not math.isfinite(value[name]):
            raise ValueError("damage probabilities must be finite numbers")
        if not 0 <= value[name] <= 1:
            raise ValueError("damage probabilities must lie in [0, 1]")
    return value


def training_config(protocol, *, panel, tied, school, reinject, seed,
                    steps=16, updates=1024, damage=None):
    """Build one scientific recipe, before adding runtime source provenance."""
    if panel not in {"factorial", "recurrence", "damage", "optimization"}:
        raise ValueError("unknown core item-6 panel")
    if type(tied) is not bool or type(reinject) is not bool:
        raise ValueError("tying and reinjection must be explicit booleans")
    if school not in (0.0, 0.3) or seed not in (20, 21):
        raise ValueError("core item 6 uses school 0/.3 and initialization 20/21")
    if type(steps) is not int or steps < 1 or type(updates) is not int or updates < 1:
        raise ValueError("steps and optimization updates must be positive integers")
    if school and steps < 4:
        raise ValueError("school every four updates is undefined for T < 4")
    damage = _damage_spec(damage, training=True, steps=steps)
    training = protocol["training"]
    school_id = str(float(school)).replace(".", "p")
    run_id = (f"{panel}_tie{int(tied)}_school{school_id}_reinj{int(reinject)}"
              f"_T{steps}_damage{int(damage is not None)}_s0_i{seed}_u{updates}")
    return {
        "run_id": run_id, "panel": panel, "phase": "item6_ablation",
        "protocol_id": protocol["protocol_id"], "family": "neuropixel",
        "vocab": 35, "height": 8, "width": 8, "split_seed": 0,
        "init_seed": seed, "learning_rate": 0.003, "updates": updates,
        "batch_size": 64, "steps": steps,
        "weight_decay": training["weight_decay"],
        "gradient_clip": training["gradient_clip"], "school_weight": float(school),
        "train_sample_seed": protocol["data"]["train_sample_seed"],
        "update_random_seed": training["update_random_seed"],
        "variant": {"tied": tied, "reinject": reinject, "freeze_pad": False},
        "training_damage": damage,
    }


def core_training_configs(protocol):
    """Return the approved 26 runs; no selection, data creation, or Torch import."""
    factorial = protocol["factorial"]
    expected = {
        "tying": [False, True], "school_weight": [0.0, 0.3],
        "reinjection": [False, True], "init_seeds": [20, 21],
        "split_seed": 0, "learning_rate": 0.003,
    }
    if any(factorial.get(key) != value for key, value in expected.items()):
        raise ValueError("the original frozen factorial differs from item 6")
    if any(protocol["training"][key] != value
           for key, value in {"updates": 1024, "batch_size": 64, "steps": 16}.items()):
        raise ValueError("the original exposure budget changed")
    configs = [
        training_config(protocol, panel="factorial", tied=tied, school=school,
                        reinject=reinject, seed=seed)
        for seed, tied, school, reinject in product(
            factorial["init_seeds"], factorial["tying"],
            factorial["school_weight"], factorial["reinjection"])
    ]
    for seed, steps in product((20, 21), (1, 4)):
        configs.append(training_config(
            protocol, panel="recurrence", tied=True, school=0.0,
            reinject=True, seed=seed, steps=steps))
    for seed in (20, 21):
        configs.append(training_config(
            protocol, panel="damage", tied=True, school=0.0,
            reinject=True, seed=seed, damage=TRAIN_DAMAGE))
    for seed, school in product((20, 21), (0.0, 0.3)):
        configs.append(training_config(
            protocol, panel="optimization", tied=True, school=school,
            reinject=True, seed=seed, updates=8192))
    if len(configs) != 26 or len({row["run_id"] for row in configs}) != 26:
        raise AssertionError("the core item-6 inventory must have 26 unique runs")
    return configs


def core_evaluation_cases(configs):
    """Prespecify clean scoring plus deployment truncation and paired lesions."""
    cases = []
    for config in configs:
        run_id = config["run_id"]
        cases.append({"evaluation_id": run_id + "__clean", "run_id": run_id,
                      "kind": "clean", "steps_override": None, "damage": None})
        baseline = (config["panel"] == "factorial"
                    and config["variant"]["tied"] and config["variant"]["reinject"]
                    and config["school_weight"] == 0.0)
        if baseline:
            for steps in (1, 4):
                cases.append({
                    "evaluation_id": run_id + f"__deploy_T{steps}", "run_id": run_id,
                    "kind": "deployment_truncation", "steps_override": steps, "damage": None,
                })
        if baseline or config["panel"] == "damage":
            cases.append({
                "evaluation_id": run_id + "__lesion", "run_id": run_id,
                "kind": "fixed_lesion", "steps_override": None,
                "damage": deepcopy(EVAL_DAMAGE),
            })
    if len(cases) != 34 or len({row["evaluation_id"] for row in cases}) != 34:
        raise AssertionError("the core inventory must have 34 evaluation cases")
    return cases


def training_model(config):
    """Construct the legacy model or a parameter-identical damage subclass."""
    from neuropixel.research.models import ResearchNCA, build_model, model_config

    arguments = dict(vocab=config["vocab"], h=config["height"], w=config["width"],
                     steps=config["steps"], **config["variant"])
    damage = _damage_spec(config.get("training_damage"), training=True, steps=config["steps"])
    if damage is None or damage["apply_probability"] == 0 or damage["erase_probability"] == 0:
        # No extra RNG draws, altered seed, or wrapper parameters in the zero-damage case.
        return build_model(config["family"], **arguments)
    if config["family"] != "neuropixel":
        raise ValueError("the approved damage intervention is an NCA intervention")
    import torch

    class TrainingDamageNCA(ResearchNCA):
        """Erase complete state vectors after update 8; never identities or weights."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.training_damage = deepcopy(damage)
            self.damage_generator = torch.Generator(device="cpu").manual_seed(damage["seed"])

        def forward(self, canvas, *args, **kwargs):
            if not self.training:
                return super().forward(canvas, *args, **kwargs)
            if kwargs.get("hook") is not None:
                raise ValueError("training damage cannot silently compose another hook")
            apply = bool(torch.rand((), generator=self.damage_generator, device="cpu")
                         < damage["apply_probability"])
            if not apply:
                return super().forward(canvas, *args, **kwargs)
            keep = torch.rand((len(canvas), 1, canvas.shape[1], canvas.shape[2]),
                              generator=self.damage_generator, device="cpu") >= damage["erase_probability"]

            def lesion(step, state):
                return state * keep.to(device=state.device) if step == damage["after_step"] else state

            kwargs["hook"] = lesion
            return super().forward(canvas, *args, **kwargs)

    # model_config forks/restores CPU RNG; the actual initialization remains unchanged.
    resolved = model_config(config["family"], **arguments)
    return TrainingDamageNCA(config["vocab"], tuple(resolved["out_pos"]),
                             steps=config["steps"], **resolved["kwargs"])


def training_metadata(config):
    """Describe the actual objective and intervention, not just the legacy factory."""
    from neuropixel.research.models import model_config

    result = model_config(
        config["family"], vocab=config["vocab"], h=config["height"], w=config["width"],
        steps=config["steps"], **config["variant"])
    school = config["school_weight"]
    result.update(
        factory="neuropixel.research.ablations.training_model",
        primary_supervision=("answer_cross_entropy_plus_occupied_token_lens_cross_entropy"
                             if school else "answer_cross_entropy_only"),
        school_weight=school, school_lens_every=4 if school else None,
        school_state_times=list(range(3, config["steps"], 4)) if school else [],
        training_damage=_damage_spec(
            config.get("training_damage"), training=True, steps=config["steps"]),
        validation_intervention="none; deterministic full firing in eval mode",
    )
    return result


@contextmanager
def training_factory_adapter(config):
    """Temporarily adapt the existing trainer; always restore both global factories.

    This adapter is deliberately process-local and sequential. Concurrent or nested
    training in the same Python process is rejected. It changes no source files.
    """
    global _FACTORY_ACTIVE
    from neuropixel.research import experiment

    if _FACTORY_ACTIVE:
        raise RuntimeError("item-6 factory adaptation is already active in this process")
    old_factory, old_metadata = experiment.model_from_config, experiment.model_config

    def factory(requested):
        if requested != config:
            raise ValueError("trainer requested a different adapted configuration")
        return training_model(requested)

    def metadata(family, **kwargs):
        expected = dict(vocab=config["vocab"], h=config["height"], w=config["width"],
                        steps=config["steps"], **config["variant"])
        if family != config["family"] or kwargs != expected:
            raise ValueError("trainer metadata request differs from the adapted configuration")
        return training_metadata(config)

    _FACTORY_ACTIVE = True
    experiment.model_from_config, experiment.model_config = factory, metadata
    try:
        yield
    finally:
        experiment.model_from_config, experiment.model_config = old_factory, old_metadata
        _FACTORY_ACTIVE = False


def evaluation_keep_mask(n, h, w, damage):
    """Generate one complete mask by example index, independent of evaluation batches."""
    import torch

    generator = torch.Generator(device="cpu").manual_seed(damage["seed"])
    return torch.rand((n, 1, h, w), generator=generator, device="cpu") >= damage["erase_probability"]


def evaluation_view(model, config, case, keep_mask=None):
    """Return an evaluation-only view; no retraining or checkpoint mutation."""
    import torch

    steps = case["steps_override"]
    effective_steps = config["steps"] if steps is None else steps
    damage = _damage_spec(case["damage"], training=False, steps=effective_steps)
    if steps is None and damage is None:
        return model
    if damage is not None:
        if keep_mask is None or keep_mask.dtype != torch.bool:
            raise ValueError("lesion evaluation requires its presaved boolean mask")
        if tuple(keep_mask.shape[1:]) != (1, config["height"], config["width"]):
            raise ValueError("evaluation lesion mask has an incompatible spatial shape")

    class EvaluationView(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = model
            self.offset = 0

        def forward(self, canvas):
            if self.training:
                raise RuntimeError("deployment interventions must only be evaluated")
            kwargs = {} if steps is None else {"steps": steps}
            if damage is not None:
                stop = self.offset + len(canvas)
                if stop > len(keep_mask):
                    raise ValueError("evaluation exceeded the prespecified mask inventory")
                batch_keep = keep_mask[self.offset:stop]
                self.offset = stop

                def lesion(step, state):
                    return (state * batch_keep.to(device=state.device)
                            if step == damage["after_step"] else state)

                kwargs["hook"] = lesion
            return self.base(canvas, **kwargs)

    return EvaluationView()
