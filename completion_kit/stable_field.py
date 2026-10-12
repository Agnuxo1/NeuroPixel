"""A separate NumPy neural-field candidate with an exact-weight contraction bound.

Historical residual NeuroPixel models are never overwritten or reinterpreted.
The bound concerns ideal real arithmetic; deployment roundoff is a separate gate.
"""
from __future__ import annotations

from fractions import Fraction
import math
from pathlib import Path
import numpy as np

from .common import read_json, write_json


def perceive(state: np.ndarray, kernels: np.ndarray) -> np.ndarray:
    """Apply one learned depthwise 3x3 correlation using zero boundary padding."""
    padded = np.pad(state, ((0, 0), (0, 0), (1, 1), (1, 1)))
    output = np.zeros_like(state)
    height, width = state.shape[-2:]
    for row in range(3):
        for column in range(3):
            output += kernels[None, :, row, column, None, None] * padded[:, :, row:row+height, column:column+width]
    return output


def perceive_backward(state: np.ndarray, kernels: np.ndarray, gradient: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return exact analytical reverse-mode derivatives of the depthwise correlation."""
    padded = np.pad(state, ((0, 0), (0, 0), (1, 1), (1, 1)))
    padded_gradient = np.zeros_like(padded)
    kernel_gradient = np.zeros_like(kernels)
    height, width = state.shape[-2:]
    for row in range(3):
        for column in range(3):
            view = padded[:, :, row:row+height, column:column+width]
            kernel_gradient[:, row, column] = np.sum(gradient * view, axis=(0, 2, 3))
            padded_gradient[:, :, row:row+height, column:column+width] += gradient * kernels[None, :, row, column, None, None]
    return padded_gradient[:, :, 1:-1, 1:-1], kernel_gradient


def recurrent_bound(mix: np.ndarray, kernels: np.ndarray) -> Fraction:
    """Conservatively bound ||mix*perception||_infinity using exact stored floats."""
    mix, kernels = np.asarray(mix), np.asarray(kernels)
    if (mix.ndim != 2 or mix.shape[0] != mix.shape[1]
            or kernels.shape != (len(mix), 3, 3)
            or not np.isfinite(mix).all() or not np.isfinite(kernels).all()):
        raise ValueError("Invalid recurrent weights")
    channel_bounds = [sum((Fraction.from_float(abs(float(value))) for value in channel.ravel()), Fraction())
                      for channel in kernels]
    return max(sum((Fraction.from_float(abs(float(value))) * channel_bounds[index]
                    for index, value in enumerate(row)), Fraction()) for row in mix)


def project_recurrence(mix: np.ndarray, kernels: np.ndarray, rho: float) -> None:
    """Scale learned mixing weights and recheck the rational bound after rounding."""
    if not 0 < rho < 1:
        raise ValueError("rho must lie strictly between zero and one")
    target = Fraction.from_float(float(rho))
    bound = recurrent_bound(mix, kernels)
    if bound <= target:
        return
    scale = float(target / bound) * (1 - 64 * np.finfo(mix.dtype).eps)
    mix *= scale
    for _ in range(32):
        if recurrent_bound(mix, kernels) <= target:
            return
        mix *= 1 - 64 * np.finfo(mix.dtype).eps
    raise ArithmeticError("Rounded projection did not satisfy the exact-weight bound")


class StableField:
    """Driven leaky tanh field, shared local computation and a local output decoder."""
    def __init__(self, input_channels: int, state_channels: int, classes: int,
                 *, steps: int = 16, alpha: float = 0.5, rho: float = 0.95,
                 seed: int = 0, constrained: bool = True):
        if min(input_channels, state_channels, classes, steps) < 1 or not 0 < alpha <= 1 or not 0 < rho < 1:
            raise ValueError("Invalid field dimensions or dynamics")
        self.steps, self.alpha, self.rho, self.constrained = steps, alpha, rho, constrained
        rng = np.random.default_rng(seed)
        self.parameters = dict(
            kernels=rng.normal(0, 0.1, (state_channels, 3, 3)),
            mix=rng.normal(0, 0.1, (state_channels, state_channels)),
            drive=rng.normal(0, 0.1, (state_channels, input_channels)),
            bias=np.zeros(state_channels),
            decoder=rng.normal(0, 0.1, (classes, state_channels)),
            output_bias=np.zeros(classes))
        self.project()

    def project(self) -> None:
        """Enforce the new candidate's constraint after every parameter update."""
        if self.constrained:
            project_recurrence(self.parameters["mix"], self.parameters["kernels"], self.rho)

    def update(self, state: np.ndarray, input_field: np.ndarray) -> np.ndarray:
        """Advance one field step; input is held fixed for the registered task."""
        p = self.parameters
        local = perceive(state, p["kernels"])
        preactivation = (np.einsum("oc,bchw->bohw", p["mix"], local)
                         + np.einsum("oc,bchw->bohw", p["drive"], input_field)
                         + p["bias"][None, :, None, None])
        return (1 - self.alpha) * state + self.alpha * np.tanh(preactivation)

    def forward(self, input_field: np.ndarray, *, steps: int | None = None,
                initial: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Reset per example unless an explicitly supplied initial state is given."""
        inputs = np.asarray(input_field, dtype=np.float64)
        if (inputs.ndim != 4 or inputs.shape[1] != self.parameters["drive"].shape[1]
                or not np.isfinite(inputs).all()):
            raise ValueError("Expected finite B,C,H,W inputs")
        shape = (len(inputs), len(self.parameters["mix"]), *inputs.shape[-2:])
        state = np.zeros(shape) if initial is None else np.asarray(initial, dtype=float).copy()
        if state.shape != shape or not np.isfinite(state).all():
            raise ValueError("Invalid initial state")
        depth = self.steps if steps is None else steps
        if not isinstance(depth, int) or depth < 0:
            raise ValueError("Nonnegative integer depth required")
        for _ in range(depth):
            state = self.update(state, inputs)
        row, column = inputs.shape[-2] // 2, inputs.shape[-1] // 2
        logits = state[:, :, row, column] @ self.parameters["decoder"].T + self.parameters["output_bias"]
        return logits, state

    def loss_gradient(self, input_field: np.ndarray, target: np.ndarray) -> tuple[float, dict]:
        """Compute multiclass cross-entropy and full BPTT parameter derivatives."""
        inputs = np.asarray(input_field, dtype=np.float64)
        target = np.asarray(target, dtype=np.int64)
        if (inputs.ndim != 4 or len(inputs) == 0 or target.shape != (len(inputs),)
                or inputs.shape[1] != self.parameters["drive"].shape[1]
                or not np.isfinite(inputs).all() or target.min() < 0
                or target.max() >= len(self.parameters["decoder"])):
            raise ValueError("Invalid supervised batch")
        p = self.parameters
        state = np.zeros((len(inputs), len(p["mix"]), *inputs.shape[-2:]))
        cache = []
        drive = np.einsum("oc,bchw->bohw", p["drive"], inputs) + p["bias"][None, :, None, None]
        for _ in range(self.steps):
            local = perceive(state, p["kernels"])
            activated = np.tanh(np.einsum("oc,bchw->bohw", p["mix"], local) + drive)
            cache.append((state, local, activated))
            state = (1 - self.alpha) * state + self.alpha * activated
        row, column = inputs.shape[-2] // 2, inputs.shape[-1] // 2
        feature = state[:, :, row, column]
        logits = feature @ p["decoder"].T + p["output_bias"]
        shifted = logits - logits.max(axis=1, keepdims=True)
        normalizer = np.log(np.exp(shifted).sum(axis=1))
        loss = float(np.mean(normalizer - shifted[np.arange(len(target)), target]))
        gradient_logits = np.exp(shifted - normalizer[:, None])
        gradient_logits[np.arange(len(target)), target] -= 1
        gradient_logits /= len(target)
        gradients = {name: np.zeros_like(value) for name, value in p.items()}
        gradients["decoder"] = gradient_logits.T @ feature
        gradients["output_bias"] = gradient_logits.sum(axis=0)
        gradient_state = np.zeros_like(state)
        gradient_state[:, :, row, column] = gradient_logits @ p["decoder"]
        for old_state, local, activated in reversed(cache):
            gradient_pre = self.alpha * gradient_state * (1 - activated**2)
            gradients["mix"] += np.einsum("bohw,bchw->oc", gradient_pre, local)
            gradients["drive"] += np.einsum("bohw,bchw->oc", gradient_pre, inputs)
            gradients["bias"] += gradient_pre.sum(axis=(0, 2, 3))
            gradient_local = np.einsum("oc,bohw->bchw", p["mix"], gradient_pre)
            gradient_old, gradient_kernel = perceive_backward(old_state, p["kernels"], gradient_local)
            gradients["kernels"] += gradient_kernel
            gradient_state = (1 - self.alpha) * gradient_state + gradient_old
        if not math.isfinite(loss) or not all(np.isfinite(g).all() for g in gradients.values()):
            raise FloatingPointError("Nonfinite loss or gradient")
        return loss, gradients

    def certificate(self) -> dict:
        """Prove an ideal-real-arithmetic sup-norm bound from the exported floats."""
        bound = recurrent_bound(self.parameters["mix"], self.parameters["kernels"])
        alpha = Fraction.from_float(float(self.alpha))
        q = 1 - alpha + alpha * bound
        return dict(recurrent_bound_rational=str(bound), q_rational=str(q), q_float=float(q),
                    strict_contraction=q < 1, norm="sup norm over all cells and channels",
                    fixed_input_unique_equilibrium=q < 1,
                    ideal_real_arithmetic=True, hardware_roundoff_certified=False,
                    memory_retention_proved=False,
                    scope="This exported leaky tanh field, fixed input, zero-padding")

    def save(self, path: Path) -> None:
        """Export every learned parameter and its exact-weight certificate."""
        write_json(path, dict(schema=1, steps=self.steps, alpha=self.alpha, rho=self.rho,
                              constrained=self.constrained,
                              parameters={name: value.tolist() for name, value in self.parameters.items()},
                              certificate=self.certificate()))

    @classmethod
    def load(cls, path: Path) -> "StableField":
        """Load a finite exported model and verify its declared dimensions and bound."""
        document = read_json(path)
        parameters = {name: np.asarray(value, dtype=np.float64) for name, value in document["parameters"].items()}
        model = cls(parameters["drive"].shape[1], len(parameters["mix"]), len(parameters["decoder"]),
                    steps=document["steps"], alpha=document["alpha"], rho=document["rho"],
                    constrained=document["constrained"])
        if set(parameters) != set(model.parameters):
            raise ValueError("Export parameter names differ")
        for name, value in parameters.items():
            if value.shape != model.parameters[name].shape or not np.isfinite(value).all():
                raise ValueError("Invalid export tensor: " + name)
        model.parameters = parameters
        if model.certificate() != document["certificate"]:
            raise ValueError("Export certificate differs")
        if model.constrained and recurrent_bound(parameters["mix"], parameters["kernels"]) > Fraction.from_float(model.rho):
            raise ValueError("Constrained export exceeds its registered rho")
        return model
