"""Nonnormal dynamics, local observability and exact finite Boolean-noise controls."""
from __future__ import annotations

import math
import numpy as np
from scipy.linalg import solve_discrete_lyapunov


def finite_difference_jacobian(update, state: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Compute a central-difference Jacobian for a SMALL deterministic state.

    This is a numerical diagnostic, not an all-state nonlinear certificate.
    Repeat with several epsilon values before interpreting its singular values.
    """
    state = np.asarray(state, dtype=np.float64)
    if state.size > 512 or epsilon <= 0 or not np.isfinite(state).all():
        raise ValueError("Use at most 512 finite state coordinates and positive epsilon")
    matrix = np.empty((state.size, state.size))
    for index in range(state.size):
        offset = np.zeros_like(state)
        offset.flat[index] = epsilon
        matrix[:, index] = (np.asarray(update(state + offset)).ravel()
                            - np.asarray(update(state - offset)).ravel()) / (2 * epsilon)
    if not np.isfinite(matrix).all():
        raise ValueError("Nonfinite finite-difference Jacobian")
    return matrix


def linear_diagnostics(jacobian: np.ndarray, readout: np.ndarray | None = None,
                       horizon: int = 32) -> dict:
    """Inspect one frozen linearization, transient powers and finite observability.

    The numerical-range rectangle uses floating-point Hermitian eigenvalues.
    Its resulting Crouzeix columns are candidate estimates, NOT certified bounds.
    Products of different Jacobians must be measured separately.
    """
    matrix = np.asarray(jacobian, dtype=np.complex128)
    if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]
            or not 1 <= len(matrix) <= 512 or horizon < 1 or not np.isfinite(matrix).all()):
        raise ValueError("Invalid small square Jacobian")
    hermitian = (matrix + matrix.conj().T) / 2
    skew = (matrix - matrix.conj().T) / (2j)
    real_limits = np.linalg.eigvalsh(hermitian)[[0, -1]].real
    imaginary_limits = np.linalg.eigvalsh(skew)[[0, -1]].real
    radius_rectangle = max(abs(complex(x, y)) for x in real_limits for y in imaginary_limits)
    power = np.eye(len(matrix), dtype=np.complex128)
    powers = []
    for depth in range(horizon + 1):
        powers.append(dict(depth=depth, operator_norm=float(np.linalg.norm(power, 2)),
                           classical_crouzeix_rectangle_estimate=(1 + math.sqrt(2)) * radius_rectangle**depth,
                           proposed_constant_two_rectangle_estimate=2 * radius_rectangle**depth))
        power = power @ matrix
    result = dict(spectral_radius=float(np.max(np.abs(np.linalg.eigvals(matrix)))),
                  one_step_norm=float(np.linalg.norm(matrix, 2)),
                  nonnormal_commutator_norm=float(np.linalg.norm(matrix.conj().T @ matrix - matrix @ matrix.conj().T, 2)),
                  approximate_numerical_range_rectangle=[real_limits.tolist(), imaginary_limits.tolist()],
                  radius_rectangle_estimate=float(radius_rectangle), powers=powers,
                  rigorous_numerical_certificate=False, nonlinear_global_stability_established=False,
                  scope="One fixed linearization, numerical approximation")
    if readout is not None:
        output = np.asarray(readout, dtype=np.complex128)
        if output.ndim != 2 or output.shape[1] != len(matrix) or not np.isfinite(output).all():
            raise ValueError("Invalid observation matrix")
        rows, power = [], np.eye(len(matrix), dtype=np.complex128)
        for _ in range(horizon):
            rows.append(output @ power)
            power = power @ matrix
        observation = np.concatenate(rows)
        singular = np.linalg.svd(observation, compute_uv=False)
        tolerance = (max(observation.shape) * np.finfo(float).eps * singular[0]
                     if len(singular) and singular[0] else 0.0)
        rank = int(np.sum(singular > tolerance))
        result["observability"] = dict(rank=rank, ambient_dimension=len(matrix),
                                       ambient_nullity=len(matrix) - rank,
                                       singular_values=singular.tolist(), tolerance=float(tolerance),
                                       semantic_causality_established=False)
    return result


def trajectory_products(jacobians: list[np.ndarray]) -> list[float]:
    """Measure J[t-1]...J[0], preserving the order of time-varying linearizations."""
    if not jacobians:
        raise ValueError("No trajectory Jacobians")
    first = np.asarray(jacobians[0], dtype=float)
    product = np.eye(first.shape[0])
    norms = []
    for matrix in jacobians:
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape != product.shape or not np.isfinite(matrix).all():
            raise ValueError("Inconsistent trajectory matrix")
        product = matrix @ product
        norms.append(float(np.linalg.norm(product, 2)))
    return norms


def lyapunov_diagnostic(matrix: np.ndarray) -> dict:
    """Solve A*P*A-P=-I for a stable finite matrix and report floating residuals."""
    matrix = np.asarray(matrix, dtype=np.complex128)
    if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]
            or not np.isfinite(matrix).all()):
        raise ValueError("Finite square matrix required")
    radius = float(np.max(np.abs(np.linalg.eigvals(matrix))))
    if radius >= 1:
        return dict(admitted=False, spectral_radius=radius, reason="Spectral radius is not below one")
    metric = solve_discrete_lyapunov(matrix.conj().T, np.eye(len(matrix)))
    metric = (metric + metric.conj().T) / 2
    residual = matrix.conj().T @ metric @ matrix - metric + np.eye(len(matrix))
    return dict(admitted=bool(np.linalg.eigvalsh(metric).min() > 0), spectral_radius=radius,
                metric_min_eigenvalue=float(np.linalg.eigvalsh(metric).min()),
                residual_norm=float(np.linalg.norm(residual, 2)),
                rigorous_numerical_certificate=False, scope="Frozen linearization only")


def binary_entropy(probability: float) -> float:
    """Return binary entropy in bits, including endpoint values."""
    if not 0 <= probability <= 1:
        raise ValueError("Probability outside [0,1]")
    if probability in (0, 1):
        return 0.0
    return -probability * math.log2(probability) - (1 - probability) * math.log2(1 - probability)


def boolean_noise_information(truth_table: np.ndarray, crossover: float) -> dict:
    """Enumerate I(f(X);Y) for uniform independent bits and independent BSC noise.

    Y is the FULL noisy cube vector; f is a deterministic ONE-BIT summary of X.
    This does not certify arbitrary neural states, multiclass labels or image noise.
    """
    values = np.asarray(truth_table)
    size = len(values)
    if (values.ndim != 1 or not 2 <= size <= 256 or size & (size - 1)
            or not np.isin(values, [0, 1]).all() or not 0 <= crossover <= 0.5):
        raise ValueError("Binary truth table for 1..8 bits and crossover in [0,0.5] required")
    bits = size.bit_length() - 1
    distances = np.array([[int(x ^ y).bit_count() for y in range(size)] for x in range(size)])
    conditional = crossover**distances * (1 - crossover)**(bits - distances)
    joint = np.stack([conditional[values == label].sum(axis=0) / size for label in (0, 1)])
    label_probability = joint.sum(axis=1, keepdims=True)
    output_probability = joint.sum(axis=0, keepdims=True)
    denominator = label_probability * output_probability
    mask = joint > 0
    information = float(np.sum(joint[mask] * np.log2(joint[mask] / denominator[mask])))
    bound = 1 - binary_entropy(crossover)
    return dict(bits=bits, crossover=crossover, output_bias=float(values.mean()),
                information_bits=information, proposed_bound_bits=bound,
                finite_numeric_check_passed=information <= bound + 1e-12,
                theorem_formalization_independently_rebuilt=False,
                scope="Finite enumeration under the exact Boolean/BSC assumptions")
