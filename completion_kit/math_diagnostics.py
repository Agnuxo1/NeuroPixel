"""Produce reproducible mathematical diagnostics without opening final test panels."""
from __future__ import annotations

import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "2"

import argparse
import json
from pathlib import Path
import numpy as np

from .common import read_json, write_json, sha256_file, require_resources
from .stable_field import StableField
from .dynamics import (finite_difference_jacobian, linear_diagnostics,
                       boolean_noise_information, lyapunov_diagnostic)


def diagnose(model_path: Path, train_path: Path) -> dict:
    """Study a declared 3x3 patch boundary problem plus finite Boolean controls.

    The patch Jacobian does not establish full-canvas observability. The exact
    contraction bound applies to every finite zero-padded canvas of this model.
    """
    require_resources()
    model = StableField.load(model_path)
    with np.load(train_path, allow_pickle=False) as archive:
        original = archive["inputs"][0:1]
    row, column = original.shape[-2] // 2, original.shape[-1] // 2
    inputs = original[:, :, row-1:row+2, column-1:column+2].astype(float)
    _, state = model.forward(inputs, steps=64)
    update = lambda value: model.update(value, inputs)
    jacobians = [finite_difference_jacobian(update, state, epsilon=e) for e in (1e-5, 1e-6, 1e-7)]
    dimension = state.size
    output = np.zeros((len(model.parameters["decoder"]), dimension))
    channels = state.shape[1]
    for c in range(channels):
        output[:, c*9+4] = model.parameters["decoder"][:, c]
    controls = {}
    for name, table in (
        ("coordinate", np.array([i & 1 for i in range(64)])),
        ("parity", np.array([i.bit_count() & 1 for i in range(64)])),
        ("majority", np.array([int(i.bit_count() >= 3) for i in range(64)]))):
        controls[name] = [boolean_noise_information(table, p) for p in (0.0, 0.05, 0.1, 0.2, 0.5)]
    return dict(model_sha256=sha256_file(model_path), train_sha256=sha256_file(train_path),
                exact_weight_contraction=model.certificate(),
                patch_linear_diagnostic=linear_diagnostics(jacobians[1], output, horizon=16),
                jacobian_epsilon_sensitivity=[float(np.max(np.abs(j-jacobians[1]))) for j in jacobians],
                patch_lyapunov_diagnostic=lyapunov_diagnostic(jacobians[1]),
                boolean_information_controls=controls, test_opened=False,
                full_canvas_observability_established=False,
                patch_scope="Center 3x3 TRAIN input crop, NEW zero boundary, 64 finite updates",
                new_mathematics_improvement_empirically_established=False)


def main() -> None:
    """Archive finite diagnostics without replacing any scientific outcome."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(args.model, args.train)
    write_json(args.output, result, exclusive=True)
    print(json.dumps({k: result[k] for k in ("model_sha256", "test_opened", "patch_scope")}, indent=2))


if __name__ == "__main__":
    main()
