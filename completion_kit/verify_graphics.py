"""Physical-graphics admission for the new candidate, using TRAIN data only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from .common import read_json, write_json, sha256_file, require_resources
from .stable_field import StableField
from .graphics_field import GraphicsField


def verify(model_path: Path, train_path: Path, required_renderer: str,
           backend: str | None = None) -> dict:
    """Compare complete states/logits/decisions at every registered finite depth."""
    require_resources()
    model = StableField.load(model_path)
    reference = StableField.load(model_path)
    reference.parameters = {name: value.astype(np.float32).astype(np.float64)
                            for name, value in reference.parameters.items()}
    reference.alpha = float(np.float32(reference.alpha))
    with np.load(train_path, allow_pickle=False) as archive:
        inputs = archive["inputs"][:16].astype(np.float32).astype(np.float64)
    graphics = GraphicsField(model, backend=backend)
    checks, result = [], {}
    try:
        renderer = graphics.metadata["renderer"]
        vendor = graphics.metadata["vendor"]
        if required_renderer.lower() not in renderer.lower():
            raise ValueError("Required physical renderer not observed: " + renderer)
        if any(name in renderer.lower() for name in ("llvmpipe", "softpipe", "software", "swrast")):
            raise ValueError("Software graphics cannot satisfy physical GPU admission")
        for depth in (0, 8, 16, 32, 64):
            native_logits, native_state = reference.forward(inputs, steps=depth)
            gpu_logits, gpu_state = graphics.forward(inputs, steps=depth)
            for name, first, second in (("state", gpu_state, native_state), ("logits", gpu_logits, native_logits)):
                allowed = 1e-4 + 1e-5 * np.abs(second)
                difference = np.abs(first - second)
                checks.append(dict(depth=depth, name=name,
                                   passed=bool(np.isfinite(first).all() and np.all(difference <= allowed)),
                                   maximum_absolute_error=float(difference.max())))
            checks.append(dict(depth=depth, name="categorical_decisions",
                               passed=bool(np.array_equal(gpu_logits.argmax(1), native_logits.argmax(1)))))
        result = dict(status="passed" if all(c["passed"] for c in checks) else "failed_preserved",
                      checks=checks, renderer=renderer, vendor=vendor, model_sha256=sha256_file(model_path),
                      train_sha256=sha256_file(train_path), reference_weight_precision="FP32 rounded weights evaluated in NumPy FP64",
                      gpu_precision="FP32", test_opened=False, physical_gpu_identity_checked=True,
                      roundoff_bound_for_all_time_established=False,
                      inference_includes_allocation_and_transfers=True)
    finally:
        graphics.close()
    return result


def main() -> None:
    """Preserve a parity receipt; refuse success when a single required check fails."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--required-renderer", default="RTX 3090")
    parser.add_argument("--backend")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = verify(args.model, args.train, args.required_renderer, args.backend)
    except Exception as error:
        result = dict(status="failed_preserved", error_type=type(error).__name__, message=str(error))
    write_json(args.output, result, exclusive=True)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
