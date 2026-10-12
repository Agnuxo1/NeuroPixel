"""Install one complete forward correction in a NEW working source version only."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

from .common import sha256_file


CORRECTED_FORWARD = '''    def forward(self, canvas: torch.Tensor, trace: bool = False, lens_every: int = 0,
                rgb: torch.Tensor | None = None, cam: torch.Tensor | None = None,
                out_pos: tuple[int, int] | None = None, steps: int | None = None, hook=None):
        """Run the requested finite depth, including a valid zero-update diagnostic."""
        depth = self.steps if steps is None else steps
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 0:
            raise ValueError("steps must be a nonnegative integer or None")
        if isinstance(lens_every, bool) or not isinstance(lens_every, int) or lens_every < 0:
            raise ValueError("lens_every must be a nonnegative integer")
        ids = F.embedding(canvas, self.dictionary()).permute(0, 3, 1, 2)
        present = canvas != 0
        if rgb is not None:
            if cam is None:
                raise ValueError("Camera pixels require an explicit camera mask")
            camera_mask = cam.unsqueeze(1)
            if self.retina is not None:
                ids = torch.where(camera_mask, self.retina(rgb), ids)
            else:
                ids = torch.cat([torch.where(camera_mask, rgb, ids[:, :3]),
                                 ids[:, 3:] * ~camera_mask], 1)
            present = present | cam
        state = self.seed(ids) * present.unsqueeze(1)
        frames = [state.detach()] if trace else None
        activity, lens = [], []
        for time_step in range(1, depth + 1):
            if lens_every and time_step % lens_every == 0:
                lens.append(self.lens_logits(state))
            hidden = torch.cat([state, self.perceive(state), ids], 1)
            increment = self.f2(F.relu(self.f1(hidden)))
            if self.training and self.fire_rate < 1:
                increment = increment * (torch.rand_like(increment[:, :1]) < self.fire_rate)
            state = state + increment
            if hook is not None:
                state = hook(time_step, state)
            activity.append(increment.abs().mean())
            if trace:
                frames.append(state.detach())
        row, column = self.out_pos if out_pos is None else out_pos
        logits = self.read(state[:, :, row, column]) @ self.dictionary().T
        logits[:, 0] = -1e4
        result = {"logits": logits,
                  "activity": torch.stack(activity).mean() if activity else state.new_zeros(()),
                  "state": state}
        if lens:
            result["lens"] = torch.stack(lens, 1)
        if trace:
            result["frames"] = torch.stack(frames, 1)
        return result
'''


def install(path: Path, expected_sha256: str) -> None:
    """Refuse stale bytes and replace only NeuroPixel.forward using AST locations."""
    if sha256_file(path) != expected_sha256:
        raise ValueError("Source changed; review before applying the scoped correction")
    raw = path.read_bytes()
    source = raw.decode("utf-8")
    tree = ast.parse(source)
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NeuroPixel"]
    functions = [node for node in classes[0].body if isinstance(node, ast.FunctionDef) and node.name == "forward"] if len(classes) == 1 else []
    if len(functions) != 1:
        raise ValueError("Expected exactly one NeuroPixel.forward")
    node = functions[0]
    lines = source.splitlines(keepends=True)
    newline = "\r\n" if b"\r\n" in raw else "\n"
    replacement = CORRECTED_FORWARD.replace("\n", newline)
    updated = "".join(lines[:node.lineno-1]) + replacement + "".join(lines[node.end_lineno:])
    ast.parse(updated)
    path.write_bytes(updated.encode("utf-8"))


def main() -> None:
    """Apply the reviewed patch only after the frozen energy recipe has been closed."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()
    install(args.model, args.expected_sha256)


if __name__ == "__main__":
    main()
