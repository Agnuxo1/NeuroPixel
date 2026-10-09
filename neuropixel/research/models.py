"""Neural references for the frozen research protocol.

All models consume visible token grids, never targets or generator metadata.
The primary experiment supplies answer cross-entropy externally. The adapted
NCA comparator is not a complete reproduction of a particular published NCA.
"""
from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from neuropixel.model import NeuroPixel, n_params


def _canvas_shape(canvas: torch.Tensor) -> tuple[int, int]:
    if canvas.ndim != 3 or canvas.dtype != torch.long:
        raise ValueError("canvas must be an int64 tensor of shape [batch, height, width]")
    if min(canvas.shape) < 1:
        raise ValueError("canvas dimensions must be nonempty")
    return canvas.shape[1], canvas.shape[2]


def _steps(requested: int | None, default: int) -> int:
    # Preserve the legacy zero-means-default convention for recurrent models.
    result = requested or default
    if not isinstance(result, int) or result < 1:
        raise ValueError("the effective number of updates must be a positive integer")
    return result


class ResearchNCA(NeuroPixel):
    """Legacy-compatible core with explicit tying, reinjection, and PAD policy.

    With tied=True, reinject=True and freeze_pad=False, the parameter names and
    forward recipe match the audited legacy NeuroPixel. In particular, this
    subclass owns its dictionary and forward implementations so a later repair
    of the legacy class's PAD lookup does not silently reinterpret these models.
    An untied decoder is an independent V-by-c_id parameter cloned from the
    effective input dictionary, retaining the same low-rank readout structure.
    """

    def __init__(
        self, vocab: int, out_pos: tuple[int, int], c_id: int = 16, c: int = 48,
        hidden: int = 128, steps: int = 16, fire_rate: float = 0.5, *,
        tied: bool = True, reinject: bool = True, freeze_pad: bool = False,
        grounded: tuple[torch.Tensor, torch.Tensor] | None = None, retina: bool = False,
    ):
        if c_id < 3 or min(vocab, c, hidden, steps) < 1:
            raise ValueError("positive dimensions and c_id >= 3 are required")
        if not 0 <= fire_rate <= 1:
            raise ValueError("fire_rate must be between zero and one")
        super().__init__(vocab, out_pos, c_id, c, hidden, steps, fire_rate, grounded, retina)
        self.tied, self.reinject, self.freeze_pad = tied, reinject, freeze_pad
        if tied:
            self.register_parameter("output_dictionary", None)
        else:
            self.output_dictionary = nn.Parameter(self.dictionary().detach().clone())

    def dictionary(self) -> torch.Tensor:
        """Explicit audited recipe; freeze_pad is an opt-in effective projection."""
        weight = self.embed.weight
        dictionary = torch.cat([
            torch.where(self.g_mask, self.g_rgb, weight[:, :3]), weight[:, 3:]
        ], dim=1)
        if self.freeze_pad:
            dictionary = torch.cat([torch.zeros_like(dictionary[:1]), dictionary[1:]], dim=0)
        return dictionary

    def decoder_dictionary(self) -> torch.Tensor:
        if self.tied:
            return self.dictionary()
        dictionary = self.output_dictionary
        if self.freeze_pad:
            dictionary = torch.cat([torch.zeros_like(dictionary[:1]), dictionary[1:]], dim=0)
        return dictionary

    def lens_logits(self, state: torch.Tensor) -> torch.Tensor:
        return self.read(state.permute(0, 2, 3, 1)) @ self.decoder_dictionary().T

    def forward(
        self, canvas: torch.Tensor, trace: bool = False, lens_every: int = 0,
        rgb: torch.Tensor | None = None, cam: torch.Tensor | None = None,
        out_pos: tuple[int, int] | None = None, steps: int | None = None, hook=None,
    ):
        _canvas_shape(canvas)
        total_steps = _steps(steps, self.steps)
        if lens_every < 0:
            raise ValueError("lens_every cannot be negative")
        # Do not delegate to nn.Embedding: legacy PAD lookup semantics are explicit.
        identities = F.embedding(canvas, self.dictionary()).permute(0, 3, 1, 2)
        present = canvas != 0
        if rgb is not None:
            if cam is None:
                raise ValueError("camera RGB requires a camera mask")
            camera_mask = cam.unsqueeze(1)
            if self.retina is not None:
                identities = torch.where(camera_mask, self.retina(rgb), identities)
            else:
                identities = torch.cat([
                    torch.where(camera_mask, rgb, identities[:, :3]),
                    identities[:, 3:] * ~camera_mask,
                ], dim=1)
            present = present | cam
        state = self.seed(identities) * present.unsqueeze(1)
        # Turning off reinjection changes only the update input, never the seed.
        update_input = identities if self.reinject else torch.zeros_like(identities)
        frames = [state.detach()] if trace else None
        activity, lens = [], []
        for t in range(1, total_steps + 1):
            if lens_every and t % lens_every == 0:
                lens.append(self.lens_logits(state))
            hidden = torch.cat([state, self.perceive(state), update_input], dim=1)
            delta = self.f2(F.relu(self.f1(hidden)))
            if self.training and self.fire_rate < 1:
                delta = delta * (torch.rand_like(delta[:, :1]) < self.fire_rate)
            state = state + delta
            if hook is not None:
                state = hook(t, state)
            activity.append(delta.abs().mean())
            if trace:
                frames.append(state.detach())
        row, col = out_pos or self.out_pos
        logits = self.read(state[:, :, row, col]) @ self.decoder_dictionary().T
        logits[:, 0] = -1e4
        result = {"logits": logits, "activity": torch.stack(activity).mean(), "state": state}
        if lens:
            result["lens"] = torch.stack(lens, dim=1)
        if trace:
            result["frames"] = torch.stack(frames, dim=1)
        return result


class SpatialConvGRU(nn.Module):
    """A local ConvGRU classifier with a persistent visible embedding field.

    The learned pointwise seed is followed by 3x3 reset/update gates and a 3x3
    candidate gate. There is no stochastic firing, auxiliary lens, or dropout.
    """

    def __init__(
        self, vocab: int, out_pos: tuple[int, int], c_id: int = 16, c: int = 24,
        steps: int = 16,
    ):
        super().__init__()
        if min(vocab, c_id, c, steps) < 1:
            raise ValueError("all ConvGRU dimensions and update counts must be positive")
        self.out_pos, self.steps = out_pos, steps
        self.embed = nn.Embedding(vocab, c_id, padding_idx=0)
        self.seed = nn.Conv2d(c_id, c, 1)
        self.gates = nn.Conv2d(c_id + c, 2 * c, 3, padding=1)
        self.candidate = nn.Conv2d(c_id + c, c, 3, padding=1)
        self.head = nn.Linear(c, vocab)

    def forward(
        self, canvas: torch.Tensor, trace: bool = False, lens_every: int = 0, *,
        out_pos: tuple[int, int] | None = None, steps: int | None = None, hook=None,
    ):
        _canvas_shape(canvas)
        if lens_every:
            raise ValueError("the ConvGRU reference has no dictionary lens")
        identities = self.embed(canvas).permute(0, 3, 1, 2)
        state = torch.tanh(self.seed(identities)) * (canvas != 0).unsqueeze(1)
        frames = [state.detach()] if trace else None
        activity = []
        for t in range(1, _steps(steps, self.steps) + 1):
            reset, update = torch.sigmoid(self.gates(torch.cat([identities, state], 1))).chunk(2, 1)
            candidate = torch.tanh(self.candidate(torch.cat([identities, reset * state], 1)))
            next_state = (1 - update) * state + update * candidate
            activity.append((next_state - state).abs().mean())
            state = next_state if hook is None else hook(t, next_state)
            if trace:
                frames.append(state.detach())
        row, col = out_pos or self.out_pos
        logits = self.head(state[:, :, row, col])
        logits[:, 0] = -1e4
        result = {"logits": logits, "activity": torch.stack(activity).mean(), "state": state}
        if trace:
            result["frames"] = torch.stack(frames, dim=1)
        return result


def grid_coordinates(h: int, w: int) -> torch.Tensor:
    """Row-major [row, column] coordinates for an h-by-w grid."""
    if min(h, w) < 1:
        raise ValueError("grid dimensions must be positive")
    return torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij"), -1).reshape(-1, 2)


def relative_position_index(h: int, w: int, positions: torch.Tensor | None = None) -> torch.Tensor:
    """Return [query, key] indices for key-minus-query 2D displacements.

    index = (delta_row + h - 1) * (2*w - 1) + delta_column + w - 1.
    Explicit coordinates allow a token permutation to carry its positions with it.
    """
    positions = grid_coordinates(h, w) if positions is None else positions
    if positions.ndim != 2 or positions.shape[1] != 2 or positions.dtype != torch.long:
        raise ValueError("positions must be int64 [tokens, 2] row/column coordinates")
    if not len(positions) or (positions < 0).any() or (positions[:, 0] >= h).any() or (positions[:, 1] >= w).any():
        raise ValueError("positions must be nonempty and inside the configured grid")
    displacement = positions.unsqueeze(0) - positions.unsqueeze(1)
    return (displacement[..., 0] + h - 1) * (2 * w - 1) + displacement[..., 1] + w - 1


class RelativeSelfAttention(nn.Module):
    """Dense attention with a separate 2D relative-bias table for each head."""

    def __init__(self, d: int, heads: int, h: int, w: int):
        super().__init__()
        if min(d, heads) < 1 or d % heads:
            raise ValueError("attention width must be positive and divisible by heads")
        self.d, self.heads, self.h, self.w = d, heads, h, w
        self.head_dim = d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.projection = nn.Linear(d, d)
        self.relative_bias = nn.Parameter(torch.zeros(heads, (2 * h - 1) * (2 * w - 1)))
        self.register_buffer("relative_index", relative_position_index(h, w), persistent=False)

    def forward(self, x: torch.Tensor, valid_keys: torch.Tensor, positions: torch.Tensor | None = None):
        batch, tokens, width = x.shape
        if width != self.d or valid_keys.shape != (batch, tokens) or valid_keys.dtype != torch.bool:
            raise ValueError("attention input and boolean key mask have incompatible shapes")
        index = self.relative_index if positions is None else relative_position_index(
            self.h, self.w, positions.to(device=x.device))
        if index.shape != (tokens, tokens):
            raise ValueError("token count does not match the supplied grid positions")
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        scores = (query @ key.transpose(-1, -2)) * self.head_dim ** -0.5
        scores = scores + self.relative_bias[:, index].unsqueeze(0)
        mask = valid_keys[:, None, None, :]
        scores = scores.masked_fill(~mask, float("-inf"))
        # An empty input has zero attention mass, not a softmax of all -infinity.
        scores = torch.where(valid_keys.any(1)[:, None, None, None], scores, torch.zeros_like(scores))
        weights = scores.softmax(-1) * mask.to(dtype=scores.dtype)
        attended = (weights @ value).transpose(1, 2).reshape(batch, tokens, width)
        return self.projection(attended)


class RelativeTransformerBlock(nn.Module):
    def __init__(self, d: int, heads: int, ff: int, h: int, w: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attention = RelativeSelfAttention(d, heads, h, w)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))

    def forward(self, x, valid_keys, positions=None):
        x = x + self.attention(self.norm1(x), valid_keys, positions)
        return x + self.ff(self.norm2(x))


class RelativeTransformer(nn.Module):
    """A two-layer spatial Transformer; empty input positions are masked as keys.

    Empty positions, including the output cell, may still issue queries. The
    original input mask is reused in each layer. There is no absolute position
    embedding or dropout. Encoder depth is set by layers, not recurrent steps.
    """

    def __init__(
        self, vocab: int, h: int, w: int, out_pos: tuple[int, int] | None = None,
        d: int = 32, layers: int = 2, heads: int = 4, ff: int = 128,
    ):
        super().__init__()
        if min(vocab, h, w, d, layers, heads, ff) < 1:
            raise ValueError("all Transformer dimensions must be positive")
        self.h, self.w = h, w
        self.out_pos = out_pos or (h - 1, w - 1)
        self.embed = nn.Embedding(vocab, d, padding_idx=0)
        self.layers = nn.ModuleList([RelativeTransformerBlock(d, heads, ff, h, w) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)
        self.register_buffer("positions", grid_coordinates(h, w), persistent=False)

    def _encode(self, tokens, positions, trace=False):
        if tokens.ndim != 2 or tokens.dtype != torch.long or min(tokens.shape) < 1:
            raise ValueError("tokens must be a nonempty int64 [batch, tokens] tensor")
        valid_keys = tokens != 0
        x = self.embed(tokens)
        frames = [x.detach()] if trace else None
        for layer in self.layers:
            x = layer(x, valid_keys, positions)
            if trace:
                frames.append(x.detach())
        return self.norm(x), frames

    def encode_tokens(self, tokens: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        """Encode explicitly ordered tokens; permute coordinates with their tokens."""
        return self._encode(tokens, positions)[0]

    def forward(
        self, canvas: torch.Tensor, trace: bool = False, lens_every: int = 0, *,
        out_pos: tuple[int, int] | None = None,
    ):
        if _canvas_shape(canvas) != (self.h, self.w):
            raise ValueError("canvas shape differs from the configured relative-position grid")
        if lens_every:
            raise ValueError("the Transformer reference has no dictionary lens")
        x, frames = self._encode(canvas.flatten(1), None, trace)
        row, col = out_pos or self.out_pos
        if not 0 <= row < self.h or not 0 <= col < self.w:
            raise ValueError("readout position is outside the configured grid")
        logits = self.head(x[:, row * self.w + col])
        logits[:, 0] = -1e4
        result = {"logits": logits, "activity": logits.new_zeros(()),
                  "state": x.transpose(1, 2).reshape(len(canvas), -1, self.h, self.w)}
        if trace:
            # These frames are embedding/block states before the final LayerNorm.
            result["frames"] = torch.stack(frames, 1).transpose(2, 3).reshape(
                len(canvas), len(frames), -1, self.h, self.w)
        return result


_DEFAULTS = {
    "neuropixel": {"c_id": 16, "c": 48, "hidden": 128, "fire_rate": 0.5,
                   "tied": True, "reinject": True, "freeze_pad": False},
    "standard_nca": {"c_id": 16, "c": 48, "hidden": 128, "fire_rate": 0.5,
                     "tied": False, "reinject": False, "freeze_pad": False},
    "convgru": {"c_id": 16, "c": 24},
    "relative_transformer": {"d": 32, "layers": 2, "heads": 4, "ff": 128},
}


def build_model(family: str, vocab: int = 35, h: int = 8, w: int = 8, steps: int = 16, **variant_kwargs):
    """Build one protocol family using visible-grid dimensions only.

    steps controls recurrent models; a Transformer uses its configured layers.
    Architecture overrides are explicit kwargs and must be recorded by callers.
    """
    if family not in _DEFAULTS:
        raise ValueError(f"unknown model family {family!r}; choose one of {tuple(_DEFAULTS)}")
    if min(vocab, h, w, steps) < 1:
        raise ValueError("vocabulary, grid dimensions and steps must be positive")
    options = {**_DEFAULTS[family], **variant_kwargs}
    out_pos = options.pop("out_pos", (h - 1, w - 1))
    if family in ("neuropixel", "standard_nca"):
        return ResearchNCA(vocab, out_pos, steps=steps, **options)
    if family == "convgru":
        return SpatialConvGRU(vocab, out_pos, steps=steps, **options)
    return RelativeTransformer(vocab, h, w, out_pos=out_pos, **options)


def model_config(family: str, vocab: int = 35, h: int = 8, w: int = 8, steps: int = 16, **variant_kwargs):
    """Return resolved defaults and actual count without consuming global RNG state."""
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        model = build_model(family, vocab, h, w, steps, **variant_kwargs)
    options = deepcopy({**_DEFAULTS[family], **variant_kwargs})
    out_pos = options.pop("out_pos", (h - 1, w - 1))
    return {"family": family, "vocab": vocab, "h": h, "w": w, "steps": steps,
            "recurrent_steps": None if family == "relative_transformer" else steps,
            "out_pos": list(out_pos), "kwargs": options, "parameters": n_params(model),
            "dropout": 0.0, "primary_supervision": "answer_cross_entropy_only"}
