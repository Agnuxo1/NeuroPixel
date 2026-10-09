"""Input-only symbolic and category controls for the original role generators.

Predictors receive only visible token IDs and the public vocabulary/query schema.
They never receive a task object, target, generator metadata, or split membership.
These are domain-informed task controls, not trained neural baselines.
"""
from __future__ import annotations

import torch

from neuropixel.task import NOUNS, PLACES, ROLES, VERBS, Vocab


def _inputs(canvas: torch.Tensor, query_pos: tuple[int, int], vocab: Vocab):
    if canvas.ndim != 3 or canvas.dtype != torch.long:
        raise ValueError("canvas must be an int64 tensor of shape [batch, height, width]")
    if not canvas.shape[0]:
        raise ValueError("canvas batch cannot be empty")
    height, width = canvas.shape[1:]
    row, col = query_pos
    if not -height <= row < height or not -width <= col < width:
        raise ValueError("query position is outside the canvas")
    row, col = row % height, col % width
    if (canvas < 0).any() or (canvas >= len(vocab)).any():
        raise ValueError("canvas contains token IDs outside the public vocabulary")
    roles = torch.tensor(vocab.ids(ROLES), device=canvas.device)
    query = canvas[:, row, col]
    if not torch.isin(query, roles).all():
        raise ValueError("every query must contain a known role token")
    return query, (row, col), roles


def _role_positions(canvas, role_ids, query_pos):
    """Find one visible fact role per example, excluding the query itself."""
    mask = canvas == role_ids[:, None, None]
    mask[:, query_pos[0], query_pos[1]] = False
    if not (mask.sum((1, 2)) == 1).all():
        raise ValueError("each fact role must occur exactly once outside the query")
    flat = mask.flatten(1).to(torch.long).argmax(1)
    return flat // canvas.shape[2], flat % canvas.shape[2]


def _row_filler_positions(canvas, rows, role_tokens):
    batch = torch.arange(canvas.shape[0], device=canvas.device)
    values = canvas[batch, rows]
    filler = (values != 0) & ~torch.isin(values, role_tokens)
    if not (filler.sum(1) == 1).all():
        raise ValueError("the matching fact row must contain exactly one filler")
    return filler.to(torch.long).argmax(1)


def adjacent_exact(
    canvas: torch.Tensor, *, query_pos: tuple[int, int] = (-1, -2), vocab: Vocab | None = None
) -> torch.Tensor:
    """Read the right neighbor of the queried fact role in an adjacent canvas."""
    vocab = Vocab() if vocab is None else vocab
    query, pos, role_tokens = _inputs(canvas, query_pos, vocab)
    rows, cols = _role_positions(canvas, query, pos)
    if (cols + 1 >= canvas.shape[2]).any():
        raise ValueError("a fact role has no right neighbor")
    prediction = canvas[torch.arange(len(canvas), device=canvas.device), rows, cols + 1]
    if ((prediction == 0) | torch.isin(prediction, role_tokens)).any():
        raise ValueError("the right neighbor is not a filler; adjacent schema required")
    return prediction


def same_row_exact(
    canvas: torch.Tensor, *, query_pos: tuple[int, int] = (-1, -2), vocab: Vocab | None = None
) -> torch.Tensor:
    """Read the sole filler in the queried fact's row, regardless of its distance."""
    vocab = Vocab() if vocab is None else vocab
    query, pos, role_tokens = _inputs(canvas, query_pos, vocab)
    rows, _ = _role_positions(canvas, query, pos)
    cols = _row_filler_positions(canvas, rows, role_tokens)
    return canvas[torch.arange(len(canvas), device=canvas.device), rows, cols]


def category_probabilities(
    canvas: torch.Tensor, *, query_pos: tuple[int, int] = (-1, -2), vocab: Vocab | None = None
) -> torch.Tensor:
    """Return a uniform distribution over visible candidates of the query category.

Only the query token and the bag of visible token identities are used. Fact role
locations, word positions, and agent/patient associations are deliberately ignored.
The public category mapping is nouns for AGENTE/PACIENTE, verbs for ACCION, and
places for LUGAR. Probability mass is uniform over distinct candidate token IDs.
"""
    vocab = Vocab() if vocab is None else vocab
    query, _, _ = _inputs(canvas, query_pos, vocab)
    category = torch.zeros((len(vocab), len(vocab)), dtype=torch.bool, device=canvas.device)
    for role, words in zip(ROLES, (NOUNS, VERBS, NOUNS, PLACES)):
        category[vocab.idx[role], vocab.ids(words)] = True
    visible = torch.zeros((len(canvas), len(vocab)), dtype=torch.bool, device=canvas.device)
    visible.scatter_(1, canvas.flatten(1), True)
    candidates = visible & category[query]
    count = candidates.sum(1, keepdim=True)
    if (count == 0).any():
        raise ValueError("a query has no visible candidate in its category")
    return candidates.to(torch.float64) / count


def category_seeded_draw(
    canvas: torch.Tensor, *, seed: int, query_pos: tuple[int, int] = (-1, -2),
    vocab: Vocab | None = None
) -> torch.Tensor:
    """Draw once per input from the category distribution with a private RNG.

Resetting this RNG for an equal probability matrix gives equal predictions. This
provides common random numbers for original/counterfactual comparisons without
consulting either target or consuming the experiment's global RNG state.
"""
    probabilities = category_probabilities(canvas, query_pos=query_pos, vocab=vocab)
    generator = torch.Generator(device=canvas.device).manual_seed(seed)
    return torch.multinomial(probabilities, 1, generator=generator).squeeze(1)


def swap_agent_patient(
    canvas: torch.Tensor, *, query_pos: tuple[int, int] = (-1, -2), vocab: Vocab | None = None
) -> torch.Tensor:
    """Swap the two noun fillers while preserving roles, query, layout and token bag.

This transformation uses visible fact rows only. Its correctness and the changed
target are checked separately against generator metadata by the evaluator.
"""
    vocab = Vocab() if vocab is None else vocab
    _, pos, role_tokens = _inputs(canvas, query_pos, vocab)
    positions = []
    for role in ("AGENTE", "PACIENTE"):
        ids = torch.full((len(canvas),), vocab.idx[role], device=canvas.device, dtype=torch.long)
        rows, _ = _role_positions(canvas, ids, pos)
        cols = _row_filler_positions(canvas, rows, role_tokens)
        positions.append((rows, cols))
    batch = torch.arange(len(canvas), device=canvas.device)
    agent, patient = positions
    agent_ids, patient_ids = canvas[batch, *agent], canvas[batch, *patient]
    nouns = torch.tensor(vocab.ids(NOUNS), device=canvas.device)
    if not (torch.isin(agent_ids, nouns) & torch.isin(patient_ids, nouns)).all():
        raise ValueError("agent and patient fillers must be public noun tokens")
    if (agent_ids == patient_ids).any():
        raise ValueError("counterfactuals require different agent and patient nouns")
    result = canvas.clone()
    result[batch, *agent], result[batch, *patient] = patient_ids, agent_ids
    return result
