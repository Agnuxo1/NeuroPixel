"""Frozen three-partition data and private CPU sampling for the neural pilot.

The original generator remains unchanged. ResearchRoleTask preserves its first
20% test triples, reserves the next 10% for validation, and trains on the final
70%. Every random draw occurs on CPU before any requested device transfer.
"""
from __future__ import annotations

from numbers import Integral

import torch

from neuropixel.task import NOUNS, PLACES, ROLES, VERBS, RoleTask


class ResearchRoleTask(RoleTask):
    """Adjacent RoleTask with the protocol's fixed 924/132/264 triple partition.

    ``seed`` controls the triple split only. Sampling requires a separate, explicit
    CPU ``torch.Generator``; model initialization and update randomness cannot become
    implicit data seeds. Public partition tuples are immutable and their cached CPU
    tensors are separate for train, validation, and test.
    """

    def __init__(self, h: int = 8, w: int = 8, seed: int = 0):
        with torch.device("cpu"):
            super().__init__(h=h, w=w, heldout_frac=0.2, seed=seed)
        self.split_seed = seed
        original_train = self.train_triples
        self.test_triples = tuple(self.test_triples)
        self.validation_triples = tuple(original_train[:132])
        self.train_triples = tuple(original_train[132:])
        self._triples_cpu = {
            "train": torch.tensor(self.train_triples, dtype=torch.long, device="cpu"),
            "validation": torch.tensor(self.validation_triples, dtype=torch.long, device="cpu"),
            "test": torch.tensor(self.test_triples, dtype=torch.long, device="cpu"),
        }
        if tuple(len(self._triples_cpu[k]) for k in ("train", "validation", "test")) != (924, 132, 264):
            raise ValueError("the frozen protocol requires the original 1,320-triple vocabulary")
        self._nouns_cpu = torch.tensor(self.v.ids(NOUNS), dtype=torch.long, device="cpu")
        self._verbs_cpu = torch.tensor(self.v.ids(VERBS), dtype=torch.long, device="cpu")
        self._places_cpu = torch.tensor(self.v.ids(PLACES), dtype=torch.long, device="cpu")
        self.role_ids = torch.tensor(self.v.ids(ROLES), dtype=torch.long, device="cpu")

    def sample(
        self, batch: int, split: str = "train", generator: torch.Generator | None = None,
        device: str | torch.device = "cpu", query_role: int | None = None,
        place_pool: list[int] | None = None, meta: bool = False,
    ):
        """Sample from one named partition on CPU, then transfer all returned tensors.

        The call signature follows RoleTask. Accepted split names are exactly ``train``,
        ``validation``, and ``test``. A missing or non-CPU generator is an error rather
        than a silent fallback to global/device randomness. Metadata has the original
        ``rows``, ``cols`` and ``fillers`` fields and follows the requested destination.
        """
        if not isinstance(batch, Integral) or isinstance(batch, bool) or batch <= 0:
            raise ValueError("batch must be a positive integer")
        if split not in self._triples_cpu:
            raise ValueError("split must be 'train', 'validation', or 'test'")
        if generator is None:
            raise ValueError("an explicit independent CPU torch.Generator is required")
        if not isinstance(generator, torch.Generator) or generator.device.type != "cpu":
            raise ValueError("sampling requires a CPU torch.Generator, independent of the output device")
        if query_role is not None and (not isinstance(query_role, Integral) or
                                       isinstance(query_role, bool) or not 0 <= query_role < len(ROLES)):
            raise ValueError("query_role must be a role index from 0 through 3")
        destination = torch.device(device)
        batch = int(batch)
        triples = self._triples_cpu[split]
        pool = self._places_cpu if place_pool is None else torch.as_tensor(
            place_pool, dtype=torch.long, device="cpu")
        if pool.ndim != 1 or len(pool) == 0 or not torch.isin(pool, self._places_cpu).all():
            raise ValueError("place_pool must contain one or more public place token IDs")

        # Preserve the original CPU sampler's random-call order and float32 layout draws.
        indices = torch.randint(len(triples), (batch,), generator=generator, device="cpu")
        selected = triples[indices]
        places = pool[torch.randint(len(pool), (batch,), generator=generator, device="cpu")]
        fillers = torch.stack([self._nouns_cpu[selected[:, 0]], self._verbs_cpu[selected[:, 1]],
                               self._nouns_cpu[selected[:, 2]], places], dim=1)
        rows = torch.rand(batch, self.h - 1, generator=generator, dtype=torch.float32,
                          device="cpu").argsort(1)[:, :len(ROLES)]
        cols = torch.randint(self.w - 1, (batch, len(ROLES)), generator=generator, device="cpu")
        index = torch.arange(batch, dtype=torch.long, device="cpu")[:, None].expand(-1, len(ROLES))
        canvas = torch.zeros(batch, self.h, self.w, dtype=torch.long, device="cpu")
        canvas[index, rows, cols] = self.role_ids.expand(batch, -1)
        canvas[index, rows, cols + 1] = fillers
        query = torch.randint(len(ROLES), (batch,), generator=generator, device="cpu")
        if query_role is not None:
            query.fill_(int(query_role))
        canvas[:, self.query_pos[0], self.query_pos[1]] = self.role_ids[query]
        target = fillers.gather(1, query[:, None]).squeeze(1)
        result = (canvas.to(destination), target.to(destination))
        if meta:
            metadata = {"rows": rows.to(destination), "cols": cols.to(destination),
                        "fillers": fillers.to(destination)}
            return *result, metadata
        return result


def frozen_dataset(
    task: ResearchRoleTask, split: str, n: int, seed: int, device: str | torch.device = "cpu"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return an equal-role dataset in ROLES order without consuming global RNG.

    The third tensor contains role indices 0/1/2/3. One private CPU generator is
    initialized once with ``seed`` and consumed sequentially for the four role blocks.
    All samples are created on CPU, then the completed dataset moves to ``device``.
    """
    if not isinstance(n, Integral) or isinstance(n, bool) or n <= 0 or n % len(ROLES):
        raise ValueError("n must be a positive integer multiple of four")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    batch = int(n) // len(ROLES)
    canvases, targets = [], []
    for role in range(len(ROLES)):
        canvas, target = task.sample(batch, split, generator, "cpu", query_role=role)
        canvases.append(canvas)
        targets.append(target)
    roles = torch.arange(len(ROLES), dtype=torch.long, device="cpu").repeat_interleave(batch)
    destination = torch.device(device)
    return (torch.cat(canvases).to(destination), torch.cat(targets).to(destination), roles.to(destination))
