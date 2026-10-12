"""Prospective graph-composition data with repeated relations and an exact oracle."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from .common import identity, sha256_file, write_json


def oracle(maps: np.ndarray, start: int, query: list[int]) -> int:
    """Follow the ordered relation composition; each relation is a permutation."""
    node = start
    for relation in query:
        node = int(maps[relation, node])
    return node


def encode(maps: np.ndarray, start: int, query: list[int], rng: np.random.Generator,
           grid: int = 8, maximum_depth: int = 4) -> np.ndarray:
    """Render source/destination/relation facts and an ordered query without labels."""
    relations, entities = maps.shape
    channels = 2 * entities + relations + maximum_depth + 3
    field = np.zeros((channels, grid, grid), dtype=np.float64)
    output = (grid // 2) * grid + grid // 2
    positions = rng.permutation([i for i in range(grid * grid) if i != output]).tolist()
    if relations * entities + 1 + len(query) > len(positions) or len(query) > maximum_depth:
        raise ValueError("Canvas or query capacity exceeded")
    relation_offset = 2 * entities
    ordinal_offset = relation_offset + relations
    start_flag, fact_flag, output_flag = ordinal_offset + maximum_depth + np.arange(3)
    at = 0
    for relation in range(relations):
        for source in range(entities):
            row, column = divmod(positions[at], grid)
            at += 1
            field[source, row, column] = 1
            field[entities + maps[relation, source], row, column] = 1
            field[relation_offset + relation, row, column] = 1
            field[fact_flag, row, column] = 1
    row, column = divmod(positions[at], grid)
    at += 1
    field[start, row, column] = 1
    field[start_flag, row, column] = 1
    for ordinal, relation in enumerate(query):
        row, column = divmod(positions[at], grid)
        at += 1
        field[relation_offset + relation, row, column] = 1
        field[ordinal_offset + ordinal, row, column] = 1
    field[output_flag, grid // 2, grid // 2] = 1
    return field


def decode_oracle(field: np.ndarray, entities: int = 12, relations: int = 3,
                  maximum_depth: int = 4) -> int:
    """Reconstruct the facts/query from the INPUT alone and compute the answer."""
    ordinal_offset = 2 * entities + relations
    start_flag, fact_flag = ordinal_offset + maximum_depth + np.arange(2)
    maps = np.full((relations, entities), -1, dtype=int)
    for row, column in zip(*np.where(field[fact_flag] == 1)):
        source = int(field[:entities, row, column].argmax())
        destination = int(field[entities:2*entities, row, column].argmax())
        relation = int(field[2*entities:ordinal_offset, row, column].argmax())
        if maps[relation, source] != -1:
            raise ValueError("Ambiguous input fact")
        maps[relation, source] = destination
    starts = np.argwhere(field[start_flag] == 1)
    if len(starts) != 1 or (maps < 0).any():
        raise ValueError("Incomplete graph or start")
    row, column = starts[0]
    start = int(field[:entities, row, column].argmax())
    query = []
    for ordinal in range(maximum_depth):
        cells = np.argwhere(field[ordinal_offset + ordinal] == 1)
        if len(cells) == 0:
            break
        if len(cells) != 1:
            raise ValueError("Ambiguous query ordinal")
        row, column = cells[0]
        query.append(int(field[2*entities:ordinal_offset, row, column].argmax()))
    return oracle(maps, start, query)


def make_dataset(output: Path, *, seed: int = 20261012, graph_counts=(96, 24, 24),
                 examples_per_graph: int = 12, entities: int = 12, relations: int = 3) -> dict:
    """Create disjoint graph groups; base/long TEST share groups and are paired."""
    if min(graph_counts) < 1 or examples_per_graph < 1 or entities < 2 or relations < 2:
        raise ValueError("Invalid dataset dimensions")
    if output.exists():
        raise FileExistsError("Dataset output exists; use a new registered directory")
    output.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    used, metadata = set(), {}
    panels = {name: [] for name in ("train", "dev", "test_base", "test_long")}
    for split, count in zip(("train", "dev", "test"), graph_counts):
        for _ in range(count):
            while True:
                maps = np.stack([rng.permutation(entities) for _ in range(relations)])
                graph_id = identity(maps.tolist())
                if graph_id not in used:
                    used.add(graph_id)
                    break
            names = ("test_base", "test_long") if split == "test" else (split,)
            for name in names:
                for _ in range(examples_per_graph):
                    depth = 4 if name == "test_long" else int(rng.integers(2, 4))
                    query = rng.integers(0, relations, size=depth).tolist()
                    start = int(rng.integers(entities))
                    field = encode(maps, start, query, rng)
                    answer = oracle(maps, start, query)
                    if decode_oracle(field, entities, relations) != answer:
                        raise AssertionError("Encoded-input oracle differs")
                    panels[name].append((field, answer, graph_id, depth))
    groups = {name: {item[2] for item in rows} for name, rows in panels.items()}
    if (groups["train"] & groups["dev"] or groups["train"] & groups["test_base"]
            or groups["dev"] & groups["test_base"]):
        raise AssertionError("Graph-group leakage")
    for name, rows in panels.items():
        path = output / (name + ".npz")
        inputs = np.stack([item[0] for item in rows])
        targets = np.array([item[1] for item in rows], dtype=np.int64)
        np.savez_compressed(path, inputs=inputs, targets=targets,
                            groups=np.array([item[2] for item in rows]),
                            depths=np.array([item[3] for item in rows], dtype=np.int64))
        metadata[name] = dict(path=path.name, sha256=sha256_file(path), examples=len(rows),
                              graph_groups=len(groups[name]))
    document = dict(schema=1, task="ordered_relation_composition", seed=seed,
                    entities=entities, relations=relations, classes=entities,
                    maximum_depth=4, grid=8, input_channels=2*entities+relations+7,
                    examples_per_graph=examples_per_graph, panels=metadata,
                    test_custody="Locally generated; not independent blind custody",
                    grouping="Exact relation-map identities; not an isomorphism-class guarantee",
                    oracle_scope="Exact symbolic input oracle, not a learned baseline",
                    long_test_is_independent_replication=False)
    write_json(output / "manifest.json", document, exclusive=True)
    return document


def main() -> None:
    """Build a new immutable development dataset without training or scoring models."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--graphs", type=int, nargs=3, default=[96, 24, 24])
    parser.add_argument("--examples-per-graph", type=int, default=12)
    args = parser.parse_args()
    print(json.dumps(make_dataset(args.output, seed=args.seed, graph_counts=args.graphs,
                                 examples_per_graph=args.examples_per_graph), indent=2))


if __name__ == "__main__":
    main()
