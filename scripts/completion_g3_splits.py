"""G3: separation units, family-grouped splits and the fixed DEV checkpoint selector.

Separation unit for the ordered relation-composition task: the *structural family* of a relation
graph, i.e. its class under any renaming of entities and any permutation of relation labels.
Renamed variants of a graph are generated inside their family, and families never cross splits.
TEST is not generated here: it belongs to the designated evaluator (guide sections 8 and 18F).
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import deque
from pathlib import Path

import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from completion_kit.common import identity, sha256_file, write_json
from completion_kit.datasets import decode_oracle, encode, oracle


def _component_code(maps: np.ndarray, inverse: np.ndarray, members: list[int]) -> tuple:
    """Smallest breadth-first relabelling code of one connected component (complete invariant)."""
    best = None
    for start in members:
        label = {start: 0}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for relation in range(maps.shape[0]):
                for nxt in (int(maps[relation, node]), int(inverse[relation, node])):
                    if nxt not in label:
                        label[nxt] = len(label)
                        queue.append(nxt)
        order = sorted(label, key=label.get)
        code = tuple(tuple(label[int(maps[r, node])] for node in order) for r in range(maps.shape[0]))
        if best is None or code < best:
            best = code
    return best


def entity_canonical(maps: np.ndarray) -> tuple:
    """Canonical form of a tuple of permutations under simultaneous entity renaming."""
    maps = np.asarray(maps, dtype=np.int64)
    relations, entities = maps.shape
    for row in maps:
        if sorted(row.tolist()) != list(range(entities)):
            raise ValueError("Every relation must be a permutation of the entities")
    inverse = np.empty_like(maps)
    for r in range(relations):
        inverse[r, maps[r]] = np.arange(entities)
    seen, codes = set(), []
    for root in range(entities):
        if root in seen:
            continue
        members, queue = [root], deque([root])
        seen.add(root)
        while queue:
            node = queue.popleft()
            for r in range(relations):
                for nxt in (int(maps[r, node]), int(inverse[r, node])):
                    if nxt not in seen:
                        seen.add(nxt)
                        members.append(nxt)
                        queue.append(nxt)
        codes.append(_component_code(maps, inverse, members))
    return tuple(sorted(codes))


def family_id(maps: np.ndarray) -> str:
    """Structural family identity: invariant to entity renaming and relation-label permutation."""
    maps = np.asarray(maps, dtype=np.int64)
    forms = [entity_canonical(maps[list(order)]) for order in itertools.permutations(range(maps.shape[0]))]
    return identity([list(map(list, component)) for component in min(forms)])


def rename(maps: np.ndarray, entity_perm: np.ndarray, relation_perm: np.ndarray) -> np.ndarray:
    """Variant of the same family: new[relation_perm[r]] = phi o maps[r] o phi^-1."""
    maps = np.asarray(maps, dtype=np.int64)
    phi = np.asarray(entity_perm, dtype=np.int64)
    out = np.empty_like(maps)
    for r in range(maps.shape[0]):
        out[relation_perm[r], phi] = phi[maps[r]]
    return out


def select_checkpoint(records: list[dict]) -> dict:
    """Fixed DEV selector: most DEV correct, then lowest DEV NLL, then earliest epoch; never by TEST or seed."""
    if not records:
        raise ValueError("No checkpoint records")
    for item in records:
        if not {"epoch", "dev_correct", "dev_nll"} <= item.keys():
            raise ValueError("Each record needs epoch, dev_correct and dev_nll")
    return min(records, key=lambda item: (-int(item["dev_correct"]), float(item["dev_nll"]), int(item["epoch"])))


def make_family_dataset(output: Path, *, seed: int, families: dict, variants: int, examples_per_variant: int,
                        entities: int = 12, relations: int = 3) -> dict:
    """Train/DEV panels whose structural families are disjoint; variants stay inside their family."""
    if output.exists():
        raise FileExistsError("Use a new registered directory")
    output.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    used_families, used_exact, panels, family_split = set(), set(), {}, {}
    for split, count in families.items():
        names = [split, split + "_long"] if split == "dev" else [split]
        for name in names:
            panels[name] = []
        for _ in range(count):
            while True:
                base = np.stack([rng.permutation(entities) for _ in range(relations)])
                fam = family_id(base)
                if fam not in used_families:
                    used_families.add(fam)
                    family_split[fam] = split
                    break
            for _ in range(variants):
                variant = rename(base, rng.permutation(entities), rng.permutation(relations))
                if family_id(variant) != fam:
                    raise AssertionError("Renaming left the family")
                used_exact.add(identity(variant.tolist()))
                for name in names:
                    for _ in range(examples_per_variant):
                        depth = 4 if name.endswith("_long") else int(rng.integers(2, 4))
                        query = rng.integers(0, relations, size=depth).tolist()
                        start = int(rng.integers(entities))
                        field = encode(variant, start, query, rng)
                        answer = oracle(variant, start, query)
                        if decode_oracle(field, entities, relations) != answer:
                            raise AssertionError("Encoded-input oracle differs")
                        panels[name].append((field.astype(np.uint8), answer, fam, identity(variant.tolist()), depth))
    families_by_panel = {name: {row[2] for row in rows} for name, rows in panels.items()}
    if families_by_panel["train"] & families_by_panel["dev"]:
        raise AssertionError("Family leakage between train and dev")
    metadata = {}
    for name, rows in panels.items():
        path = output / (name + ".npz")
        np.savez_compressed(path, inputs=np.stack([r[0] for r in rows]), targets=np.array([r[1] for r in rows], dtype=np.int64),
                            families=np.array([r[2] for r in rows]), exact_graphs=np.array([r[3] for r in rows]),
                            depths=np.array([r[4] for r in rows], dtype=np.int64))
        metadata[name] = dict(path=path.name, sha256=sha256_file(path), examples=len(rows),
                              families=len(families_by_panel[name]), exact_graphs=len({r[3] for r in rows}))
    excluded = sorted(used_families)
    (output / "excluded_families.json").write_text(json.dumps(excluded, indent=0) + "\n", encoding="utf-8", newline="\n")
    document = dict(schema=1, task="ordered_relation_composition", seed=seed, entities=entities, relations=relations,
                    input_dtype="uint8 (binary channels; cast to float before use)", families=families, variants=variants,
                    examples_per_variant=examples_per_variant, panels=metadata,
                    separation_unit="structural family: class under entity renaming and relation-label permutation",
                    excluded_families=dict(path="excluded_families.json", count=len(excluded),
                                           sha256=sha256_file(output / "excluded_families.json")),
                    dev_long_is_independent_replication=False,
                    test="Not generated here: the designated evaluator generates TEST excluding every listed family")
    write_json(output / "manifest.json", document, exclusive=True)
    return document


def audit_family_overlap(panel_paths: dict) -> dict:
    """Check structural-family overlap between panels of an existing kit dataset (exact-identity grouping)."""
    out = {}
    families = {}
    for name, path in panel_paths.items():
        data = np.load(path, allow_pickle=False)
        families[name] = set()
        for field in data["inputs"]:
            families[name].add(family_id(_maps_from_field(field)))
        out[name] = dict(examples=int(len(data["inputs"])), families=len(families[name]))
    names = sorted(families)
    out["overlaps"] = {f"{a}|{b}": len(families[a] & families[b]) for a, b in itertools.combinations(names, 2)}
    return out


def _maps_from_field(field: np.ndarray, entities: int = 12, relations: int = 3, maximum_depth: int = 4) -> np.ndarray:
    """Recover the relation maps from an encoded input alone (facts carry source, destination and relation)."""
    fact_flag = 2 * entities + relations + maximum_depth + 1
    maps = -np.ones((relations, entities), dtype=np.int64)
    for row, column in zip(*np.nonzero(field[fact_flag])):
        source = int(np.argmax(field[:entities, row, column]))
        destination = int(np.argmax(field[entities:2 * entities, row, column]))
        relation = int(np.argmax(field[2 * entities:2 * entities + relations, row, column]))
        maps[relation, source] = destination
    if (maps < 0).any():
        raise ValueError("Incomplete facts in field")
    return maps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train-families", type=int, default=96)
    parser.add_argument("--dev-families", type=int, default=24)
    parser.add_argument("--variants", type=int, default=3)
    parser.add_argument("--examples-per-variant", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(make_family_dataset(args.output, seed=args.seed,
                                         families={"train": args.train_families, "dev": args.dev_families},
                                         variants=args.variants, examples_per_variant=args.examples_per_variant), indent=2))


if __name__ == "__main__":
    main()


def make_family_test(output: Path, *, seed: int, families: int, variants: int, examples_per_variant: int,
                     excluded: list[str], entities: int = 12, relations: int = 3) -> dict:
    """Evaluator-side TEST: new structural families only (every listed family is rejected); base and long panels paired."""
    if output.exists():
        raise FileExistsError("Use a new registered directory")
    excluded_set = set(excluded)
    output.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    used, panels = set(), {"test_base": [], "test_long": []}
    for _ in range(families):
        while True:
            base = np.stack([rng.permutation(entities) for _ in range(relations)])
            fam = family_id(base)
            if fam not in excluded_set and fam not in used:
                used.add(fam)
                break
        for _ in range(variants):
            variant = rename(base, rng.permutation(entities), rng.permutation(relations))
            for name in panels:
                for _ in range(examples_per_variant):
                    depth = 4 if name == "test_long" else int(rng.integers(2, 4))
                    query = rng.integers(0, relations, size=depth).tolist()
                    start = int(rng.integers(entities))
                    field = encode(variant, start, query, rng)
                    answer = oracle(variant, start, query)
                    if decode_oracle(field, entities, relations) != answer:
                        raise AssertionError("Encoded-input oracle differs")
                    panels[name].append((field.astype(np.uint8), answer, fam, depth))
    metadata = {}
    for name, rows in panels.items():
        path = output / (name + ".npz")
        np.savez_compressed(path, inputs=np.stack([r[0] for r in rows]), targets=np.array([r[1] for r in rows], dtype=np.int64),
                            families=np.array([r[2] for r in rows]), depths=np.array([r[3] for r in rows], dtype=np.int64))
        metadata[name] = dict(path=path.name, sha256=sha256_file(path), examples=len(rows), families=len({r[2] for r in rows}))
    document = dict(schema=1, seed=seed, families=families, variants=variants, examples_per_variant=examples_per_variant,
                    excluded_families_sha256=identity(sorted(excluded_set)), overlap_with_excluded=len(used & excluded_set),
                    panels=metadata, long_test_is_independent_replication=False)
    write_json(output / "manifest.json", document, exclusive=True)
    return document
