"""Small deterministic grammar fixtures, never the item-9 study populations."""
import copy
import hashlib
import json
import random
import unittest

from neuropixel.research.complex_binding_data import (
    CONDITIONS, CONTROLS, GROUP_UNIVERSE_SIZE, canonical_group, control_weights,
    fixture_exposure_ledger, generate_scenarios, group_id, randomize_scenario, render_scenario,
    split_for_group, symbolic_answer, validate_scenario,
)


def manual_scenario():
    """A hand-specified semantic fixture, independent of the random generator."""
    bag = {"nouns": [5, 6, 7, 8], "verbs": [17, 18], "places": [27, 28]}
    truth = ((5, 17, 6, 27), (7, 18, 8, 28))
    facts = [{"event": event, "role": role, "filler": truth[event][role]}
             for event in range(2) for role in range(4)]
    return {"schema_version": 1, "group_id": group_id(bag), "bag": bag,
            "facts": facts, "layout": [{"row": row, "start": row % 6} for row in range(8)]}


def manual_canvas(event, role):
    """Literal triples provide parser truth without calling the renderer."""
    triples = [(35, 1, 5), (35, 2, 17), (35, 3, 6), (35, 4, 27),
               (36, 1, 7), (36, 2, 18), (36, 3, 8), (36, 4, 28)]
    canvas = [[0] * 8 for _ in range(10)]
    for row, triple in enumerate(triples):
        canvas[row][0:3] = triple
    canvas[9][5:7] = [35 + event, 1 + role]
    return canvas


class ComplexBindingDataTests(unittest.TestCase):
    def test_canonical_group_hash_and_membership_ignore_order(self):
        bag = manual_scenario()["bag"]
        canonical = b'{"nouns":[5,6,7,8],"places":[27,28],"verbs":[17,18]}'
        expected = hashlib.sha256(canonical).hexdigest()
        self.assertEqual(group_id(bag), expected)
        scrambled = {key: list(reversed(value)) for key, value in bag.items()}
        self.assertEqual(canonical_group(scrambled), bag)
        self.assertEqual(group_id(scrambled), expected)
        bucket = int(expected, 16) % 100
        self.assertEqual(split_for_group(bag), "train" if bucket < 70 else "validation" if bucket < 85 else "final")
        self.assertEqual(GROUP_UNIVERSE_SIZE, 623700)

    def test_group_rejects_category_collision_and_wrong_cardinality(self):
        for key, values in (("nouns", [5, 5, 7, 8]), ("nouns", [5, 6, 7]),
                            ("verbs", [17, 27]), ("places", [27, 35])):
            bag = manual_scenario()["bag"]
            bag[key] = values
            with self.subTest(key=key, values=values), self.assertRaises(ValueError):
                canonical_group(bag)

    def test_parser_matches_manual_truth_for_all_eight_queries(self):
        truth = ((5, 17, 6, 27), (7, 18, 8, 28))
        for event in range(2):
            for role in range(4):
                with self.subTest(event=event, role=role):
                    self.assertEqual(symbolic_answer(manual_canvas(event, role)), truth[event][role])

    def test_all_transform_targets_match_independent_semantic_fixture(self):
        scenario = manual_scenario()
        truth = ((5, 17, 6, 27), (7, 18, 8, 28))
        for condition in CONDITIONS:
            for event in range(2):
                for role in range(4):
                    expected = truth[event][role]
                    if condition == "swap_queried_agent_patient" and role in (0, 2):
                        expected = truth[event][2 - role]
                    elif condition == "query_switch":
                        expected = truth[1 - event][role]
                    record = render_scenario(scenario, event, role, condition)
                    with self.subTest(condition=condition, event=event, role=role):
                        self.assertEqual(record["target"], expected)
                        self.assertEqual(record["base_target"], truth[event][role])
                        self.assertEqual(record["changed_gold"], expected != truth[event][role])
                        self.assertEqual(record["group_id"], scenario["group_id"])
                        self.assertEqual(record["query_event"], 1-event if condition in ("relabel_events", "query_switch") else event)

    def test_renderer_geometry_and_shared_query_layouts(self):
        scenario = manual_scenario()
        for condition in ("base", "relabel_events", "query_switch", "layout_permutation"):
            records = [render_scenario(scenario, event, role, condition) for event in range(2) for role in range(4)]
            self.assertTrue(all(record["canvas"][:9] == records[0]["canvas"][:9] for record in records))
            for record in records:
                self.assertEqual(len(record["canvas"]), 10)
                self.assertTrue(all(len(row) == 8 for row in record["canvas"]))
                self.assertEqual(sum(x != 0 for row in record["canvas"] for x in row), 26)
                self.assertEqual(record["canvas"][9][7], 0)
        self.assertNotEqual(render_scenario(scenario, 0, 0)["canvas"][:9],
                            render_scenario(scenario, 0, 0, "layout_permutation")["canvas"][:9])

    def test_pair_ids_and_legitimate_duplicate_records(self):
        scenario = manual_scenario()
        records = [render_scenario(scenario, 0, 1, condition) for condition in CONDITIONS]
        self.assertEqual(len({record["pair_id"] for record in records}), 1)
        self.assertEqual(len({record["record_id"] for record in records}), 6)
        base_other = render_scenario(scenario, 1, 1)
        switched = render_scenario(scenario, 0, 1, "query_switch")
        self.assertEqual(base_other["canvas"], switched["canvas"])
        self.assertEqual(base_other["target"], switched["target"])
        self.assertNotEqual(base_other["record_id"], switched["record_id"])

    def test_scene_bag_invariant_but_query_switch_changes_full_input_bag(self):
        scenario = manual_scenario()
        base = render_scenario(scenario, 0, 0)
        scene = sorted(token for row in base["canvas"][:9] for token in row)
        for condition in CONDITIONS:
            record = render_scenario(scenario, 0, 0, condition)
            self.assertEqual(sorted(token for row in record["canvas"][:9] for token in row), scene)
        switched = render_scenario(scenario, 0, 0, "query_switch")
        self.assertNotEqual(sorted(sum(base["canvas"], [])), sorted(sum(switched["canvas"], [])))

    def test_controls_are_probabilities_with_exact_conditional_expectations(self):
        expected = {"symbolic": (1.0, 1.0), "role_only": (0.5, 0.5),
                    "event_category": (0.5, 0.75), "bag_category": (0.25, 0.375)}
        for control in CONTROLS:
            all_probabilities, binding_probabilities = [], []
            for event in range(2):
                for role in range(4):
                    canvas = manual_canvas(event, role)
                    weights = control_weights(canvas, control)
                    self.assertEqual(sum(weights.values()), 1.0)
                    self.assertTrue(all(0 < value <= 1 for value in weights.values()))
                    gold = ((5, 17, 6, 27), (7, 18, 8, 28))[event][role]
                    all_probabilities.append(weights.get(gold, 0))
                    if role in (0, 2):
                        binding_probabilities.append(weights.get(gold, 0))
            self.assertEqual(sum(binding_probabilities)/4, expected[control][0])
            self.assertEqual(sum(all_probabilities)/8, expected[control][1])

    def test_input_only_controls_cannot_read_targets_or_scenario_metadata(self):
        canvas = manual_canvas(0, 0)
        with self.assertRaises(TypeError):
            control_weights(canvas, "bag_category", target=5)
        with self.assertRaises(TypeError):
            symbolic_answer(canvas, metadata={"answer": 5})
        with self.assertRaises(ValueError):
            control_weights(canvas, "unregistered_oracle")

    def test_same_shortcut_distribution_faces_changed_counterfactual_gold(self):
        scenario = manual_scenario()
        base = render_scenario(scenario, 0, 0)
        swapped = render_scenario(scenario, 0, 0, "swap_queried_agent_patient")
        switched = render_scenario(scenario, 0, 0, "query_switch")
        for control in ("bag_category", "event_category"):
            self.assertEqual(control_weights(base["canvas"], control), control_weights(swapped["canvas"], control))
        self.assertEqual(control_weights(base["canvas"], "role_only"), control_weights(switched["canvas"], "role_only"))
        self.assertNotEqual(base["target"], swapped["target"])
        self.assertNotEqual(base["target"], switched["target"])

    def test_seeded_generation_reproducible_and_global_rng_independent(self):
        state = random.getstate()
        first = generate_scenarios("train", 5, 910901)
        self.assertEqual(random.getstate(), state)
        random.seed(910902)
        try:
            self.assertEqual(first, generate_scenarios("train", 5, 910901))
        finally:
            random.setstate(state)

    def test_small_fixture_partitions_unique_and_disjoint(self):
        pools = {split: generate_scenarios(split, 7, 910910 + index)
                 for index, split in enumerate(("train", "validation", "final"))}
        all_ids = []
        for split, scenarios in pools.items():
            for scenario in scenarios:
                validate_scenario(scenario)
                self.assertEqual(split_for_group(scenario["bag"]), split)
                all_ids.append(scenario["group_id"])
        self.assertEqual(len(set(all_ids)), 21)

    def test_randomization_preserves_bag_split_and_original(self):
        original = manual_scenario()
        snapshot = copy.deepcopy(original)
        left, right = random.Random(910920), random.Random(910920)
        first = randomize_scenario(original, left)
        self.assertEqual(first, randomize_scenario(original, right))
        self.assertEqual(first["bag"], original["bag"])
        self.assertEqual(first["group_id"], original["group_id"])
        self.assertEqual(split_for_group(first["bag"]), split_for_group(original["bag"]))
        self.assertNotEqual(first, original)
        self.assertEqual(original, snapshot)
        for condition in CONDITIONS:
            render_scenario(original, 0, 0, condition)
        self.assertEqual(original, snapshot)

    def test_invalid_generation_inputs_and_finite_rejection_bound(self):
        for args in (("test", 1, 1), ("train", 0, 1), ("train", 1, "1"),
                     ("final", GROUP_UNIVERSE_SIZE+1, 1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                generate_scenarios(*args)
        with self.assertRaisesRegex(RuntimeError, "rejection bound exhausted"):
            generate_scenarios("train", 2, 910930, max_attempts=1)

    def test_explicit_group_exclusion_and_fixture_ledger(self):
        first = generate_scenarios("train", 5, 910901)[0]["group_id"]
        excluded = generate_scenarios("train", 4, 910901, excluded_groups=[first])
        self.assertNotIn(first, {scenario["group_id"] for scenario in excluded})
        ledger = fixture_exposure_ledger()
        self.assertEqual(ledger, fixture_exposure_ledger())
        self.assertIn(first, ledger["excluded_groups"])
        self.assertTrue({scenario["group_id"] for scenario in excluded} <= set(ledger["excluded_groups"]))
        self.assertTrue(set(ledger["constructed_group_ids"]) <= set(ledger["excluded_groups"]))
        self.assertEqual(len(ledger["generation_runs"]), 6)
        self.assertFalse({run["seed"] for run in ledger["generation_runs"]} & set(range(91001, 91006)))
        for invalid in (None, "0"*64, ["x"*64], ["0"*63], [True]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                generate_scenarios("train", 1, 910901, excluded_groups=invalid)

    def test_invalid_canvas_shapes_query_and_output(self):
        invalid = []
        invalid.append(manual_canvas(0, 0)[:-1])
        canvas = manual_canvas(0, 0); canvas[0] = canvas[0][:-1]; invalid.append(canvas)
        for column, value in ((7, 5), (5, 1), (6, 35), (0, 17)):
            canvas = manual_canvas(0, 0); canvas[9][column] = value; invalid.append(canvas)
        for canvas in invalid:
            with self.subTest(canvas=canvas), self.assertRaises(ValueError):
                symbolic_answer(canvas)

    def test_invalid_visible_fact_grammar_categories_and_duplicates(self):
        invalid = []
        for row, column, value in ((4, 0, 35), (0, 2, 17), (2, 2, 5), (0, 3, 5), (0, 0, 37)):
            canvas = manual_canvas(0, 0); canvas[row][column] = value; invalid.append(canvas)
        canvas = manual_canvas(0, 0); canvas[7] = [0]*8; invalid.append(canvas)
        for canvas in invalid:
            with self.subTest(canvas=canvas), self.assertRaises(ValueError):
                symbolic_answer(canvas)

    def test_invalid_scenario_truth_layout_and_group_hash(self):
        invalid = []
        scenario = manual_scenario(); scenario["group_id"] = "0"*64; invalid.append(scenario)
        scenario = manual_scenario(); scenario["layout"][1]["row"] = 0; invalid.append(scenario)
        scenario = manual_scenario(); scenario["layout"][0]["start"] = 6; invalid.append(scenario)
        scenario = manual_scenario(); scenario["facts"][4]["event"] = 0; invalid.append(scenario)
        scenario = manual_scenario(); scenario["facts"][0]["filler"] = 9; invalid.append(scenario)
        for scenario in invalid:
            with self.subTest(scenario=scenario), self.assertRaises(ValueError):
                render_scenario(scenario, 0, 0)

    def test_invalid_renderer_query_and_condition(self):
        for event, role, condition in ((-1, 0, "base"), (2, 0, "base"), (0, 4, "base"),
                                       (0, 0, "unknown")):
            with self.subTest(event=event, role=role, condition=condition), self.assertRaises(ValueError):
                render_scenario(manual_scenario(), event, role, condition)

    def test_json_roundtrip_preserves_all_fixture_semantics(self):
        scenario = json.loads(json.dumps(manual_scenario(), sort_keys=True))
        records = [render_scenario(scenario, event, role, condition)
                   for condition in CONDITIONS for event in range(2) for role in range(4)]
        restored = json.loads(json.dumps(records, sort_keys=True, allow_nan=False))
        self.assertEqual(records, restored)
        self.assertEqual(len({record["record_id"] for record in records}), 48)
        self.assertEqual(len({record["pair_id"] for record in records}), 8)
        self.assertEqual(len({record["group_id"] for record in records}), 1)


if __name__ == "__main__":
    unittest.main()
