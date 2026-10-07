"""Invented raw-QA contract fixtures; no official TRAIN/DEV/TEST payloads.

Run with: python -m unittest discover -s tests -p test_babi_qa.py -v
DEVELOPMENT_FIXTURES enumerates all valid input contexts constructed below,
including OOV/overflow and unsupported-symbolic contexts. Invalid parser
mutations are preserved in the tests but are not performance examples.
No model, checkpoint, RNG, file acquisition or held-out access is performed.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "neuropixel" / "research" / "babi_qa.py"
_SPEC = importlib.util.spec_from_file_location("item10_babi_contract_module", _MODULE_PATH)
babi = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(babi)

FACT_A = "The neral is north of the vexa."
FACT_B = "The vexa is west of the tovin."
FACT_C = "The selka is east of the dovar."
FACT_D = "The neral is east of the selka."
BASE_FACTS = [FACT_A, FACT_B]
BASE_Q = "What is north of the vexa?"
INVERSE_Q = "What is south of the neral?"
OBJECT_Q = "What is neral north of?"
OTHER_Q = "What is west of the tovin?"
THIRD_Q = "What is east of the dovar?"
BRANCH_Q = "What is east of the selka?"
OOV_FACTS = ["The quorin is north of the vexa.", FACT_B]
LONG_Q = "What is north of the very very very very very very very distant vexa?"
AMBIGUOUS_FACTS = [FACT_A, "The selka is north of the vexa."]
TRANSITIVE_FACTS = [FACT_A, "The vexa is north of the tovin."]
TRANSITIVE_Q = "What is north of the tovin?"
UNSUPPORTED_FACTS = ["The neral is to the north of the vexa.", FACT_B]
UNKNOWN_Q = "Where is the quorin?"
PERMUTED_TOKENS = ["of vexa north is neral the the .", FACT_B]
QUERY_REPEATED = "What is neral neral neral neral north of the vexa?"


def _alphabetic_suffix(i):
    return chr(97 + i // 26) + chr(97 + i % 26)


# Fixed, nonadaptive background makes grouped splitting a population test.
# No random draws, real source records, or prospective study seeds are used.
GROUPING_FIXTURES = tuple(
    {"facts": [
        "The zeral" + _alphabetic_suffix(i) + " is north of the zexa" + _alphabetic_suffix(i) + ".",
        "The zexa" + _alphabetic_suffix(i) + " is west of the zovin" + _alphabetic_suffix(i) + ".",
    ], "question": "What is north of the zexa" + _alphabetic_suffix(i) + "?"}
    for i in range(128)
)


def _fixture(facts, question):
    return {"facts": list(facts), "question": question}


def _fixture_ledger():
    contexts = [
        _fixture([FACT_A], BASE_Q),
        _fixture(BASE_FACTS, BASE_Q),
        _fixture(BASE_FACTS, INVERSE_Q),
        _fixture(BASE_FACTS, OBJECT_Q),
        _fixture(BASE_FACTS, OTHER_Q),
        _fixture([FACT_A, FACT_D], BRANCH_Q),
        _fixture(BASE_FACTS + [FACT_C], THIRD_Q),
        _fixture(BASE_FACTS + [FACT_C], BASE_Q),
        _fixture(list(reversed(BASE_FACTS)), BASE_Q),
        _fixture(OOV_FACTS, BASE_Q),
        _fixture(OOV_FACTS, UNKNOWN_Q),
        _fixture(BASE_FACTS, LONG_Q),
        _fixture(AMBIGUOUS_FACTS, BASE_Q),
        _fixture(TRANSITIVE_FACTS, TRANSITIVE_Q),
        _fixture(TRANSITIVE_FACTS, BASE_Q),
        _fixture(UNSUPPORTED_FACTS, BASE_Q),
        _fixture(BASE_FACTS, UNKNOWN_Q),
        _fixture(PERMUTED_TOKENS, BASE_Q),
        _fixture(BASE_FACTS, QUERY_REPEATED),
        _fixture(["  THE neral is north of the vexa .  ", FACT_B], " WHAT is north of the vexa ? "),
        _fixture([FACT_A] * 7, BASE_Q),
        _fixture([FACT_A], "What is north of the vexa?"),
    ]
    # Exact string deduplication only: downstream novelty keys use their declared
    # tokenizer. Multiple raw strings can legitimately share one normalized key.
    unique = {}
    for item in contexts + list(GROUPING_FIXTURES):
        key = json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        unique.setdefault(key, item)
    return tuple(unique.values())


DEVELOPMENT_FIXTURES = _fixture_ledger()
DEVELOPMENT_FIXTURE_SCOPE = {
    "source": "freshly invented contract fixtures only",
    "fixed_grouping_background": 128,
    "rng": "none",
    "official_payloads": "none",
    "coverage": "all valid facts/question contexts passed to parser, encoder or controls; invalid parser mutations are not performance fixtures",
}


def _episode(facts, question, answer, supports=None):
    supports = list(range(1, len(facts) + 1)) if supports is None else supports
    lines = [str(i + 1) + " " + fact for i, fact in enumerate(facts)]
    lines.append(str(len(facts) + 1) + " " + question + "\t" + answer + "\t" + " ".join(map(str, supports)))
    return "\n".join(lines) + "\n"


def _parse(text):
    return babi.parse_babi(text, {"fixture": "item10-contract-invented-v1"})


def _records():
    # Distinct questions prevent contradictory labels for the same visible input.
    return _parse(
        _episode(BASE_FACTS, BASE_Q, "neral")
        + _episode(BASE_FACTS, INVERSE_Q, "vexa")
        + _episode(BASE_FACTS, OBJECT_Q, "vexa")
    )["records"]


def _background_records():
    text = "".join(
        _episode(item["facts"], item["question"], "zeral" + _alphabetic_suffix(i))
        for i, item in enumerate(GROUPING_FIXTURES)
    )
    return _parse(text)["records"]


def _linked_records():
    return _parse(
        "1 " + FACT_A + "\n"
        "2 " + BASE_Q + "\tneral\t1\n"
        "3 " + FACT_B + "\n"
        "4 " + OTHER_Q + "\tvexa\t3\n"
        "1 " + FACT_A + "\n"
        "2 " + BASE_Q + "\tneral\t1\n"
        "3 " + FACT_D + "\n"
        "4 " + BRANCH_Q + "\tneral\t3\n"
        + _episode(BASE_FACTS, INVERSE_Q, "vexa")
    )["records"]


class ParserContractTests(unittest.TestCase):
    def test_prior_facts_only_no_earlier_answers_or_future_facts(self):
        text = (
            "1 " + FACT_A + "\n"
            "2 " + BASE_Q + "\tneral\t1\n"
            "3 " + FACT_B + "\n"
            "4 " + OTHER_Q + "\tvexa\t3\n"
            "5 " + FACT_C + "\n"
            "6 " + THIRD_Q + "\tselka\t5\n"
        )
        rows = _parse(text)["records"]
        self.assertEqual([[f["line_id"] for f in r["facts"]] for r in rows], [[1], [1, 3], [1, 3, 5]])
        self.assertEqual([[f["text"] for f in r["facts"]] for r in rows],
                         [[FACT_A], BASE_FACTS, BASE_FACTS + [FACT_C]])
        self.assertEqual([r["question_line_id"] for r in rows], [2, 4, 6])
        self.assertEqual([r["supporting_fact_ids"] for r in rows], [[1], [3], [5]])
        self.assertEqual(len({r["episode_id"] for r in rows}), 1)
        self.assertEqual(len({r["episode_sha256"] for r in rows}), 1)
        rows[0]["facts"][0]["text"] = "mutation"
        self.assertEqual(rows[1]["facts"][0]["text"], FACT_A)

    def test_numeric_ids_are_original_positive_contiguous_statement_ids(self):
        valid = _episode(BASE_FACTS, BASE_Q, "neral")
        malformed = [
            valid.replace("1 The", "0 The", 1),
            valid.replace("1 The", "01 The", 1),
            valid.replace("1 The", "2 The", 1),
            valid.replace("2 The", "4 The", 1),
            valid.replace("3 What", "2 What", 1),
            valid.replace("1 The", "1\tThe", 1),
            valid + "\n",
        ]
        for text in malformed:
            with self.subTest(text=text), self.assertRaises(ValueError):
                _parse(text)
        reset = _parse(valid + valid)["records"]
        self.assertNotEqual(reset[0]["episode_id"], reset[1]["episode_id"])
        self.assertEqual(reset[0]["input_sha256"], reset[1]["input_sha256"])

    def test_malformed_question_fields_are_rejected(self):
        prefix = "1 " + FACT_A + "\n2 "
        suffixes = [
            BASE_Q, BASE_Q + "\tneral", BASE_Q + "\tneral\t1\textra",
            BASE_Q + "\t\t1", BASE_Q + "\ttwo words\t1",
            BASE_Q + "\tneral\t", BASE_Q.rstrip("?") + "\tneral\t1",
            " \tneral\t1", BASE_Q + "\t.\t1",
        ]
        for suffix in suffixes:
            with self.subTest(suffix=suffix), self.assertRaises(ValueError):
                _parse(prefix + suffix + "\n")
        with self.assertRaises(ValueError):
            _parse("1 " + FACT_A + "\n")

    def test_support_metadata_never_selects_facts_and_cannot_reference_questions(self):
        first = _parse(_episode(BASE_FACTS, BASE_Q, "neral", [1]))["records"][0]
        second = _parse(_episode(BASE_FACTS, BASE_Q, "neral", [2]))["records"][0]
        self.assertEqual(first["input_sha256"], second["input_sha256"])
        self.assertEqual([f["text"] for f in first["facts"]], BASE_FACTS)
        for support in ("1 1", "0", "01", "3", "4"):
            with self.subTest(support=support), self.assertRaises(ValueError):
                _parse("1 " + FACT_A + "\n2 " + BASE_Q + "\tneral\t" + support + "\n")
        with self.assertRaises(ValueError):
            _parse("1 " + FACT_A + "\n2 " + BASE_Q + "\tneral\t1\n3 " + BASE_Q + "\tneral\t2\n")

    def test_source_and_raw_line_hashes_preserve_bytes(self):
        text = _episode(BASE_FACTS, BASE_Q, "neral").replace("\n", "\r\n")
        expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
        parsed = babi.parse_babi(text, {"fixture": "crlf", "sha256": expected})
        self.assertEqual(parsed["source"]["sha256"], expected)
        raw = text.splitlines(keepends=True)
        row = parsed["records"][0]
        self.assertEqual(row["facts"][0]["raw_line_sha256"], hashlib.sha256(raw[0].encode()).hexdigest())
        self.assertEqual(row["raw_question_line_sha256"], hashlib.sha256(raw[2].encode()).hexdigest())
        with self.assertRaises(ValueError):
            babi.parse_babi(text, {"sha256": "0" * 64})

    def test_normalization_is_punctuation_preserving_and_label_blind(self):
        row = _parse(_episode(BASE_FACTS, BASE_Q, "neral"))["records"][0]
        changed = _parse(_episode(BASE_FACTS, BASE_Q, "selka", [2]))["records"][0]
        variant = _parse(_episode(
            ["  THE neral is north of the vexa .  ", FACT_B],
            " WHAT is north of the vexa ? ", "neral"
        ))["records"][0]
        self.assertEqual(babi.tokenize(FACT_A), ["the", "neral", "is", "north", "of", "the", "vexa", "."])
        self.assertEqual(babi.tokenize(BASE_Q), ["what", "is", "north", "of", "the", "vexa", "?"])
        self.assertEqual(row["input_sha256"], changed["input_sha256"])
        self.assertEqual(row["input_sha256"], variant["input_sha256"])
        self.assertNotEqual(row["record_id"], changed["record_id"])

    def test_conflicting_gold_is_retained_but_not_silently_fit(self):
        rows = _parse(_episode(BASE_FACTS, BASE_Q, "neral") + _episode(BASE_FACTS, BASE_Q, "selka"))["records"]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["input_sha256"], rows[1]["input_sha256"])
        self.assertEqual([r["answer"] for r in rows], ["neral", "selka"])
        for fit in (babi.fit_encoder, babi.fit_shortcuts, babi.build_development_split):
            with self.subTest(fit=fit.__name__), self.assertRaises(ValueError):
                fit(rows)


class GroupingContractTests(unittest.TestCase):
    def test_connected_episodes_full_stories_and_shared_prefixes_do_not_cross(self):
        linked = _linked_records()
        rows = linked + _background_records()
        split = babi.build_development_split(rows)
        self.assertEqual(split["salt"], "neuropixel-item10-dev-v1")
        self.assertEqual(split["dev_fraction"], 0.1)
        # The second episode has a different full story but the same first QA;
        # the last episode shares a full story, not the first question.
        self.assertNotEqual(linked[0]["episode_sha256"], linked[2]["episode_sha256"])
        self.assertEqual(linked[0]["input_sha256"], linked[2]["input_sha256"])
        self.assertEqual(linked[0]["episode_sha256"], linked[4]["episode_sha256"])
        self.assertEqual(len({split["record_group"][r["record_id"]] for r in linked}), 1)
        train, val = set(split["train_ids"]), set(split["validation_ids"])
        self.assertTrue(train)
        self.assertTrue(val)
        self.assertFalse(train & val)
        self.assertEqual(train | val, {r["record_id"] for r in rows})
        for group in split["groups"]:
            ids = set(group["record_ids"])
            self.assertTrue(ids <= train or ids <= val)

    def test_group_ids_and_assignments_ignore_order_and_gold(self):
        rows = _linked_records() + _background_records()
        first = babi.build_development_split(rows)
        reversed_result = babi.build_development_split(list(reversed(rows)))
        relabeled = deepcopy(rows)
        for record in relabeled:
            record["answer"] = "quorin"
        changed = babi.build_development_split(relabeled)
        for result in (reversed_result, changed):
            self.assertEqual(first["record_group"], result["record_group"])
            self.assertEqual(first["groups"], result["groups"])
            self.assertEqual(set(first["train_ids"]), set(result["train_ids"]))
            self.assertEqual(set(first["validation_ids"]), set(result["validation_ids"]))

    def test_conflict_diagnostic_keeps_both_and_empty_partition_is_explicit(self):
        conflict = _parse(_episode(BASE_FACTS, BASE_Q, "neral") + _episode(BASE_FACTS, BASE_Q, "selka"))["records"]
        split = babi.build_development_split(conflict + _background_records(), allow_conflicts=True)
        self.assertIs(split["allow_conflicts"], True)
        self.assertEqual(split["record_group"][conflict[0]["record_id"]], split["record_group"][conflict[1]["record_id"]])
        self.assertTrue({r["record_id"] for r in conflict} <= set(split["train_ids"]) | set(split["validation_ids"]))
        with self.assertRaises(ValueError):
            babi.build_development_split(_records())
        for fraction in (0, 1, -0.1, float("nan"), float("inf"), True):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                babi.build_development_split(_background_records(), dev_fraction=fraction)


class EncoderContractTests(unittest.TestCase):
    def test_literal_layout_roundtrip_preserves_order_and_punctuation(self):
        encoder = babi.fit_encoder(_records())
        result = babi.encode_input(BASE_FACTS, BASE_Q, encoder)
        self.assertEqual((encoder["height"], encoder["width"]), (4, 8))
        self.assertEqual(encoder["query_pos"], [2, 0])
        self.assertEqual(encoder["out_pos"], [3, 7])
        self.assertEqual(result["canvas"][-1], [0] * 8)
        decoded = babi.decode_input(result["canvas"], encoder)
        self.assertEqual(decoded["facts"], [
            ["the", "neral", "is", "north", "of", "the", "vexa", "."],
            ["the", "vexa", "is", "west", "of", "the", "tovin", "."],
        ])
        self.assertEqual(decoded["question"], ["what", "is", "north", "of", "the", "vexa", "?"])
        reversed_canvas = babi.encode_input(list(reversed(BASE_FACTS)), BASE_Q, encoder)["canvas"]
        self.assertEqual(reversed_canvas[0], result["canvas"][1])
        self.assertEqual(reversed_canvas[1], result["canvas"][0])
        self.assertNotEqual(reversed_canvas, result["canvas"])

    def test_encoding_is_pure_and_metadata_cannot_affect_visible_input(self):
        rows = _records()
        encoder = babi.fit_encoder(rows)
        before = deepcopy(encoder)
        facts = list(BASE_FACTS)
        first = babi.encode_input(facts, BASE_Q, encoder)
        changed = deepcopy(rows[0])
        changed["answer"] = "quorin"
        changed["supporting_fact_ids"] = [2]
        second = babi.encode_input([f["text"] for f in changed["facts"]], changed["question"], encoder)
        self.assertEqual(first, second)
        self.assertEqual(facts, BASE_FACTS)
        self.assertEqual(encoder, before)

    def test_input_oov_and_output_support_are_distinct_and_vocab_stays_frozen(self):
        encoder = babi.fit_encoder(_records())
        before = deepcopy(encoder)
        result = babi.encode_input(OOV_FACTS, BASE_Q, encoder)
        self.assertEqual(result["unknown_input_tokens"], ["quorin"])
        self.assertEqual(result["canvas"][0][1], 1)
        self.assertEqual(babi.decode_input(result["canvas"], encoder)["facts"][0][1], "<UNK>")
        self.assertNotIn("quorin", encoder["token_to_id"])
        self.assertIn("tovin", encoder["token_to_id"])
        self.assertNotIn("tovin", encoder["answer_tokens"])
        supported = [word in encoder["token_to_id"] for word in ["quorin", "tovin"]]
        score = babi.score_answers(["quorin", "tovin"], ["quorin", "tovin"],
                                  supported_mask=supported, nll=[None, 0.25])
        self.assertEqual((score["correct"], score["n"], score["cross_entropy_n"]), (1, 2, 1))
        self.assertEqual(encoder, before)

    def test_train_gold_vocabulary_admission_and_overflow_never_truncate(self):
        training = _parse(_episode(BASE_FACTS, BASE_Q, "quorin"))["records"]
        self.assertIn("quorin", babi.fit_encoder(training)["token_to_id"])
        encoder = babi.fit_encoder(_records())
        for facts, question in ((BASE_FACTS + [FACT_C], BASE_Q), (BASE_FACTS, LONG_Q)):
            with self.subTest(facts=facts, question=question), self.assertRaises(ValueError):
                babi.encode_input(facts, question, encoder)
        with self.assertRaises(ValueError):
            babi.fit_encoder(_parse(_episode([FACT_A] * 7, BASE_Q, "neral"))["records"])
        policy = dict(babi.DEFAULT_LAYOUT_POLICY, max_width=7)
        with self.assertRaises(ValueError):
            babi.fit_encoder(_records(), policy)

    def test_decoder_rejects_padding_holes_contaminated_output_and_invalid_ids(self):
        encoder = babi.fit_encoder(_records())
        canvas = babi.encode_input(BASE_FACTS, BASE_Q, encoder)["canvas"]
        mutants = []
        hole = deepcopy(canvas); hole[0][0] = 0; mutants.append(hole)
        out = deepcopy(canvas); out[-1][0] = 2; mutants.append(out)
        bad = deepcopy(canvas); bad[0][0] = len(encoder["vocabulary"]); mutants.append(bad)
        boolean = deepcopy(canvas); boolean[0][0] = True; mutants.append(boolean)
        query = deepcopy(canvas); query[-2] = [0] * encoder["width"]; mutants.append(query)
        mutants.append(canvas[:-1])
        for mutant in mutants:
            with self.subTest(mutant=mutant), self.assertRaises(ValueError):
                babi.decode_input(mutant, encoder)


class ScoringContractTests(unittest.TestCase):
    def test_manual_accuracy_per_answer_and_unsupported_denominator(self):
        result = babi.score_answers(["NERAL", "vexa", None, "<UNK>", "quorin"],
                                    ["neral", "neral", "vexa", "quorin", "quorin"],
                                    supported_mask=[True, True, True, False, False])
        self.assertEqual((result["n"], result["correct"], result["accuracy"]), (5, 1, 0.2))
        self.assertEqual(result["per_answer"]["neral"], {"n": 2, "correct": 1, "accuracy": 0.5})
        self.assertEqual(result["per_answer"]["quorin"], {"n": 2, "correct": 0, "accuracy": 0.0})
        self.assertEqual((result["supported_gold_n"], result["unsupported_gold_n"]), (3, 2))

    def test_empty_subset_has_null_accuracy_and_null_ce(self):
        result = babi.score_answers([], [], nll=[], supported_mask=[])
        self.assertEqual(result, {
            "n": 0, "correct": 0, "accuracy": None, "per_answer": {},
            "supported_gold_n": 0, "unsupported_gold_n": 0,
            "cross_entropy": None, "cross_entropy_n": 0,
        })

    def test_ce_is_over_supported_gold_only_with_explicit_none_elsewhere(self):
        result = babi.score_answers(["neral", "quorin", None], ["neral", "quorin", "vexa"],
                                    supported_mask=[True, False, True], nll=[0.25, None, 0.75])
        self.assertEqual(result["cross_entropy"], 0.5)
        self.assertEqual(result["cross_entropy_n"], 2)
        self.assertEqual(result["n"], 3)
        empty_ce = babi.score_answers(["quorin"], ["quorin"], supported_mask=[False], nll=[None])
        self.assertEqual(empty_ce["accuracy"], 0.0)
        self.assertIsNone(empty_ce["cross_entropy"])
        self.assertEqual(empty_ce["cross_entropy_n"], 0)

    def test_misalignment_nonfinite_mask_and_prediction_types_fail_closed(self):
        cases = [
            ([], ["neral"], {}),
            ([True], ["neral"], {}),
            ([3], ["neral"], {}),
            ([""], ["neral"], {}),
            (["neral"], ["two words"], {}),
            (["neral"], ["neral"], {"supported_mask": [1]}),
            (["neral"], ["neral"], {"supported_mask": []}),
            (["neral"], ["neral"], {"nll": []}),
            (["neral"], ["neral"], {"nll": [True]}),
            (["neral"], ["neral"], {"nll": [-0.1]}),
            (["neral"], ["neral"], {"nll": [float("nan")]}),
            (["neral"], ["neral"], {"nll": [float("inf")]}),
            (["neral"], ["neral"], {"nll": [None]}),
            (["quorin"], ["quorin"], {"supported_mask": [False], "nll": [0.0]}),
        ]
        for predictions, golds, kwargs in cases:
            with self.subTest(predictions=predictions, kwargs=kwargs), self.assertRaises(ValueError):
                babi.score_answers(predictions, golds, **kwargs)


class ControlContractTests(unittest.TestCase):
    def test_raw_symbolic_solver_direct_inverse_object_and_relation_order(self):
        for facts, question, answer in (
            (BASE_FACTS, BASE_Q, "neral"),
            (BASE_FACTS, INVERSE_Q, "vexa"),
            (BASE_FACTS, OBJECT_Q, "vexa"),
            (list(reversed(BASE_FACTS)), BASE_Q, "neral"),
        ):
            with self.subTest(facts=facts, question=question):
                result = babi.solve_raw(facts, question)
                self.assertEqual(result, {"status": "ok", "answer": answer, "candidates": [answer]})
        result = babi.solve_raw(TRANSITIVE_FACTS, TRANSITIVE_Q)
        self.assertEqual(result["candidates"], ["vexa"])
        self.assertNotIn("neral", result["candidates"])

    def test_raw_symbolic_ambiguity_and_unsupported_grammar_are_explicit(self):
        ambiguous = babi.solve_raw(AMBIGUOUS_FACTS, BASE_Q)
        self.assertEqual(ambiguous, {"status": "ambiguous", "answer": None, "candidates": ["neral", "selka"]})
        for facts, question in ((UNSUPPORTED_FACTS, BASE_Q), (BASE_FACTS, UNKNOWN_Q)):
            with self.subTest(facts=facts, question=question):
                result = babi.solve_raw(facts, question)
                self.assertEqual(result, {"status": "unsupported", "answer": None, "candidates": []})

    def test_majority_exact_memory_question_only_and_miss_are_distinct(self):
        model = babi.fit_shortcuts(_records())
        base = babi.predict_shortcuts(model, BASE_FACTS, BASE_Q)
        self.assertEqual(base["majority"], "vexa")
        self.assertEqual(base["exact_memory"], "neral")
        self.assertEqual(base["question_only"], "neral")
        changed = babi.predict_shortcuts(model, OOV_FACTS, BASE_Q)
        self.assertEqual(changed["exact_memory"], "vexa")
        self.assertEqual(changed["question_only"], "neral")
        missing = babi.predict_shortcuts(model, BASE_FACTS, UNKNOWN_Q)
        self.assertEqual(missing["question_only"], "vexa")

    def test_fact_frequency_ignores_question_repetitions_with_declared_exclusion(self):
        model = babi.fit_shortcuts(_records())
        base = babi.predict_shortcuts(model, BASE_FACTS, BASE_Q)
        repeated = babi.predict_shortcuts(model, BASE_FACTS, QUERY_REPEATED)
        self.assertEqual(base["fact_answer_frequency"], "vexa")
        self.assertEqual(repeated["fact_answer_frequency"], "vexa")
        self.assertEqual(base["fact_answer_frequency_excluding_query"], "neral")
        # Both candidates mentioned by this query: explicit majority fallback.
        self.assertEqual(repeated["fact_answer_frequency_excluding_query"], "vexa")
        self.assertEqual(set(model["answer_tokens"]), {"neral", "vexa"})

    def test_bow_controls_ignore_order_while_exact_memory_does_not(self):
        model = babi.fit_shortcuts(_records())
        before = deepcopy(model)
        base = babi.predict_shortcuts(model, BASE_FACTS, BASE_Q)
        reordered = babi.predict_shortcuts(model, list(reversed(BASE_FACTS)), BASE_Q)
        words_permuted = babi.predict_shortcuts(model, PERMUTED_TOKENS, BASE_Q)
        for prediction in (reordered, words_permuted):
            self.assertEqual(prediction["bow_memory"], base["bow_memory"])
            self.assertEqual(prediction["bow_naive_bayes"], base["bow_naive_bayes"])
            self.assertEqual(prediction["exact_memory"], "vexa")
        self.assertEqual(base["bow_memory"], "neral")
        babi.predict_shortcuts(model, OOV_FACTS, UNKNOWN_Q)
        self.assertEqual(model, before)


if __name__ == "__main__":
    unittest.main()
