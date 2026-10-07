"""Prospective raw-text single-answer QA contracts and transparent controls.

This module uses only the standard library. It never reads files, downloads data,
imports a model, accesses a held-out partition, or executes training. Vocabulary,
layout and shortcut fitting accept only the records explicitly passed by callers;
the caller must enforce the prospective TRAIN/DEV/final access policy.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
import re

TOKENIZER = "lowercase_word_punctuation_v1"
DEFAULT_LAYOUT_POLICY = {
    "max_height": 8, "max_width": 16, "overflow": "error", "input_oov": "unk",
    "lowercase": True, "preserve_punctuation": True,
}
_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_LINE = re.compile(r"([1-9][0-9]*) (.+)")
_HEX = re.compile(r"[0-9a-f]{64}")
_RESERVED = ["<PAD>", "<UNK>"]


def canonical_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_bytes(value):
    if not isinstance(value, bytes):
        raise TypeError("sha256_bytes requires bytes")
    return hashlib.sha256(value).hexdigest()


def canonical_sha256(value):
    return sha256_bytes(canonical_bytes(value))


def tokenize(text):
    """Lowercase lexical items and retain every non-whitespace punctuation token."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a nonempty string")
    return _TOKEN.findall(text.lower())


def _answer(text):
    tokens = tokenize(text)
    if len(tokens) != 1 or not re.fullmatch(r"\w+", tokens[0], re.UNICODE):
        raise ValueError("this contract requires one lexical answer token")
    return tokens[0]


def normalized_input(facts, question):
    if not isinstance(facts, (list, tuple)) or not facts:
        raise ValueError("at least one ordered statement is required")
    return {"facts": [tokenize(text) for text in facts], "question": tokenize(question)}


def _record_input(record):
    return normalized_input([fact["text"] for fact in record["facts"]], record["question"])


def parse_babi(text, source_identity):
    """Parse numbered statements and tab-separated question/answer/support fields.

    All prior statements in the current episode are retained, in source order.
    Earlier questions and their answers are never included as statements. Support
    indices are validated as metadata only and do not select the visible input.
    Raw-line digests include the original line ending where one is present.
    """
    if not isinstance(text, str) or not text:
        raise ValueError("source text must be nonempty UTF-8 decoded text")
    if not isinstance(source_identity, dict) or not source_identity:
        raise ValueError("a nonempty public source identity is required")
    source = json.loads(canonical_bytes(source_identity))
    actual = sha256_bytes(text.encode("utf-8"))
    if "sha256" in source and source["sha256"] != actual:
        raise ValueError("source SHA-256 does not match the supplied bytes")
    source["sha256"] = actual
    records, facts, episode_records = [], [], []
    episode_number, previous_id = 0, 0

    def finish_episode():
        if episode_number and not episode_records:
            raise ValueError("episode has no question")
        if episode_records:
            episode_hash = canonical_sha256([tokenize(f["text"]) for f in facts])
            for record in episode_records:
                record["episode_sha256"] = episode_hash

    for source_line, raw in enumerate(text.splitlines(keepends=True), 1):
        line = raw.rstrip("\r\n")
        match = _LINE.fullmatch(line)
        if match is None:
            raise ValueError(f"malformed numbered source line {source_line}")
        line_id, payload = int(match.group(1)), match.group(2)
        if line_id == 1:
            finish_episode()
            episode_number += 1
            previous_id, facts, episode_records = 0, [], []
        if episode_number == 0 or line_id != previous_id + 1:
            raise ValueError(f"noncontiguous episode line ID at source line {source_line}")
        previous_id = line_id
        raw_hash = sha256_bytes(raw.encode("utf-8"))
        if "\t" not in payload:
            if not payload.strip() or payload.rstrip().endswith("?"):
                raise ValueError("a question must have separate answer and support fields")
            facts.append({"line_id": line_id, "text": payload,
                          "source_line_number": source_line,
                          "raw_line_sha256": raw_hash})
            continue
        fields = payload.split("\t")
        if len(fields) != 3:
            raise ValueError("question lines require exactly question, answer and support fields")
        question, gold, support_field = fields
        if not question.rstrip().endswith("?"):
            raise ValueError("question text must end in a question mark")
        answer = _answer(gold)
        support_parts = support_field.split()
        if not support_parts or any(not re.fullmatch(r"[1-9][0-9]*", x) for x in support_parts):
            raise ValueError("support metadata must contain positive statement IDs")
        support = [int(x) for x in support_parts]
        known = {f["line_id"] for f in facts}
        if len(set(support)) != len(support) or not set(support) <= known:
            raise ValueError("support metadata references duplicate or non-statement IDs")
        visible = normalized_input([f["text"] for f in facts], question)
        record = {
            "record_id": canonical_sha256({"source": actual, "source_line": source_line}),
            "episode_id": canonical_sha256({"source": actual, "episode": episode_number}),
            "episode_sha256": None,
            "input_sha256": canonical_sha256(visible),
            "question_line_id": line_id, "source_line_number": source_line,
            "facts": [dict(f) for f in facts], "question": question, "answer": answer,
            "supporting_fact_ids": support, "raw_question_line_sha256": raw_hash,
        }
        records.append(record)
        episode_records.append(record)
    finish_episode()
    if not records:
        raise ValueError("source contains no single-answer QA records")
    return {"schema_version": 1, "source": source, "tokenizer": TOKENIZER, "records": records}


def _validate_records(records):
    if not isinstance(records, (list, tuple)) or not records:
        raise ValueError("a nonempty record population is required")
    ids, labels = set(), {}
    for record in records:
        for name in ("record_id", "episode_id", "episode_sha256", "input_sha256"):
            if not isinstance(record.get(name), str) or not _HEX.fullmatch(record[name]):
                raise ValueError(f"invalid record hash: {name}")
        if record["record_id"] in ids:
            raise ValueError("duplicate record ID")
        ids.add(record["record_id"])
        visible_hash = canonical_sha256(_record_input(record))
        if visible_hash != record["input_sha256"]:
            raise ValueError("visible input differs from its recorded identity")
        answer = _answer(record["answer"])
        if record["answer"] != answer:
            raise ValueError("record answer must already be normalized")
        if visible_hash in labels and labels[visible_hash] != answer:
            raise ValueError("identical visible input has conflicting answers")
        labels[visible_hash] = answer


def build_development_split(records, salt="neuropixel-item10-dev-v1", dev_fraction=0.1):
    """Keep connected episodes, duplicate full stories and duplicate inputs together.

    Grouping uses only normalized visible text and episode identity, never labels.
    Connected-component transitivity prevents a shared-prefix question from
    bridging TRAIN and DEV. It does not exclude all shared individual facts,
    templates, entities or renamed/isomorphic stories.
    """
    _validate_records(records)
    if not isinstance(salt, str) or not salt:
        raise ValueError("a nonempty prospective split salt is required")
    if isinstance(dev_fraction, bool) or not isinstance(dev_fraction, (int, float)):
        raise ValueError("dev_fraction must be numeric")
    if not math.isfinite(dev_fraction) or not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction must lie strictly between zero and one")
    parent = list(range(len(records)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        a, b = find(i), find(j)
        if a != b:
            parent[max(a, b)] = min(a, b)

    seen = {}
    for i, record in enumerate(records):
        for field in ("episode_id", "episode_sha256", "input_sha256"):
            key = (field, record[field])
            if key in seen:
                union(i, seen[key])
            else:
                seen[key] = i
    members = defaultdict(list)
    for i in range(len(records)):
        members[find(i)].append(i)
    groups, mapping = [], {}
    for indices in members.values():
        inputs = sorted({records[i]["input_sha256"] for i in indices})
        stories = sorted({records[i]["episode_sha256"] for i in indices})
        group_id = canonical_sha256({"input_sha256": inputs, "episode_sha256": stories})
        split_hash = canonical_sha256({"salt": salt, "group_id": group_id})
        fraction = int(split_hash, 16) / (1 << 256)
        partition = "validation" if fraction < dev_fraction else "train"
        record_ids = sorted(records[i]["record_id"] for i in indices)
        groups.append({"group_id": group_id, "partition": partition,
                       "record_ids": record_ids, "n": len(indices),
                       "input_sha256": inputs, "episode_sha256": stories})
        mapping.update({record_id: group_id for record_id in record_ids})
    groups.sort(key=lambda x: x["group_id"])
    assigned = {g["group_id"]: g["partition"] for g in groups}
    train = [r["record_id"] for r in records if assigned[mapping[r["record_id"]]] == "train"]
    validation = [r["record_id"] for r in records if assigned[mapping[r["record_id"]]] == "validation"]
    if not train or not validation:
        raise ValueError("hash split has an empty partition; do not silently change its salt")
    return {"schema_version": 1, "salt": salt, "dev_fraction": dev_fraction,
            "grouping": "connected_episode_full_story_and_normalized_input_v1",
            "record_group": mapping, "groups": groups, "train_ids": train,
            "validation_ids": validation,
            "population_record_ids_sha256": canonical_sha256(sorted(mapping))}


def _layout_policy(policy):
    policy = dict(DEFAULT_LAYOUT_POLICY if policy is None else policy)
    if set(policy) != set(DEFAULT_LAYOUT_POLICY):
        raise ValueError("layout policy must declare exactly the documented fields")
    for name in ("max_height", "max_width"):
        if type(policy[name]) is not int or policy[name] < 1:
            raise ValueError("layout bounds must be positive integers")
    if (policy["overflow"] != "error" or policy["input_oov"] != "unk"
            or policy["lowercase"] is not True or policy["preserve_punctuation"] is not True):
        raise ValueError("unsupported prospective layout or tokenization policy")
    return policy


def fit_encoder(train_records, layout_policy=None):
    """Fit only on the assigned training records; no DEV or final input is accepted implicitly."""
    _validate_records(train_records)
    policy = _layout_policy(layout_policy)
    texts = [_record_input(r) for r in train_records]
    max_facts = max(len(x["facts"]) for x in texts)
    height = max_facts + 2
    width = max(len(line) for x in texts for line in x["facts"] + [x["question"]])
    if height > policy["max_height"] or width > policy["max_width"]:
        raise ValueError("assigned TRAIN exceeds the predeclared layout admission caps")
    vocabulary = sorted({token for x in texts for line in x["facts"] + [x["question"]]
                         for token in line} | {r["answer"] for r in train_records})
    vocabulary = _RESERVED + vocabulary
    return {"schema_version": 1, "tokenizer": TOKENIZER, "vocabulary": vocabulary,
            "token_to_id": {token: i for i, token in enumerate(vocabulary)},
            "height": height, "width": width, "max_facts": max_facts,
            "query_pos": [height - 2, 0], "out_pos": [height - 1, width - 1],
            "policy": policy,
            "training_record_ids_sha256": canonical_sha256(sorted(r["record_id"] for r in train_records)),
            "answer_tokens": sorted({r["answer"] for r in train_records}),
            "answer_counts": dict(sorted(Counter(r["answer"] for r in train_records).items()))}


def _validate_encoder(encoder):
    if encoder.get("schema_version") != 1 or encoder.get("tokenizer") != TOKENIZER:
        raise ValueError("unsupported encoder identity")
    _layout_policy(encoder["policy"])
    vocabulary = encoder["vocabulary"]
    if (not isinstance(vocabulary, list) or vocabulary[:2] != _RESERVED
            or any(not isinstance(x, str) for x in vocabulary)
            or len(vocabulary) < 3 or len(set(vocabulary)) != len(vocabulary)
            or encoder["token_to_id"] != {t: i for i, t in enumerate(vocabulary)}):
        raise ValueError("malformed vocabulary or token ID mapping")
    h, w, maximum = encoder["height"], encoder["width"], encoder["max_facts"]
    if any(type(x) is not int or x < 1 for x in (h, w, maximum)):
        raise ValueError("invalid encoder dimensions")
    if (h != maximum + 2 or h > encoder["policy"]["max_height"]
            or w > encoder["policy"]["max_width"]
            or encoder["query_pos"] != [h - 2, 0] or encoder["out_pos"] != [h - 1, w - 1]):
        raise ValueError("encoder layout violates its frozen policy")


def encode_input(facts, question, encoder):
    """Encode only raw ordered input text; answer/support metadata cannot enter this API."""
    _validate_encoder(encoder)
    visible = normalized_input(facts, question)
    h, w = encoder["height"], encoder["width"]
    if len(visible["facts"]) > encoder["max_facts"]:
        raise ValueError("input exceeds the frozen statement-row capacity")
    if any(len(line) > w for line in visible["facts"] + [visible["question"]]):
        raise ValueError("input exceeds the frozen token-row width; truncation is forbidden")
    canvas, unknown = [[0] * w for _ in range(h)], []
    token_to_id = encoder["token_to_id"]
    rows = list(enumerate(visible["facts"])) + [(h - 2, visible["question"])]
    for row, tokens in rows:
        for col, token in enumerate(tokens):
            canvas[row][col] = token_to_id.get(token, 1)
            if token not in token_to_id:
                unknown.append(token)
    return {"canvas": canvas, "input_tokens": visible, "unknown_input_tokens": unknown}


def decode_input(canvas, encoder):
    """Recover normalized tokens, with explicit UNK where lexical identity was lost."""
    _validate_encoder(encoder)
    h, w, vocabulary = encoder["height"], encoder["width"], encoder["vocabulary"]
    if not isinstance(canvas, (list, tuple)) or len(canvas) != h:
        raise ValueError("canvas height differs from its encoder")
    decoded = []
    for row in canvas:
        if not isinstance(row, (list, tuple)) or len(row) != w:
            raise ValueError("canvas width differs from its encoder")
        if any(type(x) is not int or not 0 <= x < len(vocabulary) for x in row):
            raise ValueError("canvas contains an invalid token ID")
        first_pad = next((i for i, value in enumerate(row) if value == 0), w)
        if any(row[first_pad:]):
            raise ValueError("non-PAD token after row padding")
        decoded.append([vocabulary[x] for x in row[:first_pad]])
    if decoded[-1] or not decoded[-2]:
        raise ValueError("readout row must be empty and question row must be nonempty")
    facts = decoded[:-2]
    while facts and not facts[-1]:
        facts.pop()
    if not facts or any(not row for row in facts):
        raise ValueError("fact rows must be contiguous and nonempty")
    return {"facts": facts, "question": decoded[-2]}


def score_answers(predicted_tokens, golds, *, nll=None, supported_mask=None):
    """Exact normalized single-token answers; unsupported gold never earns UNK credit.

    All examples remain in the accuracy denominator. Cross entropy, when
    supplied, is defined only on supported labels; every unsupported entry must
    be None and every supported entry must be finite and nonnegative.
    """
    if (not isinstance(golds, (list, tuple)) or not golds
            or not isinstance(predicted_tokens, (list, tuple))
            or len(predicted_tokens) != len(golds)):
        raise ValueError("prediction and gold populations must have equal nonzero lengths")
    if any(x is not None and (not isinstance(x, str) or not x) for x in predicted_tokens):
        raise ValueError("predictions must be nonempty token strings or explicit None")
    targets = [_answer(x) for x in golds]
    supported = [True] * len(golds) if supported_mask is None else list(supported_mask)
    if len(supported) != len(golds) or any(type(x) is not bool for x in supported):
        raise ValueError("supported_mask must be one boolean per example")
    correct, per_answer = 0, {}
    for prediction, target, valid in zip(predicted_tokens, targets, supported):
        # A non-answer vocabulary item such as PAD, UNK or punctuation is wrong.
        hit = valid and isinstance(prediction, str) and prediction.lower() == target
        correct += int(hit)
        entry = per_answer.setdefault(target, {"n": 0, "correct": 0})
        entry["n"] += 1
        entry["correct"] += int(hit)
    for entry in per_answer.values():
        entry["accuracy"] = entry["correct"] / entry["n"]
    losses = []
    if nll is not None:
        if not isinstance(nll, (list, tuple)) or len(nll) != len(golds):
            raise ValueError("NLL must align with every example")
        for value, valid in zip(nll, supported):
            if not valid:
                if value is not None:
                    raise ValueError("unsupported gold must have NLL=None")
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("supported NLL must be a real number")
            if not math.isfinite(value) or value < 0:
                raise ValueError("NLL must be finite and nonnegative")
            losses.append(float(value))
    return {"n": len(golds), "correct": correct, "accuracy": correct / len(golds),
            "per_answer": dict(sorted(per_answer.items())),
            "supported_gold_n": sum(supported), "unsupported_gold_n": len(golds) - sum(supported),
            "cross_entropy": math.fsum(losses) / len(losses) if losses else None,
            "cross_entropy_n": len(losses)}


def _winner(counts, fallback):
    return min(counts, key=lambda token: (-counts[token], token)) if counts else fallback


def _question_key(question):
    return canonical_sha256(tokenize(question))


def _bag(facts, question):
    visible = normalized_input(facts, question)
    return Counter(token for row in visible["facts"] + [visible["question"]] for token in row)


def fit_shortcuts(train_records):
    """Fit transparent controls on assigned TRAIN only; no semantic parser feeds inputs.

    BOW is multinomial naive Bayes with Laplace alpha=1 for priors and token
    likelihoods. The answer category for fact frequency is the set of observed
    TRAIN answer tokens. All ties use lexical order; misses use TRAIN majority.
    """
    _validate_records(train_records)
    answers, exact, questions = Counter(), defaultdict(Counter), defaultdict(Counter)
    class_tokens, vocabulary = defaultdict(Counter), set()
    for record in train_records:
        answer = record["answer"]
        facts = [x["text"] for x in record["facts"]]
        answers[answer] += 1
        exact[record["input_sha256"]][answer] += 1
        questions[_question_key(record["question"])][answer] += 1
        bag = _bag(facts, record["question"])
        class_tokens[answer].update(bag)
        vocabulary.update(bag)
    return {"schema_version": 1, "tokenizer": TOKENIZER, "alpha": 1.0,
            "tie_rule": "lexical_ascending", "answer_counts": dict(sorted(answers.items())),
            "majority": _winner(answers, None), "answer_tokens": sorted(answers),
            "exact_counts": {k: dict(sorted(v.items())) for k, v in sorted(exact.items())},
            "question_counts": {k: dict(sorted(v.items())) for k, v in sorted(questions.items())},
            "bow_vocabulary": sorted(vocabulary),
            "bow_class_tokens": {k: dict(sorted(v.items())) for k, v in sorted(class_tokens.items())},
            "training_record_ids_sha256": canonical_sha256(sorted(r["record_id"] for r in train_records))}


def predict_shortcuts(model, facts, question):
    if model.get("schema_version") != 1 or model.get("tokenizer") != TOKENIZER or model["alpha"] != 1.0:
        raise ValueError("unsupported shortcut model")
    visible = normalized_input(facts, question)
    majority = model["majority"]
    exact = model["exact_counts"].get(canonical_sha256(visible), {})
    question_counts = model["question_counts"].get(_question_key(question), {})
    category = set(model["answer_tokens"])
    fact_counts = Counter(t for row in visible["facts"] for t in row if t in category)
    query_counts = Counter(t for t in visible["question"] if t in category)
    bow = _bag(facts, question)
    vocabulary = set(model["bow_vocabulary"])
    scores = {}
    total_examples, classes = sum(model["answer_counts"].values()), len(category)
    for answer in sorted(category):
        counts = model["bow_class_tokens"][answer]
        denominator = sum(counts.values()) + len(vocabulary)
        score = math.log((model["answer_counts"][answer] + 1) / (total_examples + classes))
        score += math.fsum(n * math.log((counts.get(token, 0) + 1) / denominator)
                           for token, n in sorted(bow.items()) if token in vocabulary)
        scores[answer] = score
    frequent = _winner(fact_counts, majority)
    # A separately declared variant tests whether simple query-entity exclusion
    # removes a frequency tie; it does not parse the requested direction.
    without_query = {t: n for t, n in fact_counts.items() if t not in query_counts}
    return {"majority": majority, "exact_memory": _winner(exact, majority),
            "question_only": _winner(question_counts, majority),
            "bow_naive_bayes": min(scores, key=lambda t: (-scores[t], t)),
            "fact_answer_frequency": frequent,
            "fact_answer_frequency_excluding_query": _winner(without_query, majority)}


_DIRECTIONS = {"north": "south", "south": "north", "east": "west", "west": "east"}
_FACT = re.compile(r"(?:the )?(\w+) is (north|south|east|west) of (?:the )?(\w+)\.")
_QUERY_SUBJECT = re.compile(r"what is (north|south|east|west) of (?:the )?(\w+)\?")
_QUERY_OBJECT = re.compile(r"what is (?:the )?(\w+) (north|south|east|west) of\?")


def solve_raw(facts, question):
    """Direct QA4 grammar control with inverse relations, without transitive closure.

    Supports 'A is north of B', 'What is north of B?' and 'What is A north of?'.
    It consumes visible strings only. Unsupported text is not repaired or
    partially solved; ambiguous candidate sets are reported rather than tied.
    """
    normalized_input(facts, question)
    edges = set()
    for text in facts:
        match = _FACT.fullmatch(" ".join(text.lower().split()))
        if match is None:
            return {"status": "unsupported", "answer": None, "candidates": []}
        subject, direction, obj = match.groups()
        edges.add((subject, direction, obj))
        edges.add((obj, _DIRECTIONS[direction], subject))
    text = " ".join(question.lower().split())
    subject_query, object_query = _QUERY_SUBJECT.fullmatch(text), _QUERY_OBJECT.fullmatch(text)
    if subject_query:
        direction, obj = subject_query.groups()
        candidates = sorted({a for a, d, b in edges if d == direction and b == obj})
    elif object_query:
        subject, direction = object_query.groups()
        candidates = sorted({b for a, d, b in edges if a == subject and d == direction})
    else:
        return {"status": "unsupported", "answer": None, "candidates": []}
    status = "ok" if len(candidates) == 1 else "ambiguous" if candidates else "unsupported"
    return {"status": status, "answer": candidates[0] if len(candidates) == 1 else None,
            "candidates": candidates}
