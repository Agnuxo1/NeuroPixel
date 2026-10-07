"""Independent offline recount of item-10 raw QA and saved prediction artifacts.

Only stdlib, NumPy and SciPy are imported. No project code, Torch, pickle, model,
checkpoint inference, download or answer substitution is permitted. Read-only
input roots must be supplied explicitly; output is exclusive. The primary
interval uses five paired training realizations. The common cluster bootstrap
is separately conditional on saved checkpoints and observed source components.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
import re

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
NUMBER_PATTERN = re.compile(r"([1-9][0-9]*) (.+)")
NLL_ATOL = 1e-10
NLL_RTOL = 1e-12
SEEDS = (60, 61, 62, 63, 64)
FAMILIES = ("neuropixel", "relative_transformer")


class AuditError(ValueError):
    pass


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(value).hexdigest()


def identity(value):
    return digest(canonical_bytes(value))


def require(condition, message):
    if not condition:
        raise AuditError(message)


def tokens(text):
    require(isinstance(text, str) and bool(text.strip()), "empty or nontext lexical input")
    return TOKEN_PATTERN.findall(text.lower())


def visible(facts, question):
    require(isinstance(facts, (list, tuple)) and bool(facts), "no visible statements")
    return {"facts": [tokens(x) for x in facts], "question": tokens(question)}


def one_answer(value):
    result = tokens(value)
    require(len(result) == 1 and re.fullmatch(r"\w+", result[0], re.UNICODE) is not None,
            "gold is not one lexical token")
    return result[0]


def parse_raw_qa(text):
    """Independent parser: visible prefix and official answer remain separate."""
    require(isinstance(text, str) and bool(text), "empty source payload")
    source_sha = digest(text.encode("utf-8"))
    episodes, current, previous, episode_number = [], None, 0, 0
    # First collect complete source episodes; then reconstruct each QA prefix.
    for physical_number, raw_line in enumerate(text.splitlines(keepends=True), 1):
        line = raw_line.rstrip("\r\n")
        match = NUMBER_PATTERN.fullmatch(line)
        require(match is not None, "invalid original numbered line")
        number, payload = int(match.group(1)), match.group(2)
        if number == 1:
            episode_number += 1
            current = {"number": episode_number, "lines": []}
            episodes.append(current)
            previous = 0
        require(current is not None and number == previous + 1,
                "noncontiguous original episode numbering")
        previous = number
        current["lines"].append((number, physical_number, payload, digest(raw_line.encode("utf-8"))))
    records = []
    for episode in episodes:
        facts, episode_rows = [], []
        for number, physical_number, payload, raw_hash in episode["lines"]:
            fields = payload.split("\t")
            if len(fields) == 1:
                require(bool(payload.strip()) and not payload.rstrip().endswith("?"),
                        "question has no separate answer/support fields")
                facts.append({"line_id": number, "text": payload,
                              "source_line_number": physical_number, "raw_line_sha256": raw_hash})
                continue
            require(len(fields) == 3, "question has incorrect tab-field count")
            question, gold, support_text = fields
            require(question.rstrip().endswith("?"), "question mark absent")
            answer = one_answer(gold)
            support_parts = support_text.split()
            require(bool(support_parts)
                    and all(re.fullmatch(r"[1-9][0-9]*", item) for item in support_parts),
                    "malformed support metadata")
            support = list(map(int, support_parts))
            require(len(support) == len(set(support))
                    and set(support) <= {f["line_id"] for f in facts},
                    "support does not refer uniquely to prior statements")
            lexical = visible([f["text"] for f in facts], question)
            row = {
                "record_id": identity({"source": source_sha, "source_line": physical_number}),
                "episode_id": identity({"source": source_sha, "episode": episode["number"]}),
                "episode_sha256": None,
                "input_sha256": identity(lexical),
                "question_line_id": number,
                "source_line_number": physical_number,
                "facts": [dict(fact) for fact in facts],
                "question": question,
                "answer": answer,
                "supporting_fact_ids": support,
                "raw_question_line_sha256": raw_hash,
            }
            episode_rows.append(row)
        require(bool(episode_rows), "source episode contains no question")
        complete_story = identity([tokens(f["text"]) for f in facts])
        for row in episode_rows:
            row["episode_sha256"] = complete_story
        records.extend(episode_rows)
    require(bool(records), "source contains no QA rows")
    return {"source_sha256": source_sha, "records": records}


def connected_components(records):
    """Graph traversal, independent of the data module's union-find implementation."""
    require(bool(records), "component population is empty")
    record_ids = [record["record_id"] for record in records]
    require(len(record_ids) == len(set(record_ids)), "duplicate record ID in component population")
    memberships = defaultdict(set)
    for i, row in enumerate(records):
        for field in ("episode_id", "episode_sha256", "input_sha256"):
            memberships[(field, row[field])].add(i)
    unseen, result = set(range(len(records))), []
    while unseen:
        pending, component = [min(unseen)], set()
        while pending:
            index = pending.pop()
            if index in component:
                continue
            component.add(index)
            unseen.discard(index)
            row = records[index]
            for field in ("episode_id", "episode_sha256", "input_sha256"):
                pending.extend(memberships[(field, row[field])] - component)
        members = sorted(component)
        inputs = sorted({records[i]["input_sha256"] for i in members})
        stories = sorted({records[i]["episode_sha256"] for i in members})
        result.append({
            "group_id": identity({"input_sha256": inputs, "episode_sha256": stories}),
            "indices": members,
            "record_ids": sorted(records[i]["record_id"] for i in members),
            "input_sha256": inputs,
            "episode_sha256": stories,
            "n": len(members),
        })
    return sorted(result, key=lambda group: group["group_id"])


def reconstruct_encoder(train_records, policy):
    """Lexical layout/vocabulary only; no symbolic relation conversion."""
    require(bool(train_records), "no assigned TRAIN records")
    normalized = [visible([f["text"] for f in r["facts"]], r["question"]) for r in train_records]
    max_facts = max(len(item["facts"]) for item in normalized)
    height = max_facts + 2
    width = max(len(row) for item in normalized for row in item["facts"] + [item["question"]])
    require(height <= policy["max_height"] and width <= policy["max_width"], "TRAIN layout admission failure")
    vocabulary = ["<PAD>", "<UNK>"] + sorted(
        {token for item in normalized for row in item["facts"] + [item["question"]] for token in row}
        | {r["answer"] for r in train_records}
    )
    return {
        "schema_version": 1,
        "tokenizer": "lowercase_word_punctuation_v1",
        "vocabulary": vocabulary,
        "token_to_id": {token: i for i, token in enumerate(vocabulary)},
        "height": height, "width": width, "max_facts": max_facts,
        "query_pos": [height - 2, 0], "out_pos": [height - 1, width - 1],
        "policy": policy,
        "training_record_ids_sha256": identity(sorted(r["record_id"] for r in train_records)),
        "answer_tokens": sorted({r["answer"] for r in train_records}),
        "answer_counts": dict(sorted(Counter(r["answer"] for r in train_records).items())),
    }


def reconstruct_canvas(facts, question, encoder):
    lexical = visible(facts, question)
    h, w = encoder["height"], encoder["width"]
    if len(facts) > encoder["max_facts"] or any(len(row) > w for row in lexical["facts"] + [lexical["question"]]):
        return None
    canvas = [[0 for _ in range(w)] for _ in range(h)]
    for row_number, row in list(enumerate(lexical["facts"])) + [(h - 2, lexical["question"])]:
        for column, token in enumerate(row):
            canvas[row_number][column] = encoder["token_to_id"].get(token, 1)
    return canvas


def paired_seed_interval(np_accuracy, tf_accuracy, t_distribution):
    """Five training-realization pairs only; no row/cluster pseudoreplication."""
    require(set(np_accuracy) == set(SEEDS) == set(tf_accuracy), "five complete paired seeds are required")
    pairs = []
    for seed in SEEDS:
        a, b = np_accuracy[seed], tf_accuracy[seed]
        require(all(isinstance(x, (int, float)) and not isinstance(x, bool)
                    and math.isfinite(x) and 0 <= x <= 1 for x in (a, b)),
                "nonfinite or invalid paired accuracy")
        pairs.append({"seed": seed, "neuropixel": a, "transformer": b, "difference": a - b})
    differences = [row["difference"] for row in pairs]
    mean = math.fsum(differences) / 5
    sd = math.sqrt(math.fsum((x - mean) ** 2 for x in differences) / 4)
    halfwidth = float(t_distribution.ppf(0.975, 4)) * sd / math.sqrt(5)
    return {
        "n": 5, "unit": "paired training realization", "df": 4,
        "direction": "NeuroPixel minus RelativeTransformer",
        "pairs": pairs, "mean": mean, "sample_sd": sd,
        "minimum": min(differences), "maximum": max(differences),
        "confidence_level": 0.95, "interval": [mean - halfwidth, mean + halfwidth],
        "interval_clipped": False,
    }


def ordinary_metrics(predictions, golds, supported=None):
    supported = [True] * len(golds) if supported is None else supported
    require(len(predictions) == len(golds) == len(supported), "metric populations differ")
    per_answer, correct = {}, 0
    for prediction, gold, valid in zip(predictions, golds, supported):
        hit = bool(valid and isinstance(prediction, str) and prediction.lower() == gold)
        correct += int(hit)
        entry = per_answer.setdefault(gold, {"n": 0, "correct": 0})
        entry["n"] += 1
        entry["correct"] += int(hit)
    for entry in per_answer.values():
        entry["accuracy"] = entry["correct"] / entry["n"]
    return {
        "n": len(golds), "correct": correct,
        "accuracy": correct / len(golds) if golds else None,
        "per_answer": dict(sorted(per_answer.items())),
        "supported_gold_n": sum(supported), "unsupported_gold_n": len(golds) - sum(supported),
        "cross_entropy": None, "cross_entropy_n": 0,
    }


def neural_metrics(arrays, rows, mask, encoder):
    selected = [i for i, flag in enumerate(mask) if flag]
    vocabulary = encoder["vocabulary"]
    predicted = [vocabulary[int(arrays["prediction_id"][i])]
                 if int(arrays["prediction_id"][i]) >= 0 else None for i in selected]
    metric = ordinary_metrics(predicted, [rows[i]["gold"] for i in selected],
                              [rows[i]["supported_gold"] for i in selected])
    losses = [float(arrays["nll"][i]) for i in selected if bool(arrays["nll_supported"][i])]
    metric.update(
        cross_entropy=math.fsum(losses) / len(losses) if losses else None,
        cross_entropy_n=len(losses), nll_supported_n=len(losses),
        gold_vocabulary_supported_n=sum(rows[i]["supported_gold"] for i in selected),
        can_encode_n=sum(rows[i].get("can_encode", True) for i in selected),
        cannot_encode_n=sum(not rows[i].get("can_encode", True) for i in selected),
        gold_unseen_as_train_answer_n=sum(rows[i]["gold"] not in encoder["answer_tokens"] for i in selected),
    )
    return metric


def recount_arrays(path, rows, encoder, np):
    """No checkpoint load: authenticate IDs/masks and recompute scores from logits."""
    resource_admission()
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    required = {"logits", "prediction_id", "gold_id", "can_encode", "gold_supported",
                "nll_supported", "nll", "record_id"}
    require(set(arrays) == required, "prediction array keys differ")
    n, v = len(rows), len(encoder["vocabulary"])
    require(arrays["logits"].dtype == np.float32 and arrays["logits"].shape == (n, v)
            and np.isfinite(arrays["logits"]).all(), "invalid float32 logits")
    for key in required - {"logits"}:
        require(arrays[key].shape == (n,), "array row alignment differs: " + key)
    for key in ("prediction_id", "gold_id"):
        require(arrays[key].dtype == np.int64, "ID dtype differs: " + key)
    for key in ("can_encode", "gold_supported", "nll_supported"):
        require(arrays[key].dtype == np.bool_, "mask dtype differs: " + key)
    require(arrays["record_id"].dtype.kind == "U"
            and arrays["record_id"].tolist() == [r["record_id"] for r in rows], "record order differs")
    require(arrays["nll"].dtype == np.float64 and np.isfinite(arrays["nll"]).all()
            and np.all(arrays["nll"] >= 0), "invalid float64 NLL")
    can_encode = np.asarray([r.get("can_encode", True) for r in rows], dtype=np.bool_)
    supported = np.asarray([r["supported_gold"] for r in rows], dtype=np.bool_)
    gold = np.asarray([r["gold_id"] if r["gold_id"] is not None else -1 for r in rows], dtype=np.int64)
    require(np.array_equal(arrays["gold_id"], gold)
            and np.array_equal(arrays["can_encode"], can_encode)
            and np.array_equal(arrays["gold_supported"], supported)
            and np.array_equal(arrays["nll_supported"], can_encode & supported),
            "saved gold/support/encodability differs from raw-source adapter")
    require(np.array_equal(arrays["prediction_id"][can_encode],
                           arrays["logits"][can_encode].argmax(1)), "saved prediction is not first argmax")
    require(np.all(arrays["prediction_id"][~can_encode] == -1)
            and np.all(arrays["logits"][~can_encode] == 0),
            "unencodable placeholders must not imply inference")
    expected_nll = np.zeros(n, dtype=np.float64)
    eligible = can_encode & supported
    if eligible.any():
        values = arrays["logits"][eligible].astype(np.float64)
        shifted = values - values.max(axis=1, keepdims=True)
        expected_nll[eligible] = (
            np.log(np.exp(shifted).sum(axis=1))
            - shifted[np.arange(int(eligible.sum())), gold[eligible]]
        )
    require(np.allclose(arrays["nll"], expected_nll, atol=NLL_ATOL, rtol=NLL_RTOL),
            "saved NLL differs from float64 stable logsumexp of saved float32 logits")
    require(np.all(arrays["nll"][~eligible] == 0), "ineligible NLL placeholders must be zero")
    correct = can_encode & supported & (arrays["prediction_id"] == gold)
    return arrays, correct, {
        "n": n, "can_encode": int(can_encode.sum()), "gold_supported": int(supported.sum()),
        "nll_supported": int(eligible.sum()), "max_abs_nll_error": float(np.max(
            np.abs(arrays["nll"] - expected_nll))) if n else 0.0,
        "nll_atol": NLL_ATOL, "nll_rtol": NLL_RTOL,
        "argmax_tie_rule": "first vocabulary index",
    }


def fit_control_state(train_records):
    """Independent transparent controls; labels are used only for fitting TRAIN."""
    answers = Counter(r["answer"] for r in train_records)
    winner = lambda counter, fallback: min(counter, key=lambda word: (-counter[word], word)) if counter else fallback
    majority = winner(answers, None)
    exact, questions, bags = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    classes, vocabulary = defaultdict(Counter), set()
    for record in train_records:
        lex = visible([f["text"] for f in record["facts"]], record["question"])
        bag = Counter(token for row in lex["facts"] + [lex["question"]] for token in row)
        label = record["answer"]
        exact[identity(lex)][label] += 1
        questions[identity(lex["question"])][label] += 1
        bags[identity(dict(bag))][label] += 1
        classes[label].update(bag)
        vocabulary.update(bag)
    return {"answers": answers, "majority": majority, "exact": exact, "questions": questions,
            "bags": bags, "classes": classes, "vocabulary": vocabulary}


def shortcut_predictions(model, facts, question):
    answers, majority = model["answers"], model["majority"]
    exact, questions, bags = model["exact"], model["questions"], model["bags"]
    classes, vocabulary = model["classes"], model["vocabulary"]
    winner = lambda counter, fallback: min(counter, key=lambda word: (-counter[word], word)) if counter else fallback
    lex = visible(facts, question)
    bag = Counter(token for row in lex["facts"] + [lex["question"]] for token in row)
    facts_frequency = Counter(token for row in lex["facts"] for token in row if token in answers)
    mentioned = set(lex["question"])
    scores = {}
    for label in sorted(answers):
        counts = classes[label]
        denominator = sum(counts.values()) + len(vocabulary)
        scores[label] = math.log((answers[label] + 1) / (sum(answers.values()) + len(answers)))
        scores[label] += math.fsum(count * math.log((counts.get(token, 0) + 1) / denominator)
                                  for token, count in sorted(bag.items()) if token in vocabulary)
    return {
        "majority": majority,
        "exact_memory": winner(exact.get(identity(lex), {}), majority),
        "question_only": winner(questions.get(identity(lex["question"]), {}), majority),
        "bow_memory": winner(bags.get(identity(dict(bag)), {}), majority),
        "bow_naive_bayes": min(scores, key=lambda token: (-scores[token], token)),
        "fact_answer_frequency": winner(facts_frequency, majority),
        "fact_answer_frequency_excluding_query": winner(
            {token: count for token, count in facts_frequency.items() if token not in mentioned}, majority),
    }


def symbolic_raw(facts, question):
    """Literal direct directional edges and their inverses; no transitive closure."""
    inverse = {"north": "south", "south": "north", "east": "west", "west": "east"}
    edges = set()
    for fact in facts:
        words = " ".join(fact.lower().split())
        match = re.fullmatch(r"(?:the )?(\w+) is (north|south|east|west) of (?:the )?(\w+)\.", words)
        if match is None:
            return {"status": "unsupported", "answer": None, "candidates": []}
        subject, relation, obj = match.groups()
        edges.add((subject, relation, obj))
        edges.add((obj, inverse[relation], subject))
    query = " ".join(question.lower().split())
    subject_form = re.fullmatch(r"what is (north|south|east|west) of (?:the )?(\w+)\?", query)
    object_form = re.fullmatch(r"what is (?:the )?(\w+) (north|south|east|west) of\?", query)
    if subject_form:
        relation, obj = subject_form.groups()
        candidates = sorted({a for a, rel, b in edges if rel == relation and b == obj})
    elif object_form:
        subject, relation = object_form.groups()
        candidates = sorted({b for a, rel, b in edges if rel == relation and a == subject})
    else:
        candidates = []
    return {
        "status": "ok" if len(candidates) == 1 else "ambiguous" if candidates else "unsupported",
        "answer": candidates[0] if len(candidates) == 1 else None,
        "candidates": candidates,
    }


def compare_json(expected, observed, path="root", *, atol=1e-12):
    if isinstance(expected, dict):
        require(isinstance(observed, dict) and set(expected) == set(observed), "JSON keys differ: " + path)
        for key in expected:
            compare_json(expected[key], observed[key], path + "." + key, atol=atol)
    elif isinstance(expected, list):
        require(isinstance(observed, list) and len(expected) == len(observed), "JSON lengths differ: " + path)
        for i, (left, right) in enumerate(zip(expected, observed)):
            compare_json(left, right, path + "[" + str(i) + "]", atol=atol)
    elif isinstance(expected, float):
        require(isinstance(observed, (float, int)) and not isinstance(observed, bool)
                and math.isfinite(observed) and math.isclose(expected, observed, rel_tol=0, abs_tol=atol),
                "numeric JSON value differs: " + path)
    else:
        require(type(expected) is type(observed) and expected == observed, "JSON value differs: " + path)


class Audit:
    def __init__(self, args, np, scipy):
        from pathlib import Path
        from datetime import datetime, timezone
        self.args, self.np, self.scipy = args, np, scipy
        self.root, self.dev = args.input.resolve(), args.development.resolve()
        self.output = args.output_dir.resolve()
        self.plan = self.read(args.plan)
        self.recipe = self.plan["recipe"]
        self.inputs, self.steps, self.run_records = {}, [], {}
        self.input_paths = {}
        self.report = {
            "schema_version": 1, "item": 10, "status": "started", "phase": args.phase,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "issues": [], "checks": self.steps, "input_files": self.inputs,
            "scope": "Independent saved-array recount; no Torch, model construction, pickle, checkpoint evaluation or raw-source answer substitution.",
            "limits": [
                "Byte hashes do not independently establish remote archive ancestry or publication time; the operational archive audit owns that evidence.",
                "Checkpoint files are hashed only, never deserialized; weight finiteness and parameter tensors are not rechecked here.",
                "Recorded input streams are matched across families and counted, not regenerated from their declared PRNG seeds.",
                "Resource and timing fields describe saved measurements and configured limits, not continuously observed physical utilization.",
                "bAbI QA4 is external to this project but synthetic; success does not establish natural-language or out-of-generator generalization.",
            ],
        }
        self.report["auditor"] = {"path": Path(__file__).name, "sha256": self.file_sha(Path(__file__))}
        self.report["runtime"] = {
            "python": __import__("platform").python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "threads": 1, "bootstrap_generator": "PCG64",
            "bootstrap_seed": 104001, "bootstrap_replicates": 2000,
        }

    @staticmethod
    def read(path):
        from pathlib import Path
        return json.loads(Path(path).read_text(encoding="utf-8"),
                          parse_constant=lambda value: (_ for _ in ()).throw(AuditError("nonfinite JSON: " + value)))

    @staticmethod
    def file_sha(path):
        result = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                result.update(block)
        return result.hexdigest()

    def record_file(self, path, label):
        path = path.resolve()
        require(path.is_file(), "missing input: " + str(path))
        actual = {"bytes": path.stat().st_size, "sha256": self.file_sha(path)}
        previous = self.inputs.get(label)
        require(previous is None or previous == actual, "input changed during audit: " + label)
        self.inputs[label] = actual
        self.input_paths[label] = path
        return actual

    def load(self, root, name, label=None):
        path = (root / name).resolve()
        require(path.is_relative_to(root), "input escapes its declared root")
        self.record_file(path, label or str(root.name) + "/" + name)
        return self.read(path)

    def reference(self, root, ref, label=None):
        require(isinstance(ref, dict) and {"path", "sha256", "bytes"} <= set(ref), "incomplete artifact reference")
        path = (root / ref["path"]).resolve()
        require(path.is_relative_to(root), "reference escapes declared root")
        actual = self.record_file(path, label or root.name + "/" + ref["path"])
        require(actual == {"bytes": ref["bytes"], "sha256": ref["sha256"]}, "artifact bytes/hash differ: " + ref["path"])
        return path

    def done(self, name, **observed):
        self.steps.append({"check": name, "status": "passed", **observed})

    def source(self):
        require(self.plan["schema_version"] == 1 and self.plan["item"] == 10
                and self.plan["status"] == "frozen" and bool(self.plan["freeze_utc"]), "plan not frozen item10")
        require(self.plan["phase"] == self.args.phase, "analysis phase differs from scientific plan")
        expected = self.plan["analysis_runtime"]
        compare_json(expected, self.report["runtime"], "analysis_runtime")
        self.record_file(self.args.plan.resolve(), "scientific_plan")
        for name, expected_hash in self.plan["implementation_sha256"].items():
            path = (self.args.source_root.resolve() / name).resolve()
            require(path.is_relative_to(self.args.source_root.resolve()), "source binding escapes root")
            actual = self.record_file(path, "source/" + name)
            require(actual["sha256"] == expected_hash, "source binding differs: " + name)
        self.done("source_and_plan_bindings", files=len(self.plan["implementation_sha256"]),
                  plan_sha256=self.inputs["scientific_plan"]["sha256"])

    def development(self):
        spec = self.plan["inputs"]["development"]
        for name, expected_hash in spec["files_sha256"].items():
            actual = self.record_file((self.dev / name).resolve(), "development/" + name)
            require(actual["sha256"] == expected_hash, "development input binding differs: " + name)
        raw_path = self.dev / "selected_qa4_train.txt"
        raw_info = self.record_file(raw_path, "development/selected_qa4_train.txt")
        selected_source = self.load(self.dev, "selected_source.json", "development/selected_source.json")
        require(selected_source["train_member"]["sha256"] == raw_info["sha256"]
                and selected_source["test_payload_extracted"] is False
                and selected_source["test_payload_parsed"] is False, "original TRAIN acquisition identity/access differs")
        raw = parse_raw_qa(raw_path.read_bytes().decode("utf-8", errors="strict"))
        parsed = self.load(self.dev, "official_train_records.json", "development/official_train_records.json")
        require(parsed["source"]["sha256"] == raw_info["sha256"] == raw["source_sha256"],
                "TRAIN source byte identity differs")
        compare_json(raw["records"], parsed["records"], "independent_TRAIN_parser")
        self.raw_train = raw["records"]
        by_id = {r["record_id"]: r for r in self.raw_train}
        split = self.load(self.dev, "development_split.json", "development/development_split.json")
        groups = connected_components(self.raw_train)
        for group in groups:
            draw = int(identity({"salt": "neuropixel-item10-dev-v1", "group_id": group["group_id"]}), 16) / (1 << 256)
            group["partition"] = "validation" if draw < 0.1 else "train"
        mapping = {rid: group["group_id"] for group in groups for rid in group["record_ids"]}
        partitions = {group["group_id"]: group["partition"] for group in groups}
        train_ids = [r["record_id"] for r in self.raw_train if partitions[mapping[r["record_id"]]] == "train"]
        validation_ids = [r["record_id"] for r in self.raw_train if partitions[mapping[r["record_id"]]] == "validation"]
        require(train_ids and validation_ids, "empty grouped development partition")
        saved_groups = [{k: v for k, v in group.items() if k != "indices"} for group in groups]
        expected_split = {
            "schema_version": 1, "salt": "neuropixel-item10-dev-v1", "dev_fraction": 0.1,
            "allow_conflicts": True, "grouping": "connected_episode_full_story_and_normalized_input_v1",
            "record_group": mapping, "groups": saved_groups, "train_ids": train_ids,
            "validation_ids": validation_ids, "population_record_ids_sha256": identity(sorted(mapping)),
        }
        compare_json(expected_split, split, "independent_development_split")
        self.train_raw_records = [by_id[rid] for rid in train_ids]
        labels = defaultdict(set)
        for row in self.raw_train:
            labels[row["input_sha256"]].add(row["answer"])
        require(all(len(value) == 1 for value in labels.values()), "TRAIN raw input has conflicting labels")
        encoder = self.load(self.dev, "encoder.json", "development/encoder.json")
        expected_encoder = reconstruct_encoder(self.train_raw_records, {
            "max_height": 8, "max_width": 16, "overflow": "error", "input_oov": "unk",
            "lowercase": True, "preserve_punctuation": True,
        })
        compare_json(expected_encoder, encoder, "independent_encoder")
        self.encoder = encoder
        encoded = self.load(self.dev, "encoded_official_train.json", "development/encoded_official_train.json")
        require(len(encoded) == len(self.raw_train), "TRAIN encoded population length differs")
        encoded_by_id = {}
        for raw_row, saved in zip(self.raw_train, encoded):
            require(saved["record_id"] == raw_row["record_id"], "TRAIN record order differs")
            canvas = reconstruct_canvas([f["text"] for f in raw_row["facts"]], raw_row["question"], encoder)
            require(canvas is not None and canvas == saved["canvas"], "TRAIN encoded canvas differs")
            supported = raw_row["answer"] in encoder["token_to_id"] and encoder["token_to_id"][raw_row["answer"]] >= 2
            require(saved["gold"] == raw_row["answer"] and saved["input_sha256"] == raw_row["input_sha256"]
                    and saved["encoded_input_sha256"] == identity(canvas)
                    and saved["supported_gold"] is supported
                    and saved["gold_id"] == (encoder["token_to_id"][raw_row["answer"]] if supported else None),
                    "TRAIN adapter target/identity differs")
            encoded_by_id[saved["record_id"]] = saved
        self.train = [encoded_by_id[rid] for rid in train_ids]
        self.validation = [encoded_by_id[rid] for rid in validation_ids]
        self.all_encoded = encoded
        self.encoded_by_id = encoded_by_id
        self.control_state = fit_control_state(self.train_raw_records)
        self.check_development_controls(by_id, train_ids, validation_ids)
        expected_identity = {
            "archive_commit": spec["archive_commit"], "path": spec["path"],
            "files_sha256": {name: spec["files_sha256"][name] for name in (
                "encoder.json", "development_split.json", "encoded_official_train.json",
                "official_train_records.json", "train_audit_summary.json", "selected_source.json", "shortcut_models.json")},
            "encoder_sha256": spec["files_sha256"]["encoder.json"],
            "train_ids_sha256": identity(train_ids), "validation_ids_sha256": identity(validation_ids),
        }
        self.data_identity, self.data_hash = expected_identity, identity(expected_identity)
        self.report["development"] = {
            "records": len(self.raw_train), "groups": len(groups), "train": len(train_ids),
            "validation": len(validation_ids), "raw_input_gold_conflicts": 0,
            "height": encoder["height"], "width": encoder["width"], "vocabulary": len(encoder["vocabulary"]),
            "data_identity_sha256": self.data_hash,
        }
        self.done("raw_TRAIN_parser_grouping_encoder", records=len(self.raw_train), groups=len(groups))

    def expected_config(self, family, seed, lr, kind):
        if kind == "memorization":
            seen, rows = set(), []
            for row in sorted(self.train, key=lambda r: r["record_id"]):
                if row["input_sha256"] not in seen:
                    rows.append(row); seen.add(row["input_sha256"])
                if len(rows) == 32:
                    break
            require(len(rows) == 32, "fewer than32 unique TRAIN inputs for memorization")
            run_id, updates = "mem_" + family + "_s58", 4096
        elif kind == "pilot":
            rows = self.train
            run_id = "pilot_" + family + "_lr" + {0.001: "0001", 0.003: "0003"}[lr] + "_s59"
            updates = 1024
        else:
            rows = self.train
            run_id, updates = "train_" + family + "_s" + str(seed), 4096
        e = self.encoder
        kwargs = {"c_id": 16, "c": 48, "hidden": 128, "steps": 16, "fire_rate": 0.5, "retina": False} if family == "neuropixel" else {
            "d": 32, "layers": 2, "heads": 4, "ff": 144}
        model = {"family": family, "vocab": len(e["vocabulary"]), "height": e["height"],
                 "width": e["width"], "out_pos": e["out_pos"], "kwargs": kwargs}
        config = {
            "run_id": run_id, "kind": kind, "family": family, "init_seed": seed,
            "firing_seed": 110000 + seed, "sample_seed": 100000 + seed,
            "learning_rate": lr, "updates": updates, "batch_size": 32,
            "weight_decay": 1e-4, "gradient_clip": 1.0, "model": model,
            "recipe_sha256": identity(self.recipe), "data_identity_sha256": self.data_hash,
            "training_record_ids": [r["record_id"] for r in rows],
            "training_record_ids_sha256": identity([r["record_id"] for r in rows]),
        }
        return config, rows

    def inspect_run(self, root, reference, expected_config, rows):
        import statistics
        np = self.np
        path = self.reference(root, reference)
        run = self.read(path)
        require(run["status"] == "completed", "incomplete run: " + expected_config["run_id"])
        compare_json(self.active_source, run["source"], "common_run_source")
        compare_json(expected_config, run["config"], "run_config")
        require(run["run_id"] == expected_config["run_id"]
                and run["config_sha256"] == identity(expected_config)
                and run["final_accessed"] is False, "run identity/access flag differs")
        completed = run["completed_updates"]
        require(type(completed) is int and 1 <= completed <= expected_config["updates"], "invalid update count")
        if expected_config["kind"] != "memorization":
            require(completed == expected_config["updates"], "fixed-budget run stopped early")
        self.reference(root, run["checkpoint"])
        config_path = self.reference(root, run["config_artifact"])
        compare_json(expected_config, self.read(config_path), "config_artifact")
        index_path = self.reference(root, run["minibatch_indices"])
        indices = np.load(index_path, allow_pickle=False)
        require(indices.dtype == np.int64 and indices.shape == (expected_config["updates"], 32)
                and (indices >= 0).all() and (indices < len(rows)).all(), "invalid saved index stream")
        require(digest(indices.tobytes(order="C")) == run["minibatch_indices_content_sha256"]
                and digest(indices[:completed].tobytes(order="C")) == run["used_minibatch_indices_sha256"],
                "saved index content hash differs")
        used = set(indices[:completed].reshape(-1).tolist())
        require(run["unique_training_record_count"] == len(used)
                and run["unique_training_raw_input_count"] == len({rows[i]["input_sha256"] for i in used})
                and run["unique_training_encoded_input_count"] == len({rows[i]["encoded_input_sha256"] for i in used}),
                "actual sampled membership counts differ")
        logfile = self.reference(root, run["training_log"])
        logs = [json.loads(line) for line in logfile.read_text(encoding="utf-8").splitlines() if line]
        expected_updates = list(range(128, completed + 1, 128))
        if completed % 128:
            expected_updates.append(completed)
        require([entry["update"] for entry in logs] == expected_updates, "training-log update schedule differs")
        for log in logs:
            require(math.isfinite(log["mean_answer_ce"]) and log["mean_answer_ce"] >= 0
                    and math.isfinite(log["last_gradient_norm"]) and log["last_gradient_norm"] >= 0,
                    "invalid recorded training statistic")
        for key, population in (("validation", self.validation), ("train_probe", self.train)):
            prediction_path = self.reference(root, run[key]["predictions"])
            arrays, _, check = recount_arrays(prediction_path, population, self.encoder, np)
            metric = neural_metrics(arrays, population, [True] * len(population), self.encoder)
            compare_json(metric, run[key]["metrics"], expected_config["run_id"] + "." + key, atol=NLL_ATOL)
        checks = run["memorization_checks"]
        if expected_config["kind"] == "memorization":
            require([check["update"] for check in checks] == expected_updates, "memorization check inventory differs")
            streak = 0
            for position, check in enumerate(checks):
                arrays, _, _ = recount_arrays(self.reference(root, check["predictions"]), rows, self.encoder, np)
                metric = neural_metrics(arrays, rows, [True] * len(rows), self.encoder)
                require(check["n"] == 32 and check["record_ids_sha256"] == expected_config["training_record_ids_sha256"],
                        "memorization population differs")
                compare_json(metric["accuracy"], check["accuracy"], "memorization_accuracy")
                compare_json(metric["cross_entropy"], check["cross_entropy"], "memorization_CE", atol=NLL_ATOL)
                passed = metric["accuracy"] == 1.0 and metric["cross_entropy"] <= 0.05
                streak = streak + 1 if passed else 0
                require(check["criterion_pass"] is passed and check["consecutive_passes"] == streak,
                        "memorization streak decision differs")
                require(position == len(checks) - 1 or streak < 2, "memorization continued after stopping criterion")
                compare_json(check, logs[position]["memorization_check"], "memorization_log", atol=NLL_ATOL)
            require(streak >= 2 and run["memorization_pass"] is True, "memorization admission failed")
        else:
            require(checks == [] and run["memorization_pass"] is None, "unexpected memorization checks")
        update_seconds = run["update_seconds"]
        require(len(update_seconds) == completed
                and all(isinstance(t, (int, float)) and math.isfinite(t) and t >= 0 for t in update_seconds),
                "invalid saved update-time population")
        compare_json(math.fsum(update_seconds) / completed, run["mean_update_seconds"], "mean_update_seconds")
        compare_json(float(statistics.median(update_seconds)), run["median_update_seconds"], "median_update_seconds")
        for key in ("training_seconds", "elapsed_total_seconds"):
            require(isinstance(run[key], (int, float)) and math.isfinite(run[key]) and run[key] >= 0,
                    "invalid recorded wall-time field")
        e = self.encoder
        expected_parameters = (29264 + 16 * len(e["vocabulary"])) if expected_config["family"] == "neuropixel" else (
            25472 + 65 * len(e["vocabulary"]) + 8 * (2 * e["height"] - 1) * (2 * e["width"] - 1) + 2080)
        require(run["parameter_count"] == expected_parameters, "recorded parameter count differs from declared architecture formula")
        require(run["train_probe_accuracy_below_095"] is (run["train_probe"]["metrics"]["accuracy"] < 0.95),
                "train probe competence flag differs")
        self.run_records[run["run_id"]] = run
        self.done("run_recount", run_id=run["run_id"], completed_updates=completed,
                  checkpoint_sha256=run["checkpoint"]["sha256"], parameter_count=expected_parameters)
        return run


    def environment(self, root, phase):
        runtime = self.load(root, phase + "_environment.json")
        expected = self.plan["runtime"]
        require(runtime["python"] == expected["python"] and runtime["packages"] == expected["packages"]
                and runtime["threads"] == expected["threads"] == 2
                and runtime["interop_threads"] == expected["interop_threads"] == 1
                and runtime["cuda_version"] is None and runtime["available_ram_gib"] >= 8.0,
                "saved training/inference runtime differs")
        return runtime

    def preflight(self, root):
        summary = self.load(root, "preflight_summary.json", "preflight/preflight_summary.json")
        require(summary["schema_version"] == 1 and summary["item"] == 10
                and summary["status"] == "completed" and summary["all_runs_completed"]
                and summary["memorization_pass"] and summary["admission_passed"]
                and summary["final_accessed"] is False, "preflight admission flags incomplete")
        require(summary["recipe_sha256"] == identity(self.recipe)
                and summary["data_identity_sha256"] == self.data_hash, "preflight dependency drift")
        compare_json(self.data_identity, summary["data_identity"], "preflight_data_identity")
        if self.args.phase == "study":
            spec = self.plan["inputs"]["preflight"]
            for name, expected in spec["files_sha256"].items():
                require(self.record_file((root / name).resolve(), "preflight/" + name)["sha256"] == expected,
                        "pinned preflight input differs: " + name)
        self.active_source = summary["source"]
        for path in ("neuropixel/model.py", "neuropixel/research/models.py",
                     "neuropixel/research/babi_qa.py", "scripts/research_babi_study.py"):
            require(summary["source"]["implementation_sha256"][path] == self.plan["implementation_sha256"][path],
                    "preflight scientific source differs: " + path)
        configs = [self.expected_config(f, 58, 0.003, "memorization") for f in FAMILIES]
        configs += [self.expected_config(f, 59, lr, "pilot") for f in FAMILIES for lr in (0.001, 0.003)]
        require(summary["ordered_run_ids"] == [c["run_id"] for c, _ in configs]
                and len(summary["runs"]) == 6, "six-run preflight inventory differs")
        runs = [self.inspect_run(root, ref, cfg, rows)
                for ref, (cfg, rows) in zip(summary["runs"], configs)]
        choices = {}
        for family in FAMILIES:
            candidates = [run for run in runs if run["config"]["kind"] == "pilot"
                          and run["config"]["family"] == family]
            selected = min(candidates, key=lambda row: (-row["validation"]["metrics"]["accuracy"],
                row["validation"]["metrics"]["cross_entropy"], row["config"]["learning_rate"]))
            choices[family] = {
                "run_id": selected["run_id"], "learning_rate": selected["config"]["learning_rate"],
                "validation": selected["validation"]["metrics"], "checkpoint_sha256": selected["checkpoint"]["sha256"],
                "config_sha256": selected["config_sha256"],
            }
        expected_selection = {
            "schema_version": 1, "item": 10,
            "rule": ["validation_accuracy_desc", "validation_ce_asc", "lr_asc"],
            "recipe_sha256": identity(self.recipe), "data_identity_sha256": self.data_hash,
            "choices": choices,
        }
        selected = self.read(self.reference(root, summary["selection"], "preflight/selection.json"))
        compare_json(expected_selection, selected, "validation_only_selection", atol=NLL_ATOL)
        self.selection = selected
        self.check_stream_pairs(runs)
        self.report["preflight"] = {
            "source": summary["source"], "runtime": self.environment(root, "preflight"),
            "run_ids": [r["run_id"] for r in runs], "selection": selected,
            "memorization": {r["run_id"]: {
                "checks": r["memorization_checks"], "completed_updates": r["completed_updates"],
                "passed": r["memorization_pass"],
            } for r in runs if r["config"]["kind"] == "memorization"},
            "validation": {r["run_id"]: r["validation"]["metrics"] for r in runs},
            "train_probe": {r["run_id"]: r["train_probe"]["metrics"] for r in runs},
        }
        self.done("preflight_selection", completed_runs=6, all_memorization_checks_recounted=True)

    def check_stream_pairs(self, runs):
        groups = defaultdict(list)
        for run in runs:
            groups[(run["config"]["kind"], run["config"]["init_seed"])].append(run)
        result = []
        for (kind, seed), members in sorted(groups.items()):
            require(len({r["config"]["training_record_ids_sha256"] for r in members}) == 1
                    and len({r["minibatch_indices_content_sha256"] for r in members}) == 1
                    and len({r["firing_rng_initial_sha256"] for r in members}) == 1,
                    "matched-family initialization-separate input/firing setup differs")
            result.append({
                "kind": kind, "seed": seed, "run_ids": [r["run_id"] for r in members],
                "planned_stream_content_sha256": members[0]["minibatch_indices_content_sha256"],
                "training_record_ids_sha256": members[0]["config"]["training_record_ids_sha256"],
                "scope": "Identical saved planned indices and initial firing state; model-specific firing consumption can differ.",
            })
        self.report.setdefault("paired_streams", []).extend(result)

    def gate_and_primary(self):
        from datetime import datetime
        manifest = self.load(self.root, "training_manifest.json", "training_manifest")
        gate = self.load(self.root, "pre_final_archive_receipt.json", "pre_final_archive_receipt")
        access = self.load(self.root, "final_access.json", "final_access")
        require(manifest["schema_version"] == 1 and manifest["item"] == 10
                and manifest["status"] == "completed" and manifest["final_accessed"] is False,
                "training inventory is not closed before final")
        source = manifest["source"]
        require(source["plan_sha256"] == self.inputs["scientific_plan"]["sha256"]
                and source["implementation_sha256"] == self.plan["implementation_sha256"]
                and source["tracked_source_clean"] is True, "primary scientific source differs")
        for phase in ("train", "final"):
            status = self.load(self.root, phase + "_status.json", phase + "_status")
            require(status["phase"] == phase and status["status"] == "completed", "phase did not complete")
            require(status["source"]["source_commit"] == source["source_commit"]
                    and status["source_after"]["source_commit"] == source["source_commit"]
                    and status["source_after"]["implementation_sha256"] == self.plan["implementation_sha256"],
                    "phase source-before/after differs")
        require(manifest["source_commit"] == source["source_commit"]
                and manifest["recipe_sha256"] == identity(self.recipe)
                and manifest["data_identity_sha256"] == self.data_hash, "primary inventory dependencies differ")
        compare_json(self.data_identity, manifest["data_identity"], "primary_data_identity")
        compare_json(self.selection, manifest["selection"], "primary_selected_recipe", atol=NLL_ATOL)
        preflight_spec = self.plan["inputs"]["preflight"]
        compare_json({"archive_commit": preflight_spec["archive_commit"], "path": preflight_spec["path"],
                      "selection_sha256": preflight_spec["files_sha256"]["selection.json"],
                      "preflight_summary_sha256": preflight_spec["files_sha256"]["preflight_summary.json"]},
                     manifest["selection_identity"], "selected_preflight_identity")
        require(gate["schema_version"] == 1 and gate["item"] == 10
                and gate["source_commit"] == source["source_commit"]
                and gate["training_manifest_sha256"] == self.inputs["training_manifest"]["sha256"]
                and gate["run_key"] == manifest["run_key"] == self.root.name,
                "archive receipt is not tied to closed inventory")
        require(re.fullmatch(r"[0-9a-f]{40}", gate["training_archive_commit"]) is not None,
                "invalid training archive commit")
        require(access["schema_version"] == 1 and access["item"] == 10 and access["status"] == "consumed"
                and access["training_manifest_sha256"] == self.inputs["training_manifest"]["sha256"]
                and access["pre_final_archive_receipt_sha256"] == self.inputs["pre_final_archive_receipt"]["sha256"]
                and access["training_archive_commit"] == gate["training_archive_commit"], "final access is not bound to archive gate")
        compare_json(source, access["source"], "final_access_source")
        time = lambda value: datetime.fromisoformat(value.replace("Z", "+00:00"))
        require(time(manifest["created_at_utc"]) <= time(gate["at_utc"]) <= time(access["consumed_at_utc"]),
                "recorded gate chronology differs")
        configs = [self.expected_config(family, seed, self.selection["choices"][family]["learning_rate"], "primary")
                   for seed in SEEDS for family in FAMILIES]
        require(manifest["ordered_run_ids"] == [config["run_id"] for config, _ in configs]
                and len(manifest["runs"]) == 10, "ten primary configurations missing or reordered")
        self.active_source, runs = source, []
        for item, (config, rows) in zip(manifest["runs"], configs):
            compare_json(config, item["config"], "manifest_config")
            require(item["run_id"] == config["run_id"] and item["config_sha256"] == identity(config),
                    "manifest run ID/config digest differs")
            run = self.inspect_run(self.root, item["summary"], config, rows)
            compare_json(item["checkpoint"], run["checkpoint"], "manifest_checkpoint")
            require(time(run["completed_at_utc"]) <= time(manifest["created_at_utc"]), "run completed after closed inventory")
            runs.append(run)
        require(len({run["checkpoint"]["path"] for run in runs}) == 10, "checkpoints alias an artifact path")
        self.check_stream_pairs(runs)
        self.primary_runs = runs
        self.report["gate"] = {
            "training_archive_commit": gate["training_archive_commit"], "manifest_at_utc": manifest["created_at_utc"],
            "gate_at_utc": gate["at_utc"], "access_at_utc": access["consumed_at_utc"],
            "source_commit": source["source_commit"], "checkpoint_files_hashed": 10,
            "remote_ancestry_independently_verified_here": False,
        }
        self.report["training_runtime"] = self.environment(self.root, "train")
        self.report["inference_runtime"] = self.environment(self.root, "final")
        self.done("closed_training_inventory_and_local_final_gate", runs=10)
        return manifest, access

    def final_population(self, access):
        from datetime import datetime
        raw_path = self.root / "final_data" / "official_test.txt"
        raw_info = self.record_file(raw_path, "final_data/official_test.txt")
        independent = parse_raw_qa(raw_path.read_bytes().decode("utf-8", errors="strict"))
        saved = self.load(self.root, "final_data/records.json", "final_data/records.json")
        require(saved["source"]["sha256"] == raw_info["sha256"] == independent["source_sha256"]
                and saved["source"]["member"] == self.recipe["test_member"], "final raw source identity differs")
        compare_json(independent["records"], saved["records"], "independent_TEST_parser")
        selected_source = self.load(self.dev, "selected_source.json", "development/selected_source.json")
        require(saved["source"]["archive_sha256"] == selected_source["archive_sha256"], "final source archive differs from TRAIN source")
        population = self.load(self.root, "final_data/population_manifest.json", "final_data/population_manifest.json")
        require(datetime.fromisoformat(access["consumed_at_utc"].replace("Z", "+00:00"))
                <= datetime.fromisoformat(population["created_at_utc"].replace("Z", "+00:00")),
                "final population was recorded before access receipt")
        for key in ("raw", "records", "encoded_records"):
            self.reference(self.root, population[key])
        raw_train = {r["input_sha256"] for r in self.train}
        enc_train = {r["encoded_input_sha256"] for r in self.train}
        raw_exposure = {r["input_sha256"] for r in self.raw_train}
        enc_exposure = {r["encoded_input_sha256"] for r in self.all_encoded}
        fixture_overflows = []
        for fixture in self.recipe["development_fixtures"]:
            require(set(fixture) == {"facts", "question"}, "non-input fixture metadata leaked into exposure keys")
            raw_key = identity(visible(fixture["facts"], fixture["question"]))
            raw_exposure.add(raw_key)
            canvas = reconstruct_canvas(fixture["facts"], fixture["question"], self.encoder)
            if canvas is None:
                fixture_overflows.append(raw_key)
            else:
                enc_exposure.add(identity(canvas))
        require([row["input_sha256"] for row in population["fixture_encoded_overflows"]] == fixture_overflows,
                "fixture encoding-failure exposure accounting differs")
        saved_rows = self.load(self.root, "final_data/encoded_records.json", "final_data/encoded_records.json")
        require(len(saved_rows) == len(independent["records"]), "final encoded population length differs")
        self.final_rows, overlap = [], Counter()
        for record, row in zip(independent["records"], saved_rows):
            facts = [f["text"] for f in record["facts"]]
            canvas = reconstruct_canvas(facts, record["question"], self.encoder)
            lex = visible(facts, record["question"])
            encoded_key = identity(canvas) if canvas is not None else None
            value = self.encoder["token_to_id"].get(record["answer"], -1)
            expected = {
                "record_id": record["record_id"], "episode_id": record["episode_id"],
                "episode_sha256": record["episode_sha256"], "input_sha256": record["input_sha256"],
                "gold": record["answer"], "can_encode": canvas is not None, "canvas": canvas,
                "encoded_input_sha256": encoded_key,
                "unknown_input_tokens": [token for line in lex["facts"] + [lex["question"]]
                    for token in line if token not in self.encoder["token_to_id"]] if canvas is not None else None,
                "supported_gold": value >= 2, "gold_id": value if value >= 2 else None,
                "novel_vs_optimization_train": record["input_sha256"] not in raw_train
                    and (canvas is None or encoded_key not in enc_train),
                "strict_novel_vs_full_exposure": record["input_sha256"] not in raw_exposure
                    and (canvas is None or encoded_key not in enc_exposure),
            }
            compare_json(expected, row, "final_adapter")
            overlap["raw_overlap_optimization_train"] += int(record["input_sha256"] in raw_train)
            overlap["encoded_overlap_optimization_train"] += int(canvas is not None and encoded_key in enc_train)
            overlap["raw_overlap_full_exposure"] += int(record["input_sha256"] in raw_exposure)
            overlap["encoded_overlap_full_exposure"] += int(canvas is not None and encoded_key in enc_exposure)
            self.final_rows.append(expected)
        self.masks = {
            "all_official": [True] * len(self.final_rows),
            "novel_vs_optimization_train": [r["novel_vs_optimization_train"] for r in self.final_rows],
            "strict_novel_vs_full_exposure": [r["strict_novel_vs_full_exposure"] for r in self.final_rows],
        }
        compare_json(self.masks, population["masks"], "final_masks")
        compare_json({name: sum(mask) for name, mask in self.masks.items()}, population["subset_counts"], "subset_counts")
        compare_json({name: sum(flag and not row["can_encode"] for flag, row in zip(mask, self.final_rows))
                      for name, mask in self.masks.items()}, population["subset_unencodable_counts"], "subset_unencodable_counts")
        require(population["record_count"] == len(self.final_rows)
                and population["unsupported_gold_count"] == sum(not r["supported_gold"] for r in self.final_rows)
                and population["unencodable_count"] == sum(not r["can_encode"] for r in self.final_rows),
                "final population denominators differ")
        issues = self.load(self.root, "final_data/encoding_issues.json", "final_data/encoding_issues.json")
        require([entry["record_id"] for entry in issues] == [r["record_id"] for r in self.final_rows if not r["can_encode"]],
                "encoding issue inventory differs")
        self.final_raw = independent["records"]
        self.groups = connected_components(self.final_raw)
        by_input = defaultdict(Counter)
        for row in self.final_raw:
            by_input[row["input_sha256"]][row["answer"]] += 1
        self.report["final_population"] = {
            "n": len(self.final_rows), "unique_raw_inputs": len(by_input), "connected_test_groups": len(self.groups),
            "component_sizes": [group["n"] for group in self.groups], "subset_counts": population["subset_counts"],
            "subset_unencodable_counts": population["subset_unencodable_counts"],
            "overlaps": dict(overlap), "unsupported_gold_count": population["unsupported_gold_count"],
            "unencodable_count": population["unencodable_count"], "fixture_unencodable_contexts": len(fixture_overflows),
            "raw_conflicting_input_count": sum(len(counts) > 1 for counts in by_input.values()),
            "grouping": "Connected original episode, complete normalized fact story, or identical normalized visible input.",
            "novelty_definition": "Raw unseen AND frozen-encoded unseen; unencodable inputs remain in applicable raw-novel subsets and score wrong.",
            "scope": "Exact lexical/input novelty only; not novel entities, relations, templates or generator structure.",
        }
        self.done("independent_final_parser_adaptation_masks", n=len(self.final_rows), groups=len(self.groups))
        return population


    def check_development_controls(self, by_id, train_ids, validation_ids):
        state = self.control_state
        expected_model = {
            "schema_version": 1, "tokenizer": "lowercase_word_punctuation_v1", "alpha": 1.0,
            "tie_rule": "lexical_ascending", "answer_counts": dict(sorted(state["answers"].items())),
            "majority": state["majority"], "answer_tokens": sorted(state["answers"]),
            "exact_counts": {k: dict(sorted(v.items())) for k, v in sorted(state["exact"].items())},
            "question_counts": {k: dict(sorted(v.items())) for k, v in sorted(state["questions"].items())},
            "bow_memory_counts": {k: dict(sorted(v.items())) for k, v in sorted(state["bags"].items())},
            "bow_vocabulary": sorted(state["vocabulary"]),
            "bow_class_tokens": {k: dict(sorted(v.items())) for k, v in sorted(state["classes"].items())},
            "training_record_ids_sha256": identity(sorted(train_ids)),
        }
        compare_json(expected_model, self.load(self.dev, "shortcut_models.json", "development/shortcut_models.json"),
                     "independent_shortcut_fit")
        expected_predictions = {}
        for record in self.raw_train:
            facts = [f["text"] for f in record["facts"]]
            pred = shortcut_predictions(state, facts, record["question"])
            symbolic = symbolic_raw(facts, record["question"])
            pred["symbolic_raw"] = symbolic["answer"] if symbolic["status"] == "ok" else "<NO_ANSWER>"
            expected_predictions[record["record_id"]] = {
                "predictions": pred, "symbolic_status": symbolic["status"], "symbolic_candidates": symbolic["candidates"]}
        compare_json(expected_predictions, self.load(self.dev, "control_predictions.json", "development/control_predictions.json"),
                     "independent_TRAIN_control_predictions")
        expected_scores = {}
        for name, ids in (("optimization_train", train_ids), ("validation", validation_ids),
                          ("official_train_all", [r["record_id"] for r in self.raw_train])):
            expected_scores[name] = {
                control: ordinary_metrics([expected_predictions[rid]["predictions"][control] for rid in ids],
                                          [by_id[rid]["answer"] for rid in ids])
                for control in next(iter(expected_predictions.values()))["predictions"]
            }
        compare_json(expected_scores, self.load(self.dev, "control_scores.json", "development/control_scores.json"),
                     "independent_TRAIN_control_counts")
        summary = self.load(self.dev, "train_audit_summary.json", "development/train_audit_summary.json")
        require(summary["status"] == "completed" and summary["test_payload_extracted"] is False
                and summary["neural_training_or_scoring_performed"] is False, "TRAIN acquisition scope/completion differs")
        compare_json(expected_scores, summary["control_scores"], "TRAIN_summary_controls")
        require(summary["record_count"] == len(self.raw_train)
                and summary["optimization_train_n"] == len(train_ids) and summary["validation_n"] == len(validation_ids)
                and summary["unique_input_count"] == len({r["input_sha256"] for r in self.raw_train})
                and summary["episode_count"] == len({r["episode_id"] for r in self.raw_train})
                and summary["train_validation_input_intersection"] == 0
                and not summary["raw_input_gold_conflicts"] and not summary["encoded_input_gold_conflicts"],
                "TRAIN audit counts/conflict admission differs")
        self.report["development_controls"] = expected_scores
        self.done("independent_TRAIN_control_fit_and_counts", controls=8, records=len(self.raw_train))

    def final_recount(self, manifest, population):
        np = self.np
        metrics = self.load(self.root, "final_metrics.json", "final_metrics")
        require(metrics["schema_version"] == 1 and metrics["item"] == 10 and metrics["status"] == "completed"
                and metrics["ordered_run_ids"] == manifest["ordered_run_ids"] and len(metrics["runs"]) == 10,
                "final run inventory differs")
        compare_json(manifest["source"], metrics["source"], "final_metrics_source")
        compare_json(population, metrics["population"], "final_population_reference")
        self.correct, self.recounted = {}, {}
        duplicate_groups = defaultdict(list)
        for i, row in enumerate(self.final_rows):
            if row["can_encode"]:
                duplicate_groups[row["encoded_input_sha256"]].append(i)
        duplicate_checks = []
        for declared, final in zip(manifest["runs"], metrics["runs"]):
            require(declared["run_id"] == final["run_id"], "final run ID differs")
            compare_json(declared["checkpoint"], final["checkpoint"], "final_checkpoint")
            path = self.reference(self.root, final["predictions"])
            arrays, correct, check = recount_arrays(path, self.final_rows, self.encoder, np)
            self.correct[final["run_id"]] = correct
            recounted = {name: neural_metrics(arrays, self.final_rows, mask, self.encoder)
                         for name, mask in self.masks.items()}
            compare_json(recounted, final["metrics"], "final_metrics." + final["run_id"], atol=NLL_ATOL)
            sidecar = self.load(self.root, "final_predictions/" + final["run_id"] + ".json")
            compare_json(final, sidecar, "final_prediction_sidecar", atol=NLL_ATOL)
            self.recounted[final["run_id"]] = recounted
            max_duplicate_logit_difference, prediction_disagreements = 0.0, 0
            for group in duplicate_groups.values():
                if len(group) > 1:
                    first = group[0]
                    max_duplicate_logit_difference = max(max_duplicate_logit_difference, float(
                        np.max(np.abs(arrays["logits"][group].astype(np.float64) - arrays["logits"][first].astype(np.float64)))))
                    prediction_disagreements += sum(int(arrays["prediction_id"][i] != arrays["prediction_id"][first]) for i in group[1:])
            duplicate_checks.append({
                "run_id": final["run_id"], "additional_duplicate_rows": sum(len(g) - 1 for g in duplicate_groups.values()),
                "prediction_disagreements": prediction_disagreements,
                "max_abs_logit_difference": max_duplicate_logit_difference,
                "scope": "Observed duplicate consistency, no approximate equality used to modify predictions.",
            })
            self.done("final_saved_array_recount", run_id=final["run_id"], **check)
        saved_controls = self.load(self.root, "final_control_predictions.json", "final_control_predictions")
        expected_controls = []
        for record in self.final_raw:
            facts = [f["text"] for f in record["facts"]]
            predictions = shortcut_predictions(self.control_state, facts, record["question"])
            symbolic = symbolic_raw(facts, record["question"])
            predictions["symbolic_raw"] = symbolic["answer"]
            expected_controls.append({
                "record_id": record["record_id"], "predictions": predictions,
                "symbolic_status": symbolic["status"], "symbolic_candidates": symbolic["candidates"],
            })
        compare_json(expected_controls, saved_controls, "independent_final_raw_controls")
        controls = {}
        for subset, mask in self.masks.items():
            indices = [i for i, present in enumerate(mask) if present]
            controls[subset] = {
                name: ordinary_metrics([expected_controls[i]["predictions"][name] for i in indices],
                                       [self.final_rows[i]["gold"] for i in indices])
                for name in expected_controls[0]["predictions"]
            }
        compare_json(controls, metrics["controls"], "final_control_counts")
        self.report["final_metrics"] = self.recounted
        self.report["controls"] = controls
        self.report["duplicate_consistency"] = duplicate_checks
        self.report["control_scope"] = (
            "Raw-text controls can answer beyond neural grid/vocabulary limits. "
            "Their original labels are never replaced by symbolic answers; all official rows remain.")
        self.done("independent_final_control_predictions_and_counts", controls=8, subsets=len(self.masks))

    def summarize(self):
        from scipy.stats import t
        by_family = {family: {seed: self.recounted["train_" + family + "_s" + str(seed)] for seed in SEEDS}
                     for family in FAMILIES}
        self.report["primary_paired_seed_comparison"] = paired_seed_interval(
            {seed: by_family["neuropixel"][seed]["all_official"]["accuracy"] for seed in SEEDS},
            {seed: by_family["relative_transformer"][seed]["all_official"]["accuracy"] for seed in SEEDS}, t)
        screen = {}
        for family in FAMILIES:
            entries = []
            for seed in SEEDS:
                all_metric = by_family[family][seed]["all_official"]
                strict = by_family[family][seed]["strict_novel_vs_full_exposure"]
                entries.append({
                    "seed": seed, "all_official_accuracy": all_metric["accuracy"],
                    "all_official_n": all_metric["n"], "all_official_ge_095": all_metric["accuracy"] >= 0.95,
                    "strict_novel_accuracy": strict["accuracy"], "strict_novel_n": strict["n"],
                    "strict_novel_ge_095": strict["accuracy"] >= 0.95 if strict["n"] else None,
                })
            enough = all(row["strict_novel_n"] > 0 for row in entries)
            passed = all(row["all_official_ge_095"] and row["strict_novel_ge_095"] is True for row in entries) if enough else None
            screen[family] = {"threshold": 0.95, "rows": entries, "all_five_both_populations_pass": passed,
                              "status": "established_for_this_screen" if passed else "not_established",
                              "empty_strict_subset": not enough}
        self.report["competence_screen"] = {
            "preregistered_architecture": "neuropixel", "families": screen,
            "scope": "Descriptive operational prerequisite, not a causal mechanism test, significance test, or replacement for historical H1.",
        }
        self.report["costs"] = {
            "unit": "seconds unless explicitly named otherwise",
            "scope": "Saved timing fields; no exclusive CPU work, peak RSS, energy or continuous RAM measurement is inferred.",
            "runs": [{
                "run_id": run["run_id"], "parameter_count": run["parameter_count"],
                "completed_updates": run["completed_updates"], "training_seconds": run["training_seconds"],
                "mean_update_seconds": run["mean_update_seconds"], "median_update_seconds": run["median_update_seconds"],
                "elapsed_total_seconds": run["elapsed_total_seconds"],
                "unique_training_record_count": run["unique_training_record_count"],
                "unique_training_raw_input_count": run["unique_training_raw_input_count"],
                "unique_training_encoded_input_count": run["unique_training_encoded_input_count"],
                "train_probe": run["train_probe"]["metrics"], "validation": run["validation"]["metrics"],
            } for run in self.primary_runs],
        }
        self.done("five_paired_training_realizations_and_descriptive_screen", n=5, df=4)

    def bootstrap(self):
        np = self.np
        group_count, repetitions = len(self.groups), 2000
        output = {
            "seed": 104001, "replicates": repetitions, "generator": "PCG64",
            "group_count": group_count, "unit": "connected TEST source component",
            "scope": "Conditional on the fixed checkpoints and observed source components; not training-seed uncertainty, causal evidence, or a multiplicity-adjusted claim.",
            "ratio_definition": "Sum of resampled component correct counts divided by sum of resampled component selected-row counts.",
            "interval": "percentile 2.5/97.5, NumPy linear quantiles",
            "common_indices": True, "zero_denominator_policy": "Any undefined replicate makes that metric/subset CI null; other endpoints remain defined.",
            "per_run": {}, "paired_differences": {},
        }
        if group_count < 2:
            output.update(status="unavailable", reason="fewer than two connected TEST components")
            self.report["cluster_bootstrap"] = output
            return
        generator = np.random.Generator(np.random.PCG64(104001))
        indices = generator.integers(group_count, size=(repetitions, group_count), dtype=np.int64)
        index_path = self.output / "bootstrap_cluster_indices.npy"
        with index_path.open("xb") as stream:
            np.save(stream, indices, allow_pickle=False)
            stream.flush()
            __import__("os").fsync(stream.fileno())
        output["indices"] = {"path": index_path.name, "bytes": index_path.stat().st_size,
                             "sha256": self.file_sha(index_path), "shape": list(indices.shape),
                             "int64_little_endian_content_sha256": digest(indices.astype("<i8", copy=False).tobytes(order="C"))}
        output["group_order"] = [{"group_id": group["group_id"], "record_ids": group["record_ids"], "n": group["n"]}
                                 for group in self.groups]
        numerators, denominators, draws = {}, {}, {}
        for subset, mask in self.masks.items():
            mask = np.asarray(mask, dtype=np.bool_)
            denominators[subset] = np.asarray([int(mask[g["indices"]].sum()) for g in self.groups], dtype=np.int64)
            for rid, correct in self.correct.items():
                numerators[(rid, subset)] = np.asarray(
                    [int((correct[g["indices"]] & mask[g["indices"]]).sum()) for g in self.groups], dtype=np.int64)
                draws[(rid, subset)] = np.full(repetitions, np.nan, dtype=np.float64)
        # Bounded chunks avoid a repetitions x records x models allocation.
        for start in range(0, repetitions, 100):
            resource_admission()
            take = indices[start:start + 100]
            for subset in self.masks:
                den = denominators[subset][take].sum(axis=1)
                valid = den > 0
                for rid in self.correct:
                    num = numerators[(rid, subset)][take].sum(axis=1)
                    values = draws[(rid, subset)][start:start + len(take)]
                    values[valid] = num[valid] / den[valid]
        def interval(values, point):
            valid = np.isfinite(values)
            return {"point": point, "defined_replicates": int(valid.sum()),
                    "undefined_replicates": int((~valid).sum()),
                    "ci95": np.quantile(values, [0.025, 0.975], method="linear").tolist() if valid.all() else None}
        for rid in self.correct:
            output["per_run"][rid] = {
                subset: interval(draws[(rid, subset)], self.recounted[rid][subset]["accuracy"])
                for subset in self.masks}
        for seed in SEEDS:
            a, b = "train_neuropixel_s" + str(seed), "train_relative_transformer_s" + str(seed)
            output["paired_differences"][str(seed)] = {}
            for subset in self.masks:
                av, bv = self.recounted[a][subset]["accuracy"], self.recounted[b][subset]["accuracy"]
                point = av - bv if av is not None and bv is not None else None
                output["paired_differences"][str(seed)][subset] = interval(
                    draws[(a, subset)] - draws[(b, subset)], point)
        # Preserve all scalar draws as well as the common indices, including
        # undefined values with an explicit finite mask instead of invalid JSON.
        draw_path = self.output / "bootstrap_accuracy_draws.npz"
        draw_keys = {"r%02d_s%02d" % (i, j): draws[(rid, subset)]
                     for i, rid in enumerate(sorted(self.correct)) for j, subset in enumerate(self.masks)}
        with draw_path.open("xb") as stream:
            np.savez_compressed(stream, **draw_keys)
            stream.flush()
            __import__("os").fsync(stream.fileno())
        output["draws"] = {"path": draw_path.name, "bytes": draw_path.stat().st_size,
                           "sha256": self.file_sha(draw_path),
                           "key_mapping": {key: {"run_id": rid, "subset": subset}
                                for i, rid in enumerate(sorted(self.correct)) for j, subset in enumerate(self.masks)
                                for key in ["r%02d_s%02d" % (i, j)]},
                           "undefined_representation": "IEEE NaN in numeric NPZ only; JSON intervals are null"}
        output["status"] = "completed"
        self.report["cluster_bootstrap"] = output
        self.done("common_component_bootstrap", replicates=repetitions, groups=group_count)


    def finish(self):
        from datetime import datetime, timezone
        for label, path in self.input_paths.items():
            require(path.stat().st_size == self.inputs[label]["bytes"]
                    and self.file_sha(path) == self.inputs[label]["sha256"], "input changed while auditing: " + label)
        self.report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        self.report["checked_input_count"] = len(self.inputs)
        self.report["resource_samples"] = list(RESOURCE_SAMPLES)
        self.report["resource_scope"] = "Admission and periodic sampled MemAvailable only; not continuous RAM or peak RSS."
        self.report["status"] = "verified"
        self.report["limits"].append(
            "The source archive and original TEST have public custody; a workflow access gate is not an independent blind test.")
        return self.report


RESOURCE_SAMPLES = []


def resource_admission():
    from datetime import datetime, timezone
    from pathlib import Path
    memory = {}
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        fields = line.split()
        if len(fields) >= 2:
            memory[fields[0].rstrip(":")] = int(fields[1]) * 1024
    available = memory.get("MemAvailable")
    require(available is not None and available >= 8 * 1024**3, "MemAvailable below the 8 GiB audit floor")
    value = {"at_utc": datetime.now(timezone.utc).isoformat(), "available_bytes": available}
    RESOURCE_SAMPLES.append(value)
    return value


def write_exclusive_json(path, value):
    import os
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def main():
    import argparse
    from datetime import datetime, timezone
    import os
    from pathlib import Path
    import sys
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "study"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    require(args.phase != "study" or args.preflight is not None, "study audit needs an explicit pinned preflight root")
    args.output_dir = args.output_dir.resolve()
    for root in (args.input, args.development, args.preflight, args.source_root):
        if root is not None:
            require(not args.output_dir.is_relative_to(root.resolve()), "audit output must be outside all read-only input/source roots")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    audit = None
    try:
        resource_admission()
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ[name] = "1"
        import numpy as np
        import scipy
        require(sys.byteorder == "little", "declared saved stream hashes require little-endian runtime")
        audit = Audit(args, np, scipy)
        audit.source()
        audit.development()
        audit.preflight(args.input.resolve() if args.phase == "preflight" else args.preflight.resolve())
        if args.phase == "study":
            manifest, access = audit.gate_and_primary()
            population = audit.final_population(access)
            audit.final_recount(manifest, population)
            audit.summarize()
            audit.bootstrap()
        report = audit.finish()
        write_exclusive_json(args.output_dir / "scientific_audit.json", report)
        print(json.dumps({"status": "verified", "checks": len(audit.steps), "issues": 0,
                          "report": str(args.output_dir / "scientific_audit.json")}), flush=True)
        return 0
    except Exception as error:
        report = audit.report if audit is not None else {
            "schema_version": 1, "item": 10, "phase": args.phase, "issues": [],
            "scope": "Failure before scientific recount initialization; no model was executed.",
        }
        report["status"] = "failed"
        report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        report["issues"].append({"type": type(error).__name__, "message": str(error)})
        report["resource_samples"] = list(RESOURCE_SAMPLES)
        write_exclusive_json(args.output_dir / "scientific_audit.json", report)
        print(json.dumps({"status": "failed", "error": str(error)}), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
