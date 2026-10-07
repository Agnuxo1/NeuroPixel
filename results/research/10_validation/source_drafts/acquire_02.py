"""Acquire and inspect only the selected official TRAIN member for item 10.

Remote bytes are data, never imported or executed. All compressed archives and
failed download attempts are retained. TAR headers may be traversed, but no TEST
payload is extracted, decoded, tokenized, passed to a model or scored here.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neuropixel.research import babi_qa as qa
import research_item9_worker as ops


def require(condition, message):
    if not condition:
        raise ValueError(message)


def download(entry, root, byte_cap, seconds):
    started = time.monotonic()
    destination = root / (entry["id"] + ".tar.gz")
    receipt = {"source": entry, "started_at_utc": ops.now(), "status": "running",
               "maximum_bytes": byte_cap, "maximum_seconds": seconds}
    ops.save(root / (entry["id"] + "_download.json"), receipt)
    try:
        request = urllib.request.Request(entry["url"], headers={"User-Agent": "NeuroPixel-scientific-validation/10",
                                                               "Accept-Encoding": "identity"})
        with urllib.request.urlopen(request, timeout=30) as response, destination.open("xb") as out:
            receipt.update(final_url=response.geturl(), http_status=response.status,
                           content_type=response.headers.get("Content-Type"),
                           content_length_header=response.headers.get("Content-Length"))
            size = 0
            while True:
                if time.monotonic() - started >= seconds:
                    raise TimeoutError("declared download deadline reached")
                block = response.read(min(1024 * 1024, byte_cap - size + 1))
                if not block:
                    break
                size += len(block)
                require(size <= byte_cap, "archive exceeds declared size limit")
                out.write(block)
        require(destination.stat().st_size > 0, "empty archive")
        with destination.open("rb") as stream:
            require(stream.read(2) == b"\x1f\x8b", "archive is not gzip data")
        receipt["status"] = "downloaded"
    except Exception as error:
        receipt.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
    finally:
        receipt.update(completed_at_utc=ops.now(), wall_seconds=time.monotonic() - started)
        if destination.exists():
            receipt.update(bytes=destination.stat().st_size, sha256=ops.sha(destination),
                           archive_filename=destination.name)
        ops.save(root / (entry["id"] + "_download.json"), receipt)
    return receipt


def train_member(archive, member_name, output, identifier, member_cap):
    headers = []
    total_uncompressed_bytes = 0
    payload = None
    with tarfile.open(archive, mode="r:gz") as tar:
        for member in tar:
            require(len(headers) < 5000, "too many archive members")
            total_uncompressed_bytes += member.size
            require(0 <= member.size and total_uncompressed_bytes <= 1024**3, "uncompressed TAR limit exceeded")
            headers.append({"name": member.name, "size": member.size, "regular_file": member.isfile()})
            if member.name != member_name:
                continue
            require(payload is None, "duplicate selected TRAIN member")
            require(member.isfile() and 0 < member.size <= member_cap, "invalid selected TRAIN member")
            stream = tar.extractfile(member)
            require(stream is not None, "TRAIN member unreadable")
            with stream:
                payload = stream.read(member_cap + 1)
            require(len(payload) == member.size and len(payload) <= member_cap, "TRAIN length differs")
    require(payload is not None, "selected original TRAIN member not present")
    # No tar.extract/extractall; only these explicitly chosen regular-file bytes are saved.
    target = output / (identifier + "_qa4_train.txt")
    target.write_bytes(payload)
    receipt = {"member": member_name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
               "train_filename": target.name, "archive_sha256": ops.sha(archive),
               "archive_members": headers, "test_payload_extracted": False,
               "scope": "Other TAR header metadata traversed; no other member payload returned to the parser."}
    ops.save(output / (identifier + "_member_receipt.json"), receipt, exclusive=True)
    return payload, receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["acquire"], required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_bytes())
    spec = plan["acquisition"]
    output = args.output / "data"
    output.mkdir(exist_ok=False)
    obtained, attempts = [], []
    for entry in spec["sources"]:
        download_receipt = download(entry, output, spec["archive_byte_cap"], spec["download_seconds_each"])
        row = {"download": download_receipt}
        if download_receipt["status"] == "downloaded":
            try:
                payload, identity = train_member(output / download_receipt["archive_filename"],
                    spec["train_member"], output, entry["id"], spec["train_member_byte_cap"])
                row["member"] = identity
                obtained.append((entry, payload, identity))
            except Exception as error:
                row["member_error"] = {"type": type(error).__name__, "message": str(error)}
        attempts.append(row)
        ops.save(output / "acquisition_receipt.json", {
            "schema_version": 1, "item": 10, "at_utc": ops.now(), "attempts": attempts,
            "test_payload_extracted": False, "scientific_model_scoring": False})
    require(obtained, "no admitted source yielded the selected original TRAIN member")
    selected, payload, identity = obtained[0]
    comparisons = [{"a": a[0]["id"], "b": b[0]["id"],
        "train_bytes_identical": a[1] == b[1],
        "archive_bytes_hash_equal": a[2]["archive_sha256"] == b[2]["archive_sha256"]}
        for i, a in enumerate(obtained) for b in obtained[i + 1:]]
    require(all(x["train_bytes_identical"] for x in comparisons),
            "available primary/mirror TRAIN payloads disagree; source choice must be reviewed")
    (output / "selected_qa4_train.txt").write_bytes(payload)
    selected_archive = output / (selected["id"] + ".tar.gz")
    # Refer to the saved source archive; do not duplicate a 15-MiB object needlessly.
    exposure = {"selected_source": selected, "train_member": identity,
        "archive_filename": selected_archive.name, "archive_sha256": ops.sha(selected_archive),
        "comparisons": comparisons, "upstream_independent_checksum_available": False,
        "test_payload_extracted": False, "test_payload_parsed": False, "model_training_started": False,
        "limits": "Public downloadable TEST shares the compressed archive; this is workflow gating, not independent blind custody."}
    ops.save(output / "selected_source.json", exposure, exclusive=True)
    parsed = qa.parse_babi(payload.decode("utf-8", errors="strict"), {
        "dataset": "bAbI original English 1k QA4", "split": "official_train",
        "member": spec["train_member"], "archive_sha256": exposure["archive_sha256"],
        "source_url": selected["url"], "member_sha256": identity["sha256"]})
    records = parsed["records"]
    ops.save(output / "official_train_records.json", parsed, exclusive=True)
    split = qa.build_development_split(records, salt=spec["development_salt"],
                                      dev_fraction=spec["development_fraction"])
    ops.save(output / "development_split.json", split, exclusive=True)
    by_id = {r["record_id"]: r for r in records}
    require(len(by_id) == len(records), "record IDs not unique")
    train = [by_id[i] for i in split["train_ids"]]
    valid = [by_id[i] for i in split["validation_ids"]]
    require(train and valid, "both optimization and validation partitions must be nonempty")
    encoder = qa.fit_encoder(train, layout_policy=spec["layout_policy"])
    ops.save(output / "encoder.json", encoder, exclusive=True)
    controls = qa.fit_shortcuts(train)
    ops.save(output / "shortcut_models.json", controls, exclusive=True)
    encoded, control_predictions = [], {}
    raw_labels = defaultdict(set)
    encoded_labels = defaultdict(set)
    oov_examples, overflows = [], []
    supported = []
    for r in records:
        facts = [f["text"] for f in r["facts"]]
        raw_labels[r["input_sha256"]].add(r["answer"])
        try:
            e = qa.encode_input(facts, r["question"], encoder)
        except ValueError as error:
            overflows.append({"record_id": r["record_id"], "message": str(error)})
            continue
        eh = qa.canonical_sha256(e["canvas"])
        encoded_labels[eh].add(r["answer"])
        if e["unknown_input_tokens"]:
            oov_examples.append({"record_id": r["record_id"], "tokens": e["unknown_input_tokens"]})
        support = r["answer"] in encoder["token_to_id"] and encoder["token_to_id"][r["answer"]] > 1
        supported.append(support)
        encoded.append({"record_id": r["record_id"], "input_sha256": r["input_sha256"],
            "encoded_input_sha256": eh, "canvas": e["canvas"], "gold": r["answer"],
            "gold_id": encoder["token_to_id"].get(r["answer"]) if support else None,
            "supported_gold": support, "partition": "train" if r["record_id"] in split["train_ids"] else "validation"})
        pred = qa.predict_shortcuts(controls, facts, r["question"])
        symbolic = qa.solve_raw(facts, r["question"])
        pred["symbolic_raw"] = symbolic["answer"] if symbolic["status"] == "ok" else "<NO_ANSWER>"
        control_predictions[r["record_id"]] = {"predictions": pred, "symbolic_status": symbolic["status"],
                                             "symbolic_candidates": symbolic["candidates"]}
    ops.save(output / "encoding_issues.json", {"input_oov": oov_examples, "overflows": overflows}, exclusive=True)
    require(not overflows, "the fixed TRAIN-derived geometry cannot encode all official TRAIN/DEV; do not resize after this receipt")
    ops.save(output / "encoded_official_train.json", encoded, exclusive=True)
    ops.save(output / "control_predictions.json", control_predictions, exclusive=True)
    control_scores = {}
    for label, rows in (("optimization_train", train), ("validation", valid), ("official_train_all", records)):
        golds = [r["answer"] for r in rows]
        names = sorted(control_predictions[rows[0]["record_id"]]["predictions"])
        control_scores[label] = {name: qa.score_answers(
            [control_predictions[r["record_id"]]["predictions"][name] for r in rows], golds)
            for name in names}
    ops.save(output / "control_scores.json", control_scores, exclusive=True)
    train_inputs = {r["input_sha256"] for r in train}
    valid_inputs = {r["input_sha256"] for r in valid}
    summary = {"schema_version": 1, "item": 10, "status": "completed", "at_utc": ops.now(),
        "selected_source_id": selected["id"], "source_comparisons": comparisons,
        "archive_sha256": exposure["archive_sha256"], "train_member_sha256": identity["sha256"],
        "record_count": len(records), "episode_count": len({r["episode_id"] for r in records}),
        "unique_input_count": len(raw_labels), "duplicate_input_rows": len(records) - len(raw_labels),
        "raw_input_gold_conflicts": {k: sorted(v) for k, v in raw_labels.items() if len(v) > 1},
        "encoded_input_gold_conflicts": {k: sorted(v) for k, v in encoded_labels.items() if len(v) > 1},
        "optimization_train_n": len(train), "validation_n": len(valid),
        "train_validation_input_intersection": len(train_inputs & valid_inputs),
        "vocabulary_size": len(encoder["vocabulary"]), "vocabulary": encoder["vocabulary"],
        "height": encoder["height"], "width": encoder["width"], "out_pos": encoder["out_pos"],
        "max_facts_official_train": max(len(r["facts"]) for r in records),
        "answer_counts": dict(sorted(Counter(r["answer"] for r in records).items())),
        "input_oov_examples": len(oov_examples), "unsupported_gold_count": len(supported) - sum(supported),
        "geometry_max_chebyshev_distance_to_readout":
             max(max(abs(i - encoder["out_pos"][0]), abs(j - encoder["out_pos"][1]))
                 for i in range(encoder["height"]) for j in range(encoder["width"])),
        "control_scores": control_scores, "test_payload_extracted": False,
        "neural_training_or_scoring_performed": False,
        "scope": "Original TRAIN-only provenance, representation, duplicate/shortcut audit. No final-performance result."}
    ops.save(output / "train_audit_summary.json", summary, exclusive=True)
    require(summary["train_validation_input_intersection"] == 0, "development overlap detected")
    print(json.dumps({k: summary[k] for k in ("status", "record_count", "optimization_train_n",
        "validation_n", "vocabulary_size", "height", "width", "input_oov_examples", "unsupported_gold_count")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
