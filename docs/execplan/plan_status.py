"""Deterministic ledger for a long ExecPlan: one task at a time, evidence before ticking.

Evidence hashes read CRLF as LF (key sha256_lf); they detect edits, not Git line-ending conversion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

STATUSES = {"pending", "in_progress", "done", "negative", "blocked_external", "blocked_resource"}
CLOSED = {"done", "negative"}
TASK_ID = re.compile(r"^[A-Za-z0-9._-]+$")
COMMIT = re.compile(r"^[0-9a-f]{7,40}$")
HEADING = re.compile(r"^(#{2,3}) (\d+[A-Z]?(?:\.\d+)?)\.? ")


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")


def sha256(path: Path) -> str:
    """SHA-256 with CRLF read as LF, so Git line-ending conversion cannot fake a change."""
    digest, pending = hashlib.sha256(), b""
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            data = pending + chunk
            # A CR at the chunk end may pair with an LF at the start of the next chunk.
            pending, data = (b"\r", data[:-1]) if data.endswith(b"\r") else (b"", data)
            digest.update(data.replace(b"\r\n", b"\n"))
    digest.update(pending)
    return digest.hexdigest()


def load(ledger: Path) -> dict:
    return json.loads(ledger.read_text(encoding="utf-8-sig"))


def save(ledger: Path, data: dict) -> None:
    """Keep one backup and replace the ledger atomically."""
    if ledger.exists():
        shutil.copy2(ledger, ledger.with_name(ledger.name + ".bak"))
    fd, temporary = tempfile.mkstemp(prefix=ledger.name + ".", dir=ledger.parent)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(temporary, ledger)


def workdir(data: dict, ledger: Path) -> Path:
    return (ledger.parent / data.get("workdir", ".")).resolve()


def index(data: dict) -> dict:
    return {task["id"]: task for task in data["tasks"]}


def problems(data: dict, ledger: Path) -> list[str]:
    """Every rule the next agent relies on to resume without guessing."""
    found = []
    if data.get("schema") != 1 or not isinstance(data.get("tasks"), list) or not data["tasks"]:
        return ["schema 1 with a non-empty tasks list required"]
    ids = [task.get("id") for task in data["tasks"]]
    if len(ids) != len(set(ids)) or not all(isinstance(i, str) and TASK_ID.match(i) for i in ids):
        found.append("task ids must be unique and match [A-Za-z0-9._-]+")
    tasks, root = index(data), workdir(data, ledger)
    for task in data["tasks"]:
        tid, status = task.get("id"), task.get("status")
        if status not in STATUSES:
            found.append(f"{tid}: unknown status {status}")
            continue
        for key in ("title", "section", "acceptance"):
            if not str(task.get(key) or "").strip():
                found.append(f"{tid}: {key} required")
        for dep in task.get("depends_on", []):
            if dep not in tasks:
                found.append(f"{tid}: unknown dependency {dep}")
            elif status in CLOSED | {"in_progress"} and tasks[dep]["status"] not in CLOSED:
                found.append(f"{tid}: {status} while dependency {dep} is {tasks[dep]['status']}")
        if status in CLOSED | {"blocked_external"}:
            evidence = task.get("evidence") or []
            if not evidence:
                found.append(f"{tid}: {status} needs evidence")
            for item in evidence:
                path = (root / item.get("path", "")).resolve()
                if not path.is_relative_to(root) or not path.is_file():
                    found.append(f"{tid}: missing evidence {item.get('path')}")
                elif sha256(path) != item.get("sha256_lf"):
                    found.append(f"{tid}: evidence changed after closing {item.get('path')}")
        if status in CLOSED and not COMMIT.match(str(task.get("commit") or "")):
            found.append(f"{tid}: closed task needs the commit of its work")
        if status in {"negative", "blocked_external", "blocked_resource"} and not str(task.get("note") or "").strip():
            found.append(f"{tid}: {status} needs a note (result, or exactly what is missing)")
    if sum(task.get("status") == "in_progress" for task in data["tasks"]) > 1:
        found.append("more than one task in_progress")
    seen, stack = set(), set()

    def visit(tid: str) -> None:
        if tid in stack:
            found.append("dependency cycle through " + tid)
            return
        if tid in seen or tid not in tasks:
            return
        stack.add(tid)
        for dep in tasks[tid].get("depends_on", []):
            visit(dep)
        stack.discard(tid)
        seen.add(tid)

    for tid in tasks:
        visit(tid)
    return found


def next_task(data: dict) -> dict | None:
    tasks = index(data)
    for task in data["tasks"]:
        if task["status"] == "pending" and all(tasks[d]["status"] in CLOSED for d in task.get("depends_on", [])):
            return task
    return None


def section_lines(guide: Path, section: str) -> tuple[int, int] | None:
    """Return 1-based inclusive line range of a guide section such as 18C or 6.2."""
    wanted = section.lstrip("§").strip()
    lines = guide.read_text(encoding="utf-8").split("\n")
    fence, start, level = False, None, None
    for number, line in enumerate(lines, 1):
        if line.startswith("```"):
            fence = not fence
            continue
        match = None if fence else HEADING.match(line)
        if start is None and match and match.group(2) == wanted:
            start, level = number, len(match.group(1))
        elif start is not None and not fence and line.startswith("#") and len(line) - len(line.lstrip("#")) <= level:
            return start, number - 1
    return (start, len(lines)) if start else None


def summary(data: dict, ledger: Path) -> dict:
    counts = {status: 0 for status in sorted(STATUSES)}
    for task in data["tasks"]:
        counts[task["status"]] = counts.get(task["status"], 0) + 1
    current = [t["id"] for t in data["tasks"] if t["status"] == "in_progress"]
    upcoming = next_task(data)
    guide = Path(data.get("guide", {}).get("path", ""))
    report = dict(counts=counts, in_progress=current, next=upcoming and upcoming["id"],
                  blocked=[f"{t['id']}: {t['status']}: {t.get('note', '')}" for t in data["tasks"]
                           if t["status"].startswith("blocked")],
                  problems=problems(data, ledger))
    if guide.is_file() and data["guide"].get("sha256_lf") and sha256(guide) != data["guide"]["sha256_lf"]:
        report["warning"] = "guide bytes differ from the registered sha256; reread the affected sections"
    focus = current[0] if current else upcoming and upcoming["id"]
    if focus and guide.is_file():
        report["focus_section_lines"] = section_lines(guide, index(data)[focus]["section"])
    open_tasks = counts["pending"] + counts["in_progress"]
    report["plan_complete"] = open_tasks == 0 and not report["problems"]
    return report


def render(data: dict, output: Path) -> None:
    """Human checklist; done items are ticked and struck through, never deleted."""
    lines = ["# Checklist (generated by plan_status.py render; do not edit by hand)", ""]
    phase = None
    for task in data["tasks"]:
        if task.get("phase") != phase:
            phase = task.get("phase")
            lines += ["", f"## {phase}", ""]
        label = f"{task['id']} {task['title']} (§{task['section'].lstrip('§')})"
        status = task["status"]
        if status == "done":
            lines.append(f"- [x] ~~{label}~~ {task.get('closed', '')} `{task.get('commit', '')[:10]}`")
        elif status == "negative":
            lines.append(f"- [x] ~~{label}~~ NEGATIVE: {task.get('note', '')} `{task.get('commit', '')[:10]}`")
        elif status == "in_progress":
            lines.append(f"- [ ] **IN PROGRESS** {label} (attempt {task.get('attempts', 1)})")
        elif status.startswith("blocked"):
            lines.append(f"- [ ] BLOCKED ({status[8:]}) {label}: {task.get('note', '')}")
        else:
            lines.append(f"- [ ] {label}")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def evidence_records(data: dict, ledger: Path, paths: list[str]) -> list[dict]:
    root, records = workdir(data, ledger), []
    for item in paths:
        path = (root / item).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise SystemExit("Evidence must be an existing file inside the workdir: " + item)
        records.append(dict(path=path.relative_to(root).as_posix(), sha256_lf=sha256(path)))
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, default=Path(__file__).with_name("tasks.json"))
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("status")
    commands.add_parser("next")
    start = commands.add_parser("start")
    start.add_argument("id")
    close = commands.add_parser("close")
    close.add_argument("id")
    close.add_argument("--status", choices=sorted(CLOSED), required=True)
    close.add_argument("--evidence", nargs="+", required=True)
    close.add_argument("--commit", required=True)
    close.add_argument("--note", default="")
    block = commands.add_parser("block")
    block.add_argument("id")
    block.add_argument("--kind", choices=["external", "resource"], required=True)
    block.add_argument("--note", required=True)
    block.add_argument("--evidence", nargs="*", default=[])
    reopen = commands.add_parser("reopen")
    reopen.add_argument("id")
    reopen.add_argument("--note", required=True)
    supersede = commands.add_parser("supersede", help="repoint closed evidence to a byte-identical archived snapshot")
    supersede.add_argument("id")
    supersede.add_argument("--path", required=True, help="evidence path recorded at closure")
    supersede.add_argument("--snapshot", required=True, help="archived copy whose sha256_lf equals the closure hash")
    supersede.add_argument("--note", required=True)
    pause = commands.add_parser("pause")
    pause.add_argument("id")
    pause.add_argument("--note", required=True)
    add = commands.add_parser("add")
    for name in ("id", "title", "phase", "section", "acceptance"):
        add.add_argument("--" + name, required=True)
    add.add_argument("--depends", nargs="*", default=[])
    add.add_argument("--before")
    add.add_argument("--blocks", help="existing pending task that must wait for the new one")
    commands.add_parser("render")
    section = commands.add_parser("section")
    section.add_argument("section")
    args = parser.parse_args()
    ledger = args.ledger.resolve()
    data = load(ledger)
    command = args.command or "status"
    tasks = index(data)
    if command in {"start", "close", "block", "reopen", "pause", "supersede"} and args.id not in tasks:
        raise SystemExit("Unknown task " + args.id)
    if command == "status":
        report = summary(data, ledger)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        sys.exit(1 if report["problems"] else 0)
    if command == "next":
        task = next_task(data)
        print(json.dumps(task, ensure_ascii=False, indent=2) if task else "No eligible pending task")
        return
    if command == "section":
        print(section_lines(Path(data["guide"]["path"]), args.section))
        return
    if command == "render":
        render(data, ledger.with_name("CHECKLIST.md"))
        return
    if command == "start":
        task = tasks[args.id]
        if any(t["status"] == "in_progress" for t in data["tasks"]):
            raise SystemExit("Another task is in progress; close or block it first")
        if task["status"] != "pending" or any(tasks[d]["status"] not in CLOSED for d in task.get("depends_on", [])):
            raise SystemExit("Task is not pending or its dependencies are not closed")
        task.update(status="in_progress", started=now(), attempts=task.get("attempts", 0) + 1)
    elif command == "close":
        task = tasks[args.id]
        if task["status"] != "in_progress" or not COMMIT.match(args.commit):
            raise SystemExit("Only the in-progress task can be closed, with a hex commit")
        if args.status == "negative" and not args.note.strip():
            raise SystemExit("A negative closure needs its result in --note")
        task.update(status=args.status, evidence=evidence_records(data, ledger, args.evidence),
                    commit=args.commit, closed=now(), note=args.note)
    elif command == "block":
        task = tasks[args.id]
        if task["status"] not in {"pending", "in_progress"}:
            raise SystemExit("Only pending or in-progress tasks can be blocked")
        if args.kind == "external" and not args.evidence:
            raise SystemExit("An external block needs the finished internal package as evidence")
        task.update(status="blocked_" + args.kind, note=args.note, blocked=now(),
                    evidence=evidence_records(data, ledger, args.evidence))
    elif command == "reopen":
        task = tasks[args.id]
        if not task["status"].startswith("blocked"):
            raise SystemExit("Only blocked tasks can be reopened")
        task.update(status="pending", note=(task.get("note", "") + " | reopened: " + args.note).strip(" |"))
    elif command == "supersede":
        # Closed evidence stays verifiable: the snapshot must reproduce the hash recorded at closure.
        task = tasks[args.id]
        if task["status"] not in CLOSED:
            raise SystemExit("Only closed tasks can have evidence superseded")
        item = next((e for e in task.get("evidence", []) if e.get("path") == args.path), None)
        if item is None:
            raise SystemExit("Path is not recorded evidence of this task: " + args.path)
        root = workdir(data, ledger)
        snapshot = (root / args.snapshot).resolve()
        if not snapshot.is_relative_to(root) or not snapshot.is_file():
            raise SystemExit("Snapshot must be an existing file inside the workdir")
        if sha256(snapshot) != item.get("sha256_lf"):
            raise SystemExit("Snapshot bytes differ from the closure hash; refusing")
        relative = snapshot.relative_to(root).as_posix()
        if any(e.get("path") == relative for e in task["evidence"]):
            raise SystemExit("Snapshot already recorded as evidence")
        item["path"] = relative
        task.setdefault("superseded", []).append(dict(original_path=args.path, snapshot=relative,
                                                      sha256_lf=item["sha256_lf"], at=now(), note=args.note))
    elif command == "pause":
        task = tasks[args.id]
        if task["status"] != "in_progress":
            raise SystemExit("Only the in-progress task can be paused")
        task.update(status="pending", note=(task.get("note", "") + " | paused: " + args.note).strip(" |"))
    elif command == "add":
        if args.id in tasks or not TASK_ID.match(args.id):
            raise SystemExit("New unique id required")
        if args.blocks and (args.blocks not in tasks or tasks[args.blocks]["status"] != "pending"):
            raise SystemExit("--blocks needs an existing pending task; pause it first if it is in progress")
        new = dict(id=args.id, title=args.title, phase=args.phase, section=args.section,
                   acceptance=args.acceptance, depends_on=args.depends, status="pending")
        position = next((i for i, t in enumerate(data["tasks"]) if t["id"] == args.before), len(data["tasks"]))
        data["tasks"].insert(position, new)
        if args.blocks:
            tasks[args.blocks].setdefault("depends_on", []).append(args.id)
    found = problems(data, ledger)
    if found:
        raise SystemExit("Refused; ledger would be invalid: " + "; ".join(found))
    data["updated"] = now()
    save(ledger, data)
    render(data, ledger.with_name("CHECKLIST.md"))
    print(json.dumps(summary(data, ledger), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
