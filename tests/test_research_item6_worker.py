"""Verify the archival boundary with a temporary local bare Git repository.

This exercises real commits and pushes only in pytest-owned directories, never a
network remote. It checks the concrete risk that preserving results could alter
the scientific source HEAD or accidentally include unrelated workspace files.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("item6_worker_test", ROOT / "scripts/research_item6_worker.py")
WORKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORKER)


def test_owned_archive_preserves_source_head_and_excludes_unrelated_files(tmp_path, monkeypatch):
    remote, source = tmp_path / "remote.git", tmp_path / "source"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    subprocess.run(["git", "init", str(source)], check=True, capture_output=True)
    (source / "README.md").write_text("Synthetic source tree\n", encoding="utf-8")

    bounded_git = WORKER.git

    def local_git(*arguments, cwd=None, deadline=None):
        return bounded_git(*arguments, cwd=source if cwd is None else cwd, deadline=deadline)

    local_git("add", "README.md")
    local_git("-c", "user.name=Test", "-c", "user.email=test@example.invalid",
              "commit", "-m", "Synthetic source")
    local_git("remote", "add", "origin", remote)
    head = local_git("rev-parse", "HEAD")
    local_git("push", "origin", head + ":refs/heads/test-source")
    monkeypatch.setattr(WORKER, "git", local_git)
    monkeypatch.setattr(WORKER, "ROOT", source)
    output = source / "runs/item6/synthetic-attempt"
    output.mkdir(parents=True)
    (source / "unrelated_work.txt").write_text("Must stay outside the owned archive\n", encoding="utf-8")
    (output / "unrelated_link.txt").symlink_to(source / "unrelated_work.txt")
    (output / ".unfinished.tmp").write_text("incomplete write", encoding="utf-8")
    WORKER.save_json(output / "worker_status.json", {"status": "running"})
    archive = WORKER.Archive(output, "synthetic-attempt", head)
    first = archive.publish()
    assert first and local_git("rev-parse", "HEAD") == head
    prefix = "results/research/06_cloud_runs/synthetic-attempt/"
    published = local_git("--git-dir", remote, "show",
                          f"refs/heads/{WORKER.RESULTS_BRANCH}:{prefix}worker_status.json")
    assert json.loads(published) == {"status": "running"}
    changed = local_git("--git-dir", remote, "diff", "--name-only", head, first).splitlines()
    assert changed and all(name.startswith(prefix) for name in changed)
    manifest = json.loads(local_git("--git-dir", remote, "show",
                                   f"{first}:{prefix}archive_manifest.json"))
    assert set(manifest["files"]) == {"worker_status.json"}
    assert manifest["snapshot_kind"] == "interim"
    assert manifest["source_commit"] == head

    WORKER.save_json(output / "worker_status.json", {"status": "completed"})
    second = archive.publish(final=True)
    assert second != first and local_git("rev-parse", "HEAD") == head
    assert local_git("--git-dir", remote, "rev-parse", second + "^") == first
    manifest = json.loads(local_git("--git-dir", remote, "show",
                                   f"{second}:{prefix}archive_manifest.json"))
    assert manifest["snapshot_kind"] == "final"
    assert manifest["files"]["worker_status.json"]["sha256"] == WORKER.file_sha256(output / "worker_status.json")
    assert (source / "unrelated_work.txt").read_text(encoding="utf-8") == "Must stay outside the owned archive\n"
