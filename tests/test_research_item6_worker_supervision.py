"""Short process-only checks for the item-6 worker's operational safeguards.

These tests use Python children and synthetic Git helpers. They do not import a
numerical runtime, launch a study, contact a remote, or change a repository.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import pytest

pytestmark = pytest.mark.skipif(os.name != "posix", reason="the worker targets Ubuntu process groups")


@pytest.fixture
def worker():
    path = Path(__file__).resolve().parents[1] / "scripts/research_item6_worker.py"
    specification = importlib.util.spec_from_file_location("item6_supervision_test_worker", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def wait_for_file(path, seconds=4):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.01)
    raise AssertionError(f"the lightweight child did not create {path.name}")


def test_low_ram_interrupts_owned_child_while_publication_is_blocked(worker, monkeypatch, tmp_path):
    ready, interrupted = tmp_path / "ready", tmp_path / "interrupted"
    child_code = (
        "from pathlib import Path\n"
        "import signal, sys, time\n"
        "def interrupted(signum, frame):\n"
        "    Path(sys.argv[2]).write_text('SIGINT', encoding='ascii')\n"
        "    raise SystemExit(0)\n"
        "signal.signal(signal.SIGINT, interrupted)\n"
        "Path(sys.argv[1]).write_text('ready', encoding='ascii')\n"
        "while True: time.sleep(0.1)\n"
    )
    low_ram = threading.Event()

    def resources():
        if low_ram.is_set():
            raise RuntimeError("synthetic available RAM is below the unchanged 8 GiB floor")
        return {"available_ram_gib": 16.0}

    monkeypatch.setattr(worker, "admission", resources)
    process = subprocess.Popen(
        [sys.executable, "-c", child_code, str(ready), str(interrupted)],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    supervisor = worker.ChildSupervisor(process, time.monotonic() + 10, interval=0.02)
    started = False
    try:
        wait_for_file(ready)
        supervisor.start()
        started = True
        low_ram.set()
        # Simulate synchronous publication holding the main thread. It never
        # polls/signals the child; only the independent supervisor can stop it.
        publication_release = threading.Event()
        assert not publication_release.wait(timeout=1.5)
        assert interrupted.read_text(encoding="ascii") == "SIGINT"
        assert supervisor.failed.is_set()
        assert supervisor.poll() == 0
        with pytest.raises(RuntimeError, match="independent child supervision"):
            supervisor.raise_if_failed()
    finally:
        supervisor.terminate()
        if started:
            supervisor.close()


def test_expired_deadline_refuses_the_next_child_before_popen(worker, monkeypatch, tmp_path):
    launches = []

    def forbidden_launch(*args, **kwargs):
        launches.append((args, kwargs))
        raise AssertionError("a later stage must not start after the deadline")

    monkeypatch.setattr(worker.subprocess, "Popen", forbidden_launch)
    monkeypatch.setattr(worker, "admission", lambda: {"available_ram_gib": 16.0})
    with (tmp_path / "log").open("w", encoding="ascii") as writer:
        with pytest.raises(TimeoutError, match="wall-time limit"):
            worker.launch_child([sys.executable, "-c", "pass"], os.environ.copy(),
                                writer, time.monotonic() - 0.01)
    assert launches == []


def test_git_timeout_stops_transport_descendant_without_waiting_for_its_pipe(worker, monkeypatch, tmp_path):
    import psutil

    helper_pid = tmp_path / "helper.pid"
    helper_code = f"import time; marker = {str(helper_pid)!r}; time.sleep(60)"
    fake_git = tmp_path / "git"
    fake_git.write_text(
        f"#!{sys.executable}\n"
        "from pathlib import Path\n"
        "import subprocess, sys, time\n"
        f"helper = subprocess.Popen([sys.executable, '-c', {helper_code!r}])\n"
        f"Path({str(helper_pid)!r}).write_text(str(helper.pid), encoding='ascii')\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    fake_git.chmod(0o700)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))
    started = time.monotonic()
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            worker.git("synthetic-blocked-transport", cwd=tmp_path,
                       deadline=time.monotonic() + 2)
        assert time.monotonic() - started < 8
        wait_for_file(helper_pid)
        pid = int(helper_pid.read_text(encoding="ascii"))
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            try:
                if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                    break
            except psutil.NoSuchProcess:
                break
            time.sleep(0.01)
        else:
            raise AssertionError("the Git transport descendant survived the operation timeout")
    finally:
        # Defensive cleanup only for the exact synthetic helper command, if an
        # implementation regression left it alive. Never signal an unrelated PID.
        if helper_pid.exists():
            try:
                helper = psutil.Process(int(helper_pid.read_text(encoding="ascii")))
                if helper.status() != psutil.STATUS_ZOMBIE and helper_code in helper.cmdline():
                    helper.kill()
                    try:
                        helper.wait(timeout=2)
                    except psutil.TimeoutExpired:
                        pass
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                pass
