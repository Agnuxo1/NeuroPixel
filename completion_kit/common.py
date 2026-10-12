"""Atomic evidence, deterministic identities and conservative resource admission."""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    """Hash file bytes without loading large archives into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    """Encode a finite JSON value with a deterministic representation."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def identity(value: Any) -> str:
    """Return the SHA-256 identity of a structured value."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def write_json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    """Write finite JSON atomically; exclusive mode refuses an existing artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_bytes(value) + b"\n"
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            # Hard linking is atomic and refuses an existing destination.
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: Path) -> Any:
    """Read JSON and reject nonstandard NaN/Infinity literals."""
    def reject(value: str) -> None:
        raise ValueError("Nonfinite JSON literal: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8-sig"),
                      parse_constant=reject)


def require_resources(min_ram_gib: float = 8.0) -> float:
    """Check actual free host RAM; this does not reserve a shared GPU."""
    if os.name == "nt":
        import ctypes
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in (
                    "total_phys", "available_phys", "total_page", "available_page",
                    "total_virtual", "available_virtual", "available_extended")]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("Cannot obtain actual Windows memory availability")
        available = status.available_phys / 2**30
    elif Path("/proc/meminfo").is_file():
        lines = Path("/proc/meminfo").read_text().splitlines()
        values = {line.split(":", 1)[0]: int(line.split()[1]) for line in lines}
        available = values["MemAvailable"] / 2**20
        # A container's remaining memory budget may be smaller than host RAM.
        for limit_name, current_name in (
            ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
            ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes")):
            limit_path, current_path = Path(limit_name), Path(current_name)
            if limit_path.is_file() and current_path.is_file():
                limit = limit_path.read_text().strip()
                if limit.isdigit():
                    remaining = (int(limit) - int(current_path.read_text().strip())) / 2**30
                    available = min(available, remaining)
    else:
        import psutil
        available = psutil.virtual_memory().available / 2**30
    if not math.isfinite(available) or available < min_ram_gib:
        raise RuntimeError(f"RAM admission refused: {available:.3f} GiB")
    return available


def verify_files(root: Path, expected: dict[str, str]) -> None:
    """Verify registered relative paths while refusing paths outside the root."""
    root = Path(root).resolve()
    for relative, digest in expected.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Invalid registered file: " + relative)
        if sha256_file(path) != digest:
            raise ValueError("Registered bytes changed: " + relative)
