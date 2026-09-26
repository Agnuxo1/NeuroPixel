"""Guardas de recursos: NeuroPixel convive con otros entrenamientos (p. ej. Kaggle).

Regla: CPU por defecto, pocos hilos y prioridad baja. La GPU solo se usa si se pide
explícitamente y está libre de verdad; y aun así con un tope de memoria.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass

import torch


@dataclass
class GpuStatus:
    util_pct: float
    used_mib: float
    total_mib: float

    @property
    def free_gib(self) -> float:
        return (self.total_mib - self.used_mib) / 1024


def gpu_status() -> GpuStatus | None:
    """Lee la GPU 0 con nvidia-smi (ve también los procesos de otros programas)."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True).stdout
        util, used, total = (float(v) for v in out.splitlines()[0].split(","))
        return GpuStatus(util, used, total)
    except (subprocess.SubprocessError, ValueError, IndexError):
        return None


def lower_priority() -> None:
    """Prioridad por debajo de lo normal para no robar CPU a otros procesos."""
    try:
        import psutil
        p = psutil.Process(os.getpid())
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    except Exception:
        pass


def check_ram(min_free_gb: float = 2.0) -> None:
    try:
        import psutil
    except ImportError:
        return
    free = psutil.virtual_memory().available / 1e9
    if free < min_free_gb:
        raise SystemExit(f"RAM libre {free:.1f} GB < {min_free_gb} GB: no arranco para no colapsar el PC")


def choose_device(requested: str = "cpu", *, threads: int = 4, min_free_gib: float = 6.0,
                  max_util_pct: float = 50.0, vram_cap_gib: float = 3.0,
                  force_gpu: bool = False) -> torch.device:
    """Devuelve el dispositivo seguro. 'cpu' por defecto; 'cuda' solo si la GPU está libre."""
    lower_priority()
    check_ram()
    torch.set_num_threads(max(1, threads))
    if requested != "cuda":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        print("[safety] CUDA no disponible -> CPU")
        return torch.device("cpu")
    st = gpu_status()
    if st is not None and not force_gpu and (st.util_pct > max_util_pct or st.free_gib < min_free_gib):
        print(f"[safety] GPU ocupada (uso {st.util_pct:.0f} %, libre {st.free_gib:.1f} GiB) -> CPU")
        return torch.device("cpu")
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    torch.cuda.set_per_process_memory_fraction(min(1.0, vram_cap_gib / total), 0)
    print(f"[safety] GPU permitida con tope de {vram_cap_gib} GiB")
    return torch.device("cuda")
