from __future__ import annotations

import csv
import io
import os
import platform
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import psutil

_GB = 1024**3


@dataclass(slots=True)
class GpuSnapshot:
    index: int
    name: str
    memory_used_mb: float
    memory_total_mb: float
    utilization_percent: float | None
    driver_version: str | None = None


@dataclass(slots=True)
class ResourceSnapshot:
    available_ram_gb: float
    total_ram_gb: float
    ram_percent: float
    cpu_percent: float
    ollama_rss_gb: float
    gpus: list[GpuSnapshot] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ResourceStats:
    sample_count: int = 0
    min_available_ram_gb: float | None = None
    peak_ram_percent: float = 0.0
    peak_cpu_percent: float = 0.0
    peak_ollama_rss_gb: float = 0.0
    peak_vram_used_mb: float = 0.0
    peak_gpu_utilization_percent: float = 0.0

    def observe(self, snapshot: ResourceSnapshot) -> None:
        self.sample_count += 1
        if self.min_available_ram_gb is None:
            self.min_available_ram_gb = snapshot.available_ram_gb
        else:
            self.min_available_ram_gb = min(
                self.min_available_ram_gb, snapshot.available_ram_gb
            )
        self.peak_ram_percent = max(self.peak_ram_percent, snapshot.ram_percent)
        self.peak_cpu_percent = max(self.peak_cpu_percent, snapshot.cpu_percent)
        self.peak_ollama_rss_gb = max(self.peak_ollama_rss_gb, snapshot.ollama_rss_gb)
        for gpu in snapshot.gpus:
            self.peak_vram_used_mb = max(self.peak_vram_used_mb, gpu.memory_used_mb)
            if gpu.utilization_percent is not None:
                self.peak_gpu_utilization_percent = max(
                    self.peak_gpu_utilization_percent, gpu.utilization_percent
                )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ResourceCollector:
    def __init__(self, gpu_telemetry: bool = True) -> None:
        self.nvidia_smi = shutil.which("nvidia-smi") if gpu_telemetry else None
        self._last_gpu_sample_at = 0.0
        self._last_gpus: list[GpuSnapshot] = []
        psutil.cpu_percent(interval=None)

    def sample(self) -> ResourceSnapshot:
        memory = psutil.virtual_memory()
        now = time.monotonic()
        if now - self._last_gpu_sample_at >= 1.0:
            self._last_gpus = self._gpu_snapshot()
            self._last_gpu_sample_at = now
        return ResourceSnapshot(
            available_ram_gb=memory.available / _GB,
            total_ram_gb=memory.total / _GB,
            ram_percent=float(memory.percent),
            cpu_percent=float(psutil.cpu_percent(interval=None)),
            ollama_rss_gb=self._ollama_rss() / _GB,
            gpus=list(self._last_gpus),
        )

    def hardware_snapshot(self) -> dict[str, Any]:
        memory = psutil.virtual_memory()
        return {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "total_ram_gb": round(memory.total / _GB, 3),
            "gpus": [asdict(item) for item in self._gpu_snapshot(include_driver=True)],
            "nvidia_smi_available": bool(self.nvidia_smi),
        }

    @staticmethod
    def _ollama_rss() -> int:
        total = 0
        for process in psutil.process_iter(["name", "memory_info"]):
            try:
                name = (process.info.get("name") or "").casefold()
                if "ollama" in name:
                    info = process.info.get("memory_info")
                    total += int(info.rss) if info else 0
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return total

    def _gpu_snapshot(self, include_driver: bool = False) -> list[GpuSnapshot]:
        if not self.nvidia_smi:
            return []
        fields = ["index", "name", "memory.used", "memory.total", "utilization.gpu"]
        if include_driver:
            fields.append("driver_version")
        try:
            proc = subprocess.run(
                [
                    self.nvidia_smi,
                    f"--query-gpu={','.join(fields)}",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
                creationflags=(
                    getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
                ),
            )
        except (OSError, subprocess.TimeoutExpired):
            return []
        if proc.returncode != 0:
            return []
        rows = csv.reader(io.StringIO(proc.stdout))
        result: list[GpuSnapshot] = []
        for row in rows:
            values = [item.strip() for item in row]
            if len(values) < 5:
                continue
            try:
                utilization = (
                    None if values[4] in {"N/A", "[N/A]", ""} else float(values[4])
                )
                result.append(
                    GpuSnapshot(
                        index=int(values[0]),
                        name=values[1],
                        memory_used_mb=float(values[2]),
                        memory_total_mb=float(values[3]),
                        utilization_percent=utilization,
                        driver_version=(
                            values[5] if include_driver and len(values) > 5 else None
                        ),
                    )
                )
            except ValueError:
                continue
        return result
