from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class BenchmarkWeights:
    keyword_quality: float = 0.35
    semantic_quality: float = 0.25
    ranking_quality: float = 0.20
    adherence: float = 0.20
    quality: float = 0.70
    operational: float = 0.30
    warm_throughput: float = 0.40
    cold_load: float = 0.30
    memory_headroom: float = 0.30

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(slots=True)
class BenchmarkConfig:
    base_url: str = "http://127.0.0.1:11434"
    output_dir: str = "benchmark-results"
    model_timeout_seconds: float = 600.0
    test_timeout_seconds: float = 300.0
    startup_timeout_seconds: float = 30.0
    min_available_ram_gb: float = 8.0
    context_length: int = 4096
    seed: int = 42
    max_output_tokens: int = 384
    sample_interval_seconds: float = 0.5
    stop_started_ollama: bool = False
    gpu_telemetry: bool = True
    model_patterns: list[str] = field(default_factory=list)
    suite_path: str | None = None
    weights: BenchmarkWeights = field(default_factory=BenchmarkWeights)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)
