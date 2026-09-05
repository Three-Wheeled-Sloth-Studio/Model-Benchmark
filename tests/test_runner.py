from model_benchmark.config import BenchmarkConfig
from model_benchmark.fixtures import BenchmarkSuite
from model_benchmark.ollama import ModelCandidate
from model_benchmark.resources import ResourceSnapshot
from model_benchmark.runner import BenchmarkRunner


class FakeClient:
    def unload(self, model: str) -> None:
        pass

    def stop_model(self, model: str) -> None:
        pass


class FakeResources:
    def sample(self) -> ResourceSnapshot:
        return ResourceSnapshot(
            available_ram_gb=0.745,
            total_ram_gb=15.843,
            ram_percent=95.3,
            cpu_percent=36.8,
            ollama_rss_gb=0.044,
            gpus=[],
        )


def test_host_preflight_block_never_receives_a_score(tmp_path) -> None:
    suite = BenchmarkSuite(
        suite_id="baseline",
        suite_version="1",
        description="test",
        cases=(),
    )
    runner = BenchmarkRunner(
        BenchmarkConfig(output_dir=str(tmp_path), min_available_ram_gb=8.0),
        suite,
        console=lambda _message: None,
    )
    runner.client = FakeClient()
    runner.resources = FakeResources()
    candidate = ModelCandidate(
        name="smollm2:360m",
        size_bytes=726_000_000,
        digest="test",
        family="llama",
        families=("llama",),
        parameter_size="360M",
        quantization_level="Q4_K_M",
        raw={},
    )

    result = runner.run_model(candidate)

    assert result["status"] == "host_resource_blocked"
    assert result["block_stage"] == "host_preflight"
    assert result["tests"] == []
    assert "cold_probe" not in result
    assert result["score"]["score_status"] == "not_evaluated"
    assert result["score"]["quality_score"] is None
    assert result["score"]["operational_score"] is None
    assert result["score"]["composite_score"] is None
    assert result["score"]["operational_components"]["cold_load_score"] is None
    assert result["resource_stats"]["min_available_ram_gb"] == 0.745
