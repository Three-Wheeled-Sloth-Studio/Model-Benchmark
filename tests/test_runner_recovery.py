import time

from model_benchmark.config import BenchmarkConfig
from model_benchmark.fixtures import BenchmarkSuite
from model_benchmark.ollama import ModelCandidate
from model_benchmark.resources import ResourceSnapshot
from model_benchmark.runner import BenchmarkRunner


class HealthyResources:
    def sample(self) -> ResourceSnapshot:
        return ResourceSnapshot(
            available_ram_gb=40.0,
            total_ram_gb=96.0,
            ram_percent=58.0,
            cpu_percent=10.0,
            ollama_rss_gb=1.0,
            gpus=[],
        )


class TimedOutClient:
    def unload(self, _model: str) -> bool:
        return True

    def generate(self, *_args, **_kwargs):
        time.sleep(0.05)
        return {"response": "OK"}

    def stop_model(self, _model: str) -> None:
        return None

    def recover_after_abort(self, _model: str, timeout: float = 30.0) -> bool:
        assert timeout > 0
        return False

    def api_available(self, timeout: float = 2.0) -> bool:
        return True


def test_cold_timeout_is_unscored_and_failed_recovery_halts_sweep(tmp_path) -> None:
    config = BenchmarkConfig(
        output_dir=str(tmp_path),
        test_timeout_seconds=0.01,
        model_timeout_seconds=1.0,
        sample_interval_seconds=0.001,
        gpu_telemetry=False,
    )
    suite = BenchmarkSuite("baseline", "1", "test", ())
    runner = BenchmarkRunner(config, suite, console=lambda _message: None)
    runner.client = TimedOutClient()
    runner.resources = HealthyResources()
    model = ModelCandidate("model-a", 1, "x", "llama", ("llama",), "8B", "Q4", {})

    result = runner.run_model(model)

    assert result["status"] == "timeout"
    assert result["failure_stage"] == "cold_load"
    assert result["recovery_status"] == "failed"
    assert result["score"]["quality_score"] is None
    assert result["score"]["operational_score"] is None
    assert result["score"]["operational_components"]["cold_load_score"] is None
    assert result["score"]["composite_score"] is None
    assert runner.sweep_abort_reason is not None
