import threading
import time

from model_benchmark import runner as runner_module
from model_benchmark.config import BenchmarkConfig
from model_benchmark.fixtures import BenchmarkSuite
from model_benchmark.ollama import ModelCandidate
from model_benchmark.resources import ResourceSnapshot, ResourceStats
from model_benchmark.runner import BenchmarkRunner


class FakeClient:
    def unload(self, model: str) -> bool:
        return True

    def stop_model(self, model: str, **_kwargs) -> bool:
        return True


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


class HealthyResources:
    def sample(self) -> ResourceSnapshot:
        return ResourceSnapshot(
            available_ram_gb=32.0,
            total_ram_gb=95.0,
            ram_percent=66.0,
            cpu_percent=20.0,
            ollama_rss_gb=0.1,
            gpus=[],
        )


def empty_suite() -> BenchmarkSuite:
    return BenchmarkSuite(
        suite_id="baseline",
        suite_version="1",
        description="test",
        cases=(),
    )


def test_host_preflight_block_never_receives_a_score(tmp_path) -> None:
    runner = BenchmarkRunner(
        BenchmarkConfig(output_dir=str(tmp_path), min_available_ram_gb=8.0),
        empty_suite(),
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


def test_chat_message_content_is_used_for_scoring_contract(tmp_path) -> None:
    class ChatClient(FakeClient):
        def __init__(self) -> None:
            self.timeout = None
            self.think = None

        def chat(self, model, prompt, **kwargs):
            self.timeout = kwargs["timeout"]
            self.think = kwargs["think"]
            return {
                "message": {
                    "content": '{"answer":"usable"}',
                    "thinking": "hidden reasoning",
                },
                "done_reason": "stop",
                "eval_count": 4,
                "eval_duration": 100_000_000,
            }

    runner = BenchmarkRunner(
        BenchmarkConfig(output_dir=str(tmp_path), sample_interval_seconds=0.01),
        empty_suite(),
        console=lambda _message: None,
    )
    client = ChatClient()
    runner.client = client
    runner.resources = HealthyResources()

    result = runner._run_generate(
        "qwen3.5:9b",
        prompt="Return JSON.",
        schema={"type": "object"},
        timeout=0.25,
        resource_stats=ResourceStats(),
        options={"temperature": 0, "num_predict": 32},
    )

    assert result["status"] == "succeeded"
    assert result["api_endpoint"] == "chat"
    assert result["response"] == '{"answer":"usable"}'
    assert result["thinking"] == "hidden reasoning"
    assert client.think is False
    assert client.timeout == 0.25


def test_request_timeout_is_not_extended_before_recovery(tmp_path, monkeypatch) -> None:
    class SlowChatClient(FakeClient):
        def __init__(self) -> None:
            self.timeout = None

        def chat(self, model, prompt, **kwargs):
            self.timeout = kwargs["timeout"]
            time.sleep(0.5)
            return {"message": {"content": "late"}}

    runner = BenchmarkRunner(
        BenchmarkConfig(output_dir=str(tmp_path), sample_interval_seconds=0.005),
        empty_suite(),
        console=lambda _message: None,
    )
    client = SlowChatClient()
    runner.client = client
    runner.resources = HealthyResources()
    monkeypatch.setattr(runner, "_recover_request", lambda _model, _thread: True)

    result = runner._run_generate(
        "slow:model",
        prompt="test",
        schema=None,
        timeout=0.2,
        resource_stats=ResourceStats(),
        options={"temperature": 0},
    )

    assert result["status"] == "timeout"
    assert client.timeout == 0.2
    assert result["wall_time_seconds"] < 0.35
    assert result["total_wall_time_seconds"] < 0.35


def test_recovery_budget_bounds_a_stuck_request(tmp_path, monkeypatch) -> None:
    class RecoveryClient(FakeClient):
        def recover_after_abort(self, model: str, timeout: float) -> bool:
            return False

    runner = BenchmarkRunner(
        BenchmarkConfig(output_dir=str(tmp_path)),
        empty_suite(),
        console=lambda _message: None,
    )
    runner.client = RecoveryClient()
    monkeypatch.setattr(runner_module, "_RECOVERY_TIMEOUT_SECONDS", 0.05)

    thread = threading.Thread(target=lambda: time.sleep(0.5), daemon=True)
    thread.start()
    started = time.monotonic()
    recovered = runner._recover_request("stuck:model", thread)
    elapsed = time.monotonic() - started

    assert recovered is False
    assert elapsed < 0.2
