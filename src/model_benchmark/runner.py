from __future__ import annotations

import fnmatch
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from model_benchmark.config import BenchmarkConfig
from model_benchmark.fixtures import BenchmarkCase, BenchmarkSuite
from model_benchmark.ollama import ModelCandidate, OllamaClient, OllamaError, skip_reason
from model_benchmark.resources import ResourceCollector, ResourceStats
from model_benchmark.scoring import aggregate_model_score, score_case

_SYSTEM_PROMPT = (
    "You are running a deterministic local benchmark. Follow the requested output schema exactly. "
    "Use only evidence present in the prompt. Do not add generic filler merely because it sounds professional."
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _unscored_score(reason: str) -> dict[str, Any]:
    """Return a schema-stable score payload for a model that was never evaluated."""
    return {
        "score_status": "not_evaluated",
        "score_reason": reason,
        "quality_score": None,
        "operational_score": None,
        "composite_score_raw": None,
        "composite_score": None,
        "quality_components": {
            "keyword_quality": None,
            "semantic_quality": None,
            "ranking_quality": None,
            "adherence": None,
        },
        "operational_components": {
            "cold_load_score": None,
            "warm_throughput_score": None,
            "memory_headroom_score": None,
            "median_warm_tokens_per_second": None,
            "cold_load_seconds": None,
            "warm_load_seconds": None,
        },
        "penalties": [],
    }


class BenchmarkRunner:
    def __init__(
        self,
        config: BenchmarkConfig,
        suite: BenchmarkSuite,
        *,
        console: Callable[[str], None] = print,
    ) -> None:
        self.config = config
        self.suite = suite
        self.console = console
        self.client = OllamaClient(config.base_url)
        self.resources = ResourceCollector(config.gpu_telemetry)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_id = f"{stamp}-{uuid4().hex[:8]}"
        self.run_dir = Path(config.output_dir) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.started_at = utc_now()
        self.started_ollama = False
        self.skipped_models: list[dict[str, str]] = []

    def prepare(self) -> list[ModelCandidate]:
        self.started_ollama = self.client.ensure_running(
            self.run_dir / "ollama-serve.log", self.config.startup_timeout_seconds
        )
        models = self.client.list_models()
        selected: list[ModelCandidate] = []
        for candidate in models:
            reason = skip_reason(candidate)
            if reason:
                self.skipped_models.append({"model": candidate.name, "reason": reason})
                continue
            if self.config.model_patterns and not any(
                fnmatch.fnmatch(candidate.name.casefold(), pattern.casefold())
                for pattern in self.config.model_patterns
            ):
                self.skipped_models.append(
                    {"model": candidate.name, "reason": "does not match requested --model pattern"}
                )
                continue
            selected.append(candidate)
        return selected

    def run_model(self, candidate: ModelCandidate) -> dict[str, Any]:
        self.console(f"\n[{candidate.name}] unloading before cold probe")
        started = time.monotonic()
        deadline = started + self.config.model_timeout_seconds
        resource_stats = ResourceStats()
        result: dict[str, Any] = {
            "record_type": "model_result",
            "schema_version": "1.0",
            "run_id": self.run_id,
            "suite_id": self.suite.suite_id,
            "suite_version": self.suite.suite_version,
            "model": {
                "name": candidate.name,
                "size_bytes": candidate.size_bytes,
                "digest": candidate.digest,
                "family": candidate.family,
                "families": list(candidate.families),
                "parameter_size": candidate.parameter_size,
                "quantization_level": candidate.quantization_level,
            },
            "started_at": utc_now(),
            "status": "running",
            "tests": [],
        }
        try:
            self.client.unload(candidate.name)
            preflight = self.resources.sample()
            resource_stats.observe(preflight)
            if preflight.available_ram_gb < self.config.min_available_ram_gb:
                result["status"] = "host_resource_blocked"
                result["block_stage"] = "host_preflight"
                result["score"] = _unscored_score("host_preflight")
                result["error"] = (
                    f"Available RAM {preflight.available_ram_gb:.2f} GB is below the "
                    f"{self.config.min_available_ram_gb:.2f} GB safety floor before model load."
                )
                return self._finish_model(result, resource_stats, started)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                result["status"] = "timeout"
                return self._finish_model(result, resource_stats, started)

            self.console(f"[{candidate.name}] cold-load probe")
            cold = self._run_generate(
                candidate.name,
                prompt="Return exactly the word OK.",
                schema=None,
                timeout=min(self.config.test_timeout_seconds, remaining),
                resource_stats=resource_stats,
                options={
                    "temperature": 0,
                    "seed": self.config.seed,
                    "num_ctx": self.config.context_length,
                    "num_predict": 8,
                },
            )
            result["cold_probe"] = cold
            if cold["status"] == "succeeded":
                result["loaded_model_state"] = self._loaded_model_state(candidate.name)
            if cold["status"] != "succeeded":
                result["status"] = cold["status"]
                return self._finish_model(result, resource_stats, started)

            for case in self.suite.cases:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    result["status"] = "timeout"
                    break
                self.console(f"[{candidate.name}] {case.category}: {case.id}")
                test_result = self._run_case(
                    candidate.name,
                    case,
                    timeout=min(self.config.test_timeout_seconds, remaining),
                    resource_stats=resource_stats,
                )
                result["tests"].append(test_result)
                if test_result["status"] == "resource_abort":
                    result["status"] = "resource_abort"
                    break
                if test_result["status"] == "timeout":
                    result["status"] = "timeout"
                    break

            if result["status"] == "running":
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    self.console(f"[{candidate.name}] representative warm probe")
                    representative = self.suite.representative_case
                    result["warm_probe"] = self._run_generate(
                        candidate.name,
                        prompt=representative.prompt,
                        schema=representative.schema,
                        timeout=min(self.config.test_timeout_seconds, remaining),
                        resource_stats=resource_stats,
                        options=self._options(),
                    )
                    if result["warm_probe"]["status"] != "succeeded":
                        result["status"] = result["warm_probe"]["status"]
                else:
                    result["status"] = "timeout"

            if result["status"] == "running":
                failed = [item for item in result["tests"] if item["status"] != "succeeded"]
                result["status"] = "partial" if failed else "succeeded"
            return self._finish_model(result, resource_stats, started)
        except (OllamaError, OSError) as error:
            result["status"] = "failed"
            result["error"] = str(error)
            return self._finish_model(result, resource_stats, started)
        finally:
            self.client.stop_model(candidate.name)

    def _run_case(
        self,
        model: str,
        case: BenchmarkCase,
        *,
        timeout: float,
        resource_stats: ResourceStats,
    ) -> dict[str, Any]:
        generated = self._run_generate(
            model,
            prompt=case.prompt,
            schema=case.schema,
            timeout=timeout,
            resource_stats=resource_stats,
            options=self._options(),
        )
        result = {
            "case_id": case.id,
            "category": case.category,
            **generated,
        }
        if generated["status"] == "succeeded":
            score = score_case(case, generated.get("response", ""))
            result["score"] = asdict(score)
        return result

    def _options(self) -> dict[str, Any]:
        return {
            "temperature": 0,
            "seed": self.config.seed,
            "num_ctx": self.config.context_length,
            "num_predict": self.config.max_output_tokens,
        }

    def _run_generate(
        self,
        model: str,
        *,
        prompt: str,
        schema: dict[str, Any] | None,
        timeout: float,
        resource_stats: ResourceStats,
        options: dict[str, Any],
    ) -> dict[str, Any]:
        response_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)
        started = time.monotonic()
        request_stats = ResourceStats()

        def invoke() -> None:
            try:
                response_queue.put(
                    (
                        "ok",
                        self.client.generate(
                            model,
                            prompt,
                            system=_SYSTEM_PROMPT,
                            schema=schema,
                            options=options,
                            keep_alive=-1,
                            timeout=timeout + 20.0,
                        ),
                    )
                )
            except Exception as error:  # daemon request boundary
                response_queue.put(("error", error))

        thread = threading.Thread(target=invoke, daemon=True)
        thread.start()
        abort_reason: str | None = None
        while thread.is_alive():
            snapshot = self.resources.sample()
            resource_stats.observe(snapshot)
            request_stats.observe(snapshot)
            if snapshot.available_ram_gb < self.config.min_available_ram_gb:
                abort_reason = "resource_abort"
                self.console(
                    f"[{model}] stopping: available RAM {snapshot.available_ram_gb:.2f} GB "
                    f"fell below {self.config.min_available_ram_gb:.2f} GB floor"
                )
                self.client.stop_model(model)
                break
            if time.monotonic() - started >= timeout:
                abort_reason = "timeout"
                self.console(f"[{model}] stopping: test exceeded {timeout:.0f} seconds")
                self.client.stop_model(model)
                break
            time.sleep(self.config.sample_interval_seconds)

        if abort_reason:
            thread.join(timeout=10)
            return {
                "status": abort_reason,
                "wall_time_seconds": round(time.monotonic() - started, 3),
                "metrics": {},
                "resource_stats": request_stats.as_dict(),
            }

        try:
            outcome, payload = response_queue.get_nowait()
        except queue.Empty:
            return {
                "status": "failed",
                "error": "generation ended without a result",
                "wall_time_seconds": round(time.monotonic() - started, 3),
                "metrics": {},
                "resource_stats": request_stats.as_dict(),
            }
        if outcome == "error":
            return {
                "status": "failed",
                "error": str(payload),
                "wall_time_seconds": round(time.monotonic() - started, 3),
                "metrics": {},
                "resource_stats": request_stats.as_dict(),
            }
        metrics = self._ollama_metrics(payload)
        return {
            "status": "succeeded",
            "response": payload.get("response", ""),
            "thinking": payload.get("thinking") or None,
            "done_reason": payload.get("done_reason"),
            "wall_time_seconds": round(time.monotonic() - started, 3),
            "metrics": metrics,
            "resource_stats": request_stats.as_dict(),
        }

    @staticmethod
    def _ollama_metrics(payload: dict[str, Any]) -> dict[str, Any]:
        def ms(name: str) -> float | None:
            value = payload.get(name)
            return round(float(value) / 1_000_000.0, 3) if value is not None else None

        eval_count = payload.get("eval_count")
        eval_duration = payload.get("eval_duration")
        prompt_count = payload.get("prompt_eval_count")
        prompt_duration = payload.get("prompt_eval_duration")
        eval_tps = (
            float(eval_count) / (float(eval_duration) / 1_000_000_000.0)
            if eval_count and eval_duration
            else None
        )
        prompt_tps = (
            float(prompt_count) / (float(prompt_duration) / 1_000_000_000.0)
            if prompt_count and prompt_duration
            else None
        )
        return {
            "total_duration_ms": ms("total_duration"),
            "load_duration_ms": ms("load_duration"),
            "prompt_eval_duration_ms": ms("prompt_eval_duration"),
            "eval_duration_ms": ms("eval_duration"),
            "prompt_eval_count": prompt_count,
            "eval_count": eval_count,
            "prompt_tokens_per_second": round(prompt_tps, 3) if prompt_tps is not None else None,
            "eval_tokens_per_second": round(eval_tps, 3) if eval_tps is not None else None,
        }

    def _loaded_model_state(self, model: str) -> dict[str, Any] | None:
        try:
            running = self.client.running_models()
        except OllamaError:
            return None
        for item in running:
            if (item.get("name") or item.get("model")) == model:
                return {
                    "size_bytes": item.get("size"),
                    "size_vram_bytes": item.get("size_vram"),
                    "context_length": item.get("context_length"),
                    "expires_at": item.get("expires_at"),
                }
        return None

    def _finish_model(
        self, result: dict[str, Any], resource_stats: ResourceStats, started: float
    ) -> dict[str, Any]:
        result["finished_at"] = utc_now()
        result["model_wall_time_seconds"] = round(time.monotonic() - started, 3)
        result["resource_stats"] = resource_stats.as_dict()
        if "score" not in result:
            result["score"] = aggregate_model_score(result, self.config)
        score = result["score"]
        if score["composite_score"] is None:
            self.console(
                f"[{result['model']['name']}] unscored status {result['status']} "
                f"stage {result.get('block_stage', 'n/a')}"
            )
        else:
            self.console(
                f"[{result['model']['name']}] composite {score['composite_score']:.1f} "
                f"quality {score['quality_score']:.1f} "
                f"operational {score['operational_score']:.1f} "
                f"status {result['status']}"
            )
        return result

    def metadata(self, selected_models: list[ModelCandidate]) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "run_id": self.run_id,
            "started_at": self.started_at,
            "suite": {
                "id": self.suite.suite_id,
                "version": self.suite.suite_version,
                "description": self.suite.description,
            },
            "config": self.config.as_dict(),
            "hardware": self.resources.hardware_snapshot(),
            "ollama_version": self.client.version(),
            "ollama_started_by_harness": self.started_ollama,
            "selected_models": [item.name for item in selected_models],
            "skipped_models": self.skipped_models,
        }

    def cleanup(self) -> None:
        if self.started_ollama and self.config.stop_started_ollama:
            self.client.stop_started_server()
