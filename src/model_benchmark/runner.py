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
from model_benchmark.ollama import (
    ModelCandidate,
    OllamaClient,
    OllamaError,
    OllamaHttpError,
    skip_reason,
)
from model_benchmark.resources import ResourceCollector, ResourceStats
from model_benchmark.scoring import aggregate_model_score, score_case

_SYSTEM_PROMPT = (
    "You are running a deterministic local benchmark. Follow the requested output schema exactly. "
    "Use only evidence present in the prompt. Do not add generic filler merely because it sounds professional."
)
_RECOVERY_TIMEOUT_SECONDS = 15.0


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
        self.sweep_abort_reason: str | None = None

    def prepare(self) -> list[ModelCandidate]:
        self.started_ollama = self.client.ensure_running(
            self.run_dir / "ollama-serve.log", self.config.startup_timeout_seconds
        )
        models = self.client.list_models()
        selected: list[ModelCandidate] = []
        explicit_selection = bool(self.config.model_patterns)
        for candidate in models:
            if explicit_selection and not any(
                fnmatch.fnmatch(candidate.name.casefold(), pattern.casefold())
                for pattern in self.config.model_patterns
            ):
                self.skipped_models.append(
                    {"model": candidate.name, "reason": "does not match requested --model pattern"}
                )
                continue
            reason = skip_reason(candidate, allow_specialized=explicit_selection)
            if reason:
                self.skipped_models.append({"model": candidate.name, "reason": reason})
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
        current_stage = "pre_cold_unload"
        try:
            if not self.client.unload(candidate.name):
                result["status"] = "failed"
                result["failure_stage"] = current_stage
                result["score"] = _unscored_score(current_stage)
                result["error"] = "Could not confirm the model was unloaded before the cold probe."
                self.sweep_abort_reason = (
                    f"Ollama could not establish a clean unloaded state for {candidate.name}."
                )
                return self._finish_model(result, resource_stats, started)

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

            current_stage = "cold_load"
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                result["status"] = "timeout"
                result["failure_stage"] = current_stage
                result["score"] = _unscored_score(current_stage)
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
            else:
                result["status"] = (
                    "unavailable"
                    if cold.get("http_status") in {404, 410}
                    else cold["status"]
                )
                result["failure_stage"] = current_stage
                result["score"] = _unscored_score(current_stage)
                if cold.get("error"):
                    result["error"] = cold["error"]
                if cold.get("http_status") is not None:
                    result["http_status"] = cold["http_status"]
                if cold.get("recovery_status"):
                    result["recovery_status"] = cold["recovery_status"]
                self._track_service_health(candidate.name, current_stage, cold)
                return self._finish_model(result, resource_stats, started)

            for case in self.suite.cases:
                current_stage = f"benchmark_case:{case.id}"
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    result["status"] = "timeout"
                    result["failure_stage"] = current_stage
                    break
                self.console(f"[{candidate.name}] {case.category}: {case.id}")
                test_result = self._run_case(
                    candidate.name,
                    case,
                    timeout=min(self.config.test_timeout_seconds, remaining),
                    resource_stats=resource_stats,
                )
                result["tests"].append(test_result)
                if test_result["status"] in {"resource_abort", "timeout"}:
                    result["status"] = test_result["status"]
                    result["failure_stage"] = current_stage
                    self._track_service_health(candidate.name, current_stage, test_result)
                    break

            if result["status"] == "running":
                current_stage = "warm_probe"
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
                        result["failure_stage"] = current_stage
                        self._track_service_health(
                            candidate.name, current_stage, result["warm_probe"]
                        )
                else:
                    result["status"] = "timeout"
                    result["failure_stage"] = current_stage

            if result["status"] == "running":
                failed = [item for item in result["tests"] if item["status"] != "succeeded"]
                result["status"] = "partial" if failed else "succeeded"
            return self._finish_model(result, resource_stats, started)
        except (OllamaError, OSError) as error:
            result["status"] = "failed"
            result["failure_stage"] = current_stage
            result["error"] = str(error)
            if isinstance(error, OllamaHttpError):
                result["http_status"] = error.status_code
                if error.status_code in {404, 410} and current_stage == "cold_load":
                    result["status"] = "unavailable"
                    result["score"] = _unscored_score(current_stage)
            if not self.client.api_available(timeout=2.0):
                self.sweep_abort_reason = (
                    f"Ollama API became unhealthy while testing {candidate.name} at {current_stage}."
                )
            return self._finish_model(result, resource_stats, started)
        finally:
            self.client.stop_model(
                candidate.name, timeout=3.0, allow_fallback=False
            )

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
                        self.client.chat(
                            model,
                            prompt,
                            system=_SYSTEM_PROMPT,
                            schema=schema,
                            options=options,
                            keep_alive=-1,
                            think=False,
                            timeout=max(0.1, timeout),
                        ),
                    )
                )
            except Exception as error:  # daemon request boundary
                response_queue.put(("error", error))

        thread = threading.Thread(target=invoke, daemon=True)
        thread.start()
        abort_reason: str | None = None
        abort_detected_at: float | None = None
        while thread.is_alive():
            snapshot = self.resources.sample()
            resource_stats.observe(snapshot)
            request_stats.observe(snapshot)
            if snapshot.available_ram_gb < self.config.min_available_ram_gb:
                abort_reason = "resource_abort"
                abort_detected_at = time.monotonic()
                self.console(
                    f"[{model}] stopping: available RAM {snapshot.available_ram_gb:.2f} GB "
                    f"fell below {self.config.min_available_ram_gb:.2f} GB floor"
                )
                break
            if time.monotonic() - started >= timeout:
                abort_reason = "timeout"
                abort_detected_at = time.monotonic()
                self.console(f"[{model}] stopping: test exceeded {timeout:.0f} seconds")
                break
            time.sleep(self.config.sample_interval_seconds)

        if abort_reason:
            if abort_detected_at is None:
                abort_detected_at = time.monotonic()
            recovery_started = time.monotonic()
            recovery_ok = self._recover_request(model, thread)
            finished = time.monotonic()
            return {
                "status": abort_reason,
                "recovery_status": "recovered" if recovery_ok else "failed",
                "wall_time_seconds": round(abort_detected_at - started, 3),
                "recovery_time_seconds": round(finished - recovery_started, 3),
                "total_wall_time_seconds": round(finished - started, 3),
                "metrics": {},
                "resource_stats": request_stats.as_dict(),
            }

        try:
            outcome, payload = response_queue.get_nowait()
        except queue.Empty:
            return {
                "status": "failed",
                "error": "generation ended without a result",
                "api_healthy_after_error": self.client.api_available(timeout=2.0),
                "wall_time_seconds": round(time.monotonic() - started, 3),
                "metrics": {},
                "resource_stats": request_stats.as_dict(),
            }
        if outcome == "error":
            result = {
                "status": "failed",
                "error": str(payload),
                "api_healthy_after_error": self.client.api_available(timeout=2.0),
                "wall_time_seconds": round(time.monotonic() - started, 3),
                "metrics": {},
                "resource_stats": request_stats.as_dict(),
            }
            if isinstance(payload, OllamaHttpError):
                result["http_status"] = payload.status_code
            return result
        metrics = self._ollama_metrics(payload)
        message = payload.get("message") or {}
        return {
            "status": "succeeded",
            "api_endpoint": "chat",
            "response": message.get("content") or payload.get("response", ""),
            "thinking": message.get("thinking") or payload.get("thinking") or None,
            "done_reason": payload.get("done_reason"),
            "wall_time_seconds": round(time.monotonic() - started, 3),
            "metrics": metrics,
            "resource_stats": request_stats.as_dict(),
        }

    def _recover_request(self, model: str, thread: threading.Thread) -> bool:
        deadline = time.monotonic() + _RECOVERY_TIMEOUT_SECONDS
        remaining = deadline - time.monotonic()
        if remaining > 0:
            self.client.stop_model(
                model,
                timeout=min(2.0, remaining),
                allow_fallback=False,
            )
        last_stop = time.monotonic()
        while thread.is_alive() and time.monotonic() < deadline:
            thread.join(timeout=min(0.25, max(0.0, deadline - time.monotonic())))
            now = time.monotonic()
            if thread.is_alive() and now - last_stop >= 2.0:
                remaining = deadline - now
                if remaining <= 0:
                    break
                self.client.stop_model(
                    model,
                    timeout=min(1.0, remaining),
                    allow_fallback=False,
                )
                last_stop = time.monotonic()
        remaining = deadline - time.monotonic()
        if thread.is_alive() or remaining <= 0:
            return False
        service_ok = self.client.recover_after_abort(model, timeout=remaining)
        return service_ok and not thread.is_alive()

    def _track_service_health(
        self, model: str, stage: str, generation_result: dict[str, Any]
    ) -> None:
        if generation_result.get("recovery_status") == "failed":
            self.sweep_abort_reason = (
                f"Ollama did not recover cleanly after {model} failed at {stage}."
            )
        elif generation_result.get("api_healthy_after_error") is False:
            self.sweep_abort_reason = (
                f"Ollama API became unhealthy after {model} failed at {stage}."
            )

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
                f"stage {result.get('failure_stage') or result.get('block_stage') or 'n/a'}"
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
            "sweep_abort_reason": self.sweep_abort_reason,
        }

    def cleanup(self) -> None:
        if self.started_ollama and self.config.stop_started_ollama:
            self.client.stop_started_server()
