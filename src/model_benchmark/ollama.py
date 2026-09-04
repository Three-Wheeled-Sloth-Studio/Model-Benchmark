from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class OllamaError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ModelCandidate:
    name: str
    size_bytes: int | None
    digest: str | None
    family: str | None
    families: tuple[str, ...]
    parameter_size: str | None
    quantization_level: str | None
    raw: dict[str, Any]


_EMBED_HINTS = (
    "embed",
    "embedding",
    "rerank",
    "reranker",
    "all-minilm",
    "bge-m3",
    "bge-large",
    "mxbai",
    "nomic-embed",
    "snowflake-arctic-embed",
)


def skip_reason(candidate: ModelCandidate) -> str | None:
    normalized = candidate.name.casefold()
    if (
        normalized.endswith(":cloud")
        or normalized.endswith("-cloud")
        or ":cloud-" in normalized
    ):
        return "cloud-only tag; baseline is local-only"
    if any(hint in normalized for hint in _EMBED_HINTS):
        return "embedding/reranking-only model; no baseline text/chat suite yet"
    return None


class OllamaClient:
    def __init__(self, base_url: str = "http://127.0.0.1:11434") -> None:
        self.base_url = base_url.rstrip("/")
        self.ollama_path = shutil.which("ollama")
        self.started_process: subprocess.Popen[bytes] | None = None
        self._log_handle = None

    def api_available(self, timeout: float = 2.0) -> bool:
        try:
            self._request("GET", "/api/tags", timeout=timeout)
        except (OllamaError, OSError):
            return False
        return True

    def ensure_running(self, log_path: Path, startup_timeout: float) -> bool:
        if self.api_available():
            return False
        if not self.ollama_path:
            raise OllamaError("Ollama API is unavailable and 'ollama' was not found on PATH.")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = log_path.open("ab")
        kwargs: dict[str, Any] = {
            "stdout": self._log_handle,
            "stderr": subprocess.STDOUT,
        }
        if os.name == "nt":
            kwargs["creationflags"] = (
                getattr(subprocess, "CREATE_NO_WINDOW", 0)
                | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            )
        else:
            kwargs["start_new_session"] = True
        self.started_process = subprocess.Popen([self.ollama_path, "serve"], **kwargs)
        deadline = time.monotonic() + startup_timeout
        while time.monotonic() < deadline:
            if self.api_available():
                return True
            if self.started_process.poll() is not None:
                break
            time.sleep(0.5)
        raise OllamaError(f"Ollama did not become ready within {startup_timeout:.0f} seconds.")

    def stop_started_server(self) -> None:
        if self.started_process and self.started_process.poll() is None:
            self.started_process.terminate()
            try:
                self.started_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.started_process.kill()
        if self._log_handle:
            self._log_handle.close()
            self._log_handle = None

    def version(self) -> str | None:
        if not self.ollama_path:
            return None
        try:
            proc = subprocess.run(
                [self.ollama_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except OSError:
            return None
        text = (proc.stdout or proc.stderr).strip()
        return text or None

    def list_models(self) -> list[ModelCandidate]:
        payload = self._request("GET", "/api/tags", timeout=10)
        result: list[ModelCandidate] = []
        for item in payload.get("models", []):
            details = item.get("details") or {}
            result.append(
                ModelCandidate(
                    name=item.get("name") or item.get("model") or "",
                    size_bytes=item.get("size"),
                    digest=item.get("digest"),
                    family=details.get("family"),
                    families=tuple(details.get("families") or ()),
                    parameter_size=details.get("parameter_size"),
                    quantization_level=details.get("quantization_level"),
                    raw=item,
                )
            )
        return [item for item in result if item.name]

    def running_models(self) -> list[dict[str, Any]]:
        return list(self._request("GET", "/api/ps", timeout=10).get("models", []))

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        system: str | None = None,
        schema: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
        keep_alive: str | int = "15m",
        timeout: float = 330.0,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": keep_alive,
        }
        if system:
            payload["system"] = system
        if schema:
            payload["format"] = schema
        if options:
            payload["options"] = options
        return self._request("POST", "/api/generate", payload=payload, timeout=timeout)

    def unload(self, model: str, timeout: float = 30.0) -> None:
        running = self.running_models()
        if not any(
            (item.get("name") or item.get("model")) == model for item in running
        ):
            return
        self.stop_model(model)
        deadline = time.monotonic() + min(timeout, 15.0)
        while time.monotonic() < deadline:
            if not any(
                (item.get("name") or item.get("model")) == model
                for item in self.running_models()
            ):
                return
            time.sleep(0.25)

    def stop_model(self, model: str) -> None:
        if self.ollama_path:
            try:
                subprocess.run(
                    [self.ollama_path, "stop", model],
                    capture_output=True,
                    timeout=10,
                    check=False,
                )
                return
            except (OSError, subprocess.TimeoutExpired):
                pass
        try:
            self.generate(model, "", keep_alive=0, options={"num_predict": 1}, timeout=10)
        except OllamaError:
            pass

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        timeout: float,
    ) -> dict[str, Any]:
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise OllamaError(f"Ollama HTTP {error.code}: {body[:500]}") from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise OllamaError(f"Ollama request failed: {error}") from error
        try:
            return json.loads(raw) if raw else {}
        except json.JSONDecodeError as error:
            raise OllamaError("Ollama returned invalid JSON.") from error
