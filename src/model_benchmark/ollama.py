from __future__ import annotations

import contextlib
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


class OllamaHttpError(OllamaError):
    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"Ollama HTTP {status_code}: {body[:500]}")


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
_CODER_HINTS = (
    "coder",
    "codellama",
    "codestral",
    "devstral",
    "starcoder",
    "codegemma",
)
_VISION_HINTS = (
    "-vl",
    ":vl",
    "vision",
    "llava",
    "bakllava",
    "moondream",
    "minicpm-v",
)
_REMOTE_PROVIDER_PREFIXES = (
    "gemini-",
    "gemini:",
    "claude-",
    "claude:",
)


def skip_reason(candidate: ModelCandidate, *, allow_specialized: bool = False) -> str | None:
    normalized = candidate.name.casefold()
    if (
        normalized.endswith(":cloud")
        or normalized.endswith("-cloud")
        or ":cloud-" in normalized
    ):
        return "cloud-only tag; baseline is local-only"
    if normalized.startswith(_REMOTE_PROVIDER_PREFIXES):
        return "remote/provider model stub; baseline measures local Ollama inference only"
    if any(hint in normalized for hint in _EMBED_HINTS):
        return "embedding/reranking-only model; no baseline text/chat suite yet"
    if not allow_specialized and any(hint in normalized for hint in _CODER_HINTS):
        return "coding-specialist model; excluded from the general structured-work baseline"
    if not allow_specialized and any(hint in normalized for hint in _VISION_HINTS):
        return "vision/multimodal-specialist model; excluded from the text-only baseline"
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

    def running_models(self, timeout: float = 10.0) -> list[dict[str, Any]]:
        return list(self._request("GET", "/api/ps", timeout=timeout).get("models", []))

    def is_model_running(self, model: str, timeout: float = 10.0) -> bool:
        return any(
            (item.get("name") or item.get("model")) == model
            for item in self.running_models(timeout=timeout)
        )

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
        """Low-level generate endpoint retained for compatibility and unload fallback."""
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

    def chat(
        self,
        model: str,
        prompt: str,
        *,
        system: str | None = None,
        schema: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
        keep_alive: str | int = "15m",
        think: bool = False,
        timeout: float = 330.0,
    ) -> dict[str, Any]:
        """Run benchmark inference through chat so thinking can be disabled reliably."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "keep_alive": keep_alive,
            "think": think,
        }
        if schema:
            payload["format"] = schema
        if options:
            payload["options"] = options
        return self._request("POST", "/api/chat", payload=payload, timeout=timeout)

    def unload(self, model: str, timeout: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        try:
            if not self.is_model_running(model, timeout=min(2.0, remaining)):
                return True
        except OllamaError:
            pass
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        self.stop_model(model, timeout=min(3.0, remaining))
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            try:
                if not self.is_model_running(model, timeout=min(1.0, max(0.1, remaining))):
                    return True
            except OllamaError:
                pass
            time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))
        return False

    def recover_after_abort(self, model: str, timeout: float = 15.0) -> bool:
        """Require a healthy API and unloaded target model within a strict recovery budget."""
        deadline = time.monotonic() + max(0.0, timeout)
        last_stop = 0.0
        while time.monotonic() < deadline:
            now = time.monotonic()
            remaining = deadline - now
            if now - last_stop >= 1.0:
                self.stop_model(
                    model,
                    timeout=min(2.0, max(0.1, remaining)),
                    allow_fallback=False,
                )
                last_stop = time.monotonic()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            if self.api_available(timeout=min(1.0, max(0.1, remaining))):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    if not self.is_model_running(
                        model, timeout=min(1.0, max(0.1, remaining))
                    ):
                        return True
                except OllamaError:
                    pass
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(0.25, remaining))
        return False

    def stop_model(
        self,
        model: str,
        *,
        timeout: float = 10.0,
        allow_fallback: bool = True,
    ) -> bool:
        deadline = time.monotonic() + max(0.0, timeout)
        if self.ollama_path:
            remaining = deadline - time.monotonic()
            if remaining > 0:
                try:
                    proc = subprocess.run(
                        [self.ollama_path, "stop", model],
                        capture_output=True,
                        timeout=max(0.1, remaining),
                        check=False,
                    )
                    if proc.returncode == 0:
                        return True
                except (OSError, subprocess.TimeoutExpired):
                    pass
        if not allow_fallback:
            return False
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        try:
            self.generate(
                model,
                "",
                keep_alive=0,
                options={"num_predict": 1},
                timeout=max(0.1, remaining),
            )
        except OllamaError:
            return False
        return True

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
            raise OllamaHttpError(error.code, body) from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise OllamaError(f"Ollama request failed: {error}") from error
        try:
            return json.loads(raw) if raw else {}
        except json.JSONDecodeError as error:
            raise OllamaError("Ollama returned invalid JSON.") from error
