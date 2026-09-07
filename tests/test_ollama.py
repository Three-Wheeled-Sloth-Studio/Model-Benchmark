from model_benchmark.ollama import ModelCandidate, skip_reason


def candidate(name: str) -> ModelCandidate:
    return ModelCandidate(name, 1, "x", "llama", ("llama",), "8B", "Q4_K_M", {})


def test_skips_embedding_cloud_and_remote_provider_models() -> None:
    assert "embedding" in skip_reason(candidate("nomic-embed-text:latest"))
    assert "cloud" in skip_reason(candidate("gpt-oss:cloud"))
    assert "cloud" in skip_reason(candidate("gpt-oss:120b-cloud"))
    assert "remote/provider" in skip_reason(candidate("gemini-3-flash-preview:latest"))
    assert skip_reason(candidate("qwen3:8b")) is None


def test_general_baseline_skips_specialized_coder_and_vision_models() -> None:
    assert "coding-specialist" in skip_reason(candidate("qwen3-coder:latest"))
    assert "coding-specialist" in skip_reason(candidate("devstral:24b"))
    assert "vision/multimodal" in skip_reason(candidate("qwen3-vl:8b"))
    assert "vision/multimodal" in skip_reason(candidate("llava:13b"))
    assert skip_reason(candidate("gemma3:12b")) is None


def test_explicit_specialized_selection_is_an_escape_hatch() -> None:
    assert skip_reason(candidate("qwen3-coder:latest"), allow_specialized=True) is None
    assert skip_reason(candidate("qwen3-vl:8b"), allow_specialized=True) is None
    assert "embedding" in skip_reason(
        candidate("nomic-embed-text:latest"), allow_specialized=True
    )
    assert "remote/provider" in skip_reason(
        candidate("gemini-3-flash-preview:latest"), allow_specialized=True
    )
