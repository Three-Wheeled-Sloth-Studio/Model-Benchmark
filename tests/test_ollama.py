from model_benchmark.ollama import ModelCandidate, skip_reason


def candidate(name: str) -> ModelCandidate:
    return ModelCandidate(name, 1, "x", "llama", ("llama",), "8B", "Q4_K_M", {})


def test_skips_embedding_and_cloud_models() -> None:
    assert "embedding" in skip_reason(candidate("nomic-embed-text:latest"))
    assert "cloud" in skip_reason(candidate("gpt-oss:cloud"))
    assert "cloud" in skip_reason(candidate("gpt-oss:120b-cloud"))
    assert skip_reason(candidate("qwen3:8b")) is None


def test_does_not_skip_general_multimodal_name() -> None:
    assert skip_reason(candidate("gemma3:12b")) is None
    assert skip_reason(candidate("llava:13b")) is None
