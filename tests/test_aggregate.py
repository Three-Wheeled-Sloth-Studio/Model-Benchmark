from model_benchmark.config import BenchmarkConfig
from model_benchmark.scoring import aggregate_model_score


def model_result(status: str = "succeeded", min_ram: float = 20.0):
    return {
        "status": status,
        "cold_probe": {"metrics": {"load_duration_ms": 5000}},
        "warm_probe": {"metrics": {"eval_tokens_per_second": 25}},
        "resource_stats": {"min_available_ram_gb": min_ram},
        "tests": [
            {
                "category": "keyword",
                "status": "succeeded",
                "score": {"quality_score": 90, "adherence_score": 95},
                "metrics": {"eval_tokens_per_second": 25},
            },
            {
                "category": "semantic",
                "status": "succeeded",
                "score": {"quality_score": 85, "adherence_score": 95},
                "metrics": {"eval_tokens_per_second": 20},
            },
            {
                "category": "ranking",
                "status": "succeeded",
                "score": {"quality_score": 88, "adherence_score": 90},
                "metrics": {"eval_tokens_per_second": 22},
            },
        ],
    }


def test_composite_is_quality_first() -> None:
    score = aggregate_model_score(model_result(), BenchmarkConfig())
    assert score["quality_score"] > score["operational_score"]
    assert 0 <= score["composite_score"] <= 100


def test_resource_abort_caps_composite() -> None:
    score = aggregate_model_score(
        model_result(status="resource_abort", min_ram=4), BenchmarkConfig(min_available_ram_gb=8)
    )
    assert score["composite_score"] <= 20
    assert "resource_abort_cap_20" in score["penalties"]
