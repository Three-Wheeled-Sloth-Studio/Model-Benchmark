import json

from model_benchmark.reports import append_jsonl, write_reports


def scored_result() -> dict:
    return {
        "model": {"name": "model-a"},
        "status": "succeeded",
        "score": {
            "composite_score": 88.0,
            "quality_score": 90.0,
            "operational_score": 83.0,
            "quality_components": {
                "keyword_quality": 92.0,
                "semantic_quality": 88.0,
                "ranking_quality": 90.0,
                "adherence": 90.0,
            },
            "operational_components": {
                "median_warm_tokens_per_second": 25.0,
                "cold_load_seconds": 4.0,
            },
            "penalties": [],
        },
        "resource_stats": {"min_available_ram_gb": 18.0},
    }


def blocked_result() -> dict:
    return {
        "model": {"name": "smollm2:360m"},
        "status": "host_resource_blocked",
        "block_stage": "host_preflight",
        "error": "Available RAM 0.75 GB is below the 8.00 GB safety floor before model load.",
        "score": {
            "score_status": "not_evaluated",
            "score_reason": "host_preflight",
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
        },
        "resource_stats": {
            "min_available_ram_gb": 0.745,
            "peak_ram_percent": 95.3,
            "peak_cpu_percent": 36.8,
            "peak_ollama_rss_gb": 0.044,
            "peak_vram_used_mb": 2173.0,
            "peak_gpu_utilization_percent": 38.0,
            "sample_count": 1,
        },
    }


def metadata() -> dict:
    return {
        "run_id": "run-1",
        "suite": {"id": "baseline", "version": "1"},
        "config": {"weights": {"quality": 0.7, "operational": 0.3}},
        "skipped_models": [],
    }


def test_reports_are_machine_and_human_readable(tmp_path) -> None:
    result = scored_result()
    append_jsonl(tmp_path / "results.jsonl", result)
    summary = write_reports(tmp_path, metadata(), [result])
    assert summary["rankings"][0]["model"] == "model-a"
    assert summary["unscored_models"] == []
    assert json.loads((tmp_path / "results.jsonl").read_text()) == result
    assert "model-a" in (tmp_path / "summary.md").read_text()
    assert (tmp_path / "run-metadata.yaml").exists()


def test_preflight_block_is_unscored_and_excluded_from_rankings(tmp_path) -> None:
    blocked = blocked_result()
    summary = write_reports(tmp_path, metadata(), [blocked])

    assert summary["rankings"] == []
    assert summary["unscored_models"][0]["model"] == "smollm2:360m"
    assert summary["unscored_models"][0]["block_stage"] == "host_preflight"
    assert summary["unscored_models"][0]["score"]["composite_score"] is None
    assert summary["unscored_models"][0]["score"]["quality_score"] is None
    assert summary["unscored_models"][0]["score"]["operational_components"]["cold_load_score"] is None

    markdown = (tmp_path / "summary.md").read_text()
    assert "No scored models" in markdown
    assert "## Unscored models" in markdown
    assert "host_preflight" in markdown
    assert "0.7" in markdown
    assert "not evidence that the model itself is too large or too slow" in markdown


def test_unscored_models_do_not_displace_scored_ranking(tmp_path) -> None:
    summary = write_reports(tmp_path, metadata(), [blocked_result(), scored_result()])
    assert [item["model"] for item in summary["rankings"]] == ["model-a"]
    assert [item["model"] for item in summary["unscored_models"]] == ["smollm2:360m"]
