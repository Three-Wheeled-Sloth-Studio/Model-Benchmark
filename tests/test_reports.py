import json

from model_benchmark.reports import append_jsonl, write_reports


def test_reports_are_machine_and_human_readable(tmp_path) -> None:
    metadata = {
        "run_id": "run-1",
        "suite": {"id": "baseline", "version": "1"},
        "config": {"weights": {"quality": 0.7, "operational": 0.3}},
        "skipped_models": [],
    }
    result = {
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
    append_jsonl(tmp_path / "results.jsonl", result)
    summary = write_reports(tmp_path, metadata, [result])
    assert summary["rankings"][0]["model"] == "model-a"
    assert json.loads((tmp_path / "results.jsonl").read_text()) == result
    assert "model-a" in (tmp_path / "summary.md").read_text()
    assert (tmp_path / "run-metadata.yaml").exists()
