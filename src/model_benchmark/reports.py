from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        handle.write("\n")


def write_reports(
    run_dir: Path, metadata: dict[str, Any], results: list[dict[str, Any]]
) -> dict[str, Any]:
    ranked = sorted(results, key=lambda item: item["score"]["composite_score"], reverse=True)
    summary = {
        "schema_version": "1.0",
        "run_id": metadata["run_id"],
        "suite": metadata["suite"],
        "weights": metadata["config"]["weights"],
        "rankings": [
            {
                "rank": index,
                "model": item["model"]["name"],
                "status": item["status"],
                "composite_score": item["score"]["composite_score"],
                "quality_score": item["score"]["quality_score"],
                "operational_score": item["score"]["operational_score"],
                "quality_components": item["score"]["quality_components"],
                "operational_components": item["score"]["operational_components"],
                "penalties": item["score"]["penalties"],
                "resource_stats": item["resource_stats"],
            }
            for index, item in enumerate(ranked, 1)
        ],
        "skipped_models": metadata["skipped_models"],
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "run-metadata.yaml").write_text(
        yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (run_dir / "summary.md").write_text(_markdown(metadata, ranked), encoding="utf-8")
    return summary


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.1f}"


def _fmt_gb_from_mb(value: Any) -> str:
    return "n/a" if value is None else f"{float(value) / 1024.0:.1f}"


def _fmt_gb_from_bytes(value: Any) -> str:
    return "n/a" if value is None else f"{float(value) / (1024.0 ** 3):.1f}"


def _markdown(metadata: dict[str, Any], ranked: list[dict[str, Any]]) -> str:
    lines = [
        "# Model Benchmark Summary",
        "",
        f"Run: `{metadata['run_id']}`  ",
        f"Suite: `{metadata['suite']['id']}@{metadata['suite']['version']}`  ",
        f"Quality / operational weighting: `{metadata['config']['weights']['quality']:.0%} / {metadata['config']['weights']['operational']:.0%}`",
        "",
        "| Rank | Model | Composite | Quality | Operational | Keyword | Semantic | Ranking | Adherence | Warm tok/s | Cold load s | Min RAM GB | Status |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for index, item in enumerate(ranked, 1):
        score = item["score"]
        q = score["quality_components"]
        o = score["operational_components"]
        resources = item.get("resource_stats") or {}
        min_ram = resources.get("min_available_ram_gb")
        lines.append(
            f"| {index} | `{item['model']['name']}` | {score['composite_score']:.1f} | "
            f"{score['quality_score']:.1f} | {score['operational_score']:.1f} | "
            f"{q['keyword_quality']:.1f} | {q['semantic_quality']:.1f} | "
            f"{q['ranking_quality']:.1f} | {q['adherence']:.1f} | "
            f"{o['median_warm_tokens_per_second']:.1f} | {o['cold_load_seconds']:.1f} | "
            f"{min_ram:.1f} | {item['status']} |"
            if min_ram is not None
            else f"| {index} | `{item['model']['name']}` | {score['composite_score']:.1f} | "
            f"{score['quality_score']:.1f} | {score['operational_score']:.1f} | "
            f"{q['keyword_quality']:.1f} | {q['semantic_quality']:.1f} | "
            f"{q['ranking_quality']:.1f} | {q['adherence']:.1f} | "
            f"{o['median_warm_tokens_per_second']:.1f} | {o['cold_load_seconds']:.1f} | n/a | "
            f"{item['status']} |"
        )
    if not ranked:
        lines.append("| - | No eligible models | - | - | - | - | - | - | - | - | - | - | - |")
    lines.extend(
        [
            "",
            "## Resource peaks",
            "",
            "| Model | Min free RAM GB | Peak Ollama RSS GB | Peak system VRAM GB | Peak CPU % | Peak GPU % | Ollama model VRAM GB |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in ranked:
        resources = item.get("resource_stats") or {}
        residency = item.get("loaded_model_state") or {}
        model_vram = residency.get("size_vram_bytes")
        lines.append(
            f"| `{item['model']['name']}` | "
            f"{_fmt(resources.get('min_available_ram_gb'))} | "
            f"{_fmt(resources.get('peak_ollama_rss_gb'))} | "
            f"{_fmt_gb_from_mb(resources.get('peak_vram_used_mb'))} | "
            f"{_fmt(resources.get('peak_cpu_percent'))} | "
            f"{_fmt(resources.get('peak_gpu_utilization_percent'))} | "
            f"{_fmt_gb_from_bytes(model_vram)} |"
        )
    lines.extend(["", "## Skipped models", ""])
    if metadata["skipped_models"]:
        for item in metadata["skipped_models"]:
            lines.append(f"- `{item['model']}`: {item['reason']}")
    else:
        lines.append("None.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Composite is the headline workstation-fit score, not a universal model leaderboard. "
            "Quality is first-class; operational fitness captures switching cost, throughput, and memory headroom. "
            "Timeout/resource penalties intentionally prevent a high-quality model from winning when it makes the host unsafe or impractical.",
            "",
        ]
    )
    return "\n".join(lines)
