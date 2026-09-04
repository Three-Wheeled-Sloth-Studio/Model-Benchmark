from __future__ import annotations

import json
import re
from dataclasses import dataclass
from statistics import mean, median
from typing import Any

from model_benchmark.config import BenchmarkConfig
from model_benchmark.fixtures import BenchmarkCase


@dataclass(frozen=True, slots=True)
class CaseScore:
    accuracy_score: float
    adherence_score: float
    quality_score: float
    details: dict[str, Any]


def _clamp(value: float) -> float:
    return round(max(0.0, min(100.0, value)), 3)


def _normalize(value: str) -> str:
    value = value.casefold().strip()
    value = re.sub(r"[^a-z0-9+#./ -]+", "", value)
    return " ".join(value.split())


def _f1(predicted: set[str], expected: set[str]) -> tuple[float, float, float]:
    if not predicted and not expected:
        return 1.0, 1.0, 1.0
    true_positive = len(predicted & expected)
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def parse_response(response_text: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as error:
        return None, str(error)
    if not isinstance(payload, dict):
        return None, "response root is not an object"
    return payload, None


def score_case(case: BenchmarkCase, response_text: str) -> CaseScore:
    payload, parse_error = parse_response(response_text)
    if payload is None:
        return CaseScore(
            accuracy_score=0.0,
            adherence_score=0.0,
            quality_score=0.0,
            details={"parse_error": parse_error},
        )
    if case.category == "keyword":
        return _score_keyword(case, payload)
    if case.category == "semantic":
        return _score_semantic(case, payload)
    if case.category == "ranking":
        return _score_ranking(case, payload)
    raise ValueError(f"unknown benchmark category: {case.category}")


def _score_keyword(case: BenchmarkCase, payload: dict[str, Any]) -> CaseScore:
    expected = case.expected
    raw_keywords = payload.get("keywords")
    if not isinstance(raw_keywords, list) or not all(isinstance(item, str) for item in raw_keywords):
        return CaseScore(0.0, 25.0, 0.0, {"error": "keywords must be a string array"})

    predicted = [_normalize(item) for item in raw_keywords if _normalize(item)]
    aliases: dict[str, str] = {}
    for canonical in expected["gold_keywords"]:
        aliases[_normalize(canonical)] = _normalize(canonical)
    for canonical, values in expected.get("accepted_aliases", {}).items():
        for value in values:
            aliases[_normalize(value)] = _normalize(canonical)

    canonical_predicted = [aliases.get(item, item) for item in predicted]
    gold = {_normalize(item) for item in expected["gold_keywords"]}
    predicted_set = set(canonical_predicted)
    precision, recall, f1 = _f1(predicted_set, gold)
    forbidden = {_normalize(item) for item in expected.get("forbidden_terms", [])}
    chaff_items = [item for item in canonical_predicted if item not in gold or item in forbidden]
    chaff_rate = len(chaff_items) / len(canonical_predicted) if canonical_predicted else 0.0
    accuracy = f1 * 100.0
    quality = accuracy - (55.0 * chaff_rate)

    max_items = int(expected.get("max_keywords", 8))
    adherence = 40.0
    adherence += 20.0 if set(payload) == {"keywords"} else 10.0
    adherence += 20.0 if len(raw_keywords) <= max_items else 0.0
    adherence += 20.0 if not any(item in forbidden for item in predicted) else 0.0

    return CaseScore(
        accuracy_score=_clamp(accuracy),
        adherence_score=_clamp(adherence),
        quality_score=_clamp(quality),
        details={
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "chaff_rate": round(chaff_rate, 4),
            "chaff_items": chaff_items,
            "predicted": predicted,
        },
    )


def _score_semantic(case: BenchmarkCase, payload: dict[str, Any]) -> CaseScore:
    expected_values = case.expected["values"]
    field_weights = case.expected.get("field_weights") or {
        key: 1.0 for key in expected_values
    }
    field_scores: dict[str, float] = {}
    for field, expected in expected_values.items():
        actual = payload.get(field)
        if isinstance(expected, list):
            predicted_set = {_normalize(str(item)) for item in actual} if isinstance(actual, list) else set()
            expected_set = {_normalize(str(item)) for item in expected}
            _, _, f1 = _f1(predicted_set, expected_set)
            field_scores[field] = f1
        elif isinstance(expected, (int, float)):
            try:
                field_scores[field] = 1.0 if float(actual) == float(expected) else 0.0
            except (TypeError, ValueError):
                field_scores[field] = 0.0
        else:
            field_scores[field] = 1.0 if _normalize(str(actual)) == _normalize(str(expected)) else 0.0

    weight_total = sum(float(field_weights.get(key, 1.0)) for key in expected_values)
    weighted = sum(
        field_scores[key] * float(field_weights.get(key, 1.0)) for key in expected_values
    )
    accuracy = 100.0 * weighted / weight_total if weight_total else 0.0

    required = set(expected_values)
    adherence = 40.0
    adherence += 30.0 * (len(required & set(payload)) / len(required) if required else 1.0)
    adherence += 20.0 if set(payload) <= required else 5.0
    type_ok = all(
        isinstance(payload.get(key), list) if isinstance(expected, list) else payload.get(key) is not None
        for key, expected in expected_values.items()
    )
    adherence += 10.0 if type_ok else 0.0

    return CaseScore(
        accuracy_score=_clamp(accuracy),
        adherence_score=_clamp(adherence),
        quality_score=_clamp(accuracy),
        details={"field_scores": {key: round(value * 100, 3) for key, value in field_scores.items()}},
    )


def _score_ranking(case: BenchmarkCase, payload: dict[str, Any]) -> CaseScore:
    ranking = payload.get("ranking")
    if not isinstance(ranking, list):
        return CaseScore(0.0, 20.0, 0.0, {"error": "ranking must be an array"})
    rows = [row for row in ranking if isinstance(row, dict)]
    ids = [str(row.get("id", "")) for row in rows]
    expected_order = [str(item) for item in case.expected["order"]]
    actual_pos = {value: index for index, value in enumerate(ids)}
    pairs = 0
    correct = 0
    for left_index, left in enumerate(expected_order):
        for right in expected_order[left_index + 1 :]:
            pairs += 1
            if left in actual_pos and right in actual_pos and actual_pos[left] < actual_pos[right]:
                correct += 1
    pairwise = correct / pairs if pairs else 1.0

    evidence_map = case.expected.get("reason_evidence", {})
    evidence_hits = 0
    evidence_total = 0
    unsupported_hits: list[str] = []
    forbidden = [_normalize(item) for item in case.expected.get("forbidden_claims", [])]
    for row in rows:
        candidate_id = str(row.get("id", ""))
        reason = _normalize(str(row.get("reason", "")))
        evidence = [_normalize(item) for item in evidence_map.get(candidate_id, [])]
        if evidence:
            evidence_total += 1
            if any(item in reason for item in evidence):
                evidence_hits += 1
        for claim in forbidden:
            if claim and claim in reason:
                unsupported_hits.append(claim)
    reasoning = evidence_hits / evidence_total if evidence_total else 1.0
    reasoning = max(0.0, reasoning - 0.25 * len(set(unsupported_hits)))
    accuracy = 100.0 * pairwise
    quality = 100.0 * (0.75 * pairwise + 0.25 * reasoning)

    exact_ids = set(ids) == set(expected_order) and len(ids) == len(expected_order)
    scores_valid = all(
        isinstance(row.get("score"), (int, float)) and 0 <= float(row["score"]) <= 100
        for row in rows
    )
    reasons_valid = all(bool(str(row.get("reason", "")).strip()) for row in rows)
    adherence = 40.0 + (30.0 if exact_ids else 0.0) + (15.0 if scores_valid else 0.0) + (15.0 if reasons_valid else 0.0)

    return CaseScore(
        accuracy_score=_clamp(accuracy),
        adherence_score=_clamp(adherence),
        quality_score=_clamp(quality),
        details={
            "pairwise_order_accuracy": round(pairwise, 4),
            "reasoning_evidence_coverage": round(reasoning, 4),
            "actual_order": ids,
            "expected_order": expected_order,
            "unsupported_claims": sorted(set(unsupported_hits)),
        },
    )


def piecewise(value: float, points: list[tuple[float, float]]) -> float:
    points = sorted(points)
    if value <= points[0][0]:
        return points[0][1]
    if value >= points[-1][0]:
        return points[-1][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:], strict=True):
        if x0 <= value <= x1:
            fraction = (value - x0) / (x1 - x0)
            return y0 + fraction * (y1 - y0)
    raise AssertionError("unreachable")


def aggregate_model_score(model_result: dict[str, Any], config: BenchmarkConfig) -> dict[str, Any]:
    successful = [item for item in model_result["tests"] if item.get("status") == "succeeded"]
    categories: dict[str, list[float]] = {"keyword": [], "semantic": [], "ranking": []}
    adherence: list[float] = []
    for item in successful:
        score = item["score"]
        categories[item["category"]].append(float(score["quality_score"]))
        adherence.append(float(score["adherence_score"]))

    keyword = mean(categories["keyword"]) if categories["keyword"] else 0.0
    semantic = mean(categories["semantic"]) if categories["semantic"] else 0.0
    ranking = mean(categories["ranking"]) if categories["ranking"] else 0.0
    adherence_score = mean(adherence) if adherence else 0.0
    w = config.weights
    quality = (
        keyword * w.keyword_quality
        + semantic * w.semantic_quality
        + ranking * w.ranking_quality
        + adherence_score * w.adherence
    )

    cold_ms = float((model_result.get("cold_probe") or {}).get("metrics", {}).get("load_duration_ms") or 0.0)
    cold_seconds = cold_ms / 1000.0
    cold_score = piecewise(
        cold_seconds,
        [(0, 100), (2, 100), (10, 80), (30, 50), (120, 10), (300, 0)],
    )
    tps_values = [
        float(item["metrics"]["eval_tokens_per_second"])
        for item in successful
        if item.get("metrics", {}).get("eval_tokens_per_second") is not None
    ]
    warm_probe_metrics = (model_result.get("warm_probe") or {}).get("metrics", {})
    warm_probe_tps = warm_probe_metrics.get("eval_tokens_per_second")
    warm_load_ms = warm_probe_metrics.get("load_duration_ms")
    if warm_probe_tps is not None:
        tps_values.append(float(warm_probe_tps))
    warm_tps = median(tps_values) if tps_values else 0.0
    throughput_score = piecewise(
        warm_tps,
        [(0, 0), (1, 10), (5, 35), (10, 55), (25, 80), (50, 100)],
    )

    resource_stats = model_result.get("resource_stats") or {}
    min_available = resource_stats.get("min_available_ram_gb")
    if min_available is None:
        memory_score = 50.0
    else:
        floor = config.min_available_ram_gb
        memory_score = piecewise(
            float(min_available),
            [(0, 0), (floor, 0), (floor * 1.25, 35), (floor * 2, 85), (floor * 3, 100)],
        )
    operational = (
        throughput_score * w.warm_throughput
        + cold_score * w.cold_load
        + memory_score * w.memory_headroom
    )
    raw_composite = quality * w.quality + operational * w.operational

    final_composite = raw_composite
    penalties: list[str] = []
    statuses = [item.get("status") for item in model_result["tests"]]
    if model_result.get("status") == "resource_abort":
        final_composite = min(final_composite, 20.0)
        penalties.append("resource_abort_cap_20")
    elif model_result.get("status") == "timeout":
        final_composite = min(final_composite, 40.0)
        penalties.append("model_timeout_cap_40")
    elif "timeout" in statuses:
        final_composite = min(final_composite, 60.0)
        penalties.append("test_timeout_cap_60")
    if min_available is not None and float(min_available) < config.min_available_ram_gb * 1.25:
        final_composite = min(final_composite, 70.0)
        penalties.append("severe_memory_pressure_cap_70")
    failure_count = sum(status != "succeeded" for status in statuses)
    if statuses and failure_count / len(statuses) > 0.25:
        final_composite = min(final_composite, 50.0)
        penalties.append("incomplete_suite_cap_50")

    return {
        "quality_score": _clamp(quality),
        "operational_score": _clamp(operational),
        "composite_score_raw": _clamp(raw_composite),
        "composite_score": _clamp(final_composite),
        "quality_components": {
            "keyword_quality": _clamp(keyword),
            "semantic_quality": _clamp(semantic),
            "ranking_quality": _clamp(ranking),
            "adherence": _clamp(adherence_score),
        },
        "operational_components": {
            "cold_load_score": _clamp(cold_score),
            "warm_throughput_score": _clamp(throughput_score),
            "memory_headroom_score": _clamp(memory_score),
            "median_warm_tokens_per_second": round(warm_tps, 3),
            "cold_load_seconds": round(cold_seconds, 3),
            "warm_load_seconds": (
                round(float(warm_load_ms) / 1000.0, 3) if warm_load_ms is not None else None
            ),
        },
        "penalties": penalties,
    }
