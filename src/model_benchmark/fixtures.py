from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    id: str
    category: str
    prompt: str
    schema: dict[str, Any]
    expected: dict[str, Any]


@dataclass(frozen=True, slots=True)
class BenchmarkSuite:
    suite_id: str
    suite_version: str
    description: str
    cases: tuple[BenchmarkCase, ...]

    @property
    def representative_case(self) -> BenchmarkCase:
        return next(case for case in self.cases if case.category == "keyword")


def load_suite(path: str | None = None) -> BenchmarkSuite:
    if path:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    else:
        resource = files("model_benchmark").joinpath("data/baseline.json")
        payload = json.loads(resource.read_text(encoding="utf-8"))
    cases = tuple(
        BenchmarkCase(
            id=item["id"],
            category=item["category"],
            prompt=item["prompt"],
            schema=item["schema"],
            expected=item["expected"],
        )
        for item in payload["cases"]
    )
    return BenchmarkSuite(
        suite_id=payload["suite_id"],
        suite_version=payload["suite_version"],
        description=payload["description"],
        cases=cases,
    )
