import json

from model_benchmark.fixtures import load_suite
from model_benchmark.scoring import score_case


def case(case_id: str):
    return next(item for item in load_suite().cases if item.id == case_id)


def test_keyword_score_rewards_signal_and_penalizes_chaff() -> None:
    fixture = case("keyword_resume_chaff")
    good = score_case(
        fixture,
        json.dumps({"keywords": ["FHIR", "HL7", "healthcare analytics", "workflow automation", "product discovery"]}),
    )
    noisy = score_case(
        fixture,
        json.dumps({"keywords": ["leadership", "experience", "team", "FHIR", "results", "stakeholders"]}),
    )
    assert good.quality_score > 80
    assert good.quality_score > noisy.quality_score + 40
    assert noisy.details["chaff_rate"] > 0.5


def test_semantic_exact_fixture_scores_high() -> None:
    fixture = case("semantic_job_posting")
    response = json.dumps(fixture.expected["values"])
    score = score_case(fixture, response)
    assert score.accuracy_score == 100
    assert score.adherence_score == 100


def test_ranking_rewards_relative_order_not_exact_numbers() -> None:
    fixture = case("ranking_product_fit")
    response = json.dumps(
        {
            "ranking": [
                {"id": "A", "score": 91, "reason": "Healthcare analytics and FHIR align directly."},
                {"id": "B", "score": 63, "reason": "Product discovery fits but ERP migration is a gap."},
                {"id": "C", "score": 38, "reason": "Consumer gaming is outside the supplied domain experience."},
            ]
        }
    )
    score = score_case(fixture, response)
    assert score.accuracy_score == 100
    assert score.quality_score >= 90
