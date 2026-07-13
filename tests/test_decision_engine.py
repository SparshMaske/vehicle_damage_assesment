from src.decision_engine import DamageAssessmentInput, combine_cost_ranges, overall_severity, route_claim


def test_no_damage_routes_straight_through() -> None:
    decision, reasoning, cost = route_claim([])
    assert decision == "Straight-Through Eligible"
    assert reasoning == "No visible damage detected."
    assert cost == "$0-$0"


def test_minor_low_count_routes_straight_through() -> None:
    items = [
        DamageAssessmentInput("scratch", "Minor", 0.8, [0, 0, 10, 10]),
        DamageAssessmentInput("dent", "Minor", 0.7, [10, 10, 20, 20]),
    ]
    decision, reasoning, _ = route_claim(items)
    assert decision == "Straight-Through Eligible"
    assert "minor" in reasoning.lower()


def test_severe_routes_to_adjuster() -> None:
    items = [DamageAssessmentInput("dent", "Severe", 0.9, [0, 0, 10, 10])]
    decision, reasoning, _ = route_claim(items)
    assert decision == "Needs Adjuster Review"
    assert "severe" in reasoning.lower()


def test_three_regions_route_to_adjuster() -> None:
    items = [
        DamageAssessmentInput("scratch", "Minor", 0.8, [0, 0, 10, 10]),
        DamageAssessmentInput("dent", "Minor", 0.7, [10, 10, 20, 20]),
        DamageAssessmentInput("broken_lamp", "Minor", 0.75, [20, 20, 30, 30]),
    ]
    decision, reasoning, _ = route_claim(items)
    assert decision == "Needs Adjuster Review"
    assert "three or more" in reasoning.lower()


def test_overall_severity_uses_highest_rank() -> None:
    items = [
        DamageAssessmentInput("scratch", "Minor", 0.8, [0, 0, 10, 10]),
        DamageAssessmentInput("dent", "Moderate", 0.7, [10, 10, 20, 20]),
    ]
    assert overall_severity(items) == "Moderate"


def test_cost_ranges_collapse_when_identical() -> None:
    items = [
        DamageAssessmentInput("scratch", "Minor", 0.8, [0, 0, 10, 10]),
        DamageAssessmentInput("scratch", "Minor", 0.7, [10, 10, 20, 20]),
    ]
    assert combine_cost_ranges(items) == "$200-$500"
