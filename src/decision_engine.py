from dataclasses import dataclass
from typing import Iterable


SEVERITY_RANK = {"Minor": 1, "Moderate": 2, "Severe": 3}

COST_LOOKUP = {
    ("scratch", "Minor"): "$200-$500",
    ("scratch", "Moderate"): "$500-$1,200",
    ("scratch", "Severe"): "$1,200-$2,500",
    ("dent", "Minor"): "$300-$700",
    ("dent", "Moderate"): "$700-$1,500",
    ("dent", "Severe"): "$1,500-$3,500",
    ("crack/shatter", "Minor"): "$250-$800",
    ("crack/shatter", "Moderate"): "$800-$1,800",
    ("crack/shatter", "Severe"): "$1,800-$4,000",
    ("bumper_damage", "Minor"): "$400-$900",
    ("bumper_damage", "Moderate"): "$900-$2,000",
    ("bumper_damage", "Severe"): "$2,000-$5,000",
    ("broken_lamp", "Minor"): "$150-$350",
    ("broken_lamp", "Moderate"): "$350-$800",
    ("broken_lamp", "Severe"): "$800-$1,500",
}


@dataclass
class DamageAssessmentInput:
    damage_type: str
    location: str
    severity: str
    confidence: float
    bbox: list[int]


def cost_for_damage(damage_type: str, severity: str) -> str:
    return COST_LOOKUP.get((damage_type, severity), "$300-$1,000")


def overall_severity(items: Iterable[DamageAssessmentInput]) -> str:
    max_rank = 0
    label = "None"
    for item in items:
        rank = SEVERITY_RANK.get(item.severity, 0)
        if rank > max_rank:
            max_rank = rank
            label = item.severity
    return label


def combine_cost_ranges(items: Iterable[DamageAssessmentInput]) -> str:
    ranges = [cost_for_damage(item.damage_type, item.severity) for item in items]
    if not ranges:
        return "$0-$0"
    if len(set(ranges)) == 1:
        return ranges[0]
    return "Varies by part and repair complexity"


def route_claim(items: list[DamageAssessmentInput]) -> tuple[str, str, str]:
    if not items:
        return (
            "Straight-Through Eligible",
            "No visible damage detected.",
            "$0-$0",
        )

    severities = [item.severity for item in items]
    count = len(items)
    overall = overall_severity(items)
    estimated_cost = combine_cost_ranges(items)

    if "Severe" in severities:
        return (
            "Needs Adjuster Review",
            "At least one severe damage region was detected.",
            estimated_cost,
        )

    if count >= 3:
        return (
            "Needs Adjuster Review",
            "Three or more damage regions increase claim complexity.",
            estimated_cost,
        )

    if "Moderate" in severities:
        return (
            "Needs Adjuster Review",
            "Moderate damage requires human review for repair scope validation.",
            estimated_cost,
        )

    if overall == "Minor" and count <= 2:
        return (
            "Straight-Through Eligible",
            "One or two minor regions detected with no severe indicators.",
            estimated_cost,
        )

    return (
        "Needs Adjuster Review",
        "Fallback review route triggered by mixed damage characteristics.",
        estimated_cost,
    )
