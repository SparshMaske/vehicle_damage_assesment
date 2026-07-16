from dataclasses import dataclass
from typing import Iterable


SEVERITY_RANK = {"Minor": 1, "Moderate": 2, "Severe": 3}

BASE_COST_RANGES = {
    "scratch": (200, 2500),
    "dent": (300, 3500),
    "crack/shatter": (250, 4000),
    "bumper_damage": (400, 5000),
    "broken_lamp": (150, 1500),
}

SEVERITY_MULTIPLIERS = {
    "Minor": (1.0, 1.5),
    "Moderate": (2.0, 3.5),
    "Severe": (4.0, 6.0),
}


@dataclass
class DamageAssessmentInput:
    damage_type: str
    location: str
    severity: str
    confidence: float
    bbox: list[int]


def cost_for_damage(damage_type: str, severity: str) -> str:
    base_min, base_max = BASE_COST_RANGES.get(damage_type, (300, 1000))
    multiplier_min, multiplier_max = SEVERITY_MULTIPLIERS.get(severity, (1.0, 1.5))
    adjusted_min = professional_round(base_min * multiplier_min)
    adjusted_max = professional_round(base_max * multiplier_max)
    return format_currency_range(adjusted_min, adjusted_max)


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
    regions = list(items)
    if not regions:
        return "$0-$0"

    min_total = 0
    max_total = 0
    for item in regions:
        base_min, base_max = BASE_COST_RANGES.get(item.damage_type, (300, 1000))
        multiplier_min, multiplier_max = SEVERITY_MULTIPLIERS.get(item.severity, (1.0, 1.5))
        min_total += base_min * multiplier_min
        max_total += base_max * multiplier_max

    complexity_multiplier = 1.0
    if len(regions) == 2:
        complexity_multiplier = 1.1
    elif len(regions) >= 3:
        complexity_multiplier = 1.2

    return format_currency_range(
        professional_round(min_total * complexity_multiplier),
        professional_round(max_total * complexity_multiplier),
    )


def route_claim(items: list[DamageAssessmentInput]) -> tuple[str, str, str]:
    if not items:
        return (
            "Straight-Through Eligible",
            "No visible damage indicators were identified in the submitted image set.",
            "$0-$0",
        )

    severities = [item.severity for item in items]
    count = len(items)
    overall = overall_severity(items)
    estimated_cost = combine_cost_ranges(items)

    if "Severe" in severities:
        return (
            "Needs Adjuster Review",
            "Structural integrity audit required due to severe damage indicators.",
            estimated_cost,
        )

    if count >= 3:
        return (
            "Needs Adjuster Review",
            "Multi-region repair complexity exceeds straight-through handling thresholds.",
            estimated_cost,
        )

    if "Moderate" in severities:
        return (
            "Needs Adjuster Review",
            "Field adjuster inspection recommended to validate repair scope and parts exposure.",
            estimated_cost,
        )

    if overall == "Minor" and count <= 2:
        return (
            "Straight-Through Eligible",
            "Low-complexity cosmetic damage profile supports straight-through triage.",
            estimated_cost,
        )

    return (
        "Needs Adjuster Review",
        "Mixed damage characteristics triggered the professional review safeguard.",
        estimated_cost,
    )


def professional_round(value: float) -> int:
    if value < 1000:
        return int(round(value / 50.0) * 50)
    return int(round(value / 100.0) * 100)


def format_currency_range(min_value: float, max_value: float) -> str:
    lower = int(min(min_value, max_value))
    upper = int(max(min_value, max_value))
    return f"${lower:,}-${upper:,}"
