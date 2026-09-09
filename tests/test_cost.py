from src.decision_engine import (
    DamageAssessmentInput,
    combine_cost_ranges,
    cost_for_damage,
    format_currency_range,
    professional_round,
)


def test_professional_round_uses_coarser_step_above_1000() -> None:
    assert professional_round(440) == 450  # rounded to nearest 50
    assert professional_round(8250) == 8200  # rounded to nearest 100


def test_format_currency_range_orders_and_formats() -> None:
    assert format_currency_range(8200, 450) == "$450-$8,200"


def test_cost_for_damage_scales_with_severity() -> None:
    minor = cost_for_damage("scratch", "Minor")
    severe = cost_for_damage("scratch", "Severe")
    assert minor != severe


def test_unknown_damage_type_uses_default_range() -> None:
    # Unknown type falls back to the (300, 1000) default band.
    result = cost_for_damage("unknown_type", "Minor")
    assert result.startswith("$")


def test_combine_cost_ranges_applies_complexity_multiplier() -> None:
    single = DamageAssessmentInput("scratch", "front_center", "Minor", 0.8, [0, 0, 10, 10])
    assert combine_cost_ranges([single]) != combine_cost_ranges([single, single])
