from src.reporting import build_text_report


def test_build_text_report_contains_plain_text_sections() -> None:
    report = build_text_report(
        {
            "routing_decision": "Needs Adjuster Review",
            "overall_severity": "Moderate",
            "estimated_cost_range": "$700-$1,500",
            "processing_mode": "mock",
            "summary": "Needs Adjuster Review due to moderate front-side damage.",
            "reasoning": "Moderate damage requires human review for repair scope validation.",
            "explanation_trace": ["Detected 1 damage region.", "Overall severity assessed as Moderate."],
            "review_flags": ["low_detection_confidence"],
            "recommended_next_actions": ["Request one additional side-angle image."],
            "damage_detections": [
                {
                    "type": "dent",
                    "location": "left_side",
                    "severity": "Moderate",
                    "confidence": 0.62,
                    "estimated_cost_range": "$700-$1,500",
                }
            ],
            "estimate_note": "Illustrative rule-based estimate only.",
        }
    )
    assert "Vehicle Damage Assessment Report" in report
    assert "Routing Decision: Needs Adjuster Review" in report
    assert "Detected Damage Regions" in report
