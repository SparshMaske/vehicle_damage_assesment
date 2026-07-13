from typing import Any


def build_text_report(result: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("Vehicle Damage Assessment Report")
    lines.append("")
    lines.append(f"Routing Decision: {result['routing_decision']}")
    lines.append(f"Overall Severity: {result['overall_severity']}")
    lines.append(f"Estimated Cost Range: {result['estimated_cost_range']}")
    lines.append(f"Processing Mode: {result['processing_mode']}")
    lines.append("")
    lines.append("Summary")
    lines.append(result["summary"])
    lines.append("")
    lines.append("Reasoning")
    lines.append(result["reasoning"])

    if result.get("explanation_trace"):
        lines.append("")
        lines.append("Explanation Trace")
        for item in result["explanation_trace"]:
            lines.append(f"- {item}")

    if result.get("review_flags"):
        lines.append("")
        lines.append("Review Flags")
        for item in result["review_flags"]:
            lines.append(f"- {item}")

    if result.get("recommended_next_actions"):
        lines.append("")
        lines.append("Recommended Next Actions")
        for item in result["recommended_next_actions"]:
            lines.append(f"- {item}")

    detections = result.get("damage_detections", [])
    lines.append("")
    lines.append("Detected Damage Regions")
    if not detections:
        lines.append("- No visible damage regions were detected.")
    else:
        for index, item in enumerate(detections, start=1):
            lines.append(
                f"- Region {index}: {item['type']} at {item['location']}, "
                f"severity {item['severity']}, confidence {item['confidence']:.2f}, "
                f"estimate {item['estimated_cost_range']}"
            )

    lines.append("")
    lines.append(f"Estimate Note: {result['estimate_note']}")
    return "\n".join(lines)
