import base64
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from src.decision_engine import DamageAssessmentInput, cost_for_damage, overall_severity, route_claim
from src.detection import DamageDetector
from src.llm_reasoner import GeminiReasoner
from src.severity import SeverityClassifier


class DamageAssessmentPipeline:
    def __init__(
        self,
        detector_weights: str | None = None,
        severity_weights: str | None = None,
    ) -> None:
        self.detector = DamageDetector(detector_weights)
        self.classifier = SeverityClassifier(severity_weights)
        self.reasoner = GeminiReasoner()

    def run(self, image: Image.Image) -> dict[str, Any]:
        detections, detection_mode = self.detector.predict(image)
        assessment_inputs: list[DamageAssessmentInput] = []
        enriched_detections: list[dict[str, Any]] = []
        modes = {detection_mode}

        for detection in detections:
            detection.bbox = clamp_bbox(detection.bbox, image.size)
            if not is_valid_bbox(detection.bbox):
                continue
            crop = image.crop(tuple(detection.bbox))
            severity_result, severity_mode = self.classifier.predict(crop)
            modes.add(severity_mode)

            assessment = DamageAssessmentInput(
                damage_type=detection.damage_type,
                location=infer_location(detection.bbox, image.size),
                severity=severity_result.label,
                confidence=detection.confidence,
                bbox=detection.bbox,
            )
            assessment_inputs.append(assessment)
            enriched_detections.append(
                {
                    "type": detection.damage_type,
                    "location": assessment.location,
                    "bbox": detection.bbox,
                    "severity": severity_result.label,
                    "confidence": round(detection.confidence, 3),
                    "estimated_cost_range": cost_for_damage(detection.damage_type, severity_result.label),
                }
            )

        routing_decision, reasoning, estimated_cost = route_claim(assessment_inputs)
        annotated = annotate_image(image, enriched_detections)
        encoded = image_to_base64(annotated)

        if not assessment_inputs:
            overall = "None"
        else:
            overall = overall_severity(assessment_inputs)

        processing_mode = "model" if modes == {"model"} else "mock"
        reasoning_payload = {
            "damage_detections": enriched_detections,
            "overall_severity": overall,
            "routing_decision": routing_decision,
            "estimated_cost_range": estimated_cost,
            "processing_mode": processing_mode,
        }
        llm_reasoning = self.reasoner.explain(reasoning_payload)

        return {
            "damage_detections": enriched_detections,
            "overall_severity": overall,
            "routing_decision": routing_decision,
            "estimated_cost_range": estimated_cost,
            "estimate_note": "Illustrative rule-based estimate only, not a real actuarial or repair-platform quote.",
            "reasoning": reasoning,
            "summary": llm_reasoning.summary,
            "explanation_trace": llm_reasoning.explanation_trace,
            "review_flags": llm_reasoning.review_flags,
            "recommended_next_actions": llm_reasoning.recommended_next_actions,
            "reasoning_provider": llm_reasoning.provider,
            "reasoning_mode": llm_reasoning.mode,
            "processing_mode": processing_mode,
            "annotated_image_base64": encoded,
        }


def clamp_bbox(bbox: list[int], image_size: tuple[int, int]) -> list[int]:
    width, height = image_size
    x1, y1, x2, y2 = (int(round(value)) for value in bbox)
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = min(max(x1, 0), width)
    x2 = min(max(x2, 0), width)
    y1 = min(max(y1, 0), height)
    y2 = min(max(y2, 0), height)
    return [x1, y1, x2, y2]


def is_valid_bbox(bbox: list[int]) -> bool:
    x1, y1, x2, y2 = bbox
    return x2 > x1 and y2 > y1


def annotate_image(image: Image.Image, detections: list[dict[str, Any]]) -> Image.Image:
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    for item in detections:
        bbox = item["bbox"]
        label = f'{item["type"]} | {item["severity"]} | {item["confidence"]:.2f}'
        draw.rectangle(bbox, outline="red", width=3)

        # Draw a filled label chip so white text stays legible over any background.
        text_bbox = draw.textbbox((0, 0), label)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        chip_top = max(0, bbox[1] - text_h - 4)
        draw.rectangle(
            [bbox[0], chip_top, bbox[0] + text_w + 6, chip_top + text_h + 4],
            fill="red",
        )
        draw.text((bbox[0] + 3, chip_top + 2), label, fill="white")
    return annotated


def image_to_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def infer_location(bbox: list[int], image_size: tuple[int, int]) -> str:
    width, height = image_size
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    horizontal = "left" if center_x < width / 3 else "right" if center_x > (2 * width) / 3 else "center"
    vertical = "front" if center_y < height / 3 else "rear" if center_y > (2 * height) / 3 else "mid"

    if vertical == "mid" and horizontal == "center":
        return "center_body"
    if vertical == "mid":
        return f"{horizontal}_side"
    if horizontal == "center":
        return f"{vertical}_center"
    return f"{vertical}_{horizontal}"


def default_pipeline() -> DamageAssessmentPipeline:
    root = Path(__file__).resolve().parent.parent
    detector_weights = root / "models" / "weights" / "detector.pt"
    severity_weights = root / "models" / "weights" / "severity.pth"
    return DamageAssessmentPipeline(
        detector_weights=str(detector_weights),
        severity_weights=str(severity_weights),
    )
