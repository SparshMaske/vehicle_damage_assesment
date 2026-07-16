import json
import os
from dataclasses import dataclass
from typing import Any

import requests


DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


@dataclass
class ReasoningOutput:
    summary: str
    explanation_trace: list[str]
    review_flags: list[str]
    recommended_next_actions: list[str]
    provider: str
    mode: str


class GeminiReasoner:
    def __init__(
        self,
        enabled: bool | None = None,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 20.0,
    ) -> None:
        self.enabled = (
            enabled
            if enabled is not None
            else os.getenv("ENABLE_GEMINI_REASONER", "false").lower() == "true"
        )
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model or os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        self.base_url = base_url or os.getenv("GEMINI_BASE_URL", DEFAULT_GEMINI_BASE_URL)
        self.timeout_seconds = timeout_seconds

    def explain(self, payload: dict[str, Any]) -> ReasoningOutput:
        if not self.enabled or not self.api_key:
            return self._fallback(payload, mode="disabled")

        try:
            return self._remote_explain(payload)
        except Exception:
            return self._fallback(payload, mode="fallback")

    def _remote_explain(self, payload: dict[str, Any]) -> ReasoningOutput:
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        prompt = build_reasoning_prompt(payload)
        body = {
            "model": self.model,
            "input": prompt,
            "store": False,
        }
        response = requests.post(
            self.base_url,
            headers=headers,
            json=body,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        content = extract_output_text(data)
        parsed = parse_reasoning_json(content)
        return ReasoningOutput(
            summary=parsed["summary"],
            explanation_trace=parsed["explanation_trace"],
            review_flags=parsed["review_flags"],
            recommended_next_actions=parsed["recommended_next_actions"],
            provider="gemini",
            mode="remote",
        )

    def _fallback(self, payload: dict[str, Any], mode: str) -> ReasoningOutput:
        detections = payload["damage_detections"]
        routing = payload["routing_decision"]
        overall = payload["overall_severity"]
        flags: list[str] = []
        actions: list[str] = []
        trace = [
            f"Detected {len(detections)} damage region(s) across the submitted evidence set.",
            f"Aggregate severity is assessed as {overall}.",
            f"System routing decision is {routing}.",
        ]

        if any(item["confidence"] < 0.65 for item in detections):
            flags.append("LOW_DETECTION_CONFIDENCE")
            actions.append("Request an additional supporting angle to validate the low-confidence damage region.")

        if any(item["severity"] == "Severe" for item in detections):
            flags.append("CRITICAL_DAMAGE_DETECTED")
            actions.append("Escalate for field adjuster inspection and structural verification.")

        if any(item["type"] in {"crack/shatter", "broken_lamp"} for item in detections):
            flags.append("SAFETY_COMPONENT_COMPROMISED")
            actions.append("Request a close-up image of the affected glass or lamp assembly for safety review.")

        if not detections:
            actions.append("Request a clearer photo set if the claimant reports visible damage not captured in the submission.")

        summary = (
            f"{routing}: {len(detections)} region(s) were identified with aggregate severity {overall}. "
            f"Professional fallback reasoning is active because Gemini reasoning is {mode}."
        )
        return ReasoningOutput(
            summary=summary,
            explanation_trace=trace,
            review_flags=flags,
            recommended_next_actions=actions,
            provider="gemini",
            mode=mode,
        )


def build_reasoning_prompt(payload: dict[str, Any]) -> str:
    schema = {
        "summary": "string",
        "explanation_trace": ["string"],
        "review_flags": ["string"],
        "recommended_next_actions": ["string"],
    }
    return (
        "You are a senior auto-insurance claims adjuster assisting a claims triage backend.\n"
        "Use only the structured evidence below.\n"
        "Focus on aggregate severity, damage localization, system routing decision, and field follow-up actions.\n"
        "Keep terminology professional and objective.\n"
        "Return strict JSON only.\n"
        f"Required schema: {json.dumps(schema)}\n"
        f"Evidence: {json.dumps(payload)}"
    )


def extract_output_text(response: dict[str, Any]) -> str:
    steps = response.get("steps", [])
    for step in reversed(steps):
        if step.get("type") == "model_output":
            content = step.get("content", [])
            if not content:
                continue
            first = content[0]
            text = first.get("text")
            if text:
                return text
    raise ValueError("Gemini response did not include output text.")


def parse_reasoning_json(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    parsed = json.loads(stripped)
    return {
        "summary": parsed.get("summary", ""),
        "explanation_trace": parsed.get("explanation_trace", []),
        "review_flags": parsed.get("review_flags", []),
        "recommended_next_actions": parsed.get("recommended_next_actions", []),
    }
