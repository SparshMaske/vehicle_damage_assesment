from src.llm_reasoner import GeminiReasoner, extract_output_text, parse_reasoning_json


def test_reasoner_falls_back_without_api_key() -> None:
    reasoner = GeminiReasoner(enabled=True, api_key=None)
    output = reasoner.explain(
        {
            "damage_detections": [
                {
                    "type": "crack/shatter",
                    "location": "front_center",
                    "bbox": [0, 0, 10, 10],
                    "severity": "Severe",
                    "confidence": 0.61,
                    "estimated_cost_range": "$1,800-$4,000",
                }
            ],
            "overall_severity": "Severe",
            "routing_decision": "Needs Adjuster Review",
            "estimated_cost_range": "$1,800-$4,000",
            "processing_mode": "mock",
        }
    )
    assert output.provider == "gemini"
    assert output.mode in {"disabled", "fallback"}
    assert "CRITICAL_DAMAGE_DETECTED" in output.review_flags


def test_parse_reasoning_json_accepts_fenced_blocks() -> None:
    content = """```json
{"summary":"ok","explanation_trace":["a"],"review_flags":["b"],"recommended_next_actions":["c"]}
```"""
    parsed = parse_reasoning_json(content)
    assert parsed["summary"] == "ok"
    assert parsed["review_flags"] == ["b"]


def test_extract_output_text_reads_interactions_response() -> None:
    response = {
        "steps": [
            {
                "type": "model_output",
                "content": [
                    {
                        "type": "text",
                        "text": "{\"summary\":\"ok\",\"explanation_trace\":[],\"review_flags\":[],\"recommended_next_actions\":[]}",
                    }
                ],
            }
        ]
    }
    assert "\"summary\":\"ok\"" in extract_output_text(response)
