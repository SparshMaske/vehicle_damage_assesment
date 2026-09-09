import base64
import os
from io import BytesIO

import requests
import streamlit as st
from PIL import Image


def _default_api_url() -> str:
    env_url = os.getenv("API_URL")
    if env_url:
        return env_url
    if hasattr(st, "secrets"):
        try:
            return st.secrets.get("api_url", "http://127.0.0.1:8000/predict/structured")
        except Exception:
            pass
    return "http://127.0.0.1:8000/predict/structured"


API_URL = _default_api_url()


def decode_image(encoded: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")


st.set_page_config(page_title="Vehicle Damage Assessment", page_icon="🚗", layout="wide")

st.title("AI-Powered Vehicle Damage Assessment")
st.caption("Claims triage demo for straight-through processing versus adjuster review.")

with st.sidebar:
    st.subheader("Pipeline")
    st.write("1. Damage detection")
    st.write("2. Severity classification")
    st.write("3. Rules-based routing decision")
    st.divider()
    st.caption(f"API endpoint: `{API_URL}`")
    st.caption("Cost ranges are illustrative and not sourced from live estimating platforms.")

left, right = st.columns([1, 1], gap="large")

with left:
    uploaded = st.file_uploader("Upload a damaged vehicle photo", type=["jpg", "jpeg", "png"])
    analyze = st.button("Analyze Claim", type="primary", use_container_width=True)

    if uploaded is not None:
        source_image = Image.open(uploaded).convert("RGB")
        st.image(source_image, caption="Uploaded image", use_container_width=True)
    else:
        source_image = None

with right:
    st.subheader("Assessment Output")
    if analyze:
        if source_image is None:
            st.warning("Upload an image before running the assessment.")
        else:
            payload = None
            try:
                with st.spinner("Running damage assessment pipeline..."):
                    uploaded.seek(0)
                    response = requests.post(
                        API_URL,
                        files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "image/png")},
                        timeout=60,
                    )
            except requests.exceptions.ConnectionError:
                st.error(
                    f"Could not reach the assessment API at `{API_URL}`. "
                    "Start it with `uvicorn api.main:app --reload`, or set the "
                    "`API_URL` environment variable to point at a running instance."
                )
            except requests.exceptions.Timeout:
                st.error("The assessment API timed out. Try again or check the API logs.")
            except requests.exceptions.RequestException as exc:
                st.error(f"Request to the assessment API failed: {exc}")
            else:
                if response.status_code != 200:
                    st.error(f"API request failed: {response.status_code} {response.text}")
                else:
                    payload = response.json()

            if payload is not None:
                annotated = decode_image(payload["annotated_image_base64"])
                st.image(annotated, caption="Annotated assessment", use_container_width=True)

                metric_a, metric_b, metric_c = st.columns(3)
                metric_a.metric("Routing", payload["routing_decision"])
                metric_b.metric("Overall Severity", payload["overall_severity"])
                metric_c.metric("Est. Cost", payload["estimated_cost_range"])

                st.caption(
                    f"Processing mode: {payload['processing_mode']} · "
                    f"Reasoning: {payload['reasoning_provider']} ({payload['reasoning_mode']})"
                )

                st.text_area("Summary", value=payload["summary"], height=120)
                st.text_area("Reasoning", value=payload["reasoning"], height=100)

                detections = payload.get("damage_detections", [])
                if detections:
                    st.write("Detected Damage Regions")
                    st.dataframe(
                        [
                            {
                                "Type": item["type"],
                                "Location": item["location"],
                                "Severity": item["severity"],
                                "Confidence": f"{item['confidence']:.2f}",
                                "Estimate": item["estimated_cost_range"],
                            }
                            for item in detections
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("No visible damage regions were detected.")

                if payload.get("review_flags"):
                    st.write("Review Flags")
                    for flag in payload["review_flags"]:
                        st.warning(flag)

                if payload.get("recommended_next_actions"):
                    st.write("Recommended Next Actions")
                    for action in payload["recommended_next_actions"]:
                        st.write(f"- {action}")

                if payload.get("explanation_trace"):
                    with st.expander("Explanation trace"):
                        for step in payload["explanation_trace"]:
                            st.write(f"- {step}")

                st.caption(payload["estimate_note"])
    else:
        st.info("Upload a vehicle image and run the assessment to see the annotated output and text-first assessment.")
