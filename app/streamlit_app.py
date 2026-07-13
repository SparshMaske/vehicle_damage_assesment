import base64
from io import BytesIO

import requests
import streamlit as st
from PIL import Image


API_URL = st.secrets.get("api_url", "http://127.0.0.1:8000/predict/structured") if hasattr(st, "secrets") else "http://127.0.0.1:8000/predict/structured"


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
            with st.spinner("Running damage assessment pipeline..."):
                uploaded.seek(0)
                response = requests.post(
                    API_URL,
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "image/png")},
                    timeout=60,
                )

            if response.status_code != 200:
                st.error(f"API request failed: {response.status_code} {response.text}")
            else:
                payload = response.json()
                annotated = decode_image(payload["annotated_image_base64"])
                st.image(annotated, caption="Annotated assessment", use_container_width=True)

                metric_a, metric_b, metric_c = st.columns(3)
                metric_a.metric("Routing", payload["routing_decision"])
                metric_b.metric("Overall Severity", payload["overall_severity"])
                metric_c.metric("Mode", payload["processing_mode"])

                st.text_area("Summary", value=payload["summary"], height=120)
                st.text_area("Reasoning", value=payload["reasoning"], height=100)
                if payload["recommended_next_actions"]:
                    st.write("Recommended Next Actions")
                    for action in payload["recommended_next_actions"]:
                        st.write(f"- {action}")
                st.caption(payload["estimate_note"])
    else:
        st.info("Upload a vehicle image and run the assessment to see the annotated output and text-first assessment.")
