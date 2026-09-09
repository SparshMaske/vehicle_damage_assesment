from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from api.main import app

client = TestClient(app)


def _png_bytes(size=(200, 200), color=(120, 120, 120)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["detector_mode"] in {"model", "mock"}
    assert body["severity_mode"] in {"model", "mock"}


def test_predict_rejects_non_image() -> None:
    response = client.post(
        "/predict",
        files={"file": ("note.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400


def test_predict_rejects_undecodable_image() -> None:
    response = client.post(
        "/predict",
        files={"file": ("broken.png", b"\x89PNG not really", "image/png")},
    )
    assert response.status_code == 400


def test_predict_returns_text_report() -> None:
    response = client.post(
        "/predict",
        files={"file": ("car.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 200
    assert "Vehicle Damage Assessment Report" in response.text


def test_predict_structured_returns_json() -> None:
    response = client.post(
        "/predict/structured",
        files={"file": ("car.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert "routing_decision" in payload
    assert "annotated_image_base64" in payload
