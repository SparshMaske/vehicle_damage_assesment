import os

from PIL import Image

from src.pipeline import (
    clamp_bbox,
    infer_location,
    is_valid_bbox,
    default_pipeline,
)


def test_clamp_bbox_orders_and_bounds_coordinates() -> None:
    # Reversed and out-of-bounds coordinates should be normalized.
    assert clamp_bbox([90, 80, 10, 20], (100, 100)) == [10, 20, 90, 80]
    assert clamp_bbox([-30, -10, 500, 500], (100, 100)) == [0, 0, 100, 100]


def test_is_valid_bbox_rejects_degenerate_regions() -> None:
    assert is_valid_bbox([0, 0, 10, 10]) is True
    assert is_valid_bbox([10, 10, 10, 20]) is False  # zero width
    assert is_valid_bbox([10, 10, 20, 10]) is False  # zero height


def test_infer_location_maps_center_body() -> None:
    assert infer_location([40, 40, 60, 60], (100, 100)) == "center_body"


def test_infer_location_maps_corner_regions() -> None:
    assert infer_location([0, 0, 10, 10], (100, 100)) == "front_left"
    assert infer_location([90, 90, 100, 100], (100, 100)) == "rear_right"


def test_pipeline_runs_end_to_end_in_mock_mode() -> None:
    pipeline = default_pipeline()  # no weights present -> mock mode
    image = Image.new("RGB", (200, 200), color=(120, 120, 120))
    result = pipeline.run(image)

    assert result["processing_mode"] == "mock"
    assert result["routing_decision"]
    assert "annotated_image_base64" in result
    for detection in result["damage_detections"]:
        x1, y1, x2, y2 = detection["bbox"]
        assert 0 <= x1 < x2 <= image.size[0]
        assert 0 <= y1 < y2 <= image.size[1]


def test_default_pipeline_honors_weight_path_env(monkeypatch) -> None:
    # Non-existent paths keep the pipeline in mock mode but prove the env
    # overrides are threaded through to the detector and classifier.
    monkeypatch.setenv("DETECTOR_WEIGHTS", "/tmp/does-not-exist-detector.pt")
    monkeypatch.setenv("SEVERITY_WEIGHTS", "/tmp/does-not-exist-severity.pth")
    pipeline = default_pipeline()
    assert pipeline.detector.model_path == "/tmp/does-not-exist-detector.pt"
    assert pipeline.classifier.model_path == "/tmp/does-not-exist-severity.pth"
    assert pipeline.detector.model is None
    assert pipeline.classifier.model is None
