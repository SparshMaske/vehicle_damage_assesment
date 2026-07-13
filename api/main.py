from io import BytesIO

from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError

from src.pipeline import default_pipeline
from src.reporting import build_text_report


class DamageDetectionResponse(BaseModel):
    type: str
    location: str
    bbox: list[int]
    severity: str
    confidence: float
    estimated_cost_range: str


class PredictionResponse(BaseModel):
    damage_detections: list[DamageDetectionResponse]
    overall_severity: str
    routing_decision: str
    estimated_cost_range: str
    estimate_note: str
    reasoning: str
    summary: str
    explanation_trace: list[str]
    review_flags: list[str]
    recommended_next_actions: list[str]
    reasoning_provider: str
    reasoning_mode: str
    processing_mode: str
    annotated_image_base64: str


app = FastAPI(title="Vehicle Damage Assessment API", version="1.0.0")
pipeline = default_pipeline()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


async def run_prediction(file: UploadFile) -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    content = await file.read()
    try:
        image = Image.open(BytesIO(content)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Unable to decode image.") from exc

    return pipeline.run(image)


@app.post("/predict", response_class=Response)
async def predict(file: UploadFile = File(...)) -> Response:
    result = await run_prediction(file)
    report = build_text_report(result)
    return Response(content=report, media_type="text/plain")


@app.post("/predict/structured", response_model=PredictionResponse)
async def predict_structured(file: UploadFile = File(...)) -> PredictionResponse:
    result = await run_prediction(file)
    return PredictionResponse(**result)
