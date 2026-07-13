from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError

from src.pipeline import default_pipeline


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
    processing_mode: str
    annotated_image_base64: str


app = FastAPI(title="Vehicle Damage Assessment API", version="1.0.0")
pipeline = default_pipeline()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    content = await file.read()
    try:
        image = Image.open(BytesIO(content)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Unable to decode image.") from exc

    result = pipeline.run(image)
    return PredictionResponse(**result)
