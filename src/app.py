from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from pathlib import Path
import logging

from predict import load_model, predict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mood_detector_api")

# Global model state - loaded once at startup, reused across requests
model_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, clean up on shutdown."""
    logger.info("Loading model...")
    MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
    model_state["model"], model_state["stats"], model_state["device"] = load_model(
        model_dir=str(MODEL_DIR)
    )
    logger.info("Model loaded successfully.")
    yield
    model_state.clear()
    logger.info("Model unloaded.")

app = FastAPI(
    title="Mood Detector API",
    description="Emotion detection from facial images. Returns one of: Angry, Happy, Sad.",
    version="1.0.0",
    lifespan=lifespan,
)

# Response Models

class PredictionResponse(BaseModel):
    emotion: str | None
    confidence: str | None
    probabilities: dict[str, float] | None
    message: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

# Endpoints

@app.get("/", response_class=HTMLResponse)
def index():
    html_path = Path(__file__).resolve().parent / "static" / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))

@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Health check endpoint.
    In production this would be hit by a load balancer every few seconds
    to confirm the container is alive and the model is loaded.
    Returns 200 if healthy, which is what orchestrators like Kubernetes
    use to decide whether to send traffic to this pod.
    """
    return {
        "status": "healthy",
        "model_loaded": "model" in model_state,
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_emotion(file: UploadFile = File(...)):
    """
    Upload a face image and receive an emotion prediction.
    Accepts JPEG or PNG. Returns emotion label, confidence, and
    per-class probability breakdown.
    """
    # Validate content type before doing any work
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. Upload JPEG or PNG."
        )

    image_bytes = await file.read()

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result = predict(
            image_bytes=image_bytes,
            model=model_state["model"],
            stats=model_state["stats"],
            device=model_state["device"],
        )
    except ValueError as e:
        # predict() raises ValueError for undecodable images
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Inference failed. Check server logs.")

    return JSONResponse(content=result)