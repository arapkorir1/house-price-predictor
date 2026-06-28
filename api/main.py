# api/main.py
# Run with: uvicorn api.main:app --reload --port 8000

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Any, Optional
import time
import logging
import sys
sys.path.insert(0, '.')

from src.predict import predict_single, predict_batch, get_model_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title       = "House Price Predictor API",
    description = "Predict Ames Iowa residential house prices",
    version     = "1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class HouseFeatures(BaseModel):
    """
    Raw house features — send whatever you have.
    Missing features are filled with 0 automatically.
    All values should be already processed (encoded, scaled).
    For a quick test, send an empty dict: {}
    """
    data: dict[str, Any] = Field(
        default={},
        example={
            "OverallQual":  7,
            "GrLivArea":    1500,
            "GarageCars":   2,
            "TotalBsmtSF":  800,
            "FullBath":     2,
            "YearBuilt":    2000,
        }
    )


class BatchRequest(BaseModel):
    data: list[dict[str, Any]]


class PredictionResponse(BaseModel):
    predicted_price:           float
    predicted_price_formatted: str
    model:                     str
    latency_ms:                float


class BatchResponse(BaseModel):
    predictions:  list[float]
    formatted:    list[str]
    n_houses:     int
    latency_ms:   float


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "House Price Predictor API",
        "docs":    "/docs",
        "health":  "/health"
    }


@app.get("/health")
def health():
    """Health check — returns model status and performance metrics."""
    try:
        info = get_model_info()
        return {"status": "healthy", "model": info}
    except Exception as e:
        raise HTTPException(status_code=503,
                             detail=f"Model not ready: {str(e)}")


@app.get("/info")
def info():
    """Full model metadata."""
    return get_model_info()


@app.post("/predict", response_model=PredictionResponse)
def predict(request: HouseFeatures):
    """
    Predict price for a single house.

    Quick test with curl:
    curl -X POST http://localhost:8000/predict \\
         -H "Content-Type: application/json" \\
         -d '{"data": {"OverallQual": 7, "GrLivArea": 1500}}'
    """
    start = time.perf_counter()
    try:
        result  = predict_single(request.data)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=422,
                             detail=f"Prediction error: {str(e)}")

    latency = (time.perf_counter() - start) * 1000
    logger.info(f"Predicted: {result['predicted_price_formatted']}  "
                f"latency={latency:.1f}ms")

    return PredictionResponse(
        predicted_price           = result['predicted_price'],
        predicted_price_formatted = result['predicted_price_formatted'],
        model                     = result['model'],
        latency_ms                = round(latency, 2)
    )


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch_endpoint(request: BatchRequest):
    """Predict prices for multiple houses at once."""
    if len(request.data) > 10000:
        raise HTTPException(status_code=400,
                             detail="Batch limit is 10,000 houses")
    start = time.perf_counter()
    try:
        result_df = predict_batch(request.data)
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=422,
                             detail=f"Batch error: {str(e)}")

    latency = (time.perf_counter() - start) * 1000
    return BatchResponse(
        predictions = result_df['predicted_price'].tolist(),
        formatted   = result_df['predicted_price_formatted'].tolist(),
        n_houses    = len(request.data),
        latency_ms  = round(latency, 2)
    )


# ── Request logging middleware ────────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start    = time.perf_counter()
    response = await call_next(request)
    latency  = (time.perf_counter() - start) * 1000
    logger.info(f"{request.method} {request.url.path} "
                f"→ {response.status_code}  {latency:.1f}ms")
    return response
