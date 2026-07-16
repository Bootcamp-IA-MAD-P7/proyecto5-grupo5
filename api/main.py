import sys
import json
from pathlib import Path
from contextlib import asynccontextmanager

for p in [Path.cwd()] + list(Path.cwd().parents):
    if (p / "pyproject.toml").exists():
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
        break

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from joblib import load

from src.config import MODELS_DIR
from api.schemas import (
    CustomerData,
    HealthResponse,
    FeedbackRequest,
    IngestRequest,
    MetricsResponse,
)

from api.database import (
    save_prediction,
    save_feedback,
    save_ingested_data,
    get_metrics,
)


pipeline = None
model_name = "random_forest"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    pipeline_path = MODELS_DIR / f"{model_name}_pipeline.joblib"
    if not pipeline_path.exists():
        raise RuntimeError(f"Pipeline not found: {pipeline_path}")
    pipeline = load(pipeline_path)
    yield
    pipeline = None


app = FastAPI(
    title="Telco Customer Churn API",
    description="API para predicción de churn de clientes Telco",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model=model_name,
        model_loaded=pipeline is not None,
    )


@app.post("/predict")
def predict(data: CustomerData):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")
    raw = data.model_dump(by_alias=True)
    df = pd.DataFrame([raw])
    proba = pipeline.predict_proba(df)[0, 1]
    pred = int(proba >= 0.5)
    pred_id = save_prediction(
        input_data=json.dumps(raw),
        prediction=pred,
        probability=round(float(proba), 4),
        model_name=model_name,
    )
    return {"prediction": pred, "probability": round(float(proba), 4)}


@app.post("/predict/batch")
def predict_batch(data_list: list[CustomerData]):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")
    records = [d.model_dump(by_alias=True) for d in data_list]
    df = pd.DataFrame(records)
    probas = pipeline.predict_proba(df)[:, 1]
    preds = (probas >= 0.5).astype(int)
    results = []
    for raw, p, prob in zip(records, preds, probas):
        pred_id = save_prediction(
            input_data=json.dumps(raw),
            prediction=int(p),
            probability=round(float(prob), 4),
            model_name=model_name,
        )
        results.append(
            {"prediction": int(p), "probability": round(float(prob), 4)}
        )
    return results


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    save_feedback(
        prediction_id=req.prediction_id,
        correct=req.correct,
        true_label=req.true_label,
    )
    return {"status": "ok", "message": "Feedback saved"}


@app.post("/data/ingest")
def ingest_data(req: IngestRequest):
    save_ingested_data(data=json.dumps(req.data), true_label=req.true_label)
    ingested_count = len(req.data) if isinstance(req.data, list) else 1
    return {"status": "ok", "message": f"{ingested_count} records ingested"}


@app.get("/metrics", response_model=MetricsResponse)
def metrics():
    return get_metrics()
