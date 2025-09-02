from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import os
from preprocessing import add_combined_feature

app = FastAPI(title="MLOps Microservice")

class PredictionRequest(BaseModel):
    features: List[float]

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.joblib")
_model = None

def get_model():
    global _model
    if _model is None and os.path.exists(MODEL_PATH):
        _model = joblib.load(MODEL_PATH)
    return _model

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(request: PredictionRequest):
    model = get_model()
    if model is None:
        return {"error": "Model not found. Train and place at artifacts/model.joblib"}
    
    pred = model.predict([request.features])[0]
    return {"prediction": str(pred)}
