from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import mlflow
import pandas as pd
import os
from preprocessing import add_combined_feature

app = FastAPI(title="MLOps Microservice")

RUN_ID = "9c68a9d825b74493a68894489cc28505"
MODEL_URI = f"runs:/{RUN_ID}/model"
_model = None

def get_model():
    global _model
    if _model is None:
        _model = mlflow.pyfunc.load_model(MODEL_URI)
    return _model

class PredictionRequest(BaseModel):
    features: List[float]

@app.get("/health")
def health():
    model = get_model()
    return {"status": "ok" if model else "model not loaded"}

@app.post("/predict")
def predict(request: PredictionRequest):
    model = get_model()
    
    feature_names = [
        'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
        'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry',
        'mean fractal dimension', 'radius error', 'texture error', 'perimeter error',
        'area error', 'smoothness error', 'compactness error', 'concavity error',
        'concave points error', 'symmetry error', 'fractal dimension error', 'worst radius',
        'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
        'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry',
        'worst fractal dimension'
    ]
    input_df = pd.DataFrame([request.features], columns=feature_names)
    
    pred = model.predict(input_df)[0]
    return {"prediction": str(pred)}
