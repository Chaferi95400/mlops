import os
from typing import List
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

load_dotenv()

app = FastAPI(title="Sentiment Analysis API")

MODEL_NAME = "jhondoee/distilbert-imdb-sentiment-analysis"

# --- SECTION CORRIGÉE ---
# On retire l'argument 'use_auth_token' qui est obsolète.
# La bibliothèque trouvera le token dans l'environnement grâce à load_dotenv().
sentiment_pipeline = pipeline(
    "sentiment-analysis", 
    model=MODEL_NAME
)
# --- FIN DE LA SECTION CORRIGÉE ---

class SentimentRequest(BaseModel):
    texts: List[str]

class SentimentResponse(BaseModel):
    label: str
    score: float

@app.get("/")
def root():
    return {"message": "Sentiment Analysis API is running"}

@app.post("/sentiment", response_model=List[SentimentResponse])
def get_sentiment(request: SentimentRequest):
    results = sentiment_pipeline(request.texts)
    return results
