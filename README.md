# MLOps - Sentiment Analysis API Project

This repository contains a complete MLOps workflow demonstrating how to fine-tune a sentiment analysis model, push it to the Hugging Face Hub, and serve it via a FastAPI microservice.

The final application is a **Sentiment Analysis API** that serves a `DistilBERT` model fine-tuned on the IMDB dataset.

**Model available on the Hugging Face Hub:**
[jhondoee/distilbert-imdb-sentiment-analysis](https://huggingface.co/jhondoee/distilbert-imdb-sentiment-analysis)

## Architecture

*   **`train_sentiment_model.py`**: A script that uses the `transformers` and `datasets` libraries to fine-tune a `distilbert-base-uncased` model. Upon completion, the trained model is automatically pushed to a dedicated repository on the Hugging Face Hub.
*   **`create_hf_repo.py`**: A utility script to programmatically create the repository on the Hub before training.
*   **`app.py`**: A FastAPI microservice that loads the fine-tuned model directly from the Hugging Face Hub using an access token. It exposes a `/sentiment` endpoint to analyze lists of texts.
*   **`.env`**: A file (ignored by Git) to securely store the Hugging Face access token.

## Setup and Execution Guide

Follow these steps to run the project from start to finish.

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### 2. Setup Environment
Create a `.env` file by copying the example file.

```bash
cp .env.example .env
```
Now, edit the `.env` file and paste your Hugging Face access token (you can create one in your [Hugging Face Settings](https://huggingface.co/settings/tokens)). It must have **write** permissions.

```text
# .env
HUGGING_FACE_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

### 3. Install Dependencies
It is recommended to use a virtual environment.
```bash
pip install -r requirements.txt
```

### 4. Create the Hugging Face Repo
This script creates the repository on the Hub where the model will be stored.
```bash
python create_hf_repo.py
```

### 5. Train and Push the Model
Run the training script. This will download the IMDB dataset, fine-tune the model, and upload it to your Hugging Face repository. This step will take a significant amount of time depending on your hardware.
```bash
python train_sentiment_model.py
```

### 6. Run the API
Once the model is successfully pushed to the Hub, you can start the API server. It will download the model on startup.
```bash
uvicorn app:app --reload
```

### 7. Test the API
Open a **new terminal** and send a test request using `curl`.
```bash
curl -X POST "http://127.0.0.1:8000/sentiment" \
-H "Content-Type: application/json" \
-d '{"texts": ["This movie is a masterpiece", "I really hated this film"]}'
```

You should receive a JSON response with the sentiment predictions:
```json
[
  {"label": "POSITIVE", "score": 0.99...},
  {"label": "NEGATIVE", "score": 0.99...}
]
```
