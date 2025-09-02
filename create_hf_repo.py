import os
from dotenv import load_dotenv
from huggingface_hub import create_repo

load_dotenv()

username = "jhondoee"
repo_name = "distilbert-imdb-sentiment-analysis"
hf_token = os.getenv("HUGGING_FACE_TOKEN")

try:
    url = create_repo(
        repo_id=f"{username}/{repo_name}",
        token=hf_token,
        exist_ok=True,
    )
    print(f"Repository created or already exists: {url}")
except Exception as e:
    print(f"Error creating repository: {e}")
