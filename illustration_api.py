from fastapi import FastAPI
import uvicorn
from joblib import load
#import tensorflow

app = FastAPI(
    title="Labs30-StorySquad-DS-Team A",
    description="An API for the illustration score",
    version="0.1",
    docs_url="/"
)

with open("data/transfer_model.joblib", "rb") as file:
    model = load(file)

if __name__ == "__main__":
    uvicorn.run(app)