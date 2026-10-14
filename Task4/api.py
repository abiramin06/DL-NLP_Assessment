from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import pipeline

app = FastAPI(title="Customer Feedback Sentiment API")

MODEL_PATH = "task4/saved_model"

sentiment_model = pipeline(
    "text-classification",
    model=MODEL_PATH,
    tokenizer=MODEL_PATH,
    device=-1
)


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1)


@app.get("/")
def home():
    return {"message": "Customer Feedback Sentiment API is running"}


@app.post("/predict")
def predict(request: PredictionRequest):
    text = request.text.strip()

    if not text:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=422,
            detail="Text cannot be empty"
        )

    result = sentiment_model(text, truncation=True)[0]

    label_map = {
        "LABEL_0": "Negative",
        "LABEL_1": "Positive"
    }

    label = label_map.get(result["label"], result["label"])

    return {
        "text": text,
        "label": label,
        "confidence": round(float(result["score"]), 4)
    }