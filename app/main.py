from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

model = joblib.load("models/model.pkl")

prediction_counter = Counter("predictions_total", "Total predictions made")

app = FastAPI(title="Heart Disease Predictor")

class Input(BaseModel):
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@app.post("/predict")
def predict(data: Input):
    df = pd.DataFrame([data.dict()])
    result = int(model.predict(df)[0])
    prediction_counter.inc()
    return {"prediction": result}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
