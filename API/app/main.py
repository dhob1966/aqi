from fastapi import FastAPI
from pydantic import BaseModel
from app.model.model import *
from app.model.model import __version__ as model_version
from pathlib import Path
import pickle
import pandas as pd
import numpy as np

app = FastAPI()

class predictors(BaseModel):
    lag1: float
    lag2: float
    lag3: float
    lag4: float
    lag5: float
    lag6: float
    lag7: float
    lag8: float
    lag9: float
    lag10: float
    lag11: float
    lag12: float
    lag13: float
    lag14: float
    lag15: float
    lag16: float
    lag17: float
    lag18: float
    lag19: float
    lag20: float

class PredictionOut(BaseModel):
    prediction: float
    air_quality: str

@app.get("/")
def home():
    return {"status": "OK", "model_version": model_version}

@app.post("/predict", response_model=PredictionOut)
def predict(item:predictors):
    df =pd.DataFrame([item.dict().values()], columns = item.dict().keys())
    prediction = model.predict(df)
    
    if prediction[0]<=50:
        bucket="good"
    elif 51 <= prediction[0] <=100:
        bucket='Satisfactory'
    elif 101<=prediction[0]<=200:
        bucket="Moderate"
    elif 201<=prediction[0]<=300:
        bucket='Poor'
    elif 301<=prediction[0]<=400:
        bucket='Very poor'
    elif prediction[0]>=401:
        bucket='Severe'
    else:
        bucket='ERROR'
    return {'prediction': prediction[0], 'air_quality' : bucket} 
