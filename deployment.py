from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel

app = FastAPI()

clf = load('iris.p')

class Prediction(BaseModel):
    data: int
    user: str

@app.get("/")
def hello_world():
    return {"message": "Hello World!"}

@app.get("/predict/", response_model=Prediction)
async def predict(req: Request):
    query_params = dict(req.query_params)
    data = [
        query_params['petallenght'],
        query_params['petalwidth'],
        query_params['sepallenght'],
        query_params['sepalwidth'],
    ]

    prediction = clf.predict(data)

    return {"data": prediction}
