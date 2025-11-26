import pandas as pd
import uvicorn
import warnings
warnings.filterwarnings("ignore")

from backend.predict import predict_new_data
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional


app = FastAPI()


class TitanicPassenger(BaseModel):
    Pclass: float
    Sex: str
    Fare: float
    Age: float
    Embarked: str
    SibSp: Optional[int] = None
    Parch: Optional[int] = None
    Ticket: Optional[str] = None
    Name: Optional[str] = None
    Cabin: Optional[str] = None
    

@app.get("/")
async def hello():
    return {"msg": "Hi there!"}
    
@app.post("/predict")
def predict(passenger: TitanicPassenger):
    model_path = "models/logistic_model.joblib"
    data = pd.DataFrame([dict(passenger)])
    predict = predict_new_data(new_data=data, model_path=model_path)
    return {"prediction": int(predict[0])}


# if __name__ == "__main__":
#     uvicorn.run("main:app", port=8000, reload=True)