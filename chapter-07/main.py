from fastapi import FastAPI, Response
from pydantic import BaseModel
import torch
import utils

app = FastAPI(bind="0.0.0.0:8080")
model = torch.jit.load("script.pt")
model.eval()

store = []

class Item(BaseModel):
    base64: str
    value: int

@app.get("/")
def read_root():
    with open("index.html", "r") as f:
        html = f.read()
    return Response(content=html, media_type="text/html")

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}

@app.post("/predict")
def predict(item: Item):
    store.append(item.base64)
    numTensor = utils.baseStrToTensor(item.base64)
    res = model(numTensor).argmax(1).item()
    return {"prediction": res}

@app.get("/store")
def get_store():
    print(store)
    return {"store": store}
