import fastapi
import torch
import base64
import numpy
import cv2
import mnist


app = fastapi.FastAPI()


@app.on_event("startup")
async def startup_event():
  app.state.digit = mnist.MyCnn()
  app.state.digit.load_state_dict(torch.load("./mnist-2.pt"))
  app.state.nums = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
  app.state.digit.eval()


@app.on_event("shutdown")
async def startup_event():
  print("Shutting down")


@app.get("/")
def on_root():
  return { "message": "Hello App" }


@app.post("/one_number")
async def on_learn_to_search(request: fastapi.Request):
  rawPayload = (await request.json())["img"]
  npArr = numpy.frombuffer(base64.b64decode(rawPayload), numpy.uint8)
  img = cv2.imdecode(npArr, cv2.IMREAD_COLOR)
  grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  grayImage = cv2.resize(grayImage, (28, 28), interpolation=cv2.INTER_LINEAR)
  npImg = numpy.expand_dims(grayImage, axis=0)
  npImgTensor = torch.tensor(npImg)
  npImgTensor.unsqueeze(dim=0).float()
  with torch.no_grad():
    logits = app.state.digit(npImgTensor)
    _, pred = torch.max(logits, dim=1)
    print("Prediction:", pred)
  # take the preds and put them in a list of objects with
  # the number and the confidence return that list
