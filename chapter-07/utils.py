import torch
import base64
import cv2
from io import BytesIO
from PIL import Image
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

def trn(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def tst(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def baseStrToTensor(strNum):
    myData = strNum.split(',')[1]
    npArr = np.frombuffer(base64.b64decode(myData), np.uint8)
    img = cv2.imdecode(npArr, cv2.IMREAD_COLOR)
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayImage = cv2.resize(grayImage, (28, 28), interpolation=cv2.INTER_LINEAR)
    npImg = np.expand_dims(grayImage, axis=0)
    npImgTensor = torch.tensor(npImg)
    return npImgTensor.unsqueeze(dim=0).float()
