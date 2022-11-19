# Guess The Image

```py
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Download train data from open datasets.
train_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64
epochs = 5

# Create data loaders.
train_dataloader = DataLoader(train_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
```

```py
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
```

```py
model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
```

```py
def train(dataloader, model, loss_fn, optimizer):
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


def test(dataloader, model, loss_fn):
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
```

```py
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)

print("Done!")
```

```py
import matplotlib.pyplot as plt

plt.imshow(training_data.data[0], cmap='gray')
plt.title('%i' % training_data.targets[0])
plt.show()
```

```py
myNumber = training_data.data[0].unsqueeze(dim=0).float()
currPred = model(myNumber)
print('currPred', currPred.argmax(1))
```

```py
import matplotlib.pyplot as plt
import numpy as np
import base64
import cv2

myData = 'data:image/png;base64,iVB...'
myData = myData.split(',')[1]

npArr = np.frombuffer(base64.b64decode(myData), np.uint8)
img = cv2.imdecode(npArr, cv2.IMREAD_COLOR)

# Convert 3 channel image (RGB) to 1 channel image (GRAY)
grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Resize to (28, 28)
grayImage = cv2.resize(grayImage, (28, 28), interpolation=cv2.INTER_LINEAR)

# Expand to numpy array dimenstion to (1, 28, 28)
img = np.expand_dims(grayImage, axis=0)

plt.imshow(grayImage, cmap='gray')
plt.show()
```

## Resources
1. 3Blue1Brown: https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
2. PyTorch get started: https://pytorch.org/get-started/locally
3. Google Colab: https://colab.research.google.com
3. PyTorch tutorial: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
