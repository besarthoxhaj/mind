# Number Battle

Follow steps from last time to train a simple MLP model. Then, using the
canvas in `index.html` test your model with real user draw images.

```py
import matplotlib.pyplot as plt
from torchvision import transforms
import cv2
from io import BytesIO
from PIL import Image
import numpy as np
import base64

def myImagePred(myData):

    myData = myData.split(',')[1]
    npArr = np.frombuffer(base64.b64decode(myData), np.uint8)
    img = cv2.imdecode(npArr, cv2.IMREAD_COLOR)

    # Convert 3 channel image (RGB) to 1 channel image (GRAY)
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize to (28, 28)
    grayImage = cv2.resize(grayImage, (28, 28), interpolation=cv2.INTER_LINEAR)

    # Expand to numpy array dimenstion to (1, 28, 28)
    npImg = np.expand_dims(grayImage, axis=0)

    plt.imshow(grayImage, cmap='gray')
    plt.show()

    npImgTensor = torch.tensor(npImg)
    model.eval()
    currPred = model(npImgTensor.unsqueeze(dim=0).float())
    print('currPred', currPred.argmax(1), currPred)
```

```py
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # 2 conv blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.Dropout2d(p=0.4)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.Dropout2d(p=0.4)
        )

        # fully connected layers
        self.linear_relu_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=25600, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(in_features=128, out_features=10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        logits = self.linear_relu_stack(x)
        return logits
```

```py
import matplotlib.pyplot as plt

plt.imshow(training_data.data[0], cmap='gray')
plt.title('%i' % training_data.targets[0])
plt.show()

foo = training_data.data[0].unsqueeze(dim=0)
myNumber = foo.unsqueeze(dim=0).float()
currPred = model(myNumber)
print('currPred', currPred.argmax(1))
```

## Resources
1. PyTorch tutorial: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
