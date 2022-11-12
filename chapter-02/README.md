# Number Battle

Follow steps from last time to train a simple MLP model. Then, using the
canvas in `index.html` test your model with real user draw images.

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
1. PyTorch tutorial: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
