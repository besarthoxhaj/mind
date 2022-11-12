# Number Battle

Follow steps from last time to train a simple MLP model. Then, using the
canvas in `index.html` test your model with real user draw images.

```py
import cv2
import numpy as np
import base64

# Recieve base64 data
myData = 'data:...'
myData = myData.split(',')[1]

# Transform
npArr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
img = cv2.imdecode(npArr, cv2.IMREAD_COLOR)

# Convert 3 channel image (RGB) to 1 channel image (GRAY)
grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Resize to (28, 28)
grayImage = cv2.resize(grayImage, (28, 28), interpolation=cv2.INTER_LINEAR)

# Expand to numpy array dimenstion to (1, 28, 28)
img = np.expand_dims(grayImage, axis=0)
```

## Resources
1. PyTorch tutorial: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
