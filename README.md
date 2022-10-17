## Machine Mind

Repository about Machine Learning and Deep Learning.

```sh
$ git clone https://github.com/besarthoxhaj/mind.git
$ cd mind/
$ mkdir work
$ docker builder prune
$ docker build .
$ docker image ls
$ docker run --rm -p 8888:8888 -v "$PWD"/work:/home/jovyan/work {image}
# Entered start.sh with args: jupyter la
# ...
# http://127.0.0.1:8888/lab
```

![000](./screenshots/000.png)

## Jupyter Notebook

Once inside the Jupyter Notebook select "Python 3" as notebook. Then
write and execute a simple command to check everything is ok.

![001](./screenshots/001.png)

```py
import torch
x = torch.rand(5, 3)
print(x)
```

![002](./screenshots/002.png)

Great! Seem everything is working as expected. Let's try now to download
the MNIST dataset and run a CNN (Convolutional Neural Network).

```py
from torchvision import datasets
from torchvision.transforms import ToTensor

train_data = datasets.MNIST(
    root = 'data',
    train = True,
    transform = ToTensor(),
    download = True,
)

test_data = datasets.MNIST(
    root = 'data',
    train = False,
    transform = ToTensor()
)
```

![003](./screenshots/003.png)

If also the last step worked you are all set. Just follow resource
number 2 to get started with CNNs.

Or simply search for simple neural networks to do anything, the internet
it's full of them.

## Resources

1. PyTorch get started: https://pytorch.org/get-started/locally
2. PyTorch CNN tutorial: https://medium.com/@nutanbhogendrasharma/pytorch-convolutional-neural-network-with-mnist-dataset-4e8a4265e118
