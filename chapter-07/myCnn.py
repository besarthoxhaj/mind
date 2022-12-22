from torch import nn

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # 2 conv blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(),
            nn.Dropout2d(p=0.4)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.Dropout2d(p=0.4)
        )

        # fully connected layers
        self.linear_relu_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(25600, 128),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        logits = self.linear_relu_stack(x)
        return logits