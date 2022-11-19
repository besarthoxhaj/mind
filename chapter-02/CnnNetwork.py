from torch import nn

class CnnNetwork(nn.Module):
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