import torch
import torchvision


class MyCnn(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = torch.nn.Sequential(
      torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
      torch.nn.ReLU(),
      torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
      torch.nn.ReLU(),
      torch.nn.Dropout2d(p=0.5)
    )
    self.conv2 = torch.nn.Sequential(
      torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
      torch.nn.ReLU(),
      torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
      torch.nn.ReLU(),
      torch.nn.Dropout2d(p=0.5)
    )
    self.linear_relu_stack = torch.nn.Sequential(
      torch.nn.Flatten(),
      torch.nn.Linear(in_features=25600, out_features=128),
      torch.nn.ReLU(),
      torch.nn.Dropout(p=0.5),
      torch.nn.Linear(in_features=128, out_features=10)
    )

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    logits = self.linear_relu_stack(x)
    return logits


if __name__ == "__main__":

  device = "cuda" if torch.cuda.is_available() else "cpu"
  trn_ds = torchvision.datasets.MNIST(root="data", train=True,  download=True, transform=torchvision.transforms.ToTensor())
  tst_ds = torchvision.datasets.MNIST(root="data", train=False, download=True, transform=torchvision.transforms.ToTensor())
  trn_dl = torch.utils.data.DataLoader(trn_ds, batch_size=512)
  tst_dl = torch.utils.data.DataLoader(tst_ds, batch_size=512)


  myCnn = MyCnn().to(device)
  loss = torch.nn.CrossEntropyLoss()
  opt = torch.torch.optim.SGD(myCnn.parameters(), lr=1e-3)


  for t in range(3):
    for idx, (x, y) in enumerate(trn_dl):
      x, y = x.to(device), y.to(device)
      logits = myCnn(x)
      l = loss(logits, y)
      opt.zero_grad()
      l.backward()
      opt.step()
      if idx % 50 == 0: print(f"Epoch: {t}, Loss: {l.item()}")

    with torch.no_grad():
      correct = 0
      total = 0
      for x, y in tst_dl:
        x, y = x.to(device), y.to(device)
        logits = myCnn(x)
        _, pred = torch.max(logits, dim=1)
        total += y.size(0)
        correct += (pred == y).sum().item()
      print(f"Epoch: {t}, Accuracy: {correct/total}")
    torch.save(myCnn.state_dict(), f"./mnist-{t}.pt")