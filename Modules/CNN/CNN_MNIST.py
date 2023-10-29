import torch.nn as nn

class CNN_MNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 16, (5,5))
        self.c2 = nn.Conv2d(16, 24, (5,5))

        self.l1 = nn.Linear(384, 64)
        self.l2 = nn.Linear(64, 10)

        self.fl = nn.Flatten()
        self.mp = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.expand(-1, 3, -1, -1)
        x = self.relu(self.c1(x)); x = self.mp(x)
        x = self.relu(self.c2(x)); x = self.mp(x)
        x = self.fl(x) # N x 600

        x = self.relu(self.l1(x))
        return self.l2(x)