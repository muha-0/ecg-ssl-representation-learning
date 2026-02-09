import torch.nn as nn

class LinearProbe(nn.Module):
    def __init__(self, in_dim, n_classes=2):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, h):
        return self.fc(h)