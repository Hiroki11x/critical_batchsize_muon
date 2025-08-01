import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Sequential(nn.Conv2d(
            3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.body = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(
            64), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Flatten(), nn.Linear(64 * 8 * 8, 128), nn.ReLU())
        self.head = nn.Linear(128, 10)

    def forward(self, x):
        x = self.embed(x)
        x = self.body(x)
        x = self.head(x)
        return x
