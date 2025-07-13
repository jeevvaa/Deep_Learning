import torch
import torch.nn as nn
import torch.nn.functional as F

class GarmentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

_net = GarmentNet()
_net.load_state_dict(torch.load("garment_model/net_weights.pth", map_location="cpu"))
_net.eval()

def predict(images):
    """ images: B x 3 x 256 x 256 with values in (0, 1) """
    mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
    std  = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
    x = (images - mean) / std
    with torch.no_grad():
        return torch.argmax(_net(x), dim=1, keepdim=True)
