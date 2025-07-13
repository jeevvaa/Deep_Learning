# compress_images.py

import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(64,128, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(128,256,4, 2, 1), nn.ReLU(True),
        )
        self.fc = nn.Linear(256*12*10, latent_dim)
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256*12*10)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256,128,4,2,1), nn.ReLU(True),
            nn.ConvTranspose2d(128,64, 4,2,1), nn.ReLU(True),
            nn.ConvTranspose2d(64,32,  4,2,1), nn.ReLU(True),
            nn.ConvTranspose2d(32,1,   4,2,1), nn.Sigmoid()
        )
    def forward(self, z):
        x = self.fc(z).view(z.size(0), 256, 12, 10)
        return self.deconv(x)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_enc = Encoder().to(device)
_dec = Decoder().to(device)

_enc.load_state_dict(torch.load('compression_model/encoder.pth', map_location=device))
_dec.load_state_dict(torch.load('compression_model/decoder.pth', map_location=device))
_enc.eval()
_dec.eval()

def encode(images: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return _enc(images.to(device)).cpu()

def decode(latents: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return _dec(latents.to(device)).cpu()
