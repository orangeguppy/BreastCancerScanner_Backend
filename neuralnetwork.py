# Command list to install flask:
# 1. Cd into the backend folder
# 2. Using wsl: python -m venv venv
# 3. Using wsl: source venv/bin/activate
# 4. Using wsl: pip3 install matplotlib
# 5. To install CORS: cd into the master folder, then pip install -U flask-cors

import torch
from torch import nn, save, load
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Define the Digit Classifier model
class TumourClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # First layer: Convolutional Layer
            nn.Conv2d(3, 32, (3,3)),
            nn.ReLU(), # Remove linearity

            # Second Layer
            nn.Conv2d(32, 64, (3,3)),
            nn.ReLU(), # Remove linearity

            # Last hidden layer
            nn.Conv2d(64, 64, (3,3)),
            nn.ReLU(), # Remove linearity

            # Pooling layer
            # Pool of square window of size=3, stride=2
            nn.MaxPool2d(3, stride=2),

            # Flatten
            nn.Flatten(),
            nn.Linear(746496, 2)
        )

    def forward(self, x):
        return self.model(x)