import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

class CustomDenseNet201(nn.Module):
    def __init__(self, dropout_prob=0.5):
        super(CustomDenseNet201, self).__init__()

        # Load pre-trained DenseNet-201
        self.densenet = torch.hub.load('pytorch/vision:v0.10.0', 'densenet201', weights="DenseNet201_Weights.IMAGENET1K_V1")

        # Apply dropout to all convolutional layers
        for name, module in self.densenet.named_children():
            if isinstance(module, nn.Conv2d):
                setattr(self.densenet, name, nn.Sequential(module, nn.Dropout2d(p=dropout_prob)))
        
        # Modify the output layer for binary classification.
        # Make the final output layer have 2 nodes, each node represents a binary class
        in_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(in_features, 2)

    def forward(self, x):
        return self.densenet(x)

# Example usage:
dropout_prob = 0.5
model = CustomDenseNet201(dropout_prob=dropout_prob)